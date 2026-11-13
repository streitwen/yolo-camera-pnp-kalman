#!/usr/bin/env python3
"""相机参数加载、装甲板角点对应与 solvePnP 位姿计算。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

CORNER_ALIASES = {
    "tl": "TL",
    "top-left": "TL",
    "top_left": "TL",
    "左上": "TL",
    "tr": "TR",
    "top-right": "TR",
    "top_right": "TR",
    "右上": "TR",
    "br": "BR",
    "bottom-right": "BR",
    "bottom_right": "BR",
    "右下": "BR",
    "bl": "BL",
    "bottom-left": "BL",
    "bottom_left": "BL",
    "左下": "BL",
}

PNP_METHODS = {
    "ippe": cv2.SOLVEPNP_IPPE,
    "iterative": cv2.SOLVEPNP_ITERATIVE,
    "ap3p": cv2.SOLVEPNP_AP3P,
}


@dataclass(frozen=True)
class CameraCalibration:
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms_reprojection_error_px: float | None = None
    source: str | None = None

    @classmethod
    def load(cls, path: Path) -> "CameraCalibration":
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            image_size = tuple(int(value) for value in data["image_size"])
            camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
            dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"标定文件字段不完整: {path}") from exc

        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError(f"image_size 非法: {image_size}")
        if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
            raise ValueError("camera_matrix 必须是有限的 3x3 矩阵")
        if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
            raise ValueError("相机焦距 fx/fy 必须大于 0")
        if dist_coeffs.size not in (4, 5, 8, 12, 14):
            raise ValueError(f"dist_coeffs 长度异常: {dist_coeffs.size}")

        return cls(
            image_size=(image_size[0], image_size[1]),
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rms_reprojection_error_px=(
                float(data["rms_reprojection_error_px"])
                if data.get("rms_reprojection_error_px") is not None
                else None
            ),
            source=str(path),
        )

    def to_dict(self) -> dict:
        return {
            "image_size": list(self.image_size),
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "rms_reprojection_error_px": self.rms_reprojection_error_px,
            "source": self.source,
        }


@dataclass(frozen=True)
class PoseResult:
    rvec: np.ndarray
    tvec: np.ndarray
    xyz_mm: np.ndarray
    distance_mm: float
    reprojection_error_px: float
    projected_points: np.ndarray
    object_points: np.ndarray


def parse_corner_order(value: str | Iterable[str]) -> tuple[str, ...]:
    """把 YOLO 关键点 1..4 的实体角顺序转换为 ``TL/TR/BR/BL``。"""
    items = value.split(",") if isinstance(value, str) else list(value)
    normalized: list[str] = []
    for item in items:
        key = str(item).strip().lower()
        if key not in CORNER_ALIASES:
            raise ValueError(
                f"未知角点名称 {item!r}，可用 TL、TR、BR、BL"
            )
        normalized.append(CORNER_ALIASES[key])

    if len(normalized) != 4 or set(normalized) != {"TL", "TR", "BR", "BL"}:
        raise ValueError(
            "角点顺序必须恰好包含 TL、TR、BR、BL 各一次，例如 TL,TR,BR,BL"
        )
    return tuple(normalized)


def corner_order_text(order: Sequence[str]) -> str:
    return ",".join(order)


def object_points_for_order(
    width_mm: float,
    height_mm: float,
    corner_order: Sequence[str],
) -> np.ndarray:
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("装甲板宽高必须大于 0")

    points = {
        "TL": (-width_mm / 2.0, -height_mm / 2.0, 0.0),
        "TR": (width_mm / 2.0, -height_mm / 2.0, 0.0),
        "BR": (width_mm / 2.0, height_mm / 2.0, 0.0),
        "BL": (-width_mm / 2.0, height_mm / 2.0, 0.0),
    }
    return np.asarray([points[name] for name in corner_order], dtype=np.float64)


def valid_image_points(detection: dict, image_size: tuple[int, int] | None = None) -> np.ndarray:
    keypoints = detection.get("kxy")
    if keypoints is None or len(keypoints) != 4:
        raise ValueError("模型没有返回 4 个关键点")
    points = np.asarray(keypoints, dtype=np.float64)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise ValueError("关键点必须为有限的 4x2 坐标")
    if image_size is not None:
        width, height = image_size
        if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width):
            raise ValueError("关键点 x 坐标超出图像")
        if np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
            raise ValueError("关键点 y 坐标超出图像")
    if cv2.contourArea(points.astype(np.float32)) < 4.0:
        raise ValueError("四个关键点面积过小，无法稳定求解 PnP")
    return points


def select_primary_detection(
    detections: Sequence[dict],
    image_size: tuple[int, int],
    previous_box: Sequence[float] | None = None,
) -> tuple[dict | None, str | None]:
    """选择主目标；有历史框时优先选离上一帧最近的，避免目标来回切换。"""
    valid: list[dict] = []
    for detection in detections:
        try:
            valid_image_points(detection, image_size)
        except ValueError:
            continue
        valid.append(detection)
    if not valid:
        if not detections:
            return None, "NO_DETECTION"
        return None, "NO_VALID_KEYPOINTS"

    if previous_box is not None:
        previous_center = np.array(
            [
                (float(previous_box[0]) + float(previous_box[2])) / 2.0,
                (float(previous_box[1]) + float(previous_box[3])) / 2.0,
            ],
            dtype=np.float64,
        )
        diagonal = float(np.hypot(image_size[0], image_size[1]))

        def continuity_score(detection: dict) -> float:
            box = detection["box"]
            center = np.array(
                [
                    (float(box[0]) + float(box[2])) / 2.0,
                    (float(box[1]) + float(box[3])) / 2.0,
                ],
                dtype=np.float64,
            )
            distance = float(np.linalg.norm(center - previous_center)) / max(diagonal, 1.0)
            # 距离是主项，置信度只用于非常接近时的次序判断。
            return distance - 0.01 * float(detection.get("conf", 0.0))

        return min(valid, key=continuity_score), None

    return max(valid, key=lambda item: float(item.get("conf", 0.0))), None


def solve_armor_pose(
    image_points: np.ndarray,
    calibration: CameraCalibration,
    *,
    armor_width_mm: float,
    armor_height_mm: float,
    corner_order: Sequence[str],
    method: str = "ippe",
) -> PoseResult:
    if method not in PNP_METHODS:
        raise ValueError(f"不支持的 PnP 方法 {method!r}，可选 {list(PNP_METHODS)}")

    object_points = object_points_for_order(
        armor_width_mm, armor_height_mm, corner_order
    )
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    if hasattr(cv2, "solvePnPGeneric"):
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points,
            image_points,
            calibration.camera_matrix,
            calibration.dist_coeffs,
            flags=PNP_METHODS[method],
        )
        for index in range(int(count)):
            rvec = np.asarray(rvecs[index], dtype=np.float64).reshape(3)
            tvec = np.asarray(tvecs[index], dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
                continue
            if float(tvec[2]) <= 0:
                continue
            projected, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                calibration.camera_matrix,
                calibration.dist_coeffs,
            )
            projected = projected.reshape(-1, 2)
            errors = np.linalg.norm(
                projected - image_points.reshape(-1, 2), axis=1
            )
            candidates.append((rvec, tvec, projected, float(np.mean(errors))))

    if not candidates:
        # 某些 OpenCV 构建对特定求解器不支持 Generic 返回，保留标准接口兜底。
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            calibration.camera_matrix,
            calibration.dist_coeffs,
            flags=PNP_METHODS[method],
        )
        if not ok:
            raise ValueError("cv2.solvePnP 求解失败")
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            raise ValueError("PnP 返回了非有限位姿")
        if float(tvec[2]) <= 0:
            raise ValueError(
                f"PnP 深度非正: z={float(tvec[2]):.2f} mm；请检查角点顺序和角点坐标"
            )
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            calibration.camera_matrix,
            calibration.dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - image_points.reshape(-1, 2), axis=1)
        candidates.append((rvec, tvec, projected, float(np.mean(errors))))

    rvec, tvec, projected, reprojection_error = min(
        candidates, key=lambda item: item[3]
    )
    distance = float(np.linalg.norm(tvec))
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("PnP 距离计算异常")

    return PoseResult(
        rvec=rvec,
        tvec=tvec,
        xyz_mm=tvec.copy(),
        distance_mm=distance,
        reprojection_error_px=reprojection_error,
        projected_points=projected,
        object_points=object_points,
    )


def project_camera_point(
    point_xyz_mm: Sequence[float],
    calibration: CameraCalibration,
    rvec: np.ndarray | None = None,
    tvec: np.ndarray | None = None,
) -> np.ndarray:
    """把相机坐标系三维点投影到图像；rvec/tvec 非空时按装甲板坐标系解释。"""
    point = np.asarray(point_xyz_mm, dtype=np.float64).reshape(1, 1, 3)
    if rvec is None:
        rotation = np.zeros((3, 1), dtype=np.float64)
    else:
        rotation = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    if tvec is None:
        translation = np.zeros((3, 1), dtype=np.float64)
    else:
        translation = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    projected, _ = cv2.projectPoints(
        point,
        rotation,
        translation,
        calibration.camera_matrix,
        calibration.dist_coeffs,
    )
    return projected.reshape(2)
