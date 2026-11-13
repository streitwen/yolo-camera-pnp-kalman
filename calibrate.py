#!/usr/bin/env python3
"""使用棋盘格图片估计相机内参、畸变系数和重投影误差。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class CalibrationComputation:
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms_reprojection_error_px: float
    per_view_errors_px: list[float]


@dataclass(frozen=True)
class DetectedBoard:
    path: Path
    image_size: tuple[int, int]
    corners: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenCV 棋盘格相机标定",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-dir", required=True, help="标定图片目录，可递归查找")
    parser.add_argument(
        "--pattern",
        nargs=2,
        type=int,
        default=(9, 6),
        metavar=("COLS", "ROWS"),
        help="棋盘内角点列数、行数；不是方格总数",
    )
    parser.add_argument("--square-mm", type=float, required=True, help="每个棋盘方格实际边长，单位 mm")
    parser.add_argument(
        "--output",
        default="configs/camera_params.json",
        help="标定参数输出路径",
    )
    parser.add_argument(
        "--preview-dir",
        default="results/calibration_previews",
        help="角点检测预览图输出目录",
    )
    parser.add_argument("--no-previews", action="store_true", help="不保存检测预览图")
    parser.add_argument(
        "--detection-mode",
        choices=["robust", "fast"],
        default="robust",
        help="robust 使用穷举和高精度搜索；fast 适合分辨率很高、已确认棋盘清晰时",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="只使用指定分辨率；默认自动选择成功检测最多的一组分辨率",
    )
    parser.add_argument("--min-images", type=int, default=6, help="最少需要多少张有效图片")
    parser.add_argument(
        "--max-rms",
        type=float,
        default=2.0,
        help="超过该 RMS 时打印警告；不会删除结果",
    )
    return parser


def find_chessboard(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
    *,
    detection_mode: str = "robust",
) -> np.ndarray | None:
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    if detection_mode == "robust" and hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
        flags |= cv2.CALIB_CB_EXHAUSTIVE
    if detection_mode == "robust" and hasattr(cv2, "CALIB_CB_ACCURACY"):
        flags |= cv2.CALIB_CB_ACCURACY

    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
        if found:
            return np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)

    fallback_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, fallback_flags)
    if not found:
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        60,
        0.001,
    )
    corners = cv2.cornerSubPix(
        gray,
        corners.astype(np.float32),
        (11, 11),
        (-1, -1),
        criteria,
    )
    return np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)


def collect_boards(
    image_dir: Path,
    pattern_size: tuple[int, int],
    *,
    preview_dir: Path | None,
    detection_mode: str = "robust",
) -> tuple[list[DetectedBoard], list[tuple[Path, str]]]:
    files = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    boards: list[DetectedBoard] = []
    failures: list[tuple[Path, str]] = []
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            failures.append((path, "图片无法读取"))
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = find_chessboard(gray, pattern_size, detection_mode=detection_mode)
        if corners is None:
            failures.append((path, "未检测到完整棋盘内角点"))
            continue

        height, width = gray.shape[:2]
        boards.append(
            DetectedBoard(
                path=path,
                image_size=(width, height),
                corners=corners,
            )
        )
        if preview_dir is not None:
            preview = image.copy()
            # OpenCV 5 的 drawChessboardCorners 对部分 findChessboardCornersSB
            # 返回的 float64 数组会触发断言，绘制前显式转换为连续 float32。
            preview_corners = np.ascontiguousarray(corners, dtype=np.float32)
            cv2.drawChessboardCorners(
                preview,
                pattern_size,
                preview_corners,
                True,
            )
            cv2.putText(
                preview,
                f"detected {pattern_size[0]}x{pattern_size[1]}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(preview_dir / f"{path.stem}_corners.jpg"), preview)
    return boards, failures


def make_object_points(
    pattern_size: tuple[int, int],
    square_size_mm: float,
) -> np.ndarray:
    columns, rows = pattern_size
    if columns < 2 or rows < 2:
        raise ValueError("棋盘内角点行列数都必须至少为 2")
    if square_size_mm <= 0:
        raise ValueError("--square-mm 必须大于 0")

    points = np.zeros((columns * rows, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points *= float(square_size_mm)
    return points


def calibrate_from_correspondences(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
) -> CalibrationComputation:
    if len(object_points) != len(image_points) or len(object_points) < 3:
        raise ValueError("标定至少需要 3 组一一对应的角点数据")
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("image_size 非法")

    objects = [
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 3)
        for points in object_points
    ]
    images = [
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        for points in image_points
    ]
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6,
    )
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objects,
        images,
        image_size,
        None,
        None,
        criteria=criteria,
    )

    per_view_errors: list[float] = []
    for index, (obj, img, rvec, tvec) in enumerate(
        zip(objects, images, rvecs, tvecs)
    ):
        projected, _ = cv2.projectPoints(
            obj,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        error = cv2.norm(img, projected, cv2.NORM_L2) / np.sqrt(len(projected))
        if not np.isfinite(error):
            raise ValueError(f"第 {index + 1} 张图片重投影误差不是有限值")
        per_view_errors.append(float(error))

    return CalibrationComputation(
        image_size=(int(image_size[0]), int(image_size[1])),
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        dist_coeffs=np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
        rms_reprojection_error_px=float(rms),
        per_view_errors_px=per_view_errors,
    )


def select_size_group(
    boards: Sequence[DetectedBoard],
    requested_size: tuple[int, int] | None,
) -> list[DetectedBoard]:
    if requested_size is not None:
        selected = [board for board in boards if board.image_size == requested_size]
        if not selected:
            sizes = sorted({board.image_size for board in boards})
            raise ValueError(
                f"没有成功图片匹配分辨率 {requested_size}；可用分辨率: {sizes}"
            )
        return selected

    groups: dict[tuple[int, int], list[DetectedBoard]] = {}
    for board in boards:
        groups.setdefault(board.image_size, []).append(board)
    best_size = max(groups, key=lambda size: (len(groups[size]), size[0] * size[1]))
    selected = groups[best_size]
    if len(groups) > 1:
        summary = ", ".join(
            f"{size[0]}x{size[1]}: {len(items)} 张"
            for size, items in sorted(groups.items())
        )
        print(f"[警告] 检测结果包含多个分辨率（{summary}）")
        print(
            f"[警告] 相机内参与分辨率绑定，本次只使用 {best_size[0]}x{best_size[1]} "
            f"的 {len(selected)} 张图片"
        )
    return selected


def write_calibration(
    output: Path,
    *,
    computation: CalibrationComputation,
    pattern_size: tuple[int, int],
    square_size_mm: float,
    boards: Sequence[DetectedBoard],
) -> dict:
    payload = {
        "image_size": list(computation.image_size),
        "camera_matrix": computation.camera_matrix.tolist(),
        "dist_coeffs": computation.dist_coeffs.tolist(),
        "rms_reprojection_error_px": computation.rms_reprojection_error_px,
        "per_view_reprojection_error_px": computation.per_view_errors_px,
        "pattern_inner_corners": list(pattern_size),
        "square_size_mm": float(square_size_mm),
        "image_count": len(boards),
        "image_files": [str(board.path) for board in boards],
        "opencv_version": cv2.__version__,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_dir = Path(args.image_dir).expanduser()
    pattern_size = (int(args.pattern[0]), int(args.pattern[1]))
    requested_size = (
        (int(args.image_size[0]), int(args.image_size[1]))
        if args.image_size
        else None
    )
    if not image_dir.is_dir():
        print(f"[错误] 图片目录不存在: {image_dir}", file=sys.stderr)
        return 2
    if args.square_mm <= 0:
        print("[错误] --square-mm 必须大于 0", file=sys.stderr)
        return 2

    preview_dir = None if args.no_previews else Path(args.preview_dir)
    print(f"[信息] 棋盘内角点: {pattern_size[0]} x {pattern_size[1]}")
    print(f"[信息] 方格边长: {args.square_mm:.3f} mm")
    boards, failures = collect_boards(
        image_dir,
        pattern_size,
        preview_dir=preview_dir,
        detection_mode=args.detection_mode,
    )
    print(f"[信息] 找到图片: {len(boards) + len(failures)}，成功: {len(boards)}，失败: {len(failures)}")
    for path, reason in failures:
        print(f"  跳过 {path.name}: {reason}")
    if not boards:
        print("[错误] 没有任何图片检测到完整棋盘", file=sys.stderr)
        return 2

    try:
        selected = select_size_group(boards, requested_size)
    except ValueError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    if len(selected) < args.min_images:
        print(
            f"[错误] 有效图片只有 {len(selected)} 张，少于 --min-images={args.min_images}",
            file=sys.stderr,
        )
        return 2

    object_template = make_object_points(pattern_size, args.square_mm)
    objects = [object_template.copy() for _ in selected]
    images = [board.corners for board in selected]
    computation = calibrate_from_correspondences(
        objects,
        images,
        selected[0].image_size,
    )

    output = Path(args.output).expanduser()
    payload = write_calibration(
        output,
        computation=computation,
        pattern_size=pattern_size,
        square_size_mm=args.square_mm,
        boards=selected,
    )

    print("\n================ 标定结果 ================")
    print(f"图像分辨率       : {computation.image_size[0]}x{computation.image_size[1]}")
    print(f"有效图片         : {len(selected)}")
    print(f"RMS 重投影误差   : {computation.rms_reprojection_error_px:.4f} px")
    print(
        f"单张误差范围     : {min(computation.per_view_errors_px):.4f} "
        f"~ {max(computation.per_view_errors_px):.4f} px"
    )
    print("相机内参 K        :")
    print(np.array2string(computation.camera_matrix, precision=3, suppress_small=True))
    print(f"畸变系数 D        : {computation.dist_coeffs}")
    print(f"标定参数文件     : {output}")
    if preview_dir is not None:
        print(f"检测预览目录     : {preview_dir}")
    if computation.rms_reprojection_error_px > args.max_rms:
        print(
            f"[警告] RMS 超过 {args.max_rms:.3f} px，建议补拍清晰、不同角度且齐全的棋盘图",
            file=sys.stderr,
        )
    print("[提醒] 必须使用同一台相机、同一图像分辨率和同一像素格式进行后续推理")
    print("==========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
