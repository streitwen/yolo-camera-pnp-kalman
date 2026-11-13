#!/usr/bin/env python3
"""相机/视频 → YOLO 四关键点 → PnP → 卡尔曼滤波与预测。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from armor_core import auto_style, draw_detections, put_text
from armor_predictor import build_predictor
from kalman_tracker import ConstantVelocityKalman, KalmanResult
from pose_math import (
    CameraCalibration,
    PoseResult,
    corner_order_text,
    parse_corner_order,
    project_camera_point,
    select_primary_detection,
    solve_armor_pose,
    valid_image_points,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class FramePacket:
    image: np.ndarray
    index: int
    timestamp_s: float
    metadata: dict[str, Any]


class FrameSource:
    description = "unknown"
    suggested_fps = 30.0
    total_frames = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False

    def open(self) -> None:
        raise NotImplementedError

    def read(self) -> FramePacket | None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class VideoOrImageSource(FrameSource):
    def __init__(self, source: Path, *, max_frames: int = 0):
        self.source = source
        self.max_frames = int(max_frames)
        self.cap: cv2.VideoCapture | None = None
        self.image_mode = source.suffix.lower() in IMAGE_SUFFIXES
        self._single_image: np.ndarray | None = None
        self._frame_index = 0
        self._started = 0.0

    def open(self) -> None:
        if not self.source.is_file():
            raise FileNotFoundError(f"输入文件不存在: {self.source}")
        if self.image_mode:
            image = cv2.imread(str(self.source), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取图片: {self.source}")
            self._single_image = image
            self.suggested_fps = 1.0
            self.total_frames = 1
            self.description = f"image {self.source}"
            return

        self.cap = cv2.VideoCapture(str(self.source))
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {self.source}")
        self.suggested_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._started = time.perf_counter()
        self.description = f"video {self.source}"

    def read(self) -> FramePacket | None:
        if self.max_frames and self._frame_index >= self.max_frames:
            return None
        if self.image_mode:
            if self._single_image is None or self._frame_index > 0:
                return None
            image = self._single_image
            packet = FramePacket(image, 0, 0.0, {"input_kind": "image"})
            self._frame_index += 1
            return packet

        assert self.cap is not None
        ok, image = self.cap.read()
        if not ok:
            return None
        index = self._frame_index
        self._frame_index += 1
        # 文件视频使用媒体时间轴，保证离线处理的卡尔曼 dt 与真实帧率一致。
        timestamp_s = index / max(self.suggested_fps, 1e-9)
        return FramePacket(
            image=image,
            index=index,
            timestamp_s=timestamp_s,
            metadata={"input_kind": "video", "media_timestamp_s": timestamp_s},
        )

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class OpenCvCameraSource(FrameSource):
    def __init__(self, camera_index: int, *, max_frames: int = 0):
        self.camera_index = int(camera_index)
        self.max_frames = int(max_frames)
        self.cap: cv2.VideoCapture | None = None
        self._frame_index = 0
        self._started = 0.0

    def open(self) -> None:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.camera_index, backend)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise ValueError(f"无法打开 OpenCV 相机 {self.camera_index}")
        self.suggested_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._started = time.perf_counter()
        self.description = f"opencv camera {self.camera_index}"

    def read(self) -> FramePacket | None:
        if self.max_frames and self._frame_index >= self.max_frames:
            return None
        assert self.cap is not None
        ok, image = self.cap.read()
        if not ok:
            return None
        index = self._frame_index
        self._frame_index += 1
        timestamp_s = time.perf_counter() - self._started
        return FramePacket(
            image=image,
            index=index,
            timestamp_s=timestamp_s,
            metadata={"input_kind": "opencv_camera"},
        )

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class MvsCameraSource(FrameSource):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.camera = None
        self._frame_index = 0
        self._started = 0.0
        self._timeout_count = 0

    def open(self) -> None:
        try:
            from mvs_camera import CameraConfig, MvsCamera
        except ImportError as exc:
            raise RuntimeError("无法导入 mvs_camera；请检查项目文件完整性") from exc

        config = CameraConfig(
            serial=self.args.serial or None,
            index=int(self.args.camera_index),
            exposure_us=float(self.args.exposure_us),
            gain_db=float(self.args.gain_db),
            pixel_format=str(self.args.pixel_format),
            roi=tuple(self.args.roi) if self.args.roi else None,
            frame_rate=self.args.camera_fps,
            timeout_ms=int(self.args.camera_timeout_ms),
        )
        self.camera = MvsCamera(config).open()
        self.camera.start()
        self.suggested_fps = float(self.args.output_fps)
        self._started = time.perf_counter()
        info = self.camera.camera_info
        self.description = (
            f"MVS {info.get('model', '')} serial={info.get('serial', '')}"
        ).strip()

    def read(self) -> FramePacket | None:
        if self.args.max_frames and self._frame_index >= self.args.max_frames:
            return None
        assert self.camera is not None
        while True:
            frame = self.camera.grab_frame()
            if frame is not None:
                self._timeout_count = 0
                index = self._frame_index
                self._frame_index += 1
                timestamp_s = max(0.0, frame.received_at - self._started)
                return FramePacket(
                    image=frame.image_bgr,
                    index=index,
                    timestamp_s=timestamp_s,
                    metadata={
                        **frame.metadata(),
                        "input_kind": "mvs_camera",
                        "camera_model": self.camera.camera_info.get("model"),
                        "camera_serial": self.camera.camera_info.get("serial"),
                        "firmware": self.camera.camera_info.get("firmware"),
                    },
                )
            self._timeout_count += 1
            if self._timeout_count > 10:
                raise TimeoutError("MVS 相机连续 10 次取流超时")

    def close(self) -> None:
        if self.camera is not None:
            self.camera.close()
            self.camera = None


def load_config_defaults(config_path: Path) -> dict[str, Any]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    mapping = {
        "model": "model",
        "calibration": "calibration",
        "conf": "conf",
        "iou": "iou",
        "imgsz": "imgsz",
        "armor_width_mm": "armor_width_mm",
        "armor_height_mm": "armor_height_mm",
        "corner_order": "corner_order",
        "pnp_method": "pnp_method",
        "max_reprojection_error_px": "max_reprojection_error_px",
        "kalman_process_noise": "process_noise",
        "kalman_measurement_noise_mm": "measurement_noise_mm",
        "kalman_gate_threshold": "gate_threshold",
        "kalman_jump_reset_mm": "jump_reset_mm",
        "prediction_seconds": "prediction_seconds",
        "target_lost_reset_s": "target_lost_reset_s",
    }
    return {
        mapping[key]: value
        for key, value in data.items()
        if key in mapping
    }


def build_parser(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    # Do not let the early ``--config`` parser reinterpret ``--conf`` as an
    # abbreviation.  The current CLI has both options and Python 3.14 keeps
    # abbreviation matching enabled unless it is disabled explicitly.
    pre_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre_parser.add_argument("--config", default="")
    known, _ = pre_parser.parse_known_args(argv)

    config_defaults: dict[str, Any] = {}
    if known.config:
        config_path = Path(known.config).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        config_defaults = load_config_defaults(config_path)

    parser = argparse.ArgumentParser(
        description="工业相机/视频装甲板 PnP 位姿与卡尔曼滤波",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=known.config, help="JSON 默认参数文件")

    group = parser.add_argument_group("输入与输出")
    group.add_argument(
        "--source",
        default="mvs",
        help="视频/图片路径、mvs（海康工业相机）或 opencv:0（普通摄像头）",
    )
    group.add_argument("--model", default="models/best.pt", help="YOLO 权重路径")
    group.add_argument("--calibration", default="configs/camera_params.json", help="相机标定 JSON")
    group.add_argument("--output", default="results/demo.mp4", help="结果视频路径")
    group.add_argument("--csv", default="", help="逐帧 CSV；为空时由输出视频名派生")
    group.add_argument("--summary", default="", help="汇总 JSON；为空时由输出视频名派生")
    group.add_argument("--representative", default="", help="代表截图；为空时由输出视频名派生")
    group.add_argument("--max-frames", type=int, default=0, help="最多处理帧数，0 表示不限")
    group.add_argument("--out-scale", type=float, default=0.5, help="输出缩放；0.5 表示 4K→1080p")
    group.add_argument("--output-fps", type=float, default=20.0, help="相机实拍结果视频帧率")
    group.add_argument("--codec", default="mp4v", help="OpenCV 视频编码器")
    group.add_argument("--display", action="store_true", help="实时显示；按 q 退出")
    group.add_argument("--preview-only", action="store_true", help="只验证检测框和关键点，不做 PnP")
    group.add_argument("--quiet", action="store_true", help="不打印逐帧进度")

    group = parser.add_argument_group("模型推理")
    group.add_argument("--engine", default="auto", choices=["auto", "onnxruntime", "ultralytics"])
    group.add_argument("--conf", type=float, default=0.25)
    group.add_argument("--iou", type=float, default=0.7)
    group.add_argument("--imgsz", type=int, default=640)
    group.add_argument("--max-det", type=int, default=30)
    group.add_argument("--device", default="cpu", help="Ultralytics 设备：cpu 或 0")
    group.add_argument("--threads", type=int, default=0, help="ONNX Runtime CPU 线程数")

    group = parser.add_argument_group("PnP")
    group.add_argument("--armor-width-mm", type=float, default=224.0, help="装甲板实体宽度")
    group.add_argument("--armor-height-mm", type=float, default=54.0, help="装甲板实体高度")
    group.add_argument(
        "--corner-order",
        default="TL,BL,BR,TR",
        help="YOLO 关键点 1,2,3,4 对应的实体角，例如 TL,TR,BR,BL",
    )
    group.add_argument("--pnp-method", default="ippe", choices=["ippe", "iterative", "ap3p"])
    group.add_argument(
        "--max-reprojection-error-px",
        type=float,
        default=20.0,
        help="超过该重投影误差时拒绝该帧位姿",
    )

    group = parser.add_argument_group("卡尔曼滤波")
    group.add_argument("--process-noise", type=float, default=800.0, help="加速度噪声 mm/s²")
    group.add_argument("--measurement-noise-mm", type=float, default=25.0, help="PnP 测量噪声")
    group.add_argument("--gate-threshold", type=float, default=25.0, help="马氏距离平方门限")
    group.add_argument("--jump-reset-mm", type=float, default=1000.0, help="明显跳变的复位门限")
    group.add_argument("--reset-after-rejections", type=int, default=3)
    group.add_argument("--target-lost-reset-s", type=float, default=1.0, help="丢测超时后清除航迹")
    group.add_argument("--prediction-seconds", type=float, default=0.2, help="未来预测时间")

    group = parser.add_argument_group("MVS 工业相机")
    group.add_argument("--serial", default="", help="相机序列号；为空时按索引选相机")
    group.add_argument("--camera-index", type=int, default=0)
    group.add_argument("--exposure-us", type=float, default=10000.0)
    group.add_argument("--gain-db", type=float, default=0.0)
    group.add_argument("--pixel-format", default="BayerRG8", choices=["BayerRG8", "BGR8", "Mono8"])
    group.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    group.add_argument("--camera-fps", type=float, default=None)
    group.add_argument("--camera-timeout-ms", type=int, default=1000)
    # set_defaults 必须放在 add_argument 之后，否则同 dest 的参数默认值会覆盖配置。
    parser.set_defaults(**config_defaults)
    return parser


def create_source(args: argparse.Namespace) -> FrameSource:
    source = str(args.source).strip()
    if source.lower() == "mvs":
        return MvsCameraSource(args)
    if source.lower().startswith("opencv:"):
        try:
            index = int(source.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError("OpenCV 相机格式应为 opencv:0") from exc
        return OpenCvCameraSource(index, max_frames=args.max_frames)
    return VideoOrImageSource(Path(source).expanduser(), max_frames=args.max_frames)


def fourcc(code: str) -> int:
    if hasattr(cv2, "VideoWriter_fourcc"):
        return cv2.VideoWriter_fourcc(*code)
    return cv2.VideoWriter.fourcc(*code)


def open_writer(path: Path, codec: str, fps: float, size: tuple[int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), fourcc(codec), float(fps), size)
    if not writer.isOpened():
        writer.release()
        return None
    return writer


def scaled_size(width: int, height: int, scale: float) -> tuple[int, int]:
    if scale <= 0:
        raise ValueError("--out-scale 必须大于 0")
    if abs(scale - 1.0) < 1e-9:
        return width, height
    return (
        max(2, int(width * scale) // 2 * 2),
        max(2, int(height * scale) // 2 * 2),
    )


def draw_status(
    frame: np.ndarray,
    *,
    style: dict,
    pose: PoseResult | None,
    kalman: KalmanResult | None,
    status: str,
    prediction_seconds: float,
    class_name: str | None,
    confidence: float | None,
) -> None:
    lines = [
        f"STATUS {status}",
        (
            f"TARGET {class_name} conf={confidence:.3f}"
            if class_name is not None and confidence is not None
            else "TARGET none"
        ),
    ]
    if pose is not None:
        lines.append(
            f"RAW XYZ {pose.xyz_mm[0]:.1f} {pose.xyz_mm[1]:.1f} {pose.xyz_mm[2]:.1f} mm"
        )
        lines.append(f"RAW DIST {pose.distance_mm:.1f} mm  REPROJ {pose.reprojection_error_px:.2f} px")
    if kalman is not None and kalman.initialized:
        lines.append(
            f"KF XYZ {kalman.position[0]:.1f} {kalman.position[1]:.1f} {kalman.position[2]:.1f} mm"
        )
        lines.append(
            f"KF V {kalman.velocity[0]:.1f} {kalman.velocity[1]:.1f} {kalman.velocity[2]:.1f} mm/s"
        )
        lines.append(
            f"PRED +{prediction_seconds:.2f}s "
            f"{kalman.predicted_future[0]:.1f} {kalman.predicted_future[1]:.1f} "
            f"{kalman.predicted_future[2]:.1f} mm"
        )

    scale = style["fscale"]
    thickness = style["tth"]
    line_height = max(20, int(32 * scale))
    padding = max(8, int(12 * scale))
    text_width = 0
    for line in lines:
        (width, _), _ = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        text_width = max(text_width, width)
    panel_width = min(frame.shape[1] - 2 * padding, text_width + 2 * padding)
    panel_height = min(
        frame.shape[0] - 2 * padding,
        line_height * len(lines) + padding,
    )
    overlay = frame.copy()
    cv2.rectangle(overlay, (padding, padding), (padding + panel_width, padding + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0.0, frame)
    for row, line in enumerate(lines):
        y = padding * 2 + (row + 1) * line_height - int(6 * scale)
        color = (0, 255, 0) if row == 0 and status == "TRACKING" else (255, 255, 255)
        put_text(frame, line, (padding * 2, y), scale, thickness, color)


def draw_pose(
    frame: np.ndarray,
    calibration: CameraCalibration,
    pose: PoseResult,
    kalman: KalmanResult | None,
    *,
    armor_width_mm: float,
    style: dict,
) -> None:
    axis_length = max(20.0, float(armor_width_mm) * 0.35)
    cv2.drawFrameAxes(
        frame,
        calibration.camera_matrix,
        calibration.dist_coeffs,
        pose.rvec,
        pose.tvec,
        axis_length,
        max(2, style["lw"]),
    )

    raw_center = project_camera_point(pose.xyz_mm, calibration)
    radius = max(4, style["radius"] + 2)
    cv2.drawMarker(
        frame,
        tuple(np.round(raw_center).astype(int)),
        (0, 255, 255),
        cv2.MARKER_CROSS,
        radius * 3,
        max(2, style["lw"]),
        cv2.LINE_AA,
    )

    if kalman is None or not kalman.initialized:
        return
    filtered_center = project_camera_point(kalman.position, calibration)
    future_center = project_camera_point(kalman.predicted_future, calibration)
    cv2.circle(
        frame,
        tuple(np.round(filtered_center).astype(int)),
        radius,
        (0, 255, 0),
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(
        frame,
        tuple(np.round(future_center).astype(int)),
        radius,
        (255, 0, 255),
        max(2, style["lw"]),
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        tuple(np.round(filtered_center).astype(int)),
        tuple(np.round(future_center).astype(int)),
        (255, 0, 255),
        max(2, style["lw"]),
        cv2.LINE_AA,
    )
    put_text(
        frame,
        "PRED",
        tuple((np.round(future_center) + np.array([radius + 3, -radius - 3])).astype(int)),
        style["fscale"] * 0.8,
        max(1, style["tth"] - 1),
        (255, 0, 255),
    )


def derived_path(output: Path, suffix: str) -> Path:
    return output.parent / f"{output.stem}{suffix}"


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def vector_or_blank(value: np.ndarray | None) -> list[float | str]:
    if value is None:
        return ["", "", ""]
    return [
        float(value[0]) if np.isfinite(value[0]) else "",
        float(value[1]) if np.isfinite(value[1]) else "",
        float(value[2]) if np.isfinite(value[2]) else "",
    ]


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    import csv

    columns = [
        "frame",
        "time_s",
        "detected",
        "pose_ok",
        "status",
        "class_name",
        "confidence",
        "raw_x_mm",
        "raw_y_mm",
        "raw_z_mm",
        "filtered_x_mm",
        "filtered_y_mm",
        "filtered_z_mm",
        "velocity_x_mm_s",
        "velocity_y_mm_s",
        "velocity_z_mm_s",
        "predicted_x_mm",
        "predicted_y_mm",
        "predicted_z_mm",
        "distance_mm",
        "reprojection_error_px",
        "kp1_x",
        "kp1_y",
        "kp2_x",
        "kp2_y",
        "kp3_x",
        "kp3_y",
        "kp4_x",
        "kp4_y",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})


def summarize(args, source, frame_count, rows, class_counts, class_conf, timings) -> dict:
    pose_rows = [row for row in rows if row["pose_ok"]]
    kf_rows = [row for row in rows if row["filtered_x_mm"] != ""]
    reprojections = [
        float(row["reprojection_error_px"])
        for row in pose_rows
        if row["reprojection_error_px"] != ""
    ]
    elapsed = float(timings["elapsed_s"])
    return {
        "input": {
            "source": str(args.source),
            "description": source.description,
            "total_frames_reported": source.total_frames,
            "processed_frames": frame_count,
        },
        "model": {
            "path": str(args.model),
            "engine": args.engine,
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "class_counts": dict(class_counts),
            "average_confidence": {
                name: class_conf[name] / count
                for name, count in class_counts.items()
            },
        },
        "pnp": {
            "armor_width_mm": args.armor_width_mm,
            "armor_height_mm": args.armor_height_mm,
            "corner_order": args.corner_order,
            "method": args.pnp_method,
            "pose_ok_frames": len(pose_rows),
            "pose_success_rate": len(pose_rows) / max(1, frame_count),
            "reprojection_mean_px": (
                float(np.mean(reprojections)) if reprojections else None
            ),
            "reprojection_max_px": (
                float(np.max(reprojections)) if reprojections else None
            ),
        },
        "kalman": {
            "state": "[x,y,z,vx,vy,vz]",
            "process_noise_accel_mm_s2": args.process_noise,
            "measurement_noise_mm": args.measurement_noise_mm,
            "gate_threshold": args.gate_threshold,
            "jump_reset_mm": args.jump_reset_mm,
            "prediction_seconds": args.prediction_seconds,
            "filtered_frames": len(kf_rows),
        },
        "runtime": {
            "elapsed_s": elapsed,
            "processed_fps": frame_count / max(elapsed, 1e-9),
            "avg_model_preprocess_ms": timings["preprocess"] / max(1, frame_count),
            "avg_model_inference_ms": timings["inference"] / max(1, frame_count),
            "avg_model_postprocess_ms": timings["postprocess"] / max(1, frame_count),
        },
        "notes": [
            "三维坐标为相机坐标系，单位 mm；X 向右，Y 向下，Z 向前。",
            "卡尔曼漏检帧只做预测，不使用伪造测量。",
            "真实演示必须使用连续相机帧或视频帧产生的 PnP 测量。",
        ],
    }


def run(args: argparse.Namespace) -> int:
    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        print(f"[错误] 找不到模型: {model_path}", file=sys.stderr)
        return 2

    corner_order = parse_corner_order(args.corner_order)
    calibration = None
    if not args.preview_only:
        calibration_path = Path(args.calibration).expanduser()
        if not calibration_path.is_file():
            print(f"[错误] 找不到相机标定: {calibration_path}", file=sys.stderr)
            print("先运行 calibrate.py，或加 --preview-only 仅检查视频、模型和角点。", file=sys.stderr)
            return 2
        calibration = CameraCalibration.load(calibration_path)

    try:
        predictor, names = build_predictor(
            model_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            engine=args.engine,
            device=args.device,
            threads=args.threads,
        )
    except Exception as exc:
        print(f"[错误] 模型加载失败: {exc}", file=sys.stderr)
        return 2

    source = create_source(args)
    output = Path(args.output).expanduser()
    csv_path = Path(args.csv).expanduser() if args.csv else derived_path(output, "_tracking.csv")
    summary_path = (
        Path(args.summary).expanduser() if args.summary else derived_path(output, "_summary.json")
    )
    representative_path = (
        Path(args.representative).expanduser()
        if args.representative
        else derived_path(output, "_representative.jpg")
    )

    print(f"[信息] 模型: {model_path} | 后端: {predictor.backend}")
    print(f"[信息] 类别: {names} | 关键点: {predictor.kpt_shape}")
    if calibration is not None:
        print(
            f"[信息] 标定: {calibration.image_size[0]}x{calibration.image_size[1]} "
            f"RMS={calibration.rms_reprojection_error_px}"
        )
    print(
        f"[信息] PnP: {args.armor_width_mm:.1f}x{args.armor_height_mm:.1f} mm "
        f"corner={corner_order_text(corner_order)} method={args.pnp_method}"
    )
    print(f"[信息] 输入: {source.description if getattr(source, 'description', None) else args.source}")

    kalman = ConstantVelocityKalman(
        process_noise_accel=args.process_noise,
        measurement_noise_mm=args.measurement_noise_mm,
        gate_threshold=args.gate_threshold,
        jump_reset_mm=args.jump_reset_mm,
        reset_after_rejections=args.reset_after_rejections,
    )

    rows: list[dict] = []
    class_counts: Counter[str] = Counter()
    class_conf: dict[str, float] = {}
    timing_totals = {
        "preprocess": 0.0,
        "inference": 0.0,
        "postprocess": 0.0,
    }
    writer: cv2.VideoWriter | None = None
    output_size: tuple[int, int] | None = None
    source_fps = 30.0
    frame_count = 0
    detected_frames = 0
    pose_frames = 0
    resets = 0
    best_representative: tuple[float, np.ndarray] | None = None
    previous_primary_box: list[float] | None = None
    display_enabled = bool(args.display)
    started = time.perf_counter()

    try:
        source.open()
        source_fps = (
            float(source.suggested_fps)
            if getattr(source, "suggested_fps", 0)
            else float(args.output_fps)
        )
        if str(args.source).lower() == "mvs" or str(args.source).lower().startswith("opencv:"):
            source_fps = float(args.output_fps)

        while True:
            packet = source.read()
            if packet is None:
                break
            frame = packet.image.copy()
            height, width = frame.shape[:2]
            if calibration is not None and calibration.image_size != (width, height):
                raise ValueError(
                    f"标定分辨率 {calibration.image_size[0]}x{calibration.image_size[1]} "
                    f"与输入帧 {width}x{height} 不一致；请用当前相机重新标定"
                )

            style = auto_style(width, height)
            detections, timing = predictor.predict(frame)
            for key in timing_totals:
                timing_totals[key] += float(timing.get(f"{key}_ms", 0.0))
            draw_detections(frame, detections, style, connect=True, show_index=True)

            primary, selection_reason = select_primary_detection(
                detections,
                (width, height),
                previous_box=previous_primary_box,
            )
            class_name = primary["name"] if primary is not None else None
            confidence = float(primary["conf"]) if primary is not None else None
            pose: PoseResult | None = None
            image_points: np.ndarray | None = None
            status = "NO_DETECTION"
            kalman_result: KalmanResult | None = None

            if detections:
                detected_frames += 1
                for detection in detections:
                    name = str(detection["name"])
                    class_counts[name] += 1
                    class_conf[name] = class_conf.get(name, 0.0) + float(detection["conf"])
            if primary is not None:
                class_name = str(primary["name"])
                confidence = float(primary["conf"])
                previous_primary_box = [float(value) for value in primary["box"]]

            if args.preview_only:
                status = "PREVIEW_DETECTION" if primary is not None else "NO_VALID_KEYPOINTS"
            elif primary is not None and calibration is not None:
                try:
                    image_points = valid_image_points(primary, (width, height))
                    pose = solve_armor_pose(
                        image_points,
                        calibration,
                        armor_width_mm=args.armor_width_mm,
                        armor_height_mm=args.armor_height_mm,
                        corner_order=corner_order,
                        method=args.pnp_method,
                    )
                    if pose.reprojection_error_px > args.max_reprojection_error_px:
                        raise ValueError(
                            f"重投影误差 {pose.reprojection_error_px:.2f} px "
                            f"超过 {args.max_reprojection_error_px:.2f} px"
                        )
                    kalman_result = kalman.update(
                        pose.xyz_mm,
                        packet.timestamp_s,
                        prediction_seconds=args.prediction_seconds,
                    )
                    pose_frames += 1
                    resets += int(kalman_result.reset)
                    status = "TRACKING" if kalman_result.updated else "KF_REJECTED"
                    draw_pose(
                        frame,
                        calibration,
                        pose,
                        kalman_result,
                        armor_width_mm=args.armor_width_mm,
                        style=style,
                    )
                except (ValueError, cv2.error) as exc:
                    print(f"[警告] 第 {packet.index} 帧 PnP 被拒绝: {exc}")
                    pose = None
                    status = "PNP_REJECTED"
                    kalman_result = kalman.predict_to(packet.timestamp_s)
            else:
                kalman_result = kalman.predict_to(packet.timestamp_s)
                if kalman.initialized and (
                    kalman_result.miss_duration_s > args.target_lost_reset_s
                ):
                    kalman.clear()
                    kalman_result = None
                    previous_primary_box = None
                    status = "TARGET_LOST_RESET"
                    resets += 1
                elif selection_reason:
                    status = selection_reason

            draw_status(
                frame,
                style=style,
                pose=pose,
                kalman=kalman_result,
                status=status,
                prediction_seconds=args.prediction_seconds,
                class_name=class_name,
                confidence=confidence,
            )

            if output_size is None:
                output_size = scaled_size(width, height, args.out_scale)
                writer = open_writer(output, args.codec, source_fps, output_size)
                if writer is None:
                    raise RuntimeError(
                        f"无法创建输出视频 {output}，请检查编码器 {args.codec!r}"
                    )
                print(
                    f"[信息] 输出: {output} | fps={source_fps:.2f} | "
                    f"{output_size[0]}x{output_size[1]} | codec={args.codec}"
                )

            output_frame = (
                frame
                if (width, height) == output_size
                else cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
            )
            assert writer is not None
            writer.write(output_frame)

            if primary is not None:
                score = float(primary["conf"])
                if best_representative is None or score > best_representative[0]:
                    best_representative = (score, output_frame.copy())

            keypoints = primary.get("kxy") if primary is not None else None
            kp_values = [""] * 8
            if keypoints and len(keypoints) == 4:
                for index, (x, y) in enumerate(keypoints):
                    kp_values[index * 2] = float(x)
                    kp_values[index * 2 + 1] = float(y)
            raw_xyz = vector_or_blank(pose.xyz_mm if pose is not None else None)
            filtered_xyz = vector_or_blank(
                kalman_result.position
                if kalman_result is not None and kalman_result.initialized
                else None
            )
            velocity = vector_or_blank(
                kalman_result.velocity
                if kalman_result is not None and kalman_result.initialized
                else None
            )
            predicted = vector_or_blank(
                kalman_result.predicted_future
                if kalman_result is not None and kalman_result.initialized
                else None
            )
            rows.append(
                {
                    "frame": packet.index,
                    "time_s": f"{packet.timestamp_s:.6f}",
                    "detected": int(primary is not None),
                    "pose_ok": int(pose is not None),
                    "status": status,
                    "class_name": class_name or "",
                    "confidence": "" if confidence is None else f"{confidence:.6f}",
                    "raw_x_mm": raw_xyz[0],
                    "raw_y_mm": raw_xyz[1],
                    "raw_z_mm": raw_xyz[2],
                    "filtered_x_mm": filtered_xyz[0],
                    "filtered_y_mm": filtered_xyz[1],
                    "filtered_z_mm": filtered_xyz[2],
                    "velocity_x_mm_s": velocity[0],
                    "velocity_y_mm_s": velocity[1],
                    "velocity_z_mm_s": velocity[2],
                    "predicted_x_mm": predicted[0],
                    "predicted_y_mm": predicted[1],
                    "predicted_z_mm": predicted[2],
                    "distance_mm": (
                        finite_or_none(pose.distance_mm) if pose is not None else ""
                    ),
                    "reprojection_error_px": (
                        finite_or_none(pose.reprojection_error_px)
                        if pose is not None
                        else ""
                    ),
                    "kp1_x": kp_values[0],
                    "kp1_y": kp_values[1],
                    "kp2_x": kp_values[2],
                    "kp2_y": kp_values[3],
                    "kp3_x": kp_values[4],
                    "kp3_y": kp_values[5],
                    "kp4_x": kp_values[6],
                    "kp4_y": kp_values[7],
                }
            )
            frame_count += 1

            if display_enabled:
                try:
                    cv2.imshow("Armor PnP + Kalman (q to quit)", output_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as exc:
                    print(f"[警告] 当前环境无法显示窗口，已关闭显示: {exc}")
                    display_enabled = False

            if not args.quiet and (frame_count % 30 == 0):
                print(
                    f"\r[进度] {frame_count}/{source.total_frames or '?'} 帧 | "
                    f"检测 {detected_frames} | PnP {pose_frames}",
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n[信息] 用户中断，正在保存已完成的结果")
    except Exception as exc:
        print(f"[错误] 运行失败: {exc}", file=sys.stderr)
        return 2
    finally:
        if writer is not None:
            writer.release()
        source.close()
        if display_enabled:
            cv2.destroyAllWindows()

    if frame_count == 0:
        print("[错误] 没有读取到任何帧", file=sys.stderr)
        return 2

    write_csv(csv_path, rows)
    if best_representative is not None:
        representative_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(representative_path),
            best_representative[1],
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )

    elapsed = time.perf_counter() - started
    summary = summarize(
        args,
        source,
        frame_count,
        rows,
        class_counts,
        class_conf,
        {
            **timing_totals,
            "elapsed_s": elapsed,
        },
    )
    summary["kalman"]["tracker_resets"] = resets
    summary["outputs"] = {
        "video": str(output),
        "csv": str(csv_path),
        "summary": str(summary_path),
        "representative": str(representative_path) if best_representative else None,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n================ PnP / Kalman 汇总 ================")
    print(f"输入 / 后端      : {source.description} / {predictor.backend}")
    print(f"处理帧数         : {frame_count}")
    print(f"含检测帧数       : {detected_frames} ({detected_frames / frame_count * 100:.1f}%)")
    if not args.preview_only:
        print(f"PnP 成功帧数     : {pose_frames} ({pose_frames / frame_count * 100:.1f}%)")
        print(f"滤波器复位次数   : {resets}")
        if summary["pnp"]["reprojection_mean_px"] is not None:
            print(f"平均重投影误差   : {summary['pnp']['reprojection_mean_px']:.3f} px")
    print(f"平均推理耗时     : {timing_totals['inference'] / frame_count:.2f} ms/帧")
    print(f"结果视频         : {output}")
    print(f"逐帧 CSV         : {csv_path}")
    print(f"汇总 JSON        : {summary_path}")
    if best_representative is not None:
        print(f"代表截图         : {representative_path}")
    else:
        print("[警告] 没有保存代表截图：全片未检测到有效目标")
    print("===================================================")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser(argv).parse_args(argv)
    except FileNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    if args.armor_width_mm <= 0 or args.armor_height_mm <= 0:
        print("[错误] 装甲板宽高必须大于 0", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
