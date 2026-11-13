#!/usr/bin/env python3
"""装甲板四关键点模型加载与推理的统一入口。"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np

from armor_core import decode_pose_output, preprocess, scale_detections

DEFAULT_NAMES = {0: "B1", 1: "B2", 2: "B3", 3: "R1", 4: "R2", 5: "R3"}


def read_names(path: Path) -> dict[int, str]:
    """读取 ONNX 元数据或 Ultralytics 模型中的类别名称。"""
    if path.suffix.lower() == ".onnx":
        try:
            import onnx

            model = onnx.load(str(path), load_external_data=False)
            for prop in model.metadata_props:
                if prop.key == "names":
                    value = ast.literal_eval(prop.value)
                    return {int(key): str(name) for key, name in value.items()}
        except Exception:
            pass
        return dict(DEFAULT_NAMES)

    from ultralytics import YOLO

    model = YOLO(str(path), task="pose")
    return {int(key): str(name) for key, name in model.names.items()}


class OnnxRuntimePredictor:
    def __init__(
        self,
        path: Path,
        names: dict[int, str],
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        threads: int = 0,
    ):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads > 0:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.names = names
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.kpt_shape = (4, 2)
        self.backend = f"ONNXRuntime {ort.__version__}"

    def predict(self, frame: np.ndarray):
        t0 = time.perf_counter()
        blob, meta = preprocess(frame, self.imgsz)
        t1 = time.perf_counter()
        pred = self.session.run(None, {self.input_name: blob})[0]
        t2 = time.perf_counter()
        detections = decode_pose_output(
            pred,
            self.names,
            self.conf,
            self.iou,
            self.max_det,
            self.kpt_shape,
        )
        detections = scale_detections(detections, meta)
        t3 = time.perf_counter()
        return detections, {
            "preprocess_ms": (t1 - t0) * 1000.0,
            "inference_ms": (t2 - t1) * 1000.0,
            "postprocess_ms": (t3 - t2) * 1000.0,
        }


class UltralyticsPredictor:
    def __init__(
        self,
        path: Path,
        names: dict[int, str],
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(path), task="pose")
        self.names = names
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.device = device
        self.kpt_shape = tuple(getattr(self.model.model, "kpt_shape", (4, 2)))
        self.backend = f"Ultralytics {getattr(self.model, 'ckpt_path', '')}".strip()

    def predict(self, frame: np.ndarray):
        t0 = time.perf_counter()
        result = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
            save=False,
        )[0]
        t1 = time.perf_counter()

        detections = []
        boxes = result.boxes
        keypoints = getattr(result, "keypoints", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            kxy = (
                keypoints.xy.cpu().numpy()
                if keypoints is not None and keypoints.xy is not None
                else None
            )
            for index in range(len(xyxy)):
                class_id = int(classes[index])
                detections.append(
                    {
                        "box": [float(value) for value in xyxy[index]],
                        "conf": float(confidences[index]),
                        "cls": class_id,
                        "name": self.names.get(class_id, str(class_id)),
                        "kxy": (
                            None
                            if kxy is None
                            else [[float(x), float(y)] for x, y in kxy[index]]
                        ),
                        "kconf": None,
                    }
                )

        speed = dict(result.speed or {})
        timings = {
            "preprocess_ms": float(speed.get("preprocess", 0.0)),
            "inference_ms": float(speed.get("inference", 0.0)),
            "postprocess_ms": float(speed.get("postprocess", 0.0)),
        }
        if not any(timings.values()):
            timings["total_ms"] = (t1 - t0) * 1000.0
        return detections, timings


def build_predictor(
    model_path: Path,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    engine: str = "auto",
    device: str = "cpu",
    threads: int = 0,
):
    names = read_names(model_path)
    selected_engine = engine
    if selected_engine == "auto":
        selected_engine = (
            "onnxruntime" if model_path.suffix.lower() == ".onnx" else "ultralytics"
        )

    common = {
        "names": names,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
    }
    if selected_engine == "onnxruntime":
        if model_path.suffix.lower() != ".onnx":
            raise ValueError("--engine onnxruntime 只能加载 .onnx 模型")
        predictor = OnnxRuntimePredictor(model_path, threads=threads, **common)
    elif selected_engine == "ultralytics":
        predictor = UltralyticsPredictor(model_path, device=device, **common)
    else:
        raise ValueError(f"未知推理引擎: {selected_engine}")

    if predictor.kpt_shape != (4, 2):
        raise ValueError(
            f"本作业要求模型输出 4 个二维关键点，实际为 {predictor.kpt_shape}"
        )
    return predictor, names
