#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""装甲板关键点推理的公共模块：前处理、ONNX 解码、NMS、画图。

PT / ONNX / TorchScript 三条推理路径共用这里的预处理与绘制代码，
保证对比实验里除了"推理引擎"以外，其它变量完全一致。
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# 类别索引 -> 颜色（BGR）
PALETTE = [
    (56, 56, 255),
    (0, 165, 255),
    (0, 200, 0),
    (255, 120, 0),
    (200, 0, 200),
    (0, 220, 220),
]

# 4 个关键点按序号着色：1 红、2 绿、3 蓝、4 黄
KPT_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
]


# --------------------------------------------------------------------------- #
# 前处理：letterbox 与归一化
# --------------------------------------------------------------------------- #
def letterbox(img: np.ndarray, new_shape: int = 640, color: int = 114):
    """把整帧等比缩放后居中填充成 (new_shape, new_shape)。

    返回 (处理后图像, 缩放比例 gain, 左右填充 dw, 上下填充 dh)。
    与 Ultralytics 的训练/推理填充方式保持一致：只缩不放（scaleup=True 时小图也放大）。
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = (new_shape - new_w) / 2
    dh = (new_shape - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(color, color, color))
    return img, r, dw, dh


def preprocess(frame: np.ndarray, imgsz: int = 640):
    """BGR 帧 -> NCHW float32 归一化张量（batch=1），并返回还原坐标所需的参数。"""
    padded, gain, dw, dh = letterbox(frame, imgsz)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    # 一次转换 + 原地归一化，避免再分配一份同样大小的临时数组
    blob = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    blob *= np.float32(1.0 / 255.0)
    blob = blob[None]
    meta = {"gain": gain, "pad": (dw, dh), "imgsz": imgsz,
            "orig_shape": frame.shape[:2]}
    return blob, meta


def scale_detections(dets: list[dict], meta: dict) -> list[dict]:
    """把网络输入尺度(640)下的坐标还原到原始视频分辨率。"""
    gain = meta["gain"]
    dw, dh = meta["pad"]
    h, w = meta["orig_shape"]
    for det in dets:
        x1, y1, x2, y2 = det["box"]
        det["box"] = [
            float(np.clip((x1 - dw) / gain, 0, w - 1)),
            float(np.clip((y1 - dh) / gain, 0, h - 1)),
            float(np.clip((x2 - dw) / gain, 0, w - 1)),
            float(np.clip((y2 - dh) / gain, 0, h - 1)),
        ]
        if det.get("kxy") is not None:
            det["kxy"] = [[float(np.clip((x - dw) / gain, 0, w - 1)),
                           float(np.clip((y - dh) / gain, 0, h - 1))]
                          for x, y in det["kxy"]]
    return dets


# --------------------------------------------------------------------------- #
# 后处理：置信度筛选 + 按类别 NMS
# --------------------------------------------------------------------------- #
def nms_class_aware(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                    iou_thr: float) -> np.ndarray:
    """按类别分别做 NMS，返回保留的下标（按置信度从高到低）。

    与 Ultralytics 的 non_max_suppression 行为一致：不同类别之间不互相抑制。
    """
    keep_all = []
    for cls_id in np.unique(classes):
        idx = np.where(classes == cls_id)[0]
        order = idx[np.argsort(-scores[idx])]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
            order = rest[iou <= iou_thr]
        keep_all.extend(keep)
    keep_all = np.array(sorted(keep_all, key=lambda i: -scores[i]), dtype=int)
    return keep_all


def decode_pose_output(pred: np.ndarray, names: dict[int, str], conf: float = 0.25,
                       iou: float = 0.7, max_det: int = 30,
                       kpt_shape: tuple[int, int] = (4, 2)) -> list[dict]:
    """解码 YOLO pose 的原始输出。

    输入 pred 形状为 (1, 4 + nc + nkpt*2, 8400)，其中：
        [0:4]        -> cx, cy, w, h
        [4:4+nc]     -> 各人类别分数（模型内部已做 sigmoid）
        [4+nc:]      -> 每个关键点的 x, y（已换算到 640 输入图像的像素坐标）
    返回的坐标仍在 640 输入尺度，需再调用 scale_detections 还原。
    """
    pred = np.asarray(pred)
    if pred.ndim == 3:
        pred = pred[0]
    pred = pred.T  # (8400, 18)

    nc = len(names)
    nk = int(kpt_shape[0])          # 关键点个数（本模型为 4）
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:4 + nc]
    kpts = pred[:, 4 + nc:4 + nc + nk * 2]
    if kpts.shape[1] != nk * 2:
        raise ValueError(f"输出通道数 {pred.shape[1]} 与类别数 {nc}、关键点数 {nk} 不匹配")

    cls_ids = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(len(cls_ids)), cls_ids]
    mask = scores >= conf
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]
    kpts = kpts[mask]

    cx, cy, w, h = boxes_xywh.T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep = nms_class_aware(boxes, scores, cls_ids, iou)[:max_det]

    dets = []
    for i in keep:
        xy = kpts[i].reshape(-1, 2)
        dets.append({
            "box": [float(v) for v in boxes[i]],
            "conf": float(scores[i]),
            "cls": int(cls_ids[i]),
            "name": names.get(int(cls_ids[i]), str(int(cls_ids[i]))),
            "kxy": [[float(x), float(y)] for x, y in xy],
            # 本模型的 ONNX 输出只有关键点坐标，没有单独的关键点置信度通道
            "kconf": None,
        })
    return dets


# --------------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------------- #
def auto_style(width: int, height: int, line_width: int = 0, kpt_radius: int = 0) -> dict:
    base = min(width, height)
    return {
        "lw": line_width if line_width > 0 else max(1, round(base / 480)),
        "radius": kpt_radius if kpt_radius > 0 else max(3, round(base / 160)),
        "fscale": max(0.4, base / 1100.0),
        "tth": max(1, round(base / 700.0 * 2)),
    }


def put_text(frame, text: str, org, fscale: float, thickness: int,
             color=(255, 255, 255)) -> None:
    """带黑色描边的文字，浅色背景上也能看清。"""
    cv2.putText(frame, text, org, FONT, fscale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, fscale, color, thickness, cv2.LINE_AA)


def draw_detections(frame, dets: Iterable[dict], style: dict, connect: bool = True,
                    show_index: bool = True) -> None:
    """画检测框、类别+置信度标签、4 个模型预测的关键点（原地修改）。"""
    lw, radius = style["lw"], style["radius"]
    fscale, tth = style["fscale"], style["tth"]

    for det in dets:
        cls_id = det["cls"]
        color = PALETTE[cls_id % len(PALETTE)]
        x1, y1, x2, y2 = (int(round(v)) for v in det["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, lw, cv2.LINE_AA)

        kxy = det.get("kxy")
        pts = [(i, float(x), float(y)) for i, (x, y) in enumerate(kxy or [])
               if not (x <= 0.5 and y <= 0.5)]

        if len(pts) >= 2 and connect:
            poly = np.array([[int(round(x)), int(round(y))] for _, x, y in pts],
                            dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [poly], len(pts) == 4, color, max(1, lw), cv2.LINE_AA)

        for i, x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            cv2.circle(frame, (xi, yi), radius, KPT_COLORS[i % len(KPT_COLORS)], -1, cv2.LINE_AA)
            cv2.circle(frame, (xi, yi), radius, (255, 255, 255), max(1, lw // 2), cv2.LINE_AA)
            if show_index:
                put_text(frame, str(i + 1), (xi + radius + 2, yi - radius - 2),
                         fscale * 0.8, max(1, tth - 1))

        label = f"{det['name']} {det['conf']:.2f}"
        (tw, th), base = cv2.getTextSize(label, FONT, fscale, tth)
        ty = y1 - 6 if y1 - th - base - 6 >= 0 else y1 + th + base + 6
        cv2.rectangle(frame, (x1, ty - th - base), (x1 + tw + 8, ty + base), color, -1)
        put_text(frame, label, (x1 + 4, ty), fscale, tth)


def match_detections(a: list[dict], b: list[dict], iou_thr: float = 0.5) -> int:
    """统计两组检测中类别相同且框 IoU >= iou_thr 的匹配对数（贪心匹配）。"""
    if not a or not b:
        return 0
    boxes_a = np.array([d["box"] for d in a], dtype=float)
    boxes_b = np.array([d["box"] for d in b], dtype=float)
    cls_a = np.array([d["cls"] for d in a])
    cls_b = np.array([d["cls"] for d in b])

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    used_b = set()
    matched = 0
    for i in range(len(a)):
        best_j, best_iou = -1, 0.0
        for j in range(len(b)):
            if j in used_b or cls_a[i] != cls_b[j]:
                continue
            xx1 = max(boxes_a[i, 0], boxes_b[j, 0])
            yy1 = max(boxes_a[i, 1], boxes_b[j, 1])
            xx2 = min(boxes_a[i, 2], boxes_b[j, 2])
            yy2 = min(boxes_a[i, 3], boxes_b[j, 3])
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            iou_val = inter / max(area_a[i] + area_b[j] - inter, 1e-9)
            if iou_val > best_iou:
                best_iou, best_j = iou_val, j
        if best_j >= 0 and best_iou >= iou_thr:
            matched += 1
            used_b.add(best_j)
    return matched
