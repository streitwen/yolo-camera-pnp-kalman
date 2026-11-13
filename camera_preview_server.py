"""Live MVS camera preview in a local browser.

This tool is intentionally separate from the inference program so the camera
can be aimed and focused without loading YOLO.  It serves a small MJPEG page on
localhost and releases the camera when the server process exits.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from http import server
from typing import ClassVar

import cv2
import numpy as np

from mvs_camera import CameraConfig, MvsCamera


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MVS 实时预览</title>
  <style>
    body { margin: 0; background: #101214; color: #eee;
           font: 16px/1.5 "Microsoft YaHei", sans-serif; }
    header { padding: 12px 18px; background: #1b1f23; }
    main { max-width: 1200px; margin: 0 auto; padding: 16px; }
    img { display: block; width: 100%; height: auto; background: #000;
          border-radius: 8px; }
    .hint { color: #ffd166; }
  </style>
</head>
<body>
  <header><b>MVS 实时预览</b>：
    将单块装甲板正面朝向镜头，保持在中央十字区域。</header>
  <main>
    <img src="/stream.mjpg" alt="实时相机画面">
    <p class="hint">当前曝光 30000 μs，增益 6 dB。关闭此页面后，
      请告诉 Codex“位置调好了”，再开始正式采集。</p>
  </main>
</body>
</html>
"""


class PreviewServer(server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    camera_lock: ClassVar[threading.Lock] = threading.Lock()
    args: argparse.Namespace


class Handler(server.BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"[HTTP] {self.address_string()} {fmt % args}", flush=True)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/stream.mjpg":
            self._stream()
            return
        self.send_error(404)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        args = self.server.args
        config = CameraConfig(
            serial=args.serial or None,
            index=args.camera_index,
            exposure_us=args.exposure_us,
            gain_db=args.gain_db,
            pixel_format=args.pixel_format,
            timeout_ms=args.timeout_ms,
        )

        # A browser may reconnect; the lock guarantees one exclusive camera
        # owner and deterministic SDK cleanup when this request ends.
        with PreviewServer.camera_lock:
            try:
                with MvsCamera(config) as camera:
                    camera.start()
                    print(
                        f"[预览] {camera.camera_info.get('model', '')} "
                        f"SN={camera.camera_info.get('serial', '')}",
                        flush=True,
                    )
                    while True:
                        frame_started = time.perf_counter()
                        frame_data = camera.grab_frame()
                        if frame_data is None:
                            continue
                        image = frame_data.image_bgr.copy()
                        height, width = image.shape[:2]
                        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        brightness = float(np.mean(grayscale))
                        frame_ms = (time.perf_counter() - frame_started) * 1000.0

                        cx, cy = width // 2, height // 2
                        size = min(width, height) // 12
                        cv2.line(image, (cx - size, cy), (cx + size, cy),
                                 (0, 255, 255), 2)
                        cv2.line(image, (cx, cy - size), (cx, cy + size),
                                 (0, 255, 255), 2)
                        cv2.putText(
                            image,
                            f"Brightness {brightness:5.1f}  "
                            f"{width}x{height}  {frame_ms:.1f} ms/frame",
                            (16, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        if args.scale != 1.0:
                            image = cv2.resize(
                                image,
                                None,
                                fx=args.scale,
                                fy=args.scale,
                                interpolation=cv2.INTER_AREA,
                            )
                        ok, encoded = cv2.imencode(
                            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85]
                        )
                        if not ok:
                            continue
                        payload = encoded.tobytes()
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(payload)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                print(f"[预览错误] {exc}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVS 相机浏览器实时预览")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--serial", default="DA4302122")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--exposure-us", type=float, default=30000.0)
    parser.add_argument("--gain-db", type=float, default=6.0)
    parser.add_argument(
        "--pixel-format", choices=("BayerRG8", "BGR8", "Mono8"),
        default="BayerRG8",
    )
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument("--scale", type=float, default=0.7)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    httpd = PreviewServer((args.host, args.port), Handler)
    httpd.args = args

    def stop_server(signum, frame) -> None:
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    print(
        f"[预览] http://{args.host}:{args.port}/ "
        f"exposure={args.exposure_us:g}us gain={args.gain_db:g}dB",
        flush=True,
    )
    try:
        httpd.serve_forever(poll_interval=0.25)
    finally:
        httpd.server_close()
        print("[预览] 已关闭", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
