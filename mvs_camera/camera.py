"""Camera lifecycle, frame acquisition, pixel conversion, and cleanup."""

from __future__ import annotations

import ctypes
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from .sdk import (
    MV_OK,
    MV_E_NODATA,
    CameraDescriptor,
    MvFrameOut,
    MvPixelConvertParamEx,
    MvsError,
    MvsSdk,
    PixelFormat,
    error_hex,
    error_name,
)


class CameraError(RuntimeError):
    """A camera state or image-data error."""


@dataclass
class CameraConfig:
    serial: str | None = None
    index: int = 0
    exposure_us: float = 1000.0
    gain_db: float = 0.0
    pixel_format: str = "BayerRG8"
    roi: tuple[int, int, int, int] | None = None
    frame_rate: float | None = None
    timeout_ms: int = 1000
    settle_frames: int = 5
    latest_images_only: bool = True


@dataclass
class FrameData:
    image_bgr: np.ndarray
    raw_image: np.ndarray
    requested_pixel_format: str
    actual_pixel_format: str
    width: int
    height: int
    frame_number: int
    device_timestamp: int
    host_timestamp: int
    frame_len: int
    exposure_us: float
    gain_db: float
    conversion_ms: float
    conversion: str
    received_at: float = field(default_factory=time.perf_counter)

    def metadata(self) -> dict[str, Any]:
        return {
            "requested_pixel_format": self.requested_pixel_format,
            "actual_pixel_format": self.actual_pixel_format,
            "width": self.width,
            "height": self.height,
            "frame_number": self.frame_number,
            "device_timestamp": self.device_timestamp,
            "host_timestamp": self.host_timestamp,
            "frame_len": self.frame_len,
            "exposure_us": self.exposure_us,
            "gain_db": self.gain_db,
            "conversion_ms": self.conversion_ms,
            "conversion": self.conversion,
        }


class MvsCamera:
    """USB3 MVS camera with deterministic cleanup.

    A successful ``MV_CC_GetImageBuffer`` always owns exactly one leased frame
    until ``MV_CC_FreeImageBuffer`` succeeds. All conversion and NumPy copies
    happen before the lease is returned.
    """

    def __init__(
        self,
        config: CameraConfig,
        *,
        sdk: MvsSdk | None = None,
        library: str | Sequence[str] | None = None,
    ):
        self.config = config
        self.sdk = sdk or MvsSdk(library)
        self._owns_sdk_init = False
        self.handle = None
        self._device_buffer = None
        self.device: CameraDescriptor | None = None
        self.opened = False
        self.grabbing = False
        self.leased: MvFrameOut | None = None
        self.actual_pixel_type = PixelFormat.VALUES[config.pixel_format]
        self.actual_pixel_format = config.pixel_format
        self.resource_log: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.camera_info: dict[str, Any] = {}

    def _record(self, step: str, code: int, ok: bool | None = None, detail: str = "") -> None:
        code = int(code) & 0xFFFFFFFF
        self.resource_log.append(
            {
                "step": step,
                "ok": ok if ok is not None else code == MV_OK,
                "code": error_hex(code),
                "name": error_name(code),
                "detail": detail,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
            print(f"[警告] {message}")

    def _safe_string(self, key: str, fallback: str = "") -> str:
        try:
            return self.sdk.get_string(self.handle, key)
        except MvsError:
            return fallback

    def _safe_float(self, key: str, fallback: float = 0.0) -> float:
        try:
            return float(self.sdk.get_float(self.handle, key).fCurValue)
        except MvsError:
            return fallback

    def _safe_int(self, key: str, fallback: int = 0) -> int:
        try:
            return self.sdk.get_int(self.handle, key)
        except MvsError:
            return fallback

    def open(self) -> "MvsCamera":
        if self.handle is not None:
            raise CameraError("相机已经打开，不能重复 open()")
        if self.config.pixel_format not in PixelFormat.VALUES:
            raise CameraError(
                f"不支持的像素格式 {self.config.pixel_format!r}，"
                f"可选: {', '.join(PixelFormat.VALUES)}"
            )
        try:
            self.sdk.initialize()
            self._owns_sdk_init = True
            devices = self.sdk.enumerate_usb()
            self.device = self._select_device(devices)
            self.handle, self._device_buffer = self.sdk.create_handle(self.device)
            self._record("CreateHandle", MV_OK)

            self.sdk.open_device(self.handle)
            self.opened = True
            self._record("OpenDevice", MV_OK)
            self._configure()
            self.camera_info = self._read_camera_info()
            return self
        except Exception:
            self.close()
            raise

    def _select_device(self, devices: list[CameraDescriptor]) -> CameraDescriptor:
        if not devices:
            raise CameraError("未发现 USB3 Vision 相机，请检查 USB 线、供电和 MVS 安装")
        if self.config.serial:
            for device in devices:
                if device.serial == self.config.serial:
                    return device
            known = ", ".join(d.serial or "<空>" for d in devices)
            raise CameraError(f"未找到序列号 {self.config.serial!r}，当前设备: {known}")
        if self.config.index < 0 or self.config.index >= len(devices):
            raise CameraError(
                f"相机索引 {self.config.index} 越界，共发现 {len(devices)} 台设备"
            )
        return devices[self.config.index]

    def _set_required_enum(self, key: str, value: str) -> None:
        self.sdk.set_enum_string(self.handle, key, value)
        self._record(key, MV_OK)

    def _set_optional_bool(self, key: str, value: bool) -> None:
        try:
            self.sdk.set_bool(self.handle, key, value)
        except MvsError as exc:
            self._warn(f"{key} 不可设置，已跳过: {exc}")

    def _configure(self) -> None:
        required = [
            ("AcquisitionMode", "Continuous"),
            ("TriggerMode", "Off"),
            ("ExposureAuto", "Off"),
            ("ExposureMode", "Timed"),
            ("GainAuto", "Off"),
        ]
        for key, value in required:
            self._set_required_enum(key, value)

        if self.config.roi is not None:
            x, y, width, height = self.config.roi
            if min(x, y, width, height) < 0 or width == 0 or height == 0:
                raise CameraError(f"ROI 非法: {self.config.roi}")
            self.sdk.set_int(self.handle, "OffsetX", 0)
            self.sdk.set_int(self.handle, "OffsetY", 0)
            self.sdk.set_int(self.handle, "Width", width)
            self.sdk.set_int(self.handle, "Height", height)
            self.sdk.set_int(self.handle, "OffsetX", x)
            self.sdk.set_int(self.handle, "OffsetY", y)
            self._record("ROI", MV_OK, detail=f"x={x},y={y},w={width},h={height}")

        self.actual_pixel_type = PixelFormat.VALUES[self.config.pixel_format]
        self.sdk.set_enum(self.handle, "PixelFormat", self.actual_pixel_type)
        self._record("PixelFormat", MV_OK, detail=self.config.pixel_format)
        self.actual_pixel_type = self.sdk.get_enum(self.handle, "PixelFormat")
        self.actual_pixel_format = PixelFormat.NAMES.get(
            self.actual_pixel_type, error_hex(self.actual_pixel_type)
        )
        if self.actual_pixel_type != PixelFormat.VALUES[self.config.pixel_format]:
            raise CameraError(
                f"请求 {self.config.pixel_format}，相机实际返回 {self.actual_pixel_format}"
            )

        self.set_exposure(self.config.exposure_us)
        self.set_gain(self.config.gain_db)

        if self.config.frame_rate is not None:
            self._set_optional_bool("AcquisitionFrameRateEnable", True)
            try:
                self.set_camera_float("AcquisitionFrameRate", self.config.frame_rate)
            except MvsError as exc:
                self._warn(f"无法设置目标帧率 {self.config.frame_rate}: {exc}")

    def _read_camera_info(self) -> dict[str, Any]:
        assert self.device is not None
        return {
            "model": self._safe_string("DeviceModelName", self.device.model),
            "serial": self._safe_string("DeviceSerialNumber", self.device.serial),
            "firmware": self._safe_string(
                "DeviceFirmwareVersion", self.device.device_version
            ),
            "device_version": self.device.device_version,
            "mvs_sdk_version": self.sdk.sdk_version,
            "pixel_format": self.actual_pixel_format,
            "roi": {
                "offset_x": self._safe_int("OffsetX"),
                "offset_y": self._safe_int("OffsetY"),
                "width": self._safe_int("Width"),
                "height": self._safe_int("Height"),
            },
            "exposure_us": self._safe_float("ExposureTime"),
            "gain_db": self._safe_float("Gain"),
            "config": asdict(self.config),
        }

    def set_exposure(self, exposure_us: float) -> None:
        self._require_opened()
        value = self.sdk.get_float(self.handle, "ExposureTime")
        if not (float(value.fMin) <= exposure_us <= float(value.fMax)):
            raise CameraError(
                f"曝光 {exposure_us} μs 超出当前范围 "
                f"[{value.fMin:.3f}, {value.fMax:.3f}] μs"
            )
        self.sdk.set_float(self.handle, "ExposureTime", exposure_us)
        actual = self.sdk.get_float(self.handle, "ExposureTime")
        if abs(float(actual.fCurValue) - exposure_us) > max(1.0, exposure_us * 0.01):
            raise CameraError(
                f"曝光回读不一致: 请求 {exposure_us} μs，实际 {actual.fCurValue} μs"
            )
        self.config.exposure_us = float(actual.fCurValue)
        self._record("ExposureTime", MV_OK, detail=f"{actual.fCurValue:.3f} us")

    def set_gain(self, gain_db: float) -> None:
        self._require_opened()
        value = self.sdk.get_float(self.handle, "Gain")
        if not (float(value.fMin) <= gain_db <= float(value.fMax)):
            raise CameraError(
                f"增益 {gain_db} dB 超出当前范围 "
                f"[{value.fMin:.3f}, {value.fMax:.3f}] dB"
            )
        self.sdk.set_float(self.handle, "Gain", gain_db)
        actual = self.sdk.get_float(self.handle, "Gain")
        self.config.gain_db = float(actual.fCurValue)
        self._record("Gain", MV_OK, detail=f"{actual.fCurValue:.3f} dB")

    def set_camera_float(self, key: str, value: float) -> None:
        self._require_opened()
        self.sdk.set_float(self.handle, key, value)

    def start(self) -> None:
        self._require_opened()
        if self.grabbing:
            return
        if (
            self.config.latest_images_only
            and self.sdk.mv_set_grab_strategy is not None
        ):
            # MV_GrabStrategy_LatestImagesOnly = 1
            code = int(self.sdk.mv_set_grab_strategy(self.handle, 1))
            if code == MV_OK:
                self._record("SetGrabStrategy_LatestImagesOnly", code)
            else:
                self._warn(
                    f"无法设置最新帧策略: {error_name(code)} ({error_hex(code)})"
                )
        self.sdk.start_grabbing(self.handle)
        self.grabbing = True
        self._record("StartGrabbing", MV_OK)

    def grab_frame(
        self,
        *,
        timeout_ms: int | None = None,
        force_pixel_type: int | None = None,
    ) -> FrameData | None:
        """Return one copied frame, ``None`` on timeout, or raise on bad data.

        ``force_pixel_type`` is only used by ``fault-test bad-format`` to inject
        an unsupported format after a real frame has been acquired. It does not
        alter the camera and the real buffer is still returned in ``finally``.
        """
        if not self.grabbing or self.handle is None:
            raise CameraError("取帧前必须调用 start()")
        if self.leased is not None:
            raise CameraError("上一个 SDK 图像缓冲区尚未归还")

        timeout = self.config.timeout_ms if timeout_ms is None else int(timeout_ms)
        code, frame = self.sdk.get_frame(self.handle, timeout)
        if code == MV_E_NODATA:
            self._record("GetImageBuffer", code, ok=False, detail="timeout")
            return None
        self.sdk.check("MV_CC_GetImageBuffer", code)
        self.leased = frame
        info = frame.stFrameInfo
        self._record(
            "GetImageBuffer",
            MV_OK,
            detail=f"frame={int(info.nFrameNum)}, {int(info.nWidth)}x{int(info.nHeight)}",
        )
        try:
            return self._process_leased_frame(frame, force_pixel_type)
        finally:
            # This is deliberately in ``finally``: conversion or format errors
            # must never leak the SDK buffer lease.
            self.sdk.free_frame(self.handle, frame)
            self.leased = None
            self._record("FreeImageBuffer", MV_OK)

    def _process_leased_frame(
        self, frame: MvFrameOut, force_pixel_type: int | None
    ) -> FrameData:
        info = frame.stFrameInfo
        width = int(info.nWidth)
        height = int(info.nHeight)
        pixel_type = int(
            info.enPixelType if force_pixel_type is None else force_pixel_type
        )
        frame_len = int(info.nFrameLen)
        metadata_fallback = False
        if not frame.pBufAddr or width <= 0 or height <= 0 or frame_len <= 0:
            raise CameraError(
                f"无效帧信息: ptr={bool(frame.pBufAddr)}, "
                f"width={width}, height={height}, len={frame_len}"
            )
        if pixel_type not in PixelFormat.NAMES:
            if pixel_type == 0:
                # Some MVS/library combinations return a valid image buffer but
                # leave enPixelType as Undefined. Only recover when the recent
                # camera readback and the frame length both prove the layout.
                fallback = self.actual_pixel_type
                fallback_channels = 3 if fallback == PixelFormat.BGR8 else 1
                fallback_length = width * height * fallback_channels
                if frame_len == fallback_length:
                    self._warn(
                        "SDK 帧的 enPixelType 为 0x00000000，"
                        f"已按相机回读格式 {self.actual_pixel_format} 和"
                        f"匹配的帧长度 {frame_len} 恢复"
                    )
                    pixel_type = fallback
                else:
                    # Some MVS versions leave several informational fields
                    # uninitialised even though the image buffer itself is
                    # valid. Width, height and PixelFormat were read back from
                    # the device before grabbing, so use those authoritative
                    # values and copy only the exact compact image length.
                    self._warn(
                        "SDK 帧元数据异常：enPixelType=0x00000000，"
                        f"nFrameLen={frame_len}，与相机回读格式 "
                        f"{self.actual_pixel_format} 的 {fallback_length} 字节不符；"
                        "已改用相机回读格式和宽高计算安全帧长度"
                    )
                    pixel_type = fallback
                    frame_len = fallback_length
                    metadata_fallback = True
            else:
                raise CameraError(
                    f"收到不支持的图像格式: {error_hex(pixel_type)}"
                    "（不会把未知字节强行解释成 OpenCV 图像）"
                )

        channels = 3 if pixel_type == PixelFormat.BGR8 else 1
        expected = width * height * channels
        if frame_len != expected:
            reported_len = frame_len
            if (
                force_pixel_type is None
                and pixel_type == self.actual_pixel_type
                and reported_len > max(expected * 4, 256 * 1024 * 1024)
            ):
                self._warn(
                    "SDK 帧的 nFrameLen 明显异常："
                    f"{reported_len} 字节；已按相机回读格式 "
                    f"{self.actual_pixel_format} 和宽高改用 {expected} 字节"
                )
                frame_len = expected
                metadata_fallback = True
            else:
                raise CameraError(
                    f"帧长度与紧凑图像不符: 格式={PixelFormat.NAMES[pixel_type]}, "
                    f"期望 {expected} 字节, 实际 {frame_len}。"
                    "请关闭 ChunkData 或使用单帧探测器核对该相机的行对齐方式。"
                )
        raw_all = np.ctypeslib.as_array(
            ctypes.cast(frame.pBufAddr, ctypes.POINTER(ctypes.c_ubyte)),
            shape=(frame_len,),
        )
        raw = raw_all[:expected].copy()

        conversion_ms = 0.0
        if pixel_type == PixelFormat.BGR8:
            raw_image = raw.reshape(height, width, 3)
            image_bgr = raw_image.copy()
            conversion = "none (camera outputs BGR8)"
        elif pixel_type == PixelFormat.BayerRG8:
            raw_image = raw.reshape(height, width)
            t0 = time.perf_counter()
            destination = ctypes.create_string_buffer(expected * 3)
            params = MvPixelConvertParamEx(
                nWidth=width,
                nHeight=height,
                enSrcPixelType=pixel_type,
                pSrcData=frame.pBufAddr,
                nSrcDataLen=frame_len,
                enDstPixelType=PixelFormat.BGR8,
                pDstBuffer=ctypes.cast(destination, ctypes.c_void_p),
                nDstBufferSize=expected * 3,
            )
            try:
                convert_code = int(
                    self.sdk.mv_convert_pixel(self.handle, ctypes.byref(params))
                )
            except MvsError as exc:
                convert_code = exc.code
            conversion_ms = (time.perf_counter() - t0) * 1000.0
            if convert_code == MV_OK:
                if int(params.nDstLen) != expected * 3:
                    raise CameraError(
                        f"像素转换长度异常: 期望 {expected * 3}, 实际 {params.nDstLen}"
                    )
                image_bgr = np.frombuffer(
                    destination.raw, dtype=np.uint8, count=int(params.nDstLen)
                ).reshape(height, width, 3).copy()
                conversion = "MV_CC_ConvertPixelTypeEx (BayerRG8 -> BGR8)"
            else:
                # Some MVS releases either do not provide the Ex conversion
                # structure expected by Python or reject this camera layout.
                # OpenCV's Bayer conversion is an explicitly supported
                # fallback and still uses the original BayerRG8 camera data.
                self._warn(
                    "MVS BayerRG8 -> BGR8 转换失败: "
                    f"{error_name(convert_code)} ({error_hex(convert_code)})；"
                    "已自动改用 OpenCV cv2.cvtColor"
                )
                t0 = time.perf_counter()
                # BayerRG8 stores one 2x2 block as R, G, G, B. OpenCV's
                # two-letter aliases are easy to misread: BayerBG2BGR is
                # also BayerRGGB2BGR, whereas BayerRG2BGR decodes BGGR.
                image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_BayerRGGB2BGR)
                conversion_ms += (time.perf_counter() - t0) * 1000.0
                conversion = "OpenCV COLOR_BayerRGGB2BGR (BayerRG8 -> BGR8)"
        else:
            raw_image = raw.reshape(height, width)
            image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
            conversion = "OpenCV COLOR_GRAY2BGR"

        timestamp = (int(info.nDevTimeStampHigh) << 32) | int(info.nDevTimeStampLow)
        return FrameData(
            image_bgr=image_bgr,
            raw_image=raw_image,
            requested_pixel_format=self.config.pixel_format,
            actual_pixel_format=PixelFormat.NAMES[pixel_type],
            width=width,
            height=height,
            frame_number=int(info.nFrameNum),
            device_timestamp=timestamp,
            host_timestamp=int(info.nHostTimeStamp),
            frame_len=frame_len,
            exposure_us=(
                self.config.exposure_us
                if metadata_fallback
                else float(info.fExposureTime or self.config.exposure_us)
            ),
            gain_db=(
                self.config.gain_db
                if metadata_fallback
                else float(info.fGain or self.config.gain_db)
            ),
            conversion_ms=conversion_ms,
            conversion=conversion,
        )

    def _require_opened(self) -> None:
        if self.handle is None or not self.opened:
            raise CameraError("相机尚未打开")

    def close(self) -> None:
        """Best-effort reverse-order cleanup; every step is recorded."""
        if getattr(self, "leased", None) is not None and self.handle is not None:
            try:
                self.sdk.free_frame(self.handle, self.leased)
                self._record("CleanupFreeImageBuffer", MV_OK)
            except Exception as exc:
                code = exc.code if isinstance(exc, MvsError) else MV_E_RESOURCE
                self._record("CleanupFreeImageBuffer", code, ok=False, detail=str(exc))
            self.leased = None

        if self.grabbing and self.handle is not None:
            try:
                self.sdk.stop_grabbing(self.handle)
                self._record("StopGrabbing", MV_OK)
            except Exception as exc:
                code = exc.code if isinstance(exc, MvsError) else MV_E_RESOURCE
                self._record("StopGrabbing", code, ok=False, detail=str(exc))
            self.grabbing = False

        if self.opened and self.handle is not None:
            try:
                self.sdk.close_device(self.handle)
                self._record("CloseDevice", MV_OK)
            except Exception as exc:
                code = exc.code if isinstance(exc, MvsError) else MV_E_RESOURCE
                self._record("CloseDevice", code, ok=False, detail=str(exc))
            self.opened = False

        if self.handle is not None:
            try:
                self.sdk.destroy_handle(self.handle)
                self._record("DestroyHandle", MV_OK)
            except Exception as exc:
                code = exc.code if isinstance(exc, MvsError) else MV_E_RESOURCE
                self._record("DestroyHandle", code, ok=False, detail=str(exc))
            self.handle = None

        if self._owns_sdk_init:
            try:
                self.sdk.finalize()
                self._record("Finalize", MV_OK)
            except Exception as exc:
                code = exc.code if isinstance(exc, MvsError) else MV_E_RESOURCE
                self._record("Finalize", code, ok=False, detail=str(exc))
            self._owns_sdk_init = False

    def __enter__(self) -> "MvsCamera":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False
