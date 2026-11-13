"""Small ``ctypes`` binding for the Hikrobot MVS camera-control SDK.

Only the USB3 Vision camera APIs used by this project are declared. Struct
layouts and constants were checked against the public MVS SDK headers. The
vendor SDK remains an external dependency and is not copied into this repo.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

MV_OK = 0
MV_E_HANDLE = 0x80000000
MV_E_PARAMETER = 0x80000004
MV_E_RESOURCE = 0x80000006
MV_E_NODATA = 0x80000007
MV_E_ACCESS_DENIED = 0x80000203

MV_USB_DEVICE = 0x00000004
MV_ACCESS_Exclusive = 1
MV_MAX_DEVICE_NUM = 256
INFO_MAX_BUFFER_SIZE = 64
MV_MAX_XML_SYMBOLIC_NUM = 64


class PixelFormat:
    """MVS ``MvGvspPixelType`` values used by the experiments."""

    Mono8 = 0x01080001
    BayerRG8 = 0x01080009
    BGR8 = 0x02180015

    NAMES = {
        Mono8: "Mono8",
        BayerRG8: "BayerRG8",
        BGR8: "BGR8",
    }
    VALUES = {name: value for value, name in NAMES.items()}


ERROR_NAMES = {
    MV_E_HANDLE: "MV_E_HANDLE",
    MV_E_PARAMETER: "MV_E_PARAMETER",
    MV_E_RESOURCE: "MV_E_RESOURCE",
    MV_E_NODATA: "MV_E_NODATA",
    MV_E_ACCESS_DENIED: "MV_E_ACCESS_DENIED",
}


def _unsigned32(value: int) -> int:
    return int(value) & 0xFFFFFFFF


def error_name(code: int) -> str:
    code = _unsigned32(code)
    if code == MV_OK:
        return "MV_OK"
    return ERROR_NAMES.get(code, "MVS_ERROR")


def error_hex(code: int) -> str:
    return f"0x{_unsigned32(code):08X}"


def format_sdk_version(value: int) -> str:
    value = _unsigned32(value)
    return ".".join(str((value >> shift) & 0xFF) for shift in (24, 16, 8, 0))


class MvsError(RuntimeError):
    """An MVS function returned a non-zero status code."""

    def __init__(self, operation: str, code: int, detail: str = ""):
        self.operation = operation
        self.code = _unsigned32(code)
        self.name = error_name(self.code)
        self.detail = detail
        message = f"{operation} 失败: {self.name} ({error_hex(self.code)})"
        if detail:
            message += f" - {detail}"
        super().__init__(message)


class MvUsb3DeviceInfo(ctypes.Structure):
    _fields_ = [
        ("CrtlInEndPoint", ctypes.c_ubyte),
        ("CrtlOutEndPoint", ctypes.c_ubyte),
        ("StreamEndPoint", ctypes.c_ubyte),
        ("EventEndPoint", ctypes.c_ubyte),
        ("idVendor", ctypes.c_ushort),
        ("idProduct", ctypes.c_ushort),
        ("nDeviceNumber", ctypes.c_uint),
        ("chDeviceGUID", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chVendorName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chModelName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chFamilyName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceVersion", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chManufacturerName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chSerialNumber", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chUserDefinedName", ctypes.c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("nbcdUSB", ctypes.c_uint),
        ("nDeviceAddress", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 2),
    ]


class MvDeviceSpecialInfo(ctypes.Union):
    _fields_ = [
        ("stUsb3VInfo", MvUsb3DeviceInfo),
    ]


class MvDeviceInfo(ctypes.Structure):
    _fields_ = [
        ("nMajorVer", ctypes.c_ushort),
        ("nMinorVer", ctypes.c_ushort),
        ("nMacAddrHigh", ctypes.c_uint),
        ("nMacAddrLow", ctypes.c_uint),
        ("nTLayerType", ctypes.c_uint),
        ("nDevTypeInfo", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 3),
        ("SpecialInfo", MvDeviceSpecialInfo),
    ]


class MvDeviceInfoList(ctypes.Structure):
    _fields_ = [
        ("nDeviceNum", ctypes.c_uint),
        ("pDeviceInfo", ctypes.POINTER(MvDeviceInfo) * MV_MAX_DEVICE_NUM),
    ]


class MvFrameChunkList(ctypes.Union):
    _fields_ = [
        ("pUnparsedChunkContent", ctypes.c_void_p),
        ("nAligning", ctypes.c_int64),
    ]


class MvFrameSubImageList(ctypes.Union):
    _fields_ = [
        ("pstSubImage", ctypes.c_void_p),
        ("nAligning", ctypes.c_int64),
    ]


class MvFrameUserPtr(ctypes.Union):
    _fields_ = [
        ("pUser", ctypes.c_void_p),
        ("nAligning", ctypes.c_int64),
    ]


class MvFrameOutInfoEx(ctypes.Structure):
    _fields_ = [
        ("nWidth", ctypes.c_ushort),
        ("nHeight", ctypes.c_ushort),
        ("enPixelType", ctypes.c_int32),
        ("nFrameNum", ctypes.c_uint),
        ("nDevTimeStampHigh", ctypes.c_uint),
        ("nDevTimeStampLow", ctypes.c_uint),
        ("nReserved0", ctypes.c_uint),
        ("nHostTimeStamp", ctypes.c_int64),
        ("nFrameLen", ctypes.c_uint),
        ("nSecondCount", ctypes.c_uint),
        ("nCycleCount", ctypes.c_uint),
        ("nCycleOffset", ctypes.c_uint),
        ("fGain", ctypes.c_float),
        ("fExposureTime", ctypes.c_float),
        ("nAverageBrightness", ctypes.c_uint),
        ("nRed", ctypes.c_uint),
        ("nGreen", ctypes.c_uint),
        ("nBlue", ctypes.c_uint),
        ("nFrameCounter", ctypes.c_uint),
        ("nTriggerIndex", ctypes.c_uint),
        ("nInput", ctypes.c_uint),
        ("nOutput", ctypes.c_uint),
        ("nOffsetX", ctypes.c_ushort),
        ("nOffsetY", ctypes.c_ushort),
        ("nChunkWidth", ctypes.c_ushort),
        ("nChunkHeight", ctypes.c_ushort),
        ("nLostPacket", ctypes.c_uint),
        ("nUnparsedChunkNum", ctypes.c_uint),
        ("UnparsedChunkList", MvFrameChunkList),
        ("nExtendWidth", ctypes.c_uint),
        ("nExtendHeight", ctypes.c_uint),
        ("nFrameLenEx", ctypes.c_uint64),
        ("nReserved1", ctypes.c_uint),
        ("nSubImageNum", ctypes.c_uint),
        ("SubImageList", MvFrameSubImageList),
        ("UserPtr", MvFrameUserPtr),
        ("nReserved", ctypes.c_uint * 26),
    ]


class MvFrameOut(ctypes.Structure):
    _fields_ = [
        ("pBufAddr", ctypes.c_void_p),
        ("stFrameInfo", MvFrameOutInfoEx),
        ("nRes", ctypes.c_uint * 16),
    ]


class MvFloatValue(ctypes.Structure):
    _fields_ = [
        ("fCurValue", ctypes.c_float),
        ("fMax", ctypes.c_float),
        ("fMin", ctypes.c_float),
        ("nReserved", ctypes.c_uint * 4),
    ]


class MvIntValueEx(ctypes.Structure):
    _fields_ = [
        ("nCurValue", ctypes.c_int64),
        ("nMax", ctypes.c_int64),
        ("nMin", ctypes.c_int64),
        ("nInc", ctypes.c_int64),
        ("nReserved", ctypes.c_uint * 16),
    ]


class MvEnumValue(ctypes.Structure):
    _fields_ = [
        ("nCurValue", ctypes.c_uint),
        ("nSupportedNum", ctypes.c_uint),
        ("nSupportValue", ctypes.c_uint * MV_MAX_XML_SYMBOLIC_NUM),
        ("nReserved", ctypes.c_uint * 4),
    ]


class MvStringValue(ctypes.Structure):
    _fields_ = [
        ("chCurValue", ctypes.c_char * 256),
        ("nMaxLength", ctypes.c_int64),
        ("nReserved", ctypes.c_uint * 2),
    ]


class MvPixelConvertParamEx(ctypes.Structure):
    _fields_ = [
        ("nWidth", ctypes.c_uint),
        ("nHeight", ctypes.c_uint),
        ("enSrcPixelType", ctypes.c_int32),
        ("pSrcData", ctypes.c_void_p),
        ("nSrcDataLen", ctypes.c_uint),
        ("enDstPixelType", ctypes.c_int32),
        ("pDstBuffer", ctypes.c_void_p),
        ("nDstLen", ctypes.c_uint),
        ("nDstBufferSize", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 4),
    ]


def _decode_fixed(buffer: Sequence[int]) -> str:
    data = bytes(buffer)
    return data.split(b"\0", 1)[0].decode("utf-8", errors="replace").strip()


@dataclass(frozen=True)
class CameraDescriptor:
    index: int
    serial: str
    model: str
    device_version: str
    transport: str
    raw_info: bytes


def _candidate_libraries(explicit: str | Sequence[str] | None) -> list[str]:
    if explicit:
        return [explicit] if isinstance(explicit, str) else list(explicit)
    env_value = os.environ.get("MVS_SDK_LIBRARY")
    if env_value:
        return [env_value]
    if os.name == "nt":
        # The current official Windows SDK names the DLL
        # ``MVCameraControl.dll`` (capital C) and installs it in the shared
        # runtime directory.  Keep the historical spelling as a fallback for
        # older installations, then try this machine's absolute location so
        # loading also works when the process PATH has not been refreshed.
        candidates = ["MVCameraControl.dll", "MvCameraControl.dll", "MVCameraControl"]
        common_x86 = os.environ.get("CommonProgramFiles(x86)")
        if common_x86:
            candidates.append(
                str(
                    Path(common_x86)
                    / "MVS"
                    / "Runtime"
                    / "Win64_x64"
                    / "MVCameraControl.dll"
                )
            )
        return candidates
    return [
        "libMvCameraControl.so",
        "/opt/MVS/lib/64/libMvCameraControl.so",
        "/opt/MVS/lib/aarch64/libMvCameraControl.so",
        "/opt/MVS/lib/32/libMvCameraControl.so",
    ]


class MvsSdk:
    """Load the vendor library once and expose checked wrappers."""

    def __init__(self, library: str | Sequence[str] | None = None):
        self.library_path = self._load_library(library)
        self._initialize_refs = 0
        self._bind()

    @staticmethod
    def _load_library(library: str | Sequence[str] | None):
        loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
        errors = []
        for candidate in _candidate_libraries(library):
            try:
                # Absolute paths make multi-install mistakes much easier to spot.
                path = str(Path(candidate).expanduser())
                return loader(path)
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")
        hint = (
            "请先安装海康 MVS SDK，并用 MVS_SDK_LIBRARY 指定动态库完整路径。"
            "Linux 常见位置是 /opt/MVS/lib/64/libMvCameraControl.so。"
        )
        raise MvsError("LoadMvsLibrary", MV_E_RESOURCE, hint + " | " + " || ".join(errors))

    def _bind(self) -> None:
        lib = self.library_path
        p = ctypes.c_void_p

        def fn(name: str, argtypes, restype=ctypes.c_int, required: bool = True):
            target = getattr(lib, name, None)
            if target is None:
                if required:
                    raise MvsError("BindMvsApi", MV_E_RESOURCE, f"动态库缺少符号 {name}")
                return None
            target.argtypes = argtypes
            target.restype = restype
            return target

        # ``MV_CC_Initialize`` is absent in some older packages.
        self.mv_initialize = fn("MV_CC_Initialize", [], required=False)
        self.mv_finalize = fn("MV_CC_Finalize", [], required=False)
        self.mv_get_sdk_version = fn(
            "MV_CC_GetSDKVersion",
            [],
            restype=ctypes.c_uint,
            required=False,
        )
        self.mv_enum = fn("MV_CC_EnumDevices", [ctypes.c_uint, ctypes.POINTER(MvDeviceInfoList)])
        self.mv_create = fn("MV_CC_CreateHandle", [ctypes.POINTER(p), ctypes.POINTER(MvDeviceInfo)])
        self.mv_destroy = fn("MV_CC_DestroyHandle", [p])
        self.mv_open = fn("MV_CC_OpenDevice", [p, ctypes.c_uint, ctypes.c_ushort])
        self.mv_close = fn("MV_CC_CloseDevice", [p])
        self.mv_start = fn("MV_CC_StartGrabbing", [p])
        self.mv_stop = fn("MV_CC_StopGrabbing", [p])
        self.mv_get_frame = fn(
            "MV_CC_GetImageBuffer", [p, ctypes.POINTER(MvFrameOut), ctypes.c_uint]
        )
        self.mv_free_frame = fn("MV_CC_FreeImageBuffer", [p, ctypes.POINTER(MvFrameOut)])
        self.mv_set_enum_string = fn(
            "MV_CC_SetEnumValueByString", [p, ctypes.c_char_p, ctypes.c_char_p]
        )
        self.mv_set_enum = fn("MV_CC_SetEnumValue", [p, ctypes.c_char_p, ctypes.c_uint])
        self.mv_set_float = fn("MV_CC_SetFloatValue", [p, ctypes.c_char_p, ctypes.c_float])
        self.mv_get_float = fn(
            "MV_CC_GetFloatValue", [p, ctypes.c_char_p, ctypes.POINTER(MvFloatValue)]
        )
        self.mv_set_int = fn("MV_CC_SetIntValueEx", [p, ctypes.c_char_p, ctypes.c_int64])
        self.mv_get_int = fn(
            "MV_CC_GetIntValueEx", [p, ctypes.c_char_p, ctypes.POINTER(MvIntValueEx)]
        )
        self.mv_get_enum = fn(
            "MV_CC_GetEnumValue", [p, ctypes.c_char_p, ctypes.POINTER(MvEnumValue)]
        )
        self.mv_set_bool = fn("MV_CC_SetBoolValue", [p, ctypes.c_char_p, ctypes.c_bool])
        self.mv_get_string = fn(
            "MV_CC_GetStringValue", [p, ctypes.c_char_p, ctypes.POINTER(MvStringValue)]
        )
        self.mv_convert_pixel = fn(
            "MV_CC_ConvertPixelTypeEx", [p, ctypes.POINTER(MvPixelConvertParamEx)]
        )
        self.mv_set_grab_strategy = fn(
            "MV_CC_SetGrabStrategy", [p, ctypes.c_int], required=False
        )

    def initialize(self) -> None:
        if self._initialize_refs == 0 and self.mv_initialize is not None:
            self.check("MV_CC_Initialize", self.mv_initialize())
        self._initialize_refs += 1

    def finalize(self) -> None:
        if self._initialize_refs <= 0:
            return
        self._initialize_refs -= 1
        if self._initialize_refs == 0 and self.mv_finalize is not None:
            self.check("MV_CC_Finalize", self.mv_finalize())

    @staticmethod
    def check(operation: str, code: int, detail: str = "") -> int:
        code = _unsigned32(code)
        if code != MV_OK:
            raise MvsError(operation, code, detail)
        return code

    def enumerate_usb(self) -> list[CameraDescriptor]:
        device_list = MvDeviceInfoList()
        self.check(
            "MV_CC_EnumDevices",
            self.mv_enum(MV_USB_DEVICE, ctypes.byref(device_list)),
        )
        result = []
        for index in range(int(device_list.nDeviceNum)):
            pointer = device_list.pDeviceInfo[index]
            if not pointer:
                continue
            info = pointer.contents
            if int(info.nTLayerType) != MV_USB_DEVICE:
                continue
            usb = info.SpecialInfo.stUsb3VInfo
            result.append(
                CameraDescriptor(
                    index=len(result),
                    serial=_decode_fixed(usb.chSerialNumber),
                    model=_decode_fixed(usb.chModelName),
                    device_version=_decode_fixed(usb.chDeviceVersion),
                    transport="USB3",
                    raw_info=ctypes.string_at(pointer, ctypes.sizeof(MvDeviceInfo)),
                )
            )
        return result

    @property
    def sdk_version(self) -> str | None:
        if self.mv_get_sdk_version is None:
            return None
        return format_sdk_version(int(self.mv_get_sdk_version()))

    def create_handle(self, descriptor: CameraDescriptor):
        buffer = (ctypes.c_ubyte * len(descriptor.raw_info)).from_buffer_copy(
            descriptor.raw_info
        )
        info_ptr = ctypes.cast(buffer, ctypes.POINTER(MvDeviceInfo))
        handle = ctypes.c_void_p()
        self.check("MV_CC_CreateHandle", self.mv_create(ctypes.byref(handle), info_ptr))
        return handle, buffer

    def open_device(self, handle) -> None:
        self.check(
            "MV_CC_OpenDevice",
            self.mv_open(handle, MV_ACCESS_Exclusive, 0),
            "设备可能正被 MVS 客户端或其他程序占用",
        )

    def close_device(self, handle) -> None:
        self.check("MV_CC_CloseDevice", self.mv_close(handle))

    def destroy_handle(self, handle) -> None:
        self.check("MV_CC_DestroyHandle", self.mv_destroy(handle))

    def start_grabbing(self, handle) -> None:
        self.check("MV_CC_StartGrabbing", self.mv_start(handle))

    def stop_grabbing(self, handle) -> None:
        self.check("MV_CC_StopGrabbing", self.mv_stop(handle))

    def get_frame(self, handle, timeout_ms: int):
        frame = MvFrameOut()
        code = _unsigned32(self.mv_get_frame(handle, ctypes.byref(frame), int(timeout_ms)))
        return code, frame

    def free_frame(self, handle, frame: MvFrameOut) -> None:
        self.check("MV_CC_FreeImageBuffer", self.mv_free_frame(handle, ctypes.byref(frame)))

    def set_enum_string(self, handle, key: str, value: str) -> None:
        self.check(
            f"SetEnum {key}",
            self.mv_set_enum_string(
                handle, key.encode("ascii"), value.encode("ascii")
            ),
        )

    def set_enum(self, handle, key: str, value: int) -> None:
        self.check(
            f"SetEnum {key}",
            self.mv_set_enum(handle, key.encode("ascii"), ctypes.c_uint(value)),
        )

    def get_enum(self, handle, key: str) -> int:
        value = MvEnumValue()
        self.check(
            f"GetEnum {key}",
            self.mv_get_enum(handle, key.encode("ascii"), ctypes.byref(value)),
        )
        return int(value.nCurValue)

    def set_float(self, handle, key: str, value: float) -> None:
        self.check(
            f"SetFloat {key}",
            self.mv_set_float(handle, key.encode("ascii"), ctypes.c_float(value)),
        )

    def get_float(self, handle, key: str) -> MvFloatValue:
        value = MvFloatValue()
        self.check(
            f"GetFloat {key}",
            self.mv_get_float(handle, key.encode("ascii"), ctypes.byref(value)),
        )
        return value

    def set_int(self, handle, key: str, value: int) -> None:
        self.check(
            f"SetInt {key}",
            self.mv_set_int(handle, key.encode("ascii"), ctypes.c_int64(value)),
        )

    def get_int(self, handle, key: str) -> int:
        value = MvIntValueEx()
        self.check(
            f"GetInt {key}",
            self.mv_get_int(handle, key.encode("ascii"), ctypes.byref(value)),
        )
        return int(value.nCurValue)

    def set_bool(self, handle, key: str, value: bool) -> None:
        self.check(
            f"SetBool {key}",
            self.mv_set_bool(handle, key.encode("ascii"), ctypes.c_bool(value)),
        )

    def get_string(self, handle, key: str) -> str:
        value = MvStringValue()
        self.check(
            f"GetString {key}",
            self.mv_get_string(handle, key.encode("ascii"), ctypes.byref(value)),
        )
        return value.chCurValue.decode("utf-8", errors="replace").split("\0", 1)[0]


def enumerate_usb_cameras(
    library: str | Sequence[str] | None = None,
) -> tuple[list[CameraDescriptor], MvsSdk]:
    sdk = MvsSdk(library)
    sdk.initialize()
    try:
        return sdk.enumerate_usb(), sdk
    except Exception:
        sdk.finalize()
        raise


if __name__ == "__main__":
    cameras, sdk_instance = enumerate_usb_cameras()
    try:
        for camera in cameras:
            print(
                f"[{camera.index}] {camera.model} | SN={camera.serial or '-'} "
                f"| firmware={camera.device_version or '-'}"
            )
        if not cameras:
            print("未发现 USB3 Vision 相机。")
    finally:
        sdk_instance.finalize()
    sys.exit(0)
