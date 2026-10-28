"""Background screen capture with latest-frame delivery and ROI crops."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CaptureConfig:
    monitor: int = 1
    fps: int = 30
    use_bettercam: bool = True
    regions: dict[str, dict[str, int]] = field(default_factory=dict)
    resolution: dict[str, int] = field(default_factory=dict)


@dataclass
class FramePacket:
    image: np.ndarray
    regions: dict[str, np.ndarray]
    timestamp: float


FORWARD_SIZE = (640, 360)  # (width, height)


def enable_dpi_awareness() -> None:
    """Per-monitor DPI awareness so capture and click coordinates are physical pixels."""
    import sys

    if sys.platform == "win32":
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (OSError, AttributeError):
            pass


def _bettercam_monitors() -> list[dict[str, Any]]:
    try:
        import bettercam

        outputs = bettercam.__factory.outputs  # one list of outputs per GPU
        result = []
        index = 0
        for adapter in outputs:
            for output in adapter:
                index += 1
                width, height = output.resolution
                result.append({
                    "index": index,
                    "id": index - 1,
                    "width": width,
                    "height": height,
                    "primary": index == 1,
                    "label": f"Display {index}: {width}x{height}" + (" (Primary)" if index == 1 else ""),
                })
        return result
    except Exception:
        return []


def get_available_monitors() -> list[dict[str, Any]]:
    """Return attached displays as 1-based indexed dicts (bettercam outputs, else mss)."""
    found = _bettercam_monitors()
    if found:
        return found
    try:
        import mss
    except ImportError:
        return []
    try:
        with mss.mss() as capture:
            monitors = capture.monitors[1:]
    except Exception:
        return []
    result = []
    for index, mon in enumerate(monitors, start=1):
        primary = mon.get("left", 0) == 0 and mon.get("top", 0) == 0
        label = f"Display {index}: {mon['width']}x{mon['height']}"
        result.append({
            "index": index,
            "id": index - 1,
            "width": mon["width"],
            "height": mon["height"],
            "primary": primary,
            "label": label + (" (Primary)" if primary else ""),
        })
    return result


def get_open_windows() -> list[dict[str, Any]]:
    """Return visible titled top-level windows (Windows only)."""
    import sys

    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found: list[dict[str, Any]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)
                found.append({"hwnd": int(hwnd), "title": buffer.value})
        return True

    user32.EnumWindows(callback, 0)
    return found


def get_window_rect(hwnd: int) -> dict[str, int] | None:
    """Client-area rectangle of a window in screen coordinates, or None."""
    import sys

    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    if not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
        return None
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width < 16 or height < 16:
        return None
    return {"left": origin.x, "top": origin.y, "width": width, "height": height}


def _put_latest(target: queue.Queue, item: Any) -> None:
    try:
        target.put_nowait(item)
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        try:
            target.put_nowait(item)
        except queue.Full:
            pass


class CaptureWorker(threading.Thread):
    def __init__(
        self,
        config: CaptureConfig,
        output_queue: queue.Queue,
        status_queue: queue.Queue | None = None,
    ):
        super().__init__(name="screen-capture", daemon=True)
        self.config = config
        self.output_queue = output_queue
        self.status_queue = status_queue
        self._stop_event = threading.Event()
        self._switch_event = threading.Event()
        self._monitor = max(1, int(config.monitor))
        self.backend = "uninitialized"
        self._window: int | None = None
        self._latest: FramePacket | None = None
        self._forward_buffer = np.empty((FORWARD_SIZE[1], FORWARD_SIZE[0], 3), dtype=np.uint8)
        self._forward_lock = threading.Lock()

    def _publish(self, image: np.ndarray) -> None:
        packet = FramePacket(image, self._crop_regions(image), time.time())
        self._latest = packet
        _put_latest(self.output_queue, packet)

    def get_full_frame(self) -> np.ndarray | None:
        """Latest full-resolution BGR frame (shared, treat as read-only)."""
        packet = self._latest
        return None if packet is None else packet.image

    def get_forward_view(self, out: np.ndarray | None = None) -> np.ndarray | None:
        """Latest frame downscaled to 640x360. Pass ``out`` to avoid any allocation."""
        image = self.get_full_frame()
        if image is None:
            return None
        import cv2

        if out is not None:
            cv2.resize(image, FORWARD_SIZE, dst=out, interpolation=cv2.INTER_AREA)
            return out
        with self._forward_lock:
            return cv2.resize(image, FORWARD_SIZE, dst=self._forward_buffer, interpolation=cv2.INTER_AREA)

    def get_roi(self, name: str) -> np.ndarray | None:
        """Zero-copy view of a configured ROI inside the latest frame."""
        image = self.get_full_frame()
        bounds = self.config.regions.get(name)
        if image is None or not bounds:
            return None
        height, width = image.shape[:2]
        x, y = max(0, int(bounds["x"])), max(0, int(bounds["y"]))
        right, bottom = min(width, x + int(bounds["w"])), min(height, y + int(bounds["h"]))
        if right <= x or bottom <= y:
            return None
        return image[y:bottom, x:right]

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def monitor(self) -> int:
        return self._monitor

    def set_monitor(self, monitor_index: int) -> None:
        """Switch the live capture to another (1-based) display."""
        monitor_index = max(1, int(monitor_index))
        if monitor_index != self._monitor:
            self._monitor = monitor_index
            self._switch_event.set()

    @property
    def window(self) -> int | None:
        return self._window

    def set_window(self, hwnd: int | None) -> None:
        """Capture a specific window (None returns to full-display capture)."""
        if hwnd != self._window:
            self._window = hwnd
            self._switch_event.set()

    def _scale_to_resolution(self, image: np.ndarray) -> np.ndarray:
        width = int(self.config.resolution.get("width", 0))
        height = int(self.config.resolution.get("height", 0))
        if width <= 0 or height <= 0 or image.shape[:2] == (height, width):
            return image
        import cv2

        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def _should_exit(self) -> bool:
        return self._stop_event.is_set() or self._switch_event.is_set()

    def _status(self, message: str) -> None:
        if self.status_queue is not None:
            _put_latest(self.status_queue, {"kind": "status", "message": message})

    def _crop_regions(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        height, width = frame.shape[:2]
        result = {}
        for name, bounds in self.config.regions.items():
            x = max(0, int(bounds.get("x", 0)))
            y = max(0, int(bounds.get("y", 0)))
            right = min(width, x + max(0, int(bounds.get("w", 0))))
            bottom = min(height, y + max(0, int(bounds.get("h", 0))))
            if right > x and bottom > y:
                result[name] = frame[y:bottom, x:right].copy()
        return result

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._switch_event.clear()
            self._run_once()
            if not self._switch_event.is_set():
                return

    def _run_once(self) -> None:
        if self._window is not None:
            self._run_window()
            return
        if self.config.use_bettercam:
            camera = None
            try:
                import bettercam

                camera = bettercam.create(output_idx=self._monitor - 1, output_color="BGR")
                camera.start(target_fps=max(1, self.config.fps), video_mode=True)
                self.backend = "bettercam"
                self._status(f"Capture backend: bettercam, monitor {self._monitor}")
            except Exception as exc:
                self._release_camera(camera)
                camera = None
                self._status(f"bettercam unavailable ({exc}); falling back to mss")
            if camera is not None:
                self._run_bettercam(camera)
                return
        self._run_mss()

    @staticmethod
    def _release_camera(camera: Any) -> None:
        if camera is None:
            return
        for name in ("stop", "release"):
            try:
                getattr(camera, name)()
            except Exception:
                pass

    def _run_bettercam(self, camera: Any) -> None:
        interval = 1.0 / max(1, self.config.fps)
        try:
            while not self._should_exit():
                started = time.monotonic()
                frame = camera.get_latest_frame()
                if frame is not None:
                    image = np.asarray(frame)
                    self._publish(image)
                self._stop_event.wait(max(0.0, interval - (time.monotonic() - started)))
        finally:
            self._release_camera(camera)

    def _run_mss(self) -> None:
        try:
            import mss
        except ImportError as exc:
            self._status(f"Screen capture unavailable: install mss ({exc})")
            return
        self.backend = "mss"
        interval = 1.0 / max(1, self.config.fps)
        try:
            with mss.mss() as capture:
                monitors = capture.monitors
                monitor_index = min(max(1, self._monitor), len(monitors) - 1)
                monitor = monitors[monitor_index]
                self._status(f"Capture backend: mss, monitor {monitor_index}")
                while not self._should_exit():
                    started = time.monotonic()
                    image = np.asarray(capture.grab(monitor))[:, :, :3].copy()
                    self._publish(image)
                    self._stop_event.wait(max(0.0, interval - (time.monotonic() - started)))
        except Exception as exc:
            self._status(f"mss capture failed: {exc}")

    def _run_window(self) -> None:
        try:
            import mss
        except ImportError as exc:
            self._status(f"Window capture unavailable: install mss ({exc})")
            return
        self.backend = "mss-window"
        interval = 1.0 / max(1, self.config.fps)
        hwnd = self._window
        self._status(f"Capture backend: window {hwnd}, scaled to match resolution")
        try:
            with mss.mss() as capture:
                while not self._should_exit():
                    started = time.monotonic()
                    rect = get_window_rect(hwnd)
                    if rect is not None:
                        image = np.asarray(capture.grab(rect))[:, :, :3]
                        image = self._scale_to_resolution(np.ascontiguousarray(image))
                        self._publish(image)
                    self._stop_event.wait(max(0.0, interval - (time.monotonic() - started)))
        except Exception as exc:
            self._status(f"window capture failed: {exc}")
