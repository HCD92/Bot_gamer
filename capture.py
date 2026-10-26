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
        self.backend = "uninitialized"

    def stop(self) -> None:
        self._stop_event.set()

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
        if self.config.use_bettercam:
            try:
                import bettercam

                camera = bettercam.create(output_idx=max(0, self.config.monitor - 1), output_color="BGR")
                camera.start(target_fps=max(1, self.config.fps), video_mode=True)
                self.backend = "bettercam"
                self._status("Capture backend: bettercam")
                self._run_bettercam(camera)
                return
            except Exception as exc:
                self._status(f"bettercam unavailable ({exc}); falling back to mss")
        self._run_mss()

    def _run_bettercam(self, camera: Any) -> None:
        interval = 1.0 / max(1, self.config.fps)
        try:
            while not self._stop_event.is_set():
                started = time.monotonic()
                frame = camera.get_latest_frame()
                if frame is not None:
                    image = np.asarray(frame)
                    _put_latest(
                        self.output_queue,
                        FramePacket(image, self._crop_regions(image), time.time()),
                    )
                self._stop_event.wait(max(0.0, interval - (time.monotonic() - started)))
        finally:
            try:
                camera.stop()
            except Exception:
                pass

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
                monitor_index = min(max(1, self.config.monitor), len(monitors) - 1)
                monitor = monitors[monitor_index]
                self._status(f"Capture backend: mss, monitor {monitor_index}")
                while not self._stop_event.is_set():
                    started = time.monotonic()
                    image = np.asarray(capture.grab(monitor))[:, :, :3].copy()
                    _put_latest(
                        self.output_queue,
                        FramePacket(image, self._crop_regions(image), time.time()),
                    )
                    self._stop_event.wait(max(0.0, interval - (time.monotonic() - started)))
        except Exception as exc:
            self._status(f"mss capture failed: {exc}")
