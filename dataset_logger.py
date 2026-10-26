"""Asynchronous 5 Hz multimodal dataset logger.

Layout: dataset/<map_name>/<session_id>/
    frames/frame_000000.jpg     forward view, 640x360, JPEG Q85
    minimap/minimap_000000.png  256x256 minimap crop
    telemetry.csv
"""

from __future__ import annotations

import csv
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

COLUMNS = [
    "frame_file", "minimap_file", "timestamp", "map_name", "faction",
    "w", "s", "a", "d", "space", "steer_label",
]
STRAIGHT, LEFT, RIGHT, BRAKE = 0, 1, 2, 3
CLASS_NAMES = ("STRAIGHT", "LEFT", "RIGHT", "BRAKE")
SAMPLE_HZ = 5
QUEUE_SIZE = 150
JPEG_QUALITY = 85
MINIMAP_SIZE = (256, 256)
FORWARD_SIZE = (640, 360)

# frame_source() -> (forward BGR frame, minimap BGR crop) or None; must return fresh copies
FrameSource = Callable[[], "tuple[np.ndarray, np.ndarray] | None"]
# key_source() -> {"w","s","a","d","space": bool, "steer": "A" | "D" | ""}
KeySource = Callable[[], dict]


def steer_label(keys: dict) -> int:
    """Brake (S or Space) outranks steering; otherwise A/D, else straight."""
    if keys.get("s") or keys.get("space"):
        return BRAKE
    steer = keys.get("steer", "")
    if steer == "A":
        return LEFT
    if steer == "D":
        return RIGHT
    return STRAIGHT


def prepare_sample(forward: np.ndarray, minimap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Copy and resize live frames to the on-disk dataset geometry."""
    if forward.shape[1::-1] != FORWARD_SIZE:
        forward = cv2.resize(forward, FORWARD_SIZE, interpolation=cv2.INTER_AREA)
    else:
        forward = forward.copy()
    return forward, cv2.resize(minimap, MINIMAP_SIZE, interpolation=cv2.INTER_AREA)


class DatasetLogger:
    def __init__(
        self,
        root: str | Path = "dataset",
        sample_hz: float = SAMPLE_HZ,
        queue_size: int = QUEUE_SIZE,
        jpeg_quality: int = JPEG_QUALITY,
    ) -> None:
        self.root = Path(root)
        self.interval = 1.0 / max(0.1, sample_hz)
        self.jpeg_quality = int(jpeg_quality)
        self.session_dir: Path | None = None
        self.map_name = ""
        self.faction = ""
        self.frame_source: FrameSource | None = None
        self.key_source: KeySource | None = None
        self.samples_written = 0
        self.dropped = 0
        self._queue_size = int(queue_size)
        self._stop = threading.Event()
        self._sampler: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=self._queue_size)
        self._lifecycle = threading.Lock()

    @property
    def active(self) -> bool:
        return self._sampler is not None

    def start(self, map_name: str, faction: str, direction: str) -> Path:
        self.stop()
        with self._lifecycle:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session = self.root / map_name / f"{faction}_{direction}_{stamp}"
            (session / "frames").mkdir(parents=True, exist_ok=True)
            (session / "minimap").mkdir(parents=True, exist_ok=True)
            self.session_dir = session
            self.map_name, self.faction = map_name, faction
            self.samples_written = self.dropped = 0
            self._queue = queue.Queue(maxsize=self._queue_size)
            self._stop.clear()
            self._writer = threading.Thread(
                target=self._write_loop, args=(session, self._queue),
                name="dataset-writer", daemon=True,
            )
            self._sampler = threading.Thread(
                target=self._sample_loop, args=(self._queue,), name="dataset-sampler", daemon=True,
            )
            self._writer.start()
            self._sampler.start()
        return session

    def _sample_loop(self, target: queue.Queue) -> None:
        index = 0
        next_tick = time.monotonic()
        while not self._stop.is_set():
            next_tick += self.interval
            try:
                sample = self.frame_source() if self.frame_source else None
            except Exception:
                sample = None
            if sample is not None:
                forward, minimap = sample
                keys = dict(self.key_source()) if self.key_source else {}
                try:
                    target.put_nowait((index, time.time(), forward, minimap, keys))
                    index += 1
                except queue.Full:
                    self.dropped += 1  # never block the sampling clock
            delay = next_tick - time.monotonic()
            if delay < 0:
                next_tick = time.monotonic()  # fell behind; resync instead of bursting
                delay = 0
            self._stop.wait(delay)

    def _write_loop(self, session: Path, source: queue.Queue) -> None:
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        with (session / "telemetry.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            handle.flush()
            while True:
                item = source.get()
                if item is None:
                    return
                index, stamp, forward, minimap, keys = item
                frame_file = f"frames/frame_{index:06d}.jpg"
                minimap_file = f"minimap/minimap_{index:06d}.png"
                try:
                    ok = cv2.imwrite(str(session / frame_file), forward, params)
                    ok = cv2.imwrite(str(session / minimap_file), minimap) and ok
                except Exception:
                    ok = False
                if not ok:
                    continue
                writer.writerow([
                    frame_file, minimap_file, f"{stamp:.3f}", self.map_name, self.faction,
                    int(bool(keys.get("w"))), int(bool(keys.get("s"))), int(bool(keys.get("a"))),
                    int(bool(keys.get("d"))), int(bool(keys.get("space"))), steer_label(keys),
                ])
                handle.flush()
                self.samples_written += 1

    def stop(self) -> Path | None:
        """Stop sampling and flush every queued frame to disk before returning."""
        with self._lifecycle:
            if self._sampler is None:
                return None
            self._stop.set()
            self._sampler.join(timeout=2.0)
            self._queue.put(None)  # unbounded wait is fine: the writer keeps draining
            if self._writer is not None:
                self._writer.join()
            self._sampler = self._writer = None
            return self.session_dir

    def discard(self) -> None:
        """Stop and delete the current session directory."""
        session = self.session_dir
        self.stop()
        if session is not None:
            shutil.rmtree(session, ignore_errors=True)
            self.session_dir = None
