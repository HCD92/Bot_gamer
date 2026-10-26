"""Template, color-mask, and optional OCR verification for game UI states."""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from capture import FramePacket, _put_latest


@dataclass(frozen=True)
class VisionConfig:
    template_dir: str = "assets/templates"
    thresholds: dict[str, float] = field(default_factory=dict)
    template_regions: dict[str, str] = field(default_factory=dict)
    ocr_regions: dict[str, str] = field(default_factory=dict)
    color_ranges: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    ocr_interval: float = 0.75

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisionConfig":
        return cls(
            template_dir=data.get("template_dir", cls.template_dir),
            thresholds=data.get("thresholds", {}),
            template_regions=data.get("template_regions", {}),
            ocr_regions=data.get("ocr_regions", {}),
            color_ranges=data.get("color_ranges", {}),
            ocr_interval=float(data.get("ocr_interval", 0.75)),
        )


@dataclass
class Detection:
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass
class VisionObservation:
    timestamp: float
    detections: dict[str, Detection]
    values: dict[str, int | float | None]

    def detected(self, name: str) -> bool:
        return name in self.detections


class VisionEngine:
    def __init__(self, config: VisionConfig, roi_bounds: dict[str, dict[str, int]]):
        self.config = config
        self.roi_bounds = roi_bounds
        self.templates: dict[str, np.ndarray] = {}
        self._load_templates()
        self._tesseract = None
        try:
            import pytesseract

            self._tesseract = pytesseract
        except ImportError:
            pass
        self._last_ocr = 0.0
        self._last_values: dict[str, int | float | None] = {}

    def _load_templates(self) -> None:
        directory = Path(self.config.template_dir)
        if not directory.exists():
            return
        for path in directory.glob("*.png"):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None and image.size:
                self.templates[path.stem] = image

    def _region_image(self, name: str, packet: FramePacket) -> tuple[np.ndarray, tuple[int, int]]:
        roi_name = self.config.template_regions.get(name)
        if roi_name and roi_name in packet.regions:
            bounds = self.roi_bounds.get(roi_name, {})
            return packet.regions[roi_name], (int(bounds.get("x", 0)), int(bounds.get("y", 0)))
        return packet.image, (0, 0)

    @staticmethod
    def _match(image: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int]] | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        th, tw = template.shape[:2]
        if gray.shape[0] < th or gray.shape[1] < tw:
            return None
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        return float(score), location

    def _detections(self, packet: FramePacket) -> dict[str, Detection]:
        found: dict[str, Detection] = {}
        for name, template in self.templates.items():
            image, offset = self._region_image(name, packet)
            match = self._match(image, template)
            if match is None:
                continue
            score, (x, y) = match
            if score >= float(self.config.thresholds.get(name, 0.86)):
                height, width = template.shape[:2]
                found[name] = Detection(
                    score,
                    (x + offset[0], y + offset[1], width, height),
                )
        for name, bounds in self.config.color_ranges.items():
            roi_name = bounds.get("region")
            if roi_name and roi_name in packet.regions:
                image = packet.regions[roi_name]
                offset_data = self.roi_bounds.get(roi_name, {})
                offset = (int(offset_data.get("x", 0)), int(offset_data.get("y", 0)))
            else:
                image, offset = packet.image, (0, 0)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower = np.asarray(bounds.get("lower", [0, 0, 0]), dtype=np.uint8)
            upper = np.asarray(bounds.get("upper", [179, 255, 255]), dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            minimum_area = int(bounds.get("min_area", 100))
            candidates = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) >= minimum_area]
            if candidates:
                x, y, width, height = max(candidates, key=lambda box: box[2] * box[3])
                found[name] = Detection(1.0, (x + offset[0], y + offset[1], width, height))
        return found

    def _read_ocr(self, packet: FramePacket) -> dict[str, int | float | None]:
        now = time.monotonic()
        if now - self._last_ocr < max(0.1, self.config.ocr_interval):
            return self._last_values.copy()
        self._last_ocr = now
        values: dict[str, int | float | None] = {}
        if self._tesseract is None:
            self._last_values = values
            return values
        for value_name, roi_name in self.config.ocr_regions.items():
            image = packet.regions.get(roi_name)
            if image is None:
                values[value_name] = None
                continue
            try:
                text = self._tesseract.image_to_string(
                    image,
                    config="--psm 7 -c tessedit_char_whitelist=0123456789.-",
                )
                match = re.search(r"-?\d+(?:\.\d+)?", text)
                if match is None:
                    values[value_name] = None
                elif "." in match.group():
                    values[value_name] = float(match.group())
                else:
                    values[value_name] = int(match.group())
            except Exception:
                values[value_name] = None
        self._last_values = values
        return values

    def process(self, packet: FramePacket) -> tuple[VisionObservation, np.ndarray, np.ndarray]:
        detections = self._detections(packet)
        values = self._read_ocr(packet)
        observation = VisionObservation(packet.timestamp, detections, values)
        annotated = packet.image.copy()
        for name, detection in detections.items():
            x, y, width, height = detection.bbox
            cv2.rectangle(annotated, (x, y), (x + width, y + height), (70, 220, 100), 2)
            cv2.putText(
                annotated,
                f"{name} {detection.confidence:.2f}",
                (x, max(18, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (70, 220, 100),
                1,
                cv2.LINE_AA,
            )
        for name, bounds in self.roi_bounds.items():
            x, y = int(bounds.get("x", 0)), int(bounds.get("y", 0))
            width, height = int(bounds.get("w", 0)), int(bounds.get("h", 0))
            if width > 0 and height > 0:
                cv2.rectangle(annotated, (x, y), (x + width, y + height), (235, 170, 55), 1)
                cv2.putText(annotated, name, (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 170, 55), 1)
        roi_preview = self._compose_roi_preview(packet.regions)
        return observation, annotated, roi_preview

    @staticmethod
    def _compose_roi_preview(regions: dict[str, np.ndarray]) -> np.ndarray:
        if not regions:
            return np.zeros((120, 240, 3), dtype=np.uint8)
        tiles = []
        for name, image in regions.items():
            thumb = cv2.resize(image, (220, 110), interpolation=cv2.INTER_AREA)
            cv2.putText(thumb, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(thumb)
        return cv2.vconcat(tiles[:4])


class VisionWorker(threading.Thread):
    def __init__(
        self,
        config: VisionConfig,
        roi_bounds: dict[str, dict[str, int]],
        frame_queue: queue.Queue,
        observation_queue: queue.Queue,
        render_queue: queue.Queue,
        status_queue: queue.Queue | None = None,
    ):
        super().__init__(name="vision", daemon=True)
        self.engine = VisionEngine(config, roi_bounds)
        self.frame_queue = frame_queue
        self.observation_queue = observation_queue
        self.render_queue = render_queue
        self.status_queue = status_queue
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        if self.status_queue is not None:
            _put_latest(self.status_queue, {"kind": "status", "message": f"Loaded {len(self.engine.templates)} templates"})
        while not self._stop_event.is_set():
            try:
                packet = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                observation, annotated, roi_preview = self.engine.process(packet)
                _put_latest(self.observation_queue, observation)
                _put_latest(
                    self.render_queue,
                    {"image": annotated, "roi_preview": roi_preview, "timestamp": packet.timestamp},
                )
            except Exception as exc:
                if self.status_queue is not None:
                    _put_latest(self.status_queue, {"kind": "error", "message": f"Vision error: {exc}"})


def load_vision_config(path: str | Path) -> tuple[VisionConfig, dict[str, dict[str, int]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return VisionConfig.from_dict(data.get("vision", {})), data.get("capture", {}).get("regions", {})
