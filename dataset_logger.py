"""Partitioned telemetry logging: dataset/<map>/<faction>_<direction>_<timestamp>/telemetry.csv."""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime
from pathlib import Path

COLUMNS = ["timestamp", "key", "is_down", "steer", "map_name", "faction"]


class DatasetLogger:
    def __init__(self, root: str | Path = "dataset") -> None:
        self.root = Path(root)
        self.session_dir: Path | None = None
        self.map_name = ""
        self.faction = ""
        self._file = None
        self._writer = None
        self._lock = threading.Lock()
        self._t0 = 0.0

    @property
    def active(self) -> bool:
        return self._writer is not None

    def start(self, map_name: str, faction: str, direction: str) -> Path:
        self.stop()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = self.root / map_name / f"{faction}_{direction}_{stamp}"
        session.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.session_dir = session
            self.map_name, self.faction = map_name, faction
            self._file = (session / "telemetry.csv").open("w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(COLUMNS)
            self._t0 = time.perf_counter()
        return session

    def log(self, key: str, is_down: bool, steer: str) -> None:
        with self._lock:
            if self._writer is None:
                return
            self._writer.writerow([
                round(time.perf_counter() - self._t0, 4), key, int(is_down), steer,
                self.map_name, self.faction,
            ])
            self._file.flush()

    def stop(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
            self._file = self._writer = None
