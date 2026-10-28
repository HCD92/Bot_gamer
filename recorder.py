"""Manual route recorder: turns live driving input into agent route segments."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

import audio

MAPS = ("Bakurani", "Ozeti", "Zestafona")
FACTIONS = ("Manticore", "Valkyra", "Lonestar")
DIRECTIONS = ("TO_FOB", "TO_HQ")

TRACKED_KEYS = {"w": "W", "s": "S", "a": "A", "d": "D", "space": "SPACE"}
MIN_SEGMENT_SECONDS = 0.05
MAX_SEGMENT_SECONDS = 8.0  # agent clamps each segment to this length


def build_segments(
    steer_changes: list[tuple[float, str]],
    end_time: float,
    min_seconds: float = MIN_SEGMENT_SECONDS,
) -> list[dict[str, Any]]:
    """Convert (timestamp, steer) transitions into consolidated segments.

    Segments shorter than ``min_seconds`` are absorbed by their neighbour so
    total drive time is preserved while accidental taps disappear.
    """
    spans: list[list[Any]] = []
    for index, (start, steer) in enumerate(steer_changes):
        stop = steer_changes[index + 1][0] if index + 1 < len(steer_changes) else end_time
        if stop > start:
            spans.append([steer, stop - start])

    def merge(items: list[list[Any]]) -> list[list[Any]]:
        merged: list[list[Any]] = []
        for steer, length in items:
            if merged and merged[-1][0] == steer:
                merged[-1][1] += length
            else:
                merged.append([steer, length])
        return merged

    spans = merge(spans)
    while len(spans) > 1:
        shortest = min(range(len(spans)), key=lambda i: spans[i][1])
        if spans[shortest][1] >= min_seconds:
            break
        length = spans.pop(shortest)[1]
        spans[max(0, shortest - 1) if shortest > 0 else 0][1] += length
        spans = merge(spans)

    segments = []
    for steer, length in spans:
        if length < min_seconds:
            continue
        while length > 0:
            chunk = min(length, MAX_SEGMENT_SECONDS)
            if chunk >= min_seconds:
                segments.append({"steer": steer, "seconds": round(chunk, 3)})
            length -= chunk
    return segments


def summarize(segments: list[dict[str, Any]]) -> dict[str, Any]:
    turns, previous = 0, ""
    for segment in segments:
        steer = segment["steer"]
        if steer and steer != previous:
            turns += 1
        previous = steer
    return {
        "total_seconds": round(sum(s["seconds"] for s in segments), 3),
        "turns": turns,
        "segments": len(segments),
    }


def split_round_trip(
    steer_changes: list[tuple[float, str]],
    end_time: float,
    turnaround_time: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split steering changes into outbound and return route segments."""
    if not steer_changes or not steer_changes[0][0] <= turnaround_time < end_time:
        raise ValueError("Turnaround must be marked during the recording")

    outbound_changes = [(stamp, steer) for stamp, steer in steer_changes if stamp < turnaround_time]
    return_steer = steer_changes[0][1]
    for stamp, steer in steer_changes:
        if stamp > turnaround_time:
            break
        return_steer = steer
    return_changes = [(turnaround_time, return_steer)] + [
        (stamp, steer) for stamp, steer in steer_changes if stamp > turnaround_time
    ]
    return (
        build_segments(outbound_changes, turnaround_time),
        build_segments(return_changes, end_time),
    )


class RouteRecorder:
    """Thread-safe recorder; feed it key events or let ``install_hooks`` do so."""

    def __init__(self) -> None:
        from dataset_logger import DatasetLogger

        self._lock = threading.Lock()
        self._down: set[str] = set()
        self._steer_order: list[str] = []
        self._changes: list[tuple[float, str]] = []
        self._started = 0.0
        self._pressed_since: dict[str, float] = {}
        self._held_totals: dict[str, float] = {}
        self.recording = False
        self.segments: list[dict[str, Any]] = []
        self.round_trip_segments: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None
        self.turnaround_time: float | None = None
        self.stats: dict[str, Any] = {}
        self.message = "Idle. F8 start, F9 stop, F10 discard."
        self._keyboard = None
        self._handles: list[Any] = []
        self.logger = DatasetLogger()
        self.logger.key_source = self.key_snapshot
        self.record_frames = True
        self.map_name, self.faction, self.direction = MAPS[0], FACTIONS[0], DIRECTIONS[0]

    def _steer(self) -> str:
        # Opposing keys cancel; otherwise the most recently pressed wins.
        if "A" in self._down and "D" in self._down:
            return ""
        for key in reversed(self._steer_order):
            if key in self._down:
                return key
        return ""

    def key_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "w": "W" in self._down, "s": "S" in self._down,
                "a": "A" in self._down, "d": "D" in self._down,
                "space": "SPACE" in self._down, "steer": self._steer(),
            }

    def start(self) -> None:
        with self._lock:
            if self.recording:
                return
            now = time.perf_counter()
            self._started = now
            self._pressed_since = {key: now for key in self._down}
            self._held_totals = {}
            self._changes = [(now, self._steer())]
            self.segments, self.stats = [], {}
            self.round_trip_segments = None
            self.turnaround_time = None
            self.recording = True
            self.message = "RECORDING..."
        if self.record_frames and self.logger.frame_source is not None:
            self.logger.start(self.map_name, self.faction, self.direction)
        audio.high_single()

    def stop(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.recording:
                return self.segments
            now = time.perf_counter()
            self.recording = False
            for key, since in self._pressed_since.items():
                self._held_totals[key] = self._held_totals.get(key, 0.0) + now - since
            self.segments = build_segments(self._changes, now)
            self.stats = summarize(self.segments)
            self.stats["held_seconds"] = {k: round(v, 3) for k, v in self._held_totals.items()}
            if self.turnaround_time is not None:
                self.round_trip_segments = split_round_trip(
                    self._changes, now, self.turnaround_time
                )
                self.stats["outbound"] = summarize(self.round_trip_segments[0])
                self.stats["return"] = summarize(self.round_trip_segments[1])
            self.message = (
                f"Stopped: {self.stats['total_seconds']:.2f}s, "
                f"{self.stats['turns']} turns, {self.stats['segments']} segments"
            )
            if self.round_trip_segments is not None:
                outbound, returning = self.round_trip_segments
                self.message += f" · Round trip: {len(outbound)} out / {len(returning)} back"
            segments = self.segments
        self.logger.stop()  # outside the lock: the sampler thread calls key_snapshot()
        return segments

    def mark_turnaround(self) -> bool:
        with self._lock:
            if not self.recording:
                self.message = "Start a recording before marking the turnaround."
                return False
            if self.turnaround_time is not None:
                self.message = "Turnaround already marked."
                return False
            self.turnaround_time = time.perf_counter()
            self.message = "Turnaround marked; drive the return leg and stop when back."
            return True

    def abort(self) -> None:
        with self._lock:
            if not self.recording:
                return
            self.recording = False
            self.segments, self.stats = [], {}
            self.round_trip_segments = None
            self.turnaround_time = None
            self.message = "Recording discarded; nothing saved."
        self.logger.discard()

    def handle_event(self, name: str, is_down: bool, timestamp: float | None = None) -> None:
        key = TRACKED_KEYS.get(name.lower())
        if key is None:
            return
        now = time.perf_counter() if timestamp is None else timestamp
        with self._lock:
            if is_down == (key in self._down):
                return  # key auto-repeat
            if is_down:
                self._down.add(key)
                if key in ("A", "D"):
                    self._steer_order.append(key)
                if self.recording:
                    self._pressed_since[key] = now
            else:
                self._down.discard(key)
                if self.recording and key in self._pressed_since:
                    since = self._pressed_since.pop(key)
                    self._held_totals[key] = self._held_totals.get(key, 0.0) + now - since
            if self.recording:
                steer = self._steer()
                if steer != self._changes[-1][1]:
                    self._changes.append((now, steer))

    def install_hooks(self) -> bool:
        try:
            import keyboard
        except Exception as exc:
            self.message = f"Recorder hotkeys unavailable: {exc}"
            return False
        try:
            self._keyboard = keyboard
            self._handles = [
                keyboard.hook(
                    lambda event: self.handle_event(event.name or "", event.event_type == "down")
                ),
                keyboard.add_hotkey("f8", self.start, suppress=False),
                keyboard.add_hotkey("f9", self.stop, suppress=False),
                keyboard.add_hotkey("f10", self.abort, suppress=False),
                keyboard.add_hotkey("f11", self.mark_turnaround, suppress=False),
            ]
        except Exception as exc:
            self.message = f"Recorder hotkeys unavailable: {exc}"
            return False
        return True

    def remove_hooks(self) -> None:
        if self._keyboard is None:
            return
        hook, *hotkeys = self._handles
        for remove, handle in [(self._keyboard.unhook, hook)] + [
            (self._keyboard.remove_hotkey, h) for h in hotkeys
        ]:
            try:
                remove(handle)
            except Exception:
                pass
        self._handles = []
        self._keyboard = None


def update_config(path: str | Path, keys: list[str], value: Any) -> None:
    """Atomically set a nested key in the JSON config."""
    update_config_many(path, [(keys, value)])


def update_config_many(path: str | Path, updates: list[tuple[list[str], Any]]) -> None:
    """Atomically set multiple nested keys in the JSON config."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    for keys, value in updates:
        node = data
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def assign_route(
    path: str | Path,
    route_name: str,
    segments: list[dict[str, Any]],
    map_name: str | None = None,
) -> None:
    if route_name not in ("to_fob", "to_hq"):
        raise ValueError(f"Unknown route: {route_name}")
    if not segments:
        raise ValueError("No recorded segments to save")
    if map_name:
        update_config(path, ["routes", map_name, route_name], segments)
    else:
        update_config(path, ["agent", "route_segments", route_name], segments)


def assign_round_trip(
    path: str | Path,
    first_route: str,
    outbound_segments: list[dict[str, Any]],
    return_segments: list[dict[str, Any]],
    map_name: str | None = None,
) -> None:
    if first_route not in ("to_fob", "to_hq"):
        raise ValueError(f"Unknown route: {first_route}")
    if not outbound_segments or not return_segments:
        raise ValueError("Both round-trip legs need recorded segments")
    second_route = "to_hq" if first_route == "to_fob" else "to_fob"
    prefix = ["routes", map_name] if map_name else ["agent", "route_segments"]
    update_config_many(path, [
        ([*prefix, first_route], outbound_segments),
        ([*prefix, second_route], return_segments),
    ])
