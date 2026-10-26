"""Low-level DirectInput-style keyboard and mouse input for Windows."""

from __future__ import annotations

import ctypes
import os
import random
import threading
import time
from typing import Iterable


KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

SCANCODES = {
    "ESC": 0x01,
    "1": 0x02,
    "2": 0x03,
    "3": 0x04,
    "4": 0x05,
    "5": 0x06,
    "6": 0x07,
    "7": 0x08,
    "8": 0x09,
    "9": 0x0A,
    "0": 0x0B,
    "TAB": 0x0F,
    "Q": 0x10,
    "W": 0x11,
    "E": 0x12,
    "R": 0x13,
    "T": 0x14,
    "Y": 0x15,
    "U": 0x16,
    "I": 0x17,
    "O": 0x18,
    "P": 0x19,
    "A": 0x1E,
    "S": 0x1F,
    "D": 0x20,
    "F": 0x21,
    "G": 0x22,
    "H": 0x23,
    "L": 0x26,
    "B": 0x30,
    "M": 0x32,
    "SPACE": 0x39,
    "F12": 0x58,
    "ENTER": 0x1C,
    "UP": 0xC8,
    "LEFT": 0xCB,
    "RIGHT": 0xCD,
    "DOWN": 0xD0,
}


ULONG_PTR = ctypes.c_size_t


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_uint16),
        ("wScan", ctypes.c_uint16),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_int32),
        ("dy", ctypes.c_int32),
        ("mouseData", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("type", ctypes.c_uint32), ("data", _INPUTUNION)]


class InputController:
    """Send scancode input and make every hold interruptible by ``halt``."""

    def __init__(self, hold_noise: float = 0.08, delay_noise: float = 0.18):
        if os.name != "nt":
            raise OSError("Native SendInput automation is available only on Windows.")
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.SendInput.argtypes = (
            ctypes.c_uint32,
            ctypes.POINTER(_INPUT),
            ctypes.c_int,
        )
        self._user32.SendInput.restype = ctypes.c_uint32
        self._halted = threading.Event()
        self._halted.set()
        self._pressed: set[str] = set()
        self._lock = threading.RLock()
        self._hold_noise = max(0.0, hold_noise)
        self._delay_noise = max(0.0, delay_noise)
        self.active_key = "--"
        self._pulsing_handbrake = False

    @property
    def halted(self) -> bool:
        return self._halted.is_set()

    def enable(self) -> None:
        self.release_all()
        self._halted.clear()

    def halt(self) -> None:
        self._halted.set()
        self.release_all()

    def emergency_stop(self, handbrake_seconds: float = 0.5) -> None:
        """Halt all synthesis, release movement keys, and pulse Space to stop the vehicle."""
        self._halted.set()
        self._pulsing_handbrake = True
        try:
            self.release_all()
            for key in ("W", "A", "D", "S"):
                try:
                    self._keyboard_event(key, key_up=True, force=True)
                except OSError:
                    pass
            self.active_key = "SPACE"
            try:
                self._keyboard_event("SPACE", force=True)
                time.sleep(max(0.0, handbrake_seconds))
            finally:
                self._pulsing_handbrake = False
                try:
                    self._keyboard_event("SPACE", key_up=True, force=True)
                except OSError:
                    pass
        finally:
            self._pulsing_handbrake = False
            self.active_key = "--"

    def _send(self, event: _INPUT, force: bool = False) -> bool:
        if self._halted.is_set() and not force:
            is_key_release = event.type == 1 and bool(event.ki.dwFlags & KEYEVENTF_KEYUP)
            if not is_key_release:
                return False
        sent = self._user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(_INPUT))
        if sent != 1:
            raise ctypes.WinError(ctypes.get_last_error())
        return True

    def _keyboard_event(self, key: str, key_up: bool = False, force: bool = False) -> bool:
        normalized = key.upper()
        if normalized not in SCANCODES:
            raise ValueError(f"Unsupported key: {key}")
        flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if key_up else 0)
        event = _INPUT(type=1, ki=_KEYBDINPUT(0, SCANCODES[normalized], flags, 0, 0))
        if not self._send(event, force):
            return False
        with self._lock:
            if key_up:
                self._pressed.discard(normalized)
            else:
                self._pressed.add(normalized)
        return True

    def _duration(self, seconds: float) -> float:
        deviation = max(0.005, seconds * self._hold_noise)
        return max(0.025, random.gauss(seconds, deviation))

    def _delay(self, seconds: float) -> None:
        if seconds > 0 and not self._halted.is_set():
            deviation = max(0.002, seconds * self._delay_noise)
            self._halted.wait(max(0.0, random.gauss(seconds, deviation)))

    def press(self, key: str, hold_seconds: float = 0.12, delay_seconds: float = 0.08) -> bool:
        if self._halted.is_set():
            return False
        normalized = key.upper()
        self.active_key = normalized
        if not self._keyboard_event(normalized):
            self.active_key = "--"
            return False
        deadline = time.monotonic() + self._duration(max(0.025, hold_seconds))
        try:
            while not self._halted.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._halted.wait(min(0.02, remaining))
        finally:
            self._keyboard_event(normalized, key_up=True)
            self.active_key = "--"
        self._delay(delay_seconds)
        return not self._halted.is_set()

    def hold(self, key: str, seconds: float, delay_seconds: float = 0.0) -> bool:
        return self.press(key, hold_seconds=seconds, delay_seconds=delay_seconds)

    def hold_keys(self, keys: Iterable[str], seconds: float) -> bool:
        normalized = tuple(key.upper() for key in keys)
        if not normalized or self._halted.is_set():
            return False
        self.active_key = "+".join(normalized)
        pressed: list[str] = []
        try:
            for key in normalized:
                if not self._keyboard_event(key):
                    return False
                pressed.append(key)
            deadline = time.monotonic() + self._duration(max(0.025, seconds))
            while not self._halted.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._halted.wait(min(0.02, remaining))
            return not self._halted.is_set()
        finally:
            for key in reversed(pressed):
                self._keyboard_event(key, key_up=True)
            self.active_key = "--"

    def tap_sequence(self, keys: Iterable[str], hold_seconds: float = 0.10) -> bool:
        for key in keys:
            if not self.press(key, hold_seconds=hold_seconds):
                return False
        return True

    def move_mouse_bezier(
        self,
        dx: int,
        dy: int,
        duration: float = 0.35,
        steps: int = 16,
        jitter_px: float = 0.35,
    ) -> bool:
        """Move relatively along a cubic curve with small endpoint-safe jitter."""
        if self._halted.is_set():
            return False
        steps = max(4, steps)
        control_1 = (dx * 0.28, dy * 0.05)
        control_2 = (dx * 0.72, dy * 0.95)
        previous_x = previous_y = 0
        self.active_key = "MOUSE"
        try:
            for index in range(1, steps + 1):
                if self._halted.is_set():
                    return False
                t = index / steps
                inverse = 1.0 - t
                x = (
                    3 * inverse**2 * t * control_1[0]
                    + 3 * inverse * t**2 * control_2[0]
                    + t**3 * dx
                )
                y = (
                    3 * inverse**2 * t * control_1[1]
                    + 3 * inverse * t**2 * control_2[1]
                    + t**3 * dy
                )
                if index < steps:
                    x += random.gauss(0.0, jitter_px)
                    y += random.gauss(0.0, jitter_px)
                move_x, move_y = round(x - previous_x), round(y - previous_y)
                previous_x, previous_y = x, y
                event = _INPUT(
                    type=0,
                    mi=_MOUSEINPUT(move_x, move_y, 0, MOUSEEVENTF_MOVE, 0, 0),
                )
                self._send(event)
                self._halted.wait(max(0.005, duration / steps))
        finally:
            self.active_key = "--"
        return not self._halted.is_set()

    def click_at(self, x: int, y: int, hold_seconds: float = 0.08) -> bool:
        """Left-click at absolute virtual-desktop pixel coordinates (DPI-aware process)."""
        if self._halted.is_set():
            return False
        width = self._user32.GetSystemMetrics(78) or 1
        height = self._user32.GetSystemMetrics(79) or 1
        left = self._user32.GetSystemMetrics(76)
        top = self._user32.GetSystemMetrics(77)
        nx = round((x - left) * 65535 / max(1, width - 1))
        ny = round((y - top) * 65535 / max(1, height - 1))
        base = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        self.active_key = "CLICK"
        try:
            if not self._send(_INPUT(type=0, mi=_MOUSEINPUT(nx, ny, 0, base | MOUSEEVENTF_MOVE, 0, 0))):
                return False
            self._halted.wait(0.06)
            if not self._send(_INPUT(type=0, mi=_MOUSEINPUT(nx, ny, 0, base | MOUSEEVENTF_LEFTDOWN, 0, 0))):
                return False
            self._halted.wait(self._duration(hold_seconds))
            self._send(_INPUT(type=0, mi=_MOUSEINPUT(nx, ny, 0, base | MOUSEEVENTF_LEFTUP, 0, 0)))
        finally:
            self.active_key = "--"
        return not self._halted.is_set()

    def release_all(self) -> None:
        with self._lock:
            pressed = tuple(self._pressed)
        for key in pressed:
            if key == "SPACE" and self._pulsing_handbrake:
                continue
            try:
                self._keyboard_event(key, key_up=True)
            except OSError:
                pass
