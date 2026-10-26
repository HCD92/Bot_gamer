"""Non-blocking Windows audio cues (silent no-ops on other platforms)."""

from __future__ import annotations

import threading

try:
    import winsound
except ImportError:  # non-Windows
    winsound = None

HIGH_HZ, HIGH_MS = 1200, 150
LOW_HZ, LOW_MS = 600, 250


def _play(tones: list[tuple[int, int]]) -> None:
    if winsound is None:
        return

    def worker() -> None:
        for frequency, duration in tones:
            try:
                winsound.Beep(frequency, duration)
            except RuntimeError:
                return

    threading.Thread(target=worker, name="audio-cue", daemon=True).start()


def high_single() -> None:
    """Recording started or pallet purchased."""
    _play([(HIGH_HZ, HIGH_MS)])


def high_double() -> None:
    """Delivery verified."""
    _play([(HIGH_HZ, HIGH_MS), (HIGH_HZ, HIGH_MS)])


def low_double() -> None:
    """Emergency stop or recovery state active."""
    _play([(LOW_HZ, LOW_MS), (LOW_HZ, LOW_MS)])
