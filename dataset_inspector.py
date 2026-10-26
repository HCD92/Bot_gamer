"""Inspect recorded driving telemetry and report training readiness.

Reads dataset/session_*/telemetry.csv. Recognised columns (case-insensitive):
  steer   - 'A'/'D'/''  (also 'left'/'right'/'straight' or 'steer'/'action'/'label' names)
  speed   - vehicle speed, 0 = stationary
  heading - compass/minimap heading in degrees (0-360)
"""
from __future__ import annotations

import csv
from pathlib import Path

SAMPLE_RATE_HZ = 5
MIN_SAMPLES = 5000
MAX_STRAIGHT_PCT = 65.0
BALANCE_RANGE = (0.7, 1.3)
OFF_ANGLE_DEG = 25.0
STEER_COLUMNS = ("steer", "action", "label", "steering")


def _steer_class(raw: str) -> str:
    value = (raw or "").strip().lower()
    if value in ("a", "left", "steer left (a)", "steer_left", "l"):
        return "left"
    if value in ("d", "right", "steer right (d)", "steer_right", "r"):
        return "right"
    return "straight"


def _float(raw) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _road_headings(headings: list[float]) -> list[float]:
    """Heading bins (15 deg) holding at least 10% of frames approximate road directions."""
    if not headings:
        return []
    bins: dict[int, int] = {}
    for h in headings:
        key = int((h % 360) // 15)
        bins[key] = bins.get(key, 0) + 1
    return [k * 15 + 7.5 for k, n in bins.items() if n >= 0.1 * len(headings)]


def count_recovery_transitions(headings: list[float]) -> int:
    roads = _road_headings(headings)
    if not roads:
        return 0
    off = False
    count = 0
    for h in headings:
        on_road = min(_angle_diff(h, r) for r in roads) <= OFF_ANGLE_DEG
        if not on_road:
            off = True
        elif off:
            count += 1
            off = False
    return count


def analyze_dataset(dataset_dir: str | Path = "dataset") -> dict:
    root = Path(dataset_dir)
    files = sorted(root.glob("session_*/telemetry.csv")) + sorted(root.glob("*/*/telemetry.csv"))
    counts = {"straight": 0, "left": 0, "right": 0}
    total = idle = 0
    headings_all: list[float] = []
    recovery = 0
    for path in files:
        session_headings: list[float] = []
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    row = {(k or "").strip().lower(): v for k, v in row.items()}
                    total += 1
                    raw = next((row[c] for c in STEER_COLUMNS if c in row), "")
                    counts[_steer_class(raw)] += 1
                    speed = _float(row.get("speed"))
                    if speed is not None and speed == 0:
                        idle += 1
                    heading = _float(row.get("heading"))
                    if heading is not None:
                        session_headings.append(heading % 360)
        except (OSError, csv.Error):
            continue
        headings_all.extend(session_headings)
        recovery += count_recovery_transitions(session_headings)
    pct = {k: (v * 100.0 / total if total else 0.0) for k, v in counts.items()}
    covered = len({int(h // 10) for h in headings_all})
    return {
        "sessions": len(files),
        "total_samples": total,
        "duration_minutes": total / SAMPLE_RATE_HZ / 60.0,
        "counts": counts,
        "pct": pct,
        "idle_frames": idle,
        "moving_frames": total - idle,
        "idle_pct": idle * 100.0 / total if total else 0.0,
        "heading_coverage_pct": covered * 100.0 / 36,
        "heading_range": (min(headings_all), max(headings_all)) if headings_all else None,
        "recovery_transitions": recovery,
    }


def get_readiness_report(dataset_dir: str | Path = "dataset") -> dict:
    """Return {'score': 0-100, 'status': str, 'advice': [str], 'metrics': dict}."""
    m = analyze_dataset(dataset_dir)
    advice: list[str] = []
    moving = m["moving_frames"]
    volume = min(moving / MIN_SAMPLES, 1.0)
    if moving < MIN_SAMPLES:
        advice.append(f"Record {MIN_SAMPLES - moving} more moving frames (need {MIN_SAMPLES}).")
    if m["idle_pct"] > 30:
        advice.append(f"{m['idle_pct']:.0f}% of frames are stationary; drive more continuously.")

    straight = m["pct"]["straight"]
    straight_ok = m["total_samples"] > 0 and straight <= MAX_STRAIGHT_PCT
    if m["total_samples"] and not straight_ok:
        advice.append(f"Straight bias {straight:.0f}% > {MAX_STRAIGHT_PCT:.0f}%; record more turns.")

    left, right = m["counts"]["left"], m["counts"]["right"]
    ratio = left / right if right else None
    balance_ok = ratio is not None and BALANCE_RANGE[0] <= ratio <= BALANCE_RANGE[1]
    need_left = need_right = False
    if not balance_ok and m["total_samples"]:
        need_left = right > 0 and (ratio is None or ratio < BALANCE_RANGE[0]) or (left == 0)
        need_right = not need_left
        advice.append(
            "Left/Right ratio out of 0.7-1.3; record more "
            + ("left" if need_left else "right") + " turns."
        )

    recovery_ok = m["recovery_transitions"] > 0
    if not recovery_ok:
        advice.append("No recovery frames (off-angle back to road heading); record corrections.")

    score = 100.0 * (0.4 * volume + 0.2 * straight_ok + 0.2 * balance_ok + 0.2 * recovery_ok)
    if moving < MIN_SAMPLES:
        status = "Low Volume"
    elif need_left:
        status = "Incomplete - Need More Left Turns"
    elif need_right:
        status = "Incomplete - Need More Right Turns"
    elif advice and not (straight_ok and recovery_ok):
        status = "Incomplete - " + ("Too Much Straight" if not straight_ok else "Need Recovery Data")
    else:
        status = "Ready for Training"
    return {"score": round(score, 1), "status": status, "advice": advice, "metrics": m}


if __name__ == "__main__":
    r = get_readiness_report()
    m = r["metrics"]
    print(f"Sessions: {m['sessions']}  Samples: {m['total_samples']}  Duration: {m['duration_minutes']:.1f} min")
    print(f"Straight {m['pct']['straight']:.1f}% | Left {m['pct']['left']:.1f}% | Right {m['pct']['right']:.1f}%")
    print(f"Idle frames: {m['idle_frames']} ({m['idle_pct']:.1f}%)  Heading coverage: {m['heading_coverage_pct']:.0f}%")
    print(f"Readiness: {r['score']}% [{r['status']}]")
    for a in r["advice"]:
        print(" -", a)
