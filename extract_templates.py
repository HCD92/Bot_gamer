"""Crop reference templates from raw screenshots into assets/templates/.

Full-screen shots are normalised to 2560x1440 first. Boxes are (x1, y1, x2, y2)
in the coordinate space of the preview width listed in `preview_w`.
"""
import sys
from pathlib import Path

import cv2

TARGET_W, TARGET_H = 2560, 1440
OUT_DIR = Path("assets/templates")
# Raw captures are ~3838 px wide; the tactical-map crop was taken at that same scale.
RAW_SCALE = TARGET_W / 3838

# name: (source file, preview_w or None for native-pixel boxes on a pre-cropped capture, box)
TEMPLATES = {
    "pallet_menu_open": ("Screenshot 2026-10-26 123111.jpg", 2000, (100, 6, 765, 66)),
    "hq_safe_zone": ("Screenshot 2026-10-26 123053.jpg", 1527, (716, 52, 812, 73)),
    "buy_supplies_prompt": ("Screenshot 2026-10-26 123053.jpg", 1527, (688, 660, 846, 680)),
    "friendly_fob_map": ("Screenshot 2026-10-26 123148.jpg", None, (152, 110, 298, 258)),
    "supplies_delivered": ("Screenshot 2026-10-26 123656.jpg", None, None),
    "delivery_payout": ("Screenshot 2026-10-26 123656.jpg", None, None),
}


def find_source(root: Path, name: str) -> Path | None:
    stem = Path(name).stem
    for variant in {stem, stem.replace(" ", "_")}:
        for ext in (".jpg", ".png", ".jpeg"):
            path = root / f"{variant}{ext}"
            if path.exists():
                return path
    return None


def normalise(image, preview_w):
    if preview_w is None:  # pre-cropped capture: only apply the raw->1440p scale
        return cv2.resize(image, None, fx=RAW_SCALE, fy=RAW_SCALE, interpolation=cv2.INTER_AREA)
    return cv2.resize(image, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


def main(root: str = ".") -> int:
    root_path = Path(root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = 0
    for name, (source, preview_w, box) in TEMPLATES.items():
        path = find_source(root_path, source)
        if path is None or box is None:
            print(f"SKIP {name}: {'source ' + source + ' not found' if path is None else 'box not defined'}")
            missing += 1
            continue
        image = normalise(cv2.imread(str(path)), preview_w)
        k = RAW_SCALE if preview_w is None else TARGET_W / preview_w
        x1, y1, x2, y2 = (round(v * k) for v in box)
        cv2.imwrite(str(OUT_DIR / f"{name}.png"), image[y1:y2, x1:x2])
        print(f"OK   {name}: {x2 - x1}x{y2 - y1} from {path.name}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
