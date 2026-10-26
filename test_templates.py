"""Check every template in assets/templates/ matches its source screenshot above 0.82."""
import sys
from pathlib import Path

import cv2

from extract_templates import TEMPLATES, find_source, normalise

THRESHOLD = 0.82


def main(root: str = ".") -> int:
    failed = 0
    for path in sorted(Path("assets/templates").glob("*.png")):
        source, preview_w, _ = TEMPLATES[path.stem]
        src = find_source(Path(root), source)
        template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        screen = cv2.cvtColor(normalise(cv2.imread(str(src)), preview_w), cv2.COLOR_BGR2GRAY)
        _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED))
        ok = score > THRESHOLD
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {path.name}: {score:.3f} at {loc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
