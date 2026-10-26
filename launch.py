"""Start the app and report otherwise-hidden failures from pythonw.exe."""

from __future__ import annotations

import ctypes
import traceback
from pathlib import Path


def main() -> int:
    try:
        from main import main as run_app

        return run_app()
    except Exception:
        details = traceback.format_exc()
        log_path = Path(__file__).with_name("launch-error.log")
        try:
            log_path.write_text(details, encoding="utf-8")
            message = (
                "The control panel could not start. "
                f"See {log_path} for details."
            )
        except OSError as exc:
            message = f"The control panel could not start:\n\n{details}\nLog error: {exc}"
        ctypes.windll.user32.MessageBoxW(
            None,
            message,
            "WARDOGS Logistics Control",
            0x10,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
