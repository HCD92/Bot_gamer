"""Application entry point and worker coordination."""

from __future__ import annotations

import ctypes
import json
import queue
import sys
from pathlib import Path

if sys.platform == "win32":
    ctypes.windll.shcore.SetProcessDpiAwareness(2)

from PyQt6.QtWidgets import QApplication

from agent import AgentConfig, LogisticsAgent
from capture import CaptureConfig, CaptureWorker
from gui import ControlPanel
from input_controller import InputController
from vision import VisionConfig, VisionWorker


def _queue() -> queue.Queue:
    return queue.Queue(maxsize=2)


def main() -> int:
    config_path = Path(__file__).with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    app = QApplication(sys.argv)
    app.setApplicationName("Wardogs Logistics Control")

    try:
        controller = InputController(
            hold_noise=float(config.get("input", {}).get("hold_noise", 0.08)),
            delay_noise=float(config.get("input", {}).get("delay_noise", 0.18)),
        )
    except OSError as exc:
        panel = ControlPanel(_queue(), _queue(), _queue(), _queue())
        panel.status_label.setText(str(exc))
        panel.start_button.setEnabled(False)
        panel.show()
        return app.exec()

    commands, telemetry = queue.Queue(), _queue()
    frames, observations, renders, status = _queue(), _queue(), _queue(), _queue()
    capture_data = dict(config.get("capture", {}))
    display_index = config.get("display", {}).get("monitor_index")
    if display_index is not None:
        capture_data["monitor"] = int(display_index)
    capture_config = CaptureConfig(**capture_data)
    vision_data = config.get("vision", {})
    template_dir = Path(vision_data.get("template_dir", "assets/templates"))
    if not template_dir.is_absolute():
        vision_data["template_dir"] = str(config_path.parent / template_dir)
    vision_config = VisionConfig.from_dict(vision_data)
    agent_config = AgentConfig.from_dict(config.get("agent", {}), str(config_path))
    capture_worker = CaptureWorker(capture_config, frames, status)
    vision_worker = VisionWorker(
        vision_config,
        capture_config.regions,
        frames,
        observations,
        renders,
        status,
    )
    agent_worker = LogisticsAgent(
        agent_config,
        controller,
        commands,
        observations,
        telemetry,
        initial_priority=config.get("agent", {}).get("initial_priority", "Build"),
    )
    panel = ControlPanel(commands, telemetry, renders, status, halt_input=controller.halt,
        capture_worker=capture_worker,
        config_path=config_path,
    )

    hotkey_module = None
    hotkey_handles = []
    try:
        import keyboard

        hotkey_module = keyboard

        def emergency_hotkey() -> None:
            controller.halt()
            commands.put({"kind": "emergency"})

        hotkey_handles.append(keyboard.add_hotkey("f12", emergency_hotkey, suppress=False))
        hotkey_handles.append(keyboard.add_hotkey("esc", emergency_hotkey, suppress=False))
        status.put({"kind": "status", "message": "F12 or Esc: emergency stop"})
    except Exception as exc:
        status.put({"kind": "status", "message": f"Global hotkeys unavailable: {exc}"})

    capture_worker.start()
    vision_worker.start()
    agent_worker.start()

    def shutdown() -> None:
        controller.halt()
        capture_worker.stop()
        vision_worker.stop()
        agent_worker.stop()
        if hotkey_module is not None:
            for handle in hotkey_handles:
                try:
                    hotkey_module.remove_hotkey(handle)
                except Exception:
                    pass
        for worker in (capture_worker, vision_worker, agent_worker):
            worker.join(timeout=2.0)

    app.aboutToQuit.connect(shutdown)
    panel.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
