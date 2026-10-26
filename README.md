# WARDOGS Logistics Control

Windows desktop automation scaffold for a calibrated logistics loop. It uses native `SendInput` scancodes, live capture, configurable image templates/OCR, and a timeout-guarded finite state machine. The bot deliberately does not invent a route: configure and test route segments for your map and resolution before enabling input.

## Requirements

- Windows 10/11, Python 3.10+, and the game running in windowed or borderless mode.
- A game resolution matching the configured capture ROIs; `config.json` targets Windowed Fullscreen 2560x1440 (the process enforces DPI awareness).
- Tesseract OCR installed separately and available on `PATH` if you want speed and damage OCR. Delivery is verified by the `delivery_complete` template or the interaction prompt clearing, not by the bottom-left scoreboard (match score, not FOB stock).

## Quick Launch

On Windows, double-click `launch.bat`. It creates the virtual environment if needed, checks/installs dependencies, and opens the control panel without leaving a terminal window open. Python 3.10 or newer must be installed.

To launch manually from a terminal instead:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py main.py
```

`bettercam` is preferred for capture; `mss` is the fallback. Both capture the selected display. Install or run on Windows because input injection uses `user32.dll`; on other operating systems the panel opens with automation disabled.

## DPI And Display Setup

1. Set Windows display scaling to 100% while calibrating, or recalculate every ROI in `config.json` using the coordinates reported by the capture backend.
2. Use a fixed game resolution and borderless/windowed mode. Keep the game in the foreground before pressing Start; `SendInput` affects the foreground application.
3. Do not run the game at a different resolution after calibration. Re-capture templates and update the ROI rectangles whenever the resolution or HUD scale changes.
4. `mss` ROI coordinates are display pixels. `bettercam` uses the selected output (`monitor: 1` selects the first display). Confirm the preview and ROI outlines before enabling the FSM.

## Template And OCR Calibration

Create `assets/templates/` and capture tightly cropped PNGs from the same game resolution and UI scale. Name the files exactly as the detector signals:

- `pallet_menu_open.png`: stable header or panel shown after pressing B.
- `friendly_fob.png`: a friendly FOB marker visible on the tactical map.
- `delivery_zone.png`: the delivery-zone HUD indicator.
- `unload_complete.png`: the completed unload state (not an in-progress bar).
- `hq_zone.png`: an unambiguous HQ loading/depot indicator.

Use lossless screenshots, avoid animated/glowing edges, and keep each crop at its on-screen pixel size. Adjust per-template confidence thresholds and `template_regions` in `config.json`. The speedometer and damage ROIs should contain only their respective readouts. Template thresholds are clamped to 0.82–0.88.

Adjust all `capture.regions` coordinates for your display. In particular, `tactical_map` should cover the map opened by M, and the HUD regions should exclude neighboring numbers/text where possible. The live preview draws ROI rectangles and template matches; blue outlines identify ROIs and green outlines identify detections.

## Route Calibration

The default `agent.route_segments.to_fob` and `to_hq` are empty, so Start will enter recovery and stop before driving. Configure each as a list of short, tested segments, for example:

```json
"route_segments": {
	"to_fob": [{"steer": "", "seconds": 1.5}, {"steer": "D", "seconds": 0.4}],
	"to_hq": [{"steer": "", "seconds": 1.2}, {"steer": "A", "seconds": 0.3}]
}
```

Segments hold W with the optional A/D steering key. These are fixed, map-specific steering instructions, not autonomous road or waypoint recognition. Keep segments short and conservative. The agent still requires `delivery_zone` or `hq_zone` visual confirmation before transitioning; a route that ends without its destination marker enters recovery. Set `pallet_keys` to the actual menu controls for each supply type and confirm the menu behavior manually.

## Controls And Safety

- Start begins the buy, map-check, calibrated drive, verified delivery, return, and repeat cycle. **Run batch** runs the same cycle and stops after the selected number of verified deliveries (default 10); the batch progress is shown in the Deliveries counter.
- Pause releases held inputs and enters a paused state; Resume continues the previous FSM state. Stop returns to idle.
- F12 or Esc is a global emergency stop. It releases held keys and returns the FSM to idle. Esc is not sent by the agent.
- Recovery attempts a short reverse/steer maneuver up to three times, then stops. A damage threshold, missing visual confirmation, missing calibrated route, or state timeout also prevents continued automation.
- The bot does not bypass game protections, modify game files, or access game memory. Use only where desktop automation is permitted.

## Test Sequences

1. **Static checks:** `py -m compileall -q .` and `py -m json.tool config.json`.
2. **Capture-only:** launch with the game visible; confirm the live preview updates, the selected monitor is correct, and every ROI outline encloses the intended UI element.
3. **Vision-only:** populate the five template files, verify their green boxes in the preview, and confirm OCR values are stable while stationary. Do not start the FSM yet.
4. **Input safety:** with the game foregrounded and the vehicle parked, click Start only after route and menu controls are calibrated; verify the buy menu opens. Press Pause, Stop, then F12/Esc in separate checks and confirm input releases immediately.
5. **Delivery loop:** use a safe test route and low speed; verify menu selection, map marker, destination indicator, pallet drop, unload completion, and the delivery_complete/prompt-cleared signal before allowing the return leg.
6. **Recovery:** test with an empty route or missing template first. The expected result is recovery/idle without an unverified drive or delivery.

The automated keys are global input. Keep a hand on the physical keyboard and use F12/Esc or Stop if the game state diverges from the preview.