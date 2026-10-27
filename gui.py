"""PyQt6 control panel for the logistics automation workers."""

from __future__ import annotations

import json
import queue
import sys
import time
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
import audio
from capture import get_available_monitors, get_open_windows
from dataset_inspector import MIN_SAMPLES, get_readiness_report
from train_model import DATASET_DIR, MODELS_DIR, TrainingThread, TrainParams, describe_device
from recorder import DIRECTIONS, FACTIONS, MAPS, RouteRecorder, assign_route, update_config
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QTabWidget,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from ui.dashboard import GaugeCard, RouteSegmentEditor, StatusIndicator


class PlotWidget(QWidget):
    """Minimal line plot (no extra dependency) for training curves."""

    COLORS = ("#e53935", "#43a047", "#1e88e5", "#fb8c00")

    def __init__(self, title: str, series: list[str], y_max: float | None = None):
        super().__init__()
        self.title = title
        self.names = series
        self.y_max = y_max
        self.data: list[list[float]] = [[] for _ in series]
        self.setMinimumHeight(180)

    def add_point(self, *values: float) -> None:
        for line, value in zip(self.data, values):
            line.append(float(value))
        self.update()

    def clear(self) -> None:
        self.data = [[] for _ in self.names]
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101719"))
        margin = 34
        plot_w, plot_h = self.width() - margin - 10, self.height() - 2 * margin + 8
        painter.setPen(QColor("#9da9a7"))
        painter.drawText(8, 16, self.title)
        painter.setPen(QPen(QColor("#344344")))
        painter.drawRect(margin, margin - 8, plot_w, plot_h)
        points = [v for line in self.data for v in line]
        if not points or plot_w <= 0 or plot_h <= 0:
            return
        top = self.y_max if self.y_max is not None else max(max(points), 1e-6) * 1.1
        count = max(len(line) for line in self.data)
        painter.setPen(QColor("#9da9a7"))
        painter.drawText(2, margin, f"{top:.2f}")
        painter.drawText(2, margin - 8 + plot_h, "0")
        for index, line in enumerate(self.data):
            color = QColor(self.COLORS[index % len(self.COLORS)])
            painter.setPen(QPen(color, 2))
            previous = None
            for step, value in enumerate(line):
                x = margin + (plot_w * step / max(1, count - 1) if count > 1 else plot_w / 2)
                y = margin - 8 + plot_h * (1 - min(value, top) / top)
                if previous is not None:
                    painter.drawLine(int(previous[0]), int(previous[1]), int(x), int(y))
                painter.drawEllipse(int(x) - 2, int(y) - 2, 4, 4)
                previous = (x, y)
            painter.drawText(margin + 8 + index * 110, self.height() - 8, self.names[index])


class ControlPanel(QMainWindow):
    def __init__(
        self,
        commands: queue.Queue,
        telemetry: queue.Queue,
        render_queue: queue.Queue,
        status_queue: queue.Queue,
        halt_input=None,
        capture_worker=None,
        config_path=None,
        vision_worker=None,
        model_callback=None,
        input_controller=None,
    ):
        super().__init__()
        self.commands = commands
        self.telemetry = telemetry
        self.render_queue = render_queue
        self.status_queue = status_queue
        self.halt_input = halt_input
        self.capture_worker = capture_worker
        self.config_path = Path(config_path) if config_path else None
        self.vision_worker = vision_worker
        self.model_callback = model_callback
        self.input_controller = input_controller
        self.active_model_path: str | None = None
        self.trainer: TrainingThread | None = None
        self._last_analysis = 0.0
        self._latest_frame = None
        self._latest_values: dict = {}
        self._latest_scores: dict = {}
        self._latest_detections: set[str] = set()
        self._last_logged_state = None
        self._last_logged_action = None
        self._last_logged_deliveries = 0
        self._last_logged_status = None
        self._last_logged_detections: set[str] = set()
        self._last_logged_values: dict = {}
        self._last_ocr_errors: dict = {}
        self.hotkeys_available: bool | None = None
        self._loading_routes = False
        self._latest_roi = None
        self._latest_map_frame = None
        self._ui_config = self._load_config_data()
        self.recorder = RouteRecorder()
        self._hooks_ok = self.recorder.install_hooks()
        self._paused = False
        self.setWindowTitle("Wardogs Logistics Control")
        self.resize(1720, 1080)
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)
        self.setStyleSheet("""
            QWidget { background: #07121a; color: #d7e8ee; }
            QGroupBox {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0b1c27, stop:1 #08141d);
                border: 1px solid #1b5367; border-radius: 2px;
                margin-top: 12px; padding: 8px; font-weight: 700;
                color: #71d5ed;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 5px; }
            QLabel { color: #d7e8ee; }
            QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget {
                background: #071019; color: #d7e8ee;
                border: 1px solid #235367; padding: 4px; selection-background-color: #145142;
            }
            QComboBox QAbstractItemView { background: #071019; color: #d7e8ee; }
            QPushButton {
                background: #102731; color: #cdebf2; border: 1px solid #2b697d;
                padding: 7px 12px; font-weight: 700; min-height: 23px;
            }
            QPushButton:hover { border-color: #54f28b; color: #54f28b; }
            QPushButton:disabled { color: #59717a; border-color: #263b43; }
            QProgressBar { background: #050d13; color: #d7e8ee; border: 1px solid #235367; text-align: center; }
            QProgressBar::chunk { background: #32cf80; }
            QHeaderView::section { background: #0c202a; color: #76cfe5; border: 0; padding: 4px; }
            QTabWidget::pane { border: 1px solid #1b5367; background: #07121a; }
            QTabBar::tab {
                background: #0a1922; color: #8daab5; border: 1px solid #1b5367;
                padding: 11px 20px; margin-right: 5px; min-width: 250px;
                font-weight: 700;
            }
            QTabBar::tab:selected { color: #54f28b; border: 2px solid #54f28b; background: #09241f; }
        """)

        header = QHBoxLayout()
        logo = QLabel()
        logo_path = Path(__file__).with_name("assets") / "ui" / "header_logo_title.png"
        logo_pixmap = QPixmap(str(logo_path))
        if not logo_pixmap.isNull():
            logo_pixmap = logo_pixmap.copy(0, 0, logo_pixmap.width(), min(52, logo_pixmap.height()))
            logo.setPixmap(logo_pixmap.scaledToHeight(48, Qt.TransformationMode.SmoothTransformation))
        else:
            logo.setText("WARDOGS  /  LOGISTICS CONTROL")
            logo.setStyleSheet("font-size: 22px; font-weight: 800; color: #e8f4f7;")
        header.addWidget(logo)
        header.addStretch(1)
        self.header_status = QLabel("SYSTEM INITIALIZING")
        self.header_status.setStyleSheet("color: #ffc247; font-weight: 700;")
        header.addWidget(self.header_status)
        layout.addLayout(header)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        self.live_tab = self._build_live_dashboard()
        tabs.addTab(self.live_tab, "LIVE BOT CONTROL & VISUALIZATION")
        tabs.addTab(self._build_recorder_tab(), "ROUTE RECORDER & DATA COLLECTOR")
        tabs.addTab(self._build_training_tab(), "NEURAL NET TRAINING")
        self.map_combo.blockSignals(True)
        self.map_combo.setCurrentText(self.route_map_combo.currentText())
        self.map_combo.blockSignals(False)
        layout.addWidget(tabs, 1)
        self.setCentralWidget(central)

    def _build_live_dashboard(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(8)

        top = QHBoxLayout()
        camera_box = QGroupBox("LIVE CAMERA / VISION")
        camera_layout = QVBoxLayout(camera_box)
        self.preview = QLabel("WAITING FOR LIVE CAPTURE")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(420, 250)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setStyleSheet("background: #03090d; color: #8faab6; border: 1px solid #1b5367;")
        camera_layout.addWidget(self.preview, 1)
        camera_footer = QHBoxLayout()
        self.confidence_label = QLabel("MATCH CONFIDENCE: --")
        self.confidence_label.setWordWrap(True)
        self.confidence_label.setStyleSheet("color: #85d9ec;")
        camera_footer.addWidget(self.confidence_label, 1)
        self.roi_toggle = QCheckBox("ROI / MATCH OVERLAYS")
        self.roi_toggle.setChecked(True)
        self.roi_toggle.toggled.connect(self._toggle_rois)
        camera_footer.addWidget(self.roi_toggle)
        camera_layout.addLayout(camera_footer)
        self.roi_preview = QLabel("ROI DIAGNOSTICS")
        self.roi_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.roi_preview.setMinimumHeight(72)
        self.roi_preview.setMaximumHeight(120)
        self.roi_preview.setStyleSheet("background: #03090d; color: #8faab6; border: 1px solid #1b5367;")
        camera_layout.addWidget(self.roi_preview)
        top.addWidget(camera_box, 3)

        environment = QGroupBox("ENVIRONMENT / STATE")
        environment_layout = QVBoxLayout(environment)
        gauges = QHBoxLayout()
        self.speed_gauge = GaugeCard("SPEEDOMETER · OCR", 100)
        self.damage_gauge = GaugeCard("VEHICLE DAMAGE · OCR", 100)
        gauges.addWidget(self.speed_gauge)
        gauges.addWidget(self.damage_gauge)
        environment_layout.addLayout(gauges)
        self.map_preview = QLabel("TACTICAL MAP / MINIMAP AWAITING CAPTURE")
        self.map_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.map_preview.setMinimumHeight(130)
        self.map_preview.setMaximumHeight(230)
        self.map_preview.setStyleSheet("background: #03090d; color: #8faab6; border: 1px solid #1b5367;")
        environment_layout.addWidget(self.map_preview, 1)
        top.addWidget(environment, 2)
        layout.addLayout(top, 4)

        info_row = QHBoxLayout()
        telemetry_box = QGroupBox("LIVE TELEMETRY")
        telemetry_layout = QGridLayout(telemetry_box)
        self.state_label = QLabel("STATE_IDLE")
        self.action_label = QLabel("Standby")
        self.key_label = QLabel("--")
        self.handbrake_label = QLabel("HALTED")
        self.deliveries_label = QLabel("0")
        self.duration_label = QLabel("0.0 s")
        self.telemetry_progress = QProgressBar()
        self.telemetry_progress.setRange(0, 100)
        self.telemetry_progress.setFormat("DELIVERY PROGRESS")
        entries = (
            ("FSM STATE", self.state_label), ("CURRENT ACTION", self.action_label),
            ("KEY INPUTS", self.key_label), ("INPUT HALT / BRAKE", self.handbrake_label),
            ("DELIVERIES", self.deliveries_label), ("LOOP TIME", self.duration_label),
        )
        for index, (title, value) in enumerate(entries):
            title_label = QLabel(title)
            title_label.setStyleSheet("color: #7aa4b1; font-size: 10px; font-weight: 700;")
            value.setStyleSheet("color: #e2f5f8; font-size: 14px; font-weight: 700;")
            telemetry_layout.addWidget(title_label, index // 3 * 2, index % 3)
            telemetry_layout.addWidget(value, index // 3 * 2 + 1, index % 3)
        telemetry_layout.addWidget(self.telemetry_progress, 4, 0, 1, 3)
        telemetry_note = QLabel(
            "Brake telemetry is not exposed by the backend; emergency Space-pulse status is shown when active."
        )
        telemetry_note.setStyleSheet("color: #ffc247; font-size: 10px;")
        telemetry_layout.addWidget(telemetry_note, 5, 0, 1, 3)
        info_row.addWidget(telemetry_box, 3)

        detection_column = QVBoxLayout()
        self.hud_box = QGroupBox("HUD ROI STATUS")
        hud_layout = QVBoxLayout(self.hud_box)
        self.hud_indicators = {
            name: StatusIndicator(name) for name in (
                "Speedometer", "Damage", "Tactical Map", "Pallet Menu", "Delivery Zone", "HQ Zone"
            )
        }
        for indicator in self.hud_indicators.values():
            hud_layout.addWidget(indicator)
        self.template_box = QGroupBox("LIVE TEMPLATE MATCHES")
        template_layout = QGridLayout(self.template_box)
        self.template_labels: dict[str, QLabel] = {}
        for row, name in enumerate(
            ("pallet_menu_open", "friendly_fob", "delivery_zone", "unload_complete", "hq_zone")
        ):
            name_label = QLabel(name.upper())
            name_label.setStyleSheet("color: #8fc8d6; font-weight: 600;")
            confidence_bar = QProgressBar()
            confidence_bar.setRange(0, 100)
            confidence_bar.setFormat("NO SCORE")
            result_label = QLabel("NOT LOADED")
            self.template_labels[name] = result_label
            template_layout.addWidget(name_label, row, 0)
            template_layout.addWidget(confidence_bar, row, 1)
            template_layout.addWidget(result_label, row, 2)
            if not hasattr(self, "template_bars"):
                self.template_bars = {}
            self.template_bars[name] = confidence_bar
        detection_column.addWidget(self.hud_box)
        detection_column.addWidget(self.template_box)
        info_row.addLayout(detection_column, 2)
        layout.addLayout(info_row, 2)

        controls_row = QHBoxLayout()
        control_box = QGroupBox("PRIMARY CONTROLS")
        control_layout = QHBoxLayout(control_box)
        self.start_button = QPushButton("START")
        self.start_button.setStyleSheet("background: #0d3324; color: #54f28b; border: 2px solid #54f28b; font-size: 18px; min-height: 54px;")
        self.pause_button = QPushButton("PAUSE")
        self.pause_button.setStyleSheet("background: #352a12; color: #ffc247; border: 2px solid #c99a35; font-size: 18px; min-height: 54px;")
        self.stop_button = QPushButton("STOP")
        self.stop_button.setStyleSheet("background: #3a1118; color: #ff5964; border: 2px solid #ff5964; font-size: 18px; min-height: 54px;")
        self.start_button.clicked.connect(lambda: self.commands.put({"kind": "start"}))
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop)
        for button in (self.start_button, self.pause_button, self.stop_button):
            control_layout.addWidget(button, 1)
        controls_row.addWidget(control_box, 2)

        batch_box = QGroupBox("BATCH RUN")
        batch_layout = QGridLayout(batch_box)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 1000)
        self.batch_size.setValue(10)
        self.batch_size.setToolTip("Stop after this many verified deliveries")
        self.batch_button = QPushButton("RUN BATCH")
        self.batch_button.setStyleSheet("color: #54f28b; border-color: #54f28b;")
        self.batch_button.clicked.connect(
            lambda: self.commands.put({"kind": "start", "deliveries": self.batch_size.value()})
        )
        self.batch_count_label = QLabel("COMPLETED DELIVERIES: 0")
        batch_layout.addWidget(QLabel("DELIVERIES TO PERFORM"), 0, 0)
        batch_layout.addWidget(self.batch_size, 0, 1)
        batch_layout.addWidget(self.batch_button, 1, 0, 1, 2)
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setFormat("VERIFIED DELIVERY PROGRESS")
        batch_layout.addWidget(self.batch_progress, 2, 0, 1, 2)
        batch_layout.addWidget(self.batch_count_label, 3, 0, 1, 2)
        controls_row.addWidget(batch_box, 1)
        layout.addLayout(controls_row)

        config_row = QHBoxLayout()
        display_box = QGroupBox("DISPLAY SETTINGS")
        display_layout = QGridLayout(display_box)
        display_layout.addWidget(QLabel("CAPTURE DISPLAY"), 0, 0)
        self.monitor_combo = QComboBox()
        self._populate_monitors()
        self.monitor_combo.currentIndexChanged.connect(self._monitor_changed)
        display_layout.addWidget(self.monitor_combo, 0, 1)
        display_layout.addWidget(QLabel("CAPTURE WINDOW"), 1, 0)
        self.window_combo = QComboBox()
        self.window_combo.setMinimumWidth(180)
        display_layout.addWidget(self.window_combo, 1, 1)
        refresh_windows = QPushButton("REFRESH")
        refresh_windows.clicked.connect(self._populate_windows)
        display_layout.addWidget(refresh_windows, 1, 2)
        self._populate_windows()
        self.window_combo.currentIndexChanged.connect(self._window_changed)
        config_row.addWidget(display_box, 2)

        route_box = QGroupBox("ROUTE & AGENT CONTROLS")
        route_layout = QVBoxLayout(route_box)
        map_row = QHBoxLayout()
        map_row.addWidget(QLabel("ACTIVE MAP"))
        self.route_map_combo = QComboBox()
        self.route_map_combo.addItems(MAPS)
        self.route_map_combo.setCurrentText(
            str(self._ui_config.get("agent", {}).get("active_map", MAPS[0]))
        )
        self.route_map_combo.currentTextChanged.connect(self._dashboard_map_changed)
        map_row.addWidget(self.route_map_combo)
        self.route_status_label = QLabel("Route status: checking")
        map_row.addWidget(self.route_status_label, 1)
        route_layout.addLayout(map_row)
        route_editors = QHBoxLayout()
        self.route_fob_editor = RouteSegmentEditor("TO FOB")
        self.route_hq_editor = RouteSegmentEditor("TO HQ")
        self.route_fob_editor.changed.connect(lambda segments: self._save_route_segments("to_fob", segments))
        self.route_hq_editor.changed.connect(lambda segments: self._save_route_segments("to_hq", segments))
        route_editors.addWidget(self.route_fob_editor)
        route_editors.addWidget(self.route_hq_editor)
        route_layout.addLayout(route_editors)
        config_row.addWidget(route_box, 5)

        safety_box = QGroupBox("STATUS / SAFETY")
        safety_layout = QVBoxLayout(safety_box)
        self.safety_indicators = {
            name: StatusIndicator(name) for name in (
                "Game window focused", "Templates loaded", "Route configured",
                "OCR available", "Emergency stop · F12 / ESC"
            )
        }
        for indicator in self.safety_indicators.values():
            safety_layout.addWidget(indicator)
        config_row.addWidget(safety_box, 2)
        layout.addLayout(config_row)

        lower = QHBoxLayout()
        pallet_box = QGroupBox("PALLET KEYS / PURCHASE CONFIG")
        pallet_layout = QVBoxLayout(pallet_box)
        priority_row = QHBoxLayout()
        priority_row.addWidget(QLabel("SUPPLY PRIORITY"))
        self.priority = QComboBox()
        self.priority.addItems(["Build", "Ammo", "Fuel", "Mechanical"])
        self.priority.currentTextChanged.connect(
            lambda value: self.commands.put({"kind": "priority", "value": value})
        )
        priority_row.addWidget(self.priority)
        pallet_layout.addLayout(priority_row)
        self.pallet_key_note = QLabel(
            "LOCK L  ·  MENU B  ·  SELECT via configured menu_clicks  ·  UNLOAD G / F\n"
            "Direct pallet hotkeys are not implemented; card selection uses configured screen coordinates."
        )
        self.pallet_key_note.setWordWrap(True)
        self.pallet_key_note.setStyleSheet("color: #ffc247;")
        pallet_layout.addWidget(self.pallet_key_note)
        lower.addWidget(pallet_box, 2)

        audio_box = QGroupBox("AUDIO CUES")
        audio_layout = QVBoxLayout(audio_box)
        audio_available = audio.winsound is not None
        audio_state = "ENABLED · WINDOWS BEEPS" if audio_available else "UNAVAILABLE · PLATFORM NO-OP"
        for text in (
            f"RECORDING STARTED / PURCHASE  ·  HIGH SINGLE  ·  {audio_state}",
            "DELIVERY VERIFIED  ·  HIGH DOUBLE",
            "EMERGENCY STOP / RECOVERY  ·  LOW DOUBLE",
        ):
            label = QLabel(text)
            label.setWordWrap(True)
            audio_layout.addWidget(label)
        cue_note = QLabel("Fixed cues from audio.py; per-cue enable/volume settings are not implemented.")
        cue_note.setWordWrap(True)
        cue_note.setStyleSheet("color: #ffc247;")
        audio_layout.addWidget(cue_note)
        lower.addWidget(audio_box, 2)

        terminal_box = QGroupBox("SYSTEM TERMINAL / EVENT STREAM")
        terminal_layout = QVBoxLayout(terminal_box)
        self.terminal = QPlainTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setMaximumBlockCount(400)
        self.terminal.setMinimumHeight(105)
        terminal_layout.addWidget(self.terminal)
        lower.addWidget(terminal_box, 4)
        layout.addLayout(lower)

        self.status_label = QLabel("Starting capture, vision, and FSM workers")
        self.status_label.setStyleSheet("color: #8faab6;")
        layout.addWidget(self.status_label)
        self._load_route_editors()
        scroll.setWidget(page)
        return scroll

    def _load_config_data(self) -> dict:
        if self.config_path is None:
            return {}
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _route_segments(self, route_name: str) -> list[dict]:
        map_name = self.route_map_combo.currentText() if hasattr(self, "route_map_combo") else ""
        scoped = self._ui_config.get("routes", {}).get(map_name, {})
        legacy = self._ui_config.get("agent", {}).get("route_segments", {})
        return list(scoped.get(route_name) or legacy.get(route_name) or [])

    def _load_route_editors(self) -> None:
        if not hasattr(self, "route_fob_editor"):
            return
        self._loading_routes = True
        self.route_fob_editor.set_segments(self._route_segments("to_fob"))
        self.route_hq_editor.set_segments(self._route_segments("to_hq"))
        self._loading_routes = False
        if self.config_path is None:
            self.route_status_label.setText("Route config unavailable")
        else:
            count = len(self._route_segments("to_fob"))
            self.route_status_label.setText(
                f"TO_FOB: {count} segment(s) · TO_HQ: {len(self._route_segments('to_hq'))}"
            )

    def _save_route_segments(self, route_name: str, segments: list[dict]) -> None:
        if self._loading_routes:
            return
        if self.config_path is None:
            self.route_status_label.setText("Route config unavailable; edits not saved")
            return
        map_name = self.route_map_combo.currentText()
        try:
            update_config(self.config_path, ["routes", map_name, route_name], segments)
        except (OSError, ValueError) as exc:
            self.route_status_label.setText(f"Could not save route: {exc}")
            self._record_log(f"ERROR: route configuration save failed: {exc}")
            return
        routes = self._ui_config.setdefault("routes", {})
        routes.setdefault(map_name, {})[route_name] = segments
        self.route_status_label.setText(
            f"TO_FOB: {len(self._route_segments('to_fob'))} segment(s) · "
            f"TO_HQ: {len(self._route_segments('to_hq'))}"
        )

    def _dashboard_map_changed(self, map_name: str) -> None:
        if hasattr(self, "map_combo") and self.map_combo.currentText() != map_name:
            self.map_combo.setCurrentText(map_name)
        else:
            self._on_map_changed(map_name)

    def _record_log(self, message: str) -> None:
        if not hasattr(self, "terminal"):
            return
        self.terminal.appendPlainText(f"{time.strftime('%H:%M:%S')}  {message}")

    def _template_sources(self, alias: str) -> tuple[str, ...]:
        signals = self._ui_config.get("agent", {}).get("signals", {})
        configured = {
            "pallet_menu_open": ("pallet_menu_open",),
            "friendly_fob": (signals.get("friendly_fob", "friendly_fob"), "friendly_fob"),
            "delivery_zone": (signals.get("delivery_zone", "delivery_zone"), "delivery_zone"),
            "unload_complete": (
                signals.get("delivery_complete", "delivery_complete"),
                "delivery_complete", "supplies_delivered", "delivery_payout",
            ),
            "hq_zone": (signals.get("hq_zone", "hq_zone"), "hq_zone", "hq_safe_zone"),
        }
        return tuple(dict.fromkeys(configured[alias]))

    def _refresh_templates(self, template_names: set[str]) -> None:
        for alias, label in self.template_labels.items():
            available = [name for name in self._template_sources(alias) if name in template_names]
            source = next((name for name in available if name in self._latest_scores), None)
            if source is not None:
                score = float(self._latest_scores[source])
                detected = source in self._latest_detections
                state = "DETECTED" if detected else "SEARCHING"
                color = "#54f28b" if detected else "#ffc247"
                self.template_bars[alias].setValue(max(0, min(100, round(score * 100))))
                self.template_bars[alias].setFormat(f"{score:.3f}")
                label.setText(f"{state} · {score:.3f}")
                label.setStyleSheet(f"color: {color}; font-weight: 700;")
            elif available:
                self.template_bars[alias].setValue(0)
                self.template_bars[alias].setFormat("NO SCORE")
                label.setText("NO CURRENT SCORE")
                label.setStyleSheet("color: #ffc247;")
            else:
                self.template_bars[alias].setValue(0)
                self.template_bars[alias].setFormat("NOT LOADED")
                label.setText("NOT LOADED")
                label.setStyleSheet("color: #ff5964;")

    def _set_detector_indicator(self, title: str, alias: str) -> None:
        indicator = self.hud_indicators[title]
        available = any(name in self._template_names for name in self._template_sources(alias))
        source = next(
            (name for name in self._template_sources(alias) if name in self._latest_scores),
            None,
        )
        if not available:
            indicator.set_status("missing", "TEMPLATE NOT LOADED")
        elif source is None:
            indicator.set_status("warning", "WAITING FOR MATCH SCORE")
        elif source in self._latest_detections:
            indicator.set_status("healthy", "DETECTED")
        else:
            indicator.set_status("warning", "SEARCHING")

    def _focus_state(self) -> tuple[str, str]:
        hwnd = self.window_combo.currentData()
        if hwnd is None:
            return "warning", "NOT TARGETED · FULL DISPLAY"
        if sys.platform != "win32":
            return "neutral", "FOCUS CHECK UNAVAILABLE"
        import ctypes

        focused = int(ctypes.windll.user32.GetForegroundWindow()) == int(hwnd)
        return ("healthy", "FOCUSED") if focused else ("warning", "NOT FOCUSED")

    def _refresh_dashboard_status(self) -> None:
        engine = self.vision_worker.engine if self.vision_worker is not None else None
        self._template_names = set(engine.templates) if engine is not None else set()
        count = len(self._template_names)
        self.safety_indicators["Templates loaded"].set_status(
            "healthy" if count else "missing", f"{count} LOADED"
        )
        route_ready = bool(self._route_segments("to_fob")) or self.active_model_path is not None
        self.safety_indicators["Route configured"].set_status(
            "healthy" if route_ready else "missing",
            "TO_FOB / MODEL READY" if route_ready else "TO_FOB MISSING",
        )
        if engine is None or engine._tesseract is None:
            self.safety_indicators["OCR available"].set_status("missing", "PYTESSERACT UNAVAILABLE")
        elif any(
            "tesseract" in message.lower()
            and ("not installed" in message.lower() or "not found" in message.lower())
            for message in self._last_ocr_errors.values()
        ):
            self.safety_indicators["OCR available"].set_status("missing", "TESSERACT ENGINE ERROR")
        elif self._last_ocr_errors:
            self.safety_indicators["OCR available"].set_status("warning", "OCR ERROR · SEE TERMINAL")
        elif self._latest_values.get("speed") is None and self._latest_values.get("vehicle_damage") is None:
            self.safety_indicators["OCR available"].set_status("warning", "AVAILABLE · NO CURRENT VALUE")
        else:
            self.safety_indicators["OCR available"].set_status("healthy", "AVAILABLE")
        if self.hotkeys_available is True:
            self.safety_indicators["Emergency stop · F12 / ESC"].set_status("healthy", "HOTKEYS ACTIVE")
        elif self.hotkeys_available is False:
            self.safety_indicators["Emergency stop · F12 / ESC"].set_status("missing", "HOTKEYS UNAVAILABLE")
        else:
            self.safety_indicators["Emergency stop · F12 / ESC"].set_status("warning", "CHECKING HOTKEYS")
        focus_level, focus_detail = self._focus_state()
        self.safety_indicators["Game window focused"].set_status(focus_level, focus_detail)

        regions = self.capture_worker.config.regions if self.capture_worker is not None else {}
        for title, value_name, roi_name in (
            ("Speedometer", "speed", "speedometer"),
            ("Damage", "vehicle_damage", "damage"),
        ):
            indicator = self.hud_indicators[title]
            if not (
                engine and engine._tesseract and roi_name in regions
                and engine.config.ocr_regions.get(value_name) == roi_name
            ):
                indicator.set_status("missing", "OCR / ROI UNAVAILABLE")
            elif value_name in self._last_ocr_errors:
                indicator.set_status("warning", "OCR ERROR · SEE TERMINAL")
            elif self._latest_values.get(value_name) is None:
                indicator.set_status("warning", "WAITING FOR OCR")
            else:
                indicator.set_status("healthy", f"{self._latest_values[value_name]} READ")

        map_available = any(name in regions for name in ("minimap", "tactical_map"))
        map_has_frame = self._latest_map_frame is not None
        self.hud_indicators["Tactical Map"].set_status(
            "healthy" if map_available and map_has_frame else "warning" if map_available else "missing",
            "LIVE CAPTURE" if map_available and map_has_frame else "WAITING FOR CAPTURE" if map_available else "ROI NOT CONFIGURED",
        )
        for title, alias in (
            ("Pallet Menu", "pallet_menu_open"),
            ("Delivery Zone", "delivery_zone"),
            ("HQ Zone", "hq_zone"),
        ):
            self._set_detector_indicator(title, alias)

    def _refresh_dashboard_values(self, telemetry: dict | None = None) -> None:
        if telemetry is not None:
            state = telemetry.get("state", "STATE_IDLE")
            if state != self._last_logged_state:
                self._record_log(f"FSM state: {self._last_logged_state or 'UNKNOWN'} -> {state}")
                self._last_logged_state = state
            self.state_label.setText(state)
            self._paused = state == "STATE_PAUSED"
            self.pause_button.setText("RESUME" if self._paused else "PAUSE")
            action = telemetry.get("action", "Standby")
            self.action_label.setText(action)
            if action != self._last_logged_action and action not in ("Standby", ""):
                self._record_log(f"Agent action: {action}")
                self._last_logged_action = action
            self.key_label.setText(telemetry.get("active_key", "--"))
            deliveries = int(telemetry.get("deliveries", 0))
            target = int(telemetry.get("delivery_target", 0))
            if deliveries > self._last_logged_deliveries:
                self._record_log(f"Delivery verified · count {deliveries}")
            self._last_logged_deliveries = deliveries
            self.deliveries_label.setText(f"{deliveries} / {target}" if target > 0 else str(deliveries))
            self.batch_count_label.setText(f"COMPLETED DELIVERIES: {deliveries}")
            self.duration_label.setText(f"{telemetry.get('loop_seconds', 0.0):.1f} s")
            if target > 0:
                self.telemetry_progress.setValue(round(100 * min(deliveries, target) / target))
                self.telemetry_progress.setFormat(f"DELIVERY PROGRESS · {deliveries}/{target}")
                self.batch_progress.setValue(round(100 * min(deliveries, target) / target))
                self.batch_progress.setFormat(f"VERIFIED DELIVERY PROGRESS · {deliveries}/{target}")
            else:
                self.telemetry_progress.setValue(0)
                self.telemetry_progress.setFormat("NO DELIVERY TARGET")
                self.batch_progress.setValue(0)
                self.batch_progress.setFormat("NO BATCH TARGET")
            selected = telemetry.get("priority")
            if selected and self.priority.currentText() != selected:
                self.priority.blockSignals(True)
                self.priority.setCurrentText(selected)
                self.priority.blockSignals(False)

        halted = getattr(self.input_controller, "halted", None)
        pulsing = getattr(self.input_controller, "_pulsing_handbrake", False)
        if pulsing:
            self.handbrake_label.setText("EMERGENCY SPACE PULSE ACTIVE")
            self.handbrake_label.setStyleSheet("color: #ff5964; font-weight: 700;")
        elif halted is True:
            self.handbrake_label.setText("INPUT HALTED · BRAKE UNKNOWN")
            self.handbrake_label.setStyleSheet("color: #54f28b; font-weight: 700;")
        elif halted is False:
            self.handbrake_label.setText("INPUT ENABLED · BRAKE UNKNOWN")
            self.handbrake_label.setStyleSheet("color: #ffc247; font-weight: 700;")
        else:
            self.handbrake_label.setText("BACKEND UNAVAILABLE")
            self.handbrake_label.setStyleSheet("color: #ff5964; font-weight: 700;")

        speed = self._latest_values.get("speed")
        damage = self._latest_values.get("vehicle_damage")
        max_damage = float(self._ui_config.get("agent", {}).get("max_damage", 75))
        self.speed_gauge.set_value(speed)
        self.damage_gauge.set_value(damage, warning_at=max_damage)
        self._refresh_templates(getattr(self, "_template_names", set()))
        self._refresh_dashboard_status()

        if self.capture_worker is not None:
            map_frame = self.capture_worker.get_roi("minimap")
            if map_frame is None:
                map_frame = self.capture_worker.get_roi("tactical_map")
            self._latest_map_frame = map_frame
            self._set_image(self.map_preview, map_frame)

    def _consume_render(self, render: dict) -> None:
        self._latest_frame = render.get("image")
        self._latest_roi = render.get("roi_preview")
        self._latest_scores = render.get("scores") or {}
        self._latest_detections = set(render.get("detected", ()))
        self._latest_values = render.get("values") or {}
        errors = render.get("ocr_errors") or {}
        for name, message in errors.items():
            if self._last_ocr_errors.get(name) != message:
                self._record_log(f"OCR error · {name}: {message}")
        for name in set(self._last_ocr_errors) - set(errors):
            self._record_log(f"OCR recovered · {name}")
        self._last_ocr_errors = dict(errors)
        self._set_image(self.preview, self._latest_frame)
        self._set_image(self.roi_preview, self._latest_roi)
        self._refresh_templates(getattr(self, "_template_names", set()))
        current_detections = self._latest_detections & getattr(self, "_template_names", set())
        for name in sorted(current_detections - self._last_logged_detections):
            score = self._latest_scores.get(name)
            confidence = f" · {float(score):.3f}" if score is not None else ""
            self._record_log(f"Template match: {name}{confidence}")
        self._last_logged_detections = current_detections
        for name, value in self._latest_values.items():
            if value is not None and value != self._last_logged_values.get(name):
                self._record_log(f"OCR {name}: {value}")
                self._last_logged_values[name] = value
        self._refresh_dashboard_values()

    def _toggle_rois(self, enabled: bool) -> None:
        if self.vision_worker is not None:
            self.vision_worker.engine.show_rois = enabled

    def _populate_monitors(self) -> None:
        monitors = get_available_monitors()
        current = self.capture_worker.monitor if self.capture_worker else 1
        self.monitor_combo.blockSignals(True)
        self.monitor_combo.clear()
        for monitor in monitors:
            self.monitor_combo.addItem(monitor["label"], monitor["index"])
        position = self.monitor_combo.findData(current)
        if position >= 0:
            self.monitor_combo.setCurrentIndex(position)
        self.monitor_combo.setEnabled(bool(monitors))
        self.monitor_combo.blockSignals(False)

    def _populate_windows(self) -> None:
        current = self.window_combo.currentData()
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        self.window_combo.addItem("Full display", None)
        for win in sorted(get_open_windows(), key=lambda w: w["title"].lower()):
            self.window_combo.addItem(win["title"], win["hwnd"])
        position = self.window_combo.findData(current)
        self.window_combo.setCurrentIndex(max(0, position))
        self.window_combo.blockSignals(False)

    def _window_changed(self, _index: int) -> None:
        if self.capture_worker is not None:
            self.capture_worker.set_window(self.window_combo.currentData())

    def _monitor_changed(self, _index: int) -> None:
        monitor_index = self.monitor_combo.currentData()
        if monitor_index is None:
            return
        if self.config_path is not None:
            try:
                update_config(self.config_path, ["display", "monitor_index"], int(monitor_index))
            except (OSError, ValueError) as exc:
                self.status_label.setText(f"Could not save display setting: {exc}")
        if self.capture_worker is not None:
            self.capture_worker.set_monitor(int(monitor_index))

    def _build_recorder_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.rec_status = QLabel(self.recorder.message)
        self.rec_status.setStyleSheet("font-weight: bold;")
        layout.addWidget(QLabel(
            "Global hotkeys: F8 start, F9 stop, F10 discard. Tracks W/S/A/D/Space; "
            "steering is saved, W is held automatically by the agent."
        ))
        layout.addWidget(self.rec_status)
        self.record_frames = QCheckBox("Record Synchronized Training Frames")
        self.record_frames.setChecked(True)
        self.record_frames.toggled.connect(lambda on: setattr(self.recorder, "record_frames", on))
        layout.addWidget(self.record_frames)
        controls = QHBoxLayout()
        self.rec_start = QPushButton("Start (F8)")
        self.rec_stop = QPushButton("Stop (F9)")
        self.rec_discard = QPushButton("Discard (F10)")
        self.rec_start.clicked.connect(self.recorder.start)
        self.rec_stop.clicked.connect(self.recorder.stop)
        self.rec_discard.clicked.connect(self.recorder.abort)
        for button in (self.rec_start, self.rec_stop, self.rec_discard):
            controls.addWidget(button)
        layout.addLayout(controls)
        selectors = QHBoxLayout()
        self.map_combo, self.faction_combo, self.direction_combo = QComboBox(), QComboBox(), QComboBox()
        self.map_combo.addItems(MAPS)
        self.faction_combo.addItems(FACTIONS)
        self.direction_combo.addItems(DIRECTIONS)
        for label, combo in (("Map", self.map_combo), ("Faction", self.faction_combo),
                             ("Leg", self.direction_combo)):
            selectors.addWidget(QLabel(label))
            selectors.addWidget(combo)
        layout.addLayout(selectors)
        self.map_combo.currentTextChanged.connect(self._on_map_changed)
        self.rec_preview = QPlainTextEdit()
        self.rec_preview.setReadOnly(True)
        self.rec_preview.setPlaceholderText("Recorded segments (JSON) appear here")
        layout.addWidget(self.rec_preview, 1)
        buttons = QHBoxLayout()
        self.assign_fob = QPushButton("Assign to TO_FOB Route")
        self.assign_hq = QPushButton("Assign to TO_HQ Route")
        self.assign_fob.clicked.connect(lambda: self._assign("to_fob"))
        self.assign_hq.clicked.connect(lambda: self._assign("to_hq"))
        buttons.addWidget(self.assign_fob)
        buttons.addWidget(self.assign_hq)
        layout.addLayout(buttons)
        card = QGroupBox("Dataset Readiness")
        card_layout = QVBoxLayout(card)
        self.ds_progress = QProgressBar()
        self.ds_progress.setRange(0, MIN_SAMPLES)
        self.ds_progress.setFormat(f"%v / {MIN_SAMPLES} frames")
        self.ds_dist = QLabel("Straight: 0% | Left: 0% | Right: 0%")
        self.ds_samples = QLabel("Samples: 0")
        self.ds_status = QLabel("[Need More Turns]")
        self.ds_status.setStyleSheet("font-weight: bold;")
        self.ds_advice = QLabel("")
        self.ds_advice.setWordWrap(True)
        analyze = QPushButton("Analyze Recorded Data")
        analyze.clicked.connect(self._analyze_dataset)
        for widget in (self.ds_samples, self.ds_progress, self.ds_dist, self.ds_status, self.ds_advice, analyze):
            card_layout.addWidget(widget)
        layout.addWidget(card)
        self._rec_was_recording = False
        return tab

    def _build_training_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.device_label = QLabel(describe_device())
        self.device_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.device_label)

        form = QHBoxLayout()
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(10)
        self.batch = QSpinBox()
        self.batch.setRange(1, 1024)
        self.batch.setValue(32)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(6)
        self.lr.setRange(1e-6, 1.0)
        self.lr.setSingleStep(1e-4)
        self.lr.setValue(1e-3)
        self.backbone = QComboBox()
        self.backbone.addItems(["convnet", "resnet18"])
        for label, widget in (("Epochs", self.epochs), ("Batch size", self.batch),
                              ("Learning rate", self.lr), ("Backbone", self.backbone)):
            form.addWidget(QLabel(label))
            form.addWidget(widget)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.train_button = QPushButton("Start Training")
        self.train_stop_button = QPushButton("Stop Training")
        self.train_stop_button.setEnabled(False)
        self.train_button.clicked.connect(self._start_training)
        self.train_stop_button.clicked.connect(self._stop_training)
        buttons.addWidget(self.train_button)
        buttons.addWidget(self.train_stop_button)
        layout.addLayout(buttons)

        self.train_progress = QProgressBar()
        layout.addWidget(self.train_progress)
        plots = QHBoxLayout()
        self.loss_plot = PlotWidget("Loss", ["train loss", "val loss"])
        self.acc_plot = PlotWidget("Validation accuracy (%)", ["val acc"], y_max=100.0)
        plots.addWidget(self.loss_plot)
        plots.addWidget(self.acc_plot)
        layout.addLayout(plots, 1)
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumHeight(110)
        layout.addWidget(self.train_log)

        brain = QGroupBox("Brain Manager")
        brain_layout = QHBoxLayout(brain)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(320)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_models)
        activate = QPushButton("Set Active Model")
        activate.clicked.connect(self._activate_model)
        self.model_status = QLabel("Active model: none (recorded routes)")
        brain_layout.addWidget(self.model_combo, 1)
        brain_layout.addWidget(refresh)
        brain_layout.addWidget(activate)
        layout.addWidget(brain)
        layout.addWidget(self.model_status)
        self._refresh_models()
        return tab

    def _refresh_models(self) -> None:
        current = self.model_combo.currentData()
        self.model_combo.clear()
        self.model_combo.addItem("None (recorded routes)", None)
        for path in sorted(MODELS_DIR.glob("*.pt"), reverse=True):
            self.model_combo.addItem(path.name, str(path))
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(current)))

    def _activate_model(self) -> None:
        if self.model_callback is None:
            self.model_status.setText("Model activation unavailable")
            return
        try:
            selected = self.model_combo.currentData()
            self.model_status.setText(self.model_callback(selected))
            self.active_model_path = selected
        except Exception as exc:
            self.model_status.setText(f"Could not load model: {exc}")

    def _start_training(self) -> None:
        if self.trainer is not None and self.trainer.isRunning():
            return
        self.loss_plot.clear()
        self.acc_plot.clear()
        self.train_log.clear()
        self.train_progress.setValue(0)
        params = TrainParams(
            epochs=self.epochs.value(),
            batch_size=self.batch.value(),
            learning_rate=self.lr.value(),
            backbone=self.backbone.currentText(),
            dataset_root=str(self.recorder.logger.root),
        )
        self.trainer = TrainingThread(params, self)
        self.trainer.progress.connect(self.train_progress.setValue)
        self.trainer.epoch_done.connect(self._on_epoch)
        self.trainer.log.connect(self.train_log.appendPlainText)
        self.trainer.finished_ok.connect(self._on_train_finished)
        self.trainer.failed.connect(self._on_train_failed)
        self.trainer.finished.connect(lambda: self._set_training_ui(False))
        self._set_training_ui(True)
        self.trainer.start()

    def _stop_training(self) -> None:
        if self.trainer is not None:
            self.trainer.request_stop()

    def _set_training_ui(self, running: bool) -> None:
        self.train_button.setEnabled(not running)
        self.train_stop_button.setEnabled(running)

    def _on_epoch(self, stats: dict) -> None:
        self.loss_plot.add_point(stats["train_loss"], stats["val_loss"])
        self.acc_plot.add_point(stats["val_acc"])

    def _on_train_finished(self, path: str) -> None:
        self.train_log.appendPlainText(f"Saved best checkpoint: {path}")
        self._refresh_models()
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(path)))

    def _on_train_failed(self, message: str) -> None:
        self.train_log.appendPlainText(message)

    def _analyze_dataset(self) -> None:
        report = get_readiness_report(self.recorder.logger.root)
        m = report["metrics"]
        self.ds_samples.setText(f"Samples: {m['total_samples']} across {m['sessions']} sessions")
        self.ds_progress.setValue(min(m["moving_frames"], MIN_SAMPLES))
        p = m["pct"]
        self.ds_dist.setText(
            f"Straight: {p['straight']:.0f}% | Left: {p['left']:.0f}% | Right: {p['right']:.0f}%"
            f" | Brake: {p['brake']:.0f}%  (score {report['score']:.0f}%, {m['duration_minutes']:.1f} min)"
        )
        status = report["status"]
        if status.startswith("Ready"):
            badge = "[Ready to Train]"
        elif "Turns" in status or "Straight" in status:
            badge = "[Need More Turns]"
        else:
            badge = f"[{status}]"
        self.ds_status.setText(badge)
        self.ds_advice.setText("\n".join(report["advice"]))

    def _on_map_changed(self, map_name: str) -> None:
        if hasattr(self, "route_map_combo") and self.route_map_combo.currentText() != map_name:
            self.route_map_combo.blockSignals(True)
            self.route_map_combo.setCurrentText(map_name)
            self.route_map_combo.blockSignals(False)
        if self.config_path is not None:
            try:
                update_config(self.config_path, ["agent", "active_map"], map_name)
            except (OSError, ValueError) as exc:
                self.rec_status.setText(f"Could not save active map: {exc}")
                self._record_log(f"ERROR: could not save active map: {exc}")
                return
            self._ui_config.setdefault("agent", {})["active_map"] = map_name
        self._load_route_editors()

    def _assign(self, route_name: str) -> None:
        if self.config_path is None:
            QMessageBox.warning(self, "Route Recorder", "No config file available.")
            return
        try:
            assign_route(self.config_path, route_name, self.recorder.segments, self.map_combo.currentText())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Route Recorder", str(exc))
            return
        self._ui_config = self._load_config_data()
        self._load_route_editors()
        self.rec_status.setText(f"Saved {len(self.recorder.segments)} segments to {self.map_combo.currentText()}/{route_name}")

    def _refresh_recorder(self) -> None:
        recording = self.recorder.recording
        self.recorder.map_name = self.map_combo.currentText()
        self.recorder.faction = self.faction_combo.currentText()
        self.recorder.direction = self.direction_combo.currentText()
        for combo in (self.map_combo, self.faction_combo, self.direction_combo):
            combo.setEnabled(not recording)
        if recording:
            self.rec_status.setText("● RECORDING (F9 stop, F10 discard)")
            self.rec_status.setStyleSheet("font-weight: bold; color: #e53935;")
        elif self._rec_was_recording:
            self.rec_status.setText(self.recorder.message)
            self.rec_status.setStyleSheet("font-weight: bold;")
            self.rec_preview.setPlainText(json.dumps(self.recorder.segments, indent=2))
        self._rec_was_recording = recording
        self.rec_start.setEnabled(not recording)
        self.rec_stop.setEnabled(recording)
        self.rec_discard.setEnabled(recording)
        now = time.monotonic()
        if now - self._last_analysis > (2.0 if recording else 10.0):
            self._last_analysis = now
            self._analyze_dataset()
        self.assign_fob.setEnabled(not recording and bool(self.recorder.segments))
        self.assign_hq.setEnabled(not recording and bool(self.recorder.segments))

    def _toggle_pause(self) -> None:
        if self._paused:
            self.commands.put({"kind": "resume"})
        else:
            if self.halt_input is not None:
                self.halt_input()
            self.commands.put({"kind": "pause"})

    def _stop(self) -> None:
        if self.halt_input is not None:
            self.halt_input()
        self.commands.put({"kind": "stop"})

    @staticmethod
    def _set_image(label: QLabel, image) -> None:
        if image is None or getattr(image, "size", 0) == 0 or label.size().isEmpty():
            return
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            fmt = QImage.Format.Format_RGB888
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format.Format_RGB888
        height, width = image.shape[:2]
        qimage = QImage(image.data, width, height, image.strides[0], fmt).copy()
        label.setPixmap(QPixmap.fromImage(qimage).scaled(
            label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def _refresh(self) -> None:
        self._refresh_recorder()
        latest_telemetry = None
        try:
            while True:
                latest_telemetry = self.telemetry.get_nowait()
        except queue.Empty:
            pass
        self._refresh_dashboard_values(latest_telemetry)
        try:
            while True:
                render = self.render_queue.get_nowait()
                self._consume_render(render)
                scores = self._latest_scores
                hits = self._latest_detections
                self.confidence_label.setText(
                    "MATCHES  ·  " + (
                        "  |  ".join(
                            f"{name}: {value:.2f}{' ✓' if name in hits else ''}"
                            for name, value in sorted(scores.items())
                        ) or "NO CURRENT SCORES"
                    )
                )
        except queue.Empty:
            pass
        try:
            while True:
                status = self.status_queue.get_nowait()
                message = status.get("message", "")
                if message:
                    self.status_label.setText(message)
                    self.header_status.setText(message.upper())
                    if message != self._last_logged_status:
                        self._record_log(message)
                        self._last_logged_status = message
                if "hotkeys_available" in status:
                    self.hotkeys_available = bool(status["hotkeys_available"])
        except queue.Empty:
            pass
        self._refresh_dashboard_status()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._latest_frame is not None:
            self._set_image(self.preview, self._latest_frame)
        if self._latest_roi is not None:
            self._set_image(self.roi_preview, self._latest_roi)
        if self._latest_map_frame is not None:
            self._set_image(self.map_preview, self._latest_map_frame)

    def closeEvent(self, event) -> None:
        if self.trainer is not None and self.trainer.isRunning():
            self.trainer.request_stop()
            self.trainer.wait(5000)
        self.recorder.stop()
        self.recorder.remove_hooks()
        if self.halt_input is not None:
            self.halt_input()
        self.commands.put({"kind": "stop"})
        event.accept()
