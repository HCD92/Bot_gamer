"""PyQt6 control panel for the logistics automation workers."""

from __future__ import annotations

import json
import queue
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from capture import get_available_monitors
from recorder import RouteRecorder, assign_route, update_config
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QMessageBox,
    QPlainTextEdit,
    QTabWidget,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


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
    ):
        super().__init__()
        self.commands = commands
        self.telemetry = telemetry
        self.render_queue = render_queue
        self.status_queue = status_queue
        self.halt_input = halt_input
        self.capture_worker = capture_worker
        self.config_path = Path(config_path) if config_path else None
        self.recorder = RouteRecorder()
        self._hooks_ok = self.recorder.install_hooks()
        self._paused = False
        self.setWindowTitle("Wardogs Logistics Control")
        self.resize(1100, 760)
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        toolbar = QHBoxLayout()
        self.priority = QComboBox()
        self.priority.addItems(["Build", "Ammo", "Fuel", "Mechanical"])
        self.priority.currentTextChanged.connect(
            lambda value: self.commands.put({"kind": "priority", "value": value})
        )
        toolbar.addWidget(QLabel("Supply priority"))
        toolbar.addWidget(self.priority)
        toolbar.addWidget(QLabel("Batch deliveries"))
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 1000)
        self.batch_size.setValue(10)
        self.batch_size.setToolTip("Stop after this many verified deliveries")
        toolbar.addWidget(self.batch_size)
        toolbar.addStretch(1)
        self.start_button = QPushButton("Start")
        self.batch_button = QPushButton("Run batch")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.start_button.clicked.connect(lambda: self.commands.put({"kind": "start"}))
        self.batch_button.clicked.connect(
            lambda: self.commands.put({"kind": "start", "deliveries": self.batch_size.value()})
        )
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop)
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.batch_button)
        toolbar.addWidget(self.pause_button)
        toolbar.addWidget(self.stop_button)
        layout.addLayout(toolbar)

        display_box = QGroupBox("Display Settings")
        display_layout = QHBoxLayout(display_box)
        display_layout.addWidget(QLabel("Capture display"))
        self.monitor_combo = QComboBox()
        self._populate_monitors()
        self.monitor_combo.currentIndexChanged.connect(self._monitor_changed)
        display_layout.addWidget(self.monitor_combo, 1)
        layout.addWidget(display_box)

        self.preview = QLabel("Waiting for capture")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(800, 480)
        self.preview.setStyleSheet("background: #101719; color: #9da9a7; border: 1px solid #344344;")
        layout.addWidget(self.preview, 1)

        self.roi_preview = QLabel("ROI previews")
        self.roi_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.roi_preview.setFixedHeight(150)
        self.roi_preview.setStyleSheet("background: #101719; color: #9da9a7; border: 1px solid #344344;")
        layout.addWidget(self.roi_preview)

        grid = QGridLayout()
        self.state_label = QLabel("STATE_IDLE")
        self.action_label = QLabel("Standby")
        self.key_label = QLabel("--")
        self.deliveries_label = QLabel("0")
        self.duration_label = QLabel("0.0 s")
        self.status_label = QLabel("Starting workers")
        entries = (
            ("State", self.state_label),
            ("Action", self.action_label),
            ("Active input", self.key_label),
            ("Deliveries", self.deliveries_label),
            ("Loop duration", self.duration_label),
            ("Runtime", self.status_label),
        )
        for index, (title, value) in enumerate(entries):
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(QLabel(title), index // 3, (index % 3) * 2)
            grid.addWidget(value, index // 3, (index % 3) * 2 + 1)
        layout.addLayout(grid)
        tabs = QTabWidget()
        tabs.addTab(central, "Control")
        tabs.addTab(self._build_recorder_tab(), "Route Recorder")
        self.setCentralWidget(tabs)

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
            "Global hotkeys: F8 start, F9 stop, Esc abort. Tracks W/S/A/D/Space; "
            "steering is saved, W is held automatically by the agent."
        ))
        layout.addWidget(self.rec_status)
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
        self._rec_was_recording = False
        return tab

    def _assign(self, route_name: str) -> None:
        if self.config_path is None:
            QMessageBox.warning(self, "Route Recorder", "No config file available.")
            return
        try:
            assign_route(self.config_path, route_name, self.recorder.segments)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Route Recorder", str(exc))
            return
        self.rec_status.setText(f"Saved {len(self.recorder.segments)} segments to {route_name}")

    def _refresh_recorder(self) -> None:
        recording = self.recorder.recording
        if recording:
            self.rec_status.setText("● RECORDING (F9 stop, Esc abort)")
            self.rec_status.setStyleSheet("font-weight: bold; color: #e53935;")
        elif self._rec_was_recording:
            self.rec_status.setText(self.recorder.message)
            self.rec_status.setStyleSheet("font-weight: bold;")
            self.rec_preview.setPlainText(json.dumps(self.recorder.segments, indent=2))
        self._rec_was_recording = recording
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
        if image is None or image.size == 0:
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
        try:
            while True:
                telemetry = self.telemetry.get_nowait()
                self.state_label.setText(telemetry.get("state", "STATE_IDLE"))
                self._paused = telemetry.get("state") == "STATE_PAUSED"
                self.pause_button.setText("Resume" if self._paused else "Pause")
                self.action_label.setText(telemetry.get("action", "Standby"))
                self.key_label.setText(telemetry.get("active_key", "--"))
                deliveries = telemetry.get("deliveries", 0)
                target = telemetry.get("delivery_target", 0)
                self.deliveries_label.setText(
                    f"{deliveries} / {target}" if target > 0 else str(deliveries)
                )
                self.duration_label.setText(f"{telemetry.get('loop_seconds', 0.0):.1f} s")
                selected = telemetry.get("priority")
                if selected and self.priority.currentText() != selected:
                    self.priority.blockSignals(True)
                    self.priority.setCurrentText(selected)
                    self.priority.blockSignals(False)
        except queue.Empty:
            pass
        try:
            while True:
                render = self.render_queue.get_nowait()
                self._set_image(self.preview, render.get("image"))
                self._set_image(self.roi_preview, render.get("roi_preview"))
        except queue.Empty:
            pass
        try:
            while True:
                status = self.status_queue.get_nowait()
                self.status_label.setText(status.get("message", ""))
        except queue.Empty:
            pass

    def closeEvent(self, event) -> None:
        self.recorder.remove_hooks()
        if self.halt_input is not None:
            self.halt_input()
        self.commands.put({"kind": "stop"})
        event.accept()
