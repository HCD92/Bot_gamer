"""PyQt6 control panel for the logistics automation workers."""

from __future__ import annotations

import queue

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
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
    ):
        super().__init__()
        self.commands = commands
        self.telemetry = telemetry
        self.render_queue = render_queue
        self.status_queue = status_queue
        self.halt_input = halt_input
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
        toolbar.addStretch(1)
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.start_button.clicked.connect(lambda: self.commands.put({"kind": "start"}))
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop)
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.pause_button)
        toolbar.addWidget(self.stop_button)
        layout.addLayout(toolbar)

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
        self.setCentralWidget(central)

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
        try:
            while True:
                telemetry = self.telemetry.get_nowait()
                self.state_label.setText(telemetry.get("state", "STATE_IDLE"))
                self._paused = telemetry.get("state") == "STATE_PAUSED"
                self.pause_button.setText("Resume" if self._paused else "Pause")
                self.action_label.setText(telemetry.get("action", "Standby"))
                self.key_label.setText(telemetry.get("active_key", "--"))
                self.deliveries_label.setText(str(telemetry.get("deliveries", 0)))
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
        if self.halt_input is not None:
            self.halt_input()
        self.commands.put({"kind": "stop"})
        event.accept()
