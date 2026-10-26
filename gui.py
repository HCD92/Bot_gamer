"""PyQt6 control panel for the logistics automation workers."""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
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
    QTabWidget,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


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
        self.trainer: TrainingThread | None = None
        self._last_analysis = 0.0
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
        display_layout.addWidget(QLabel("Window"))
        self.window_combo = QComboBox()
        self.window_combo.setMinimumWidth(260)
        display_layout.addWidget(self.window_combo, 1)
        refresh_windows = QPushButton("Refresh")
        refresh_windows.clicked.connect(self._populate_windows)
        display_layout.addWidget(refresh_windows)
        self._populate_windows()
        self.window_combo.currentIndexChanged.connect(self._window_changed)
        self.roi_toggle = QCheckBox("Show ROI boxes and confidence")
        self.roi_toggle.setChecked(True)
        self.roi_toggle.toggled.connect(self._toggle_rois)
        display_layout.addWidget(self.roi_toggle)
        layout.addWidget(display_box)

        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setWordWrap(True)
        layout.addWidget(self.confidence_label)

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
        tabs.addTab(self._build_training_tab(), "Neural Net Training")
        self.setCentralWidget(tabs)

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
            "Global hotkeys: F8 start, F9 stop, Esc abort. Tracks W/S/A/D/Space; "
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
        self.rec_discard = QPushButton("Discard (Esc)")
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
            self.model_status.setText(self.model_callback(self.model_combo.currentData()))
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
        if self.config_path is None:
            return
        try:
            update_config(self.config_path, ["agent", "active_map"], map_name)
        except (OSError, ValueError) as exc:
            self.rec_status.setText(f"Could not save active map: {exc}")

    def _assign(self, route_name: str) -> None:
        if self.config_path is None:
            QMessageBox.warning(self, "Route Recorder", "No config file available.")
            return
        try:
            assign_route(self.config_path, route_name, self.recorder.segments, self.map_combo.currentText())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Route Recorder", str(exc))
            return
        self.rec_status.setText(f"Saved {len(self.recorder.segments)} segments to {self.map_combo.currentText()}/{route_name}")

    def _refresh_recorder(self) -> None:
        recording = self.recorder.recording
        self.recorder.map_name = self.map_combo.currentText()
        self.recorder.faction = self.faction_combo.currentText()
        self.recorder.direction = self.direction_combo.currentText()
        for combo in (self.map_combo, self.faction_combo, self.direction_combo):
            combo.setEnabled(not recording)
        if recording:
            self.rec_status.setText("● RECORDING (F9 stop, Esc abort)")
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
                scores = render.get("scores") or {}
                hits = set(render.get("detected", ()))
                self.confidence_label.setText(
                    "  |  ".join(
                        f"{name}: {value:.2f}{' ✓' if name in hits else ''}"
                        for name, value in sorted(scores.items())
                    ) or "Confidence: --"
                )
        except queue.Empty:
            pass
        try:
            while True:
                status = self.status_queue.get_nowait()
                self.status_label.setText(status.get("message", ""))
        except queue.Empty:
            pass

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
