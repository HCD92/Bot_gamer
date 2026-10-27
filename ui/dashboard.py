"""Reusable widgets for the live logistics dashboard."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)


class StatusIndicator(QLabel):
    """Compact status line with explicit healthy, warning, and missing states."""

    COLORS = {
        "healthy": "#54f28b",
        "warning": "#ffc247",
        "missing": "#ff5964",
        "neutral": "#8faab6",
    }

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.title = title
        self.set_status("neutral", "CHECKING")

    def set_status(self, level: str, detail: str) -> None:
        color = self.COLORS.get(level, self.COLORS["neutral"])
        marker = {"healthy": "●", "warning": "▲", "missing": "■"}.get(level, "○")
        self.setText(f"{marker}  {self.title.upper()}   {detail.upper()}")
        self.setStyleSheet(f"color: {color}; font-weight: 600;")


class GaugeCard(QGroupBox):
    """Native numeric gauge; its displayed value always comes from OCR."""

    def __init__(self, title: str, scale_max: int, parent: QWidget | None = None):
        super().__init__(title, parent)
        self.scale_max = max(1, scale_max)
        layout = QVBoxLayout(self)
        self.value_label = QLabel("--")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setStyleSheet(
            "font-size: 30px; font-weight: 700; color: #54f28b; letter-spacing: 2px;"
        )
        self.bar = QProgressBar()
        self.bar.setRange(0, self.scale_max)
        self.bar.setTextVisible(False)
        self.source_label = QLabel("OCR · WAITING FOR READING")
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_label.setStyleSheet("color: #8faab6; font-size: 10px;")
        layout.addWidget(self.value_label)
        layout.addWidget(self.bar)
        layout.addWidget(self.source_label)
        self.set_value(None)

    def set_value(self, value: int | float | None, *, warning_at: float | None = None) -> None:
        if value is None:
            self.value_label.setText("--")
            self.bar.setValue(0)
            self.source_label.setText("OCR · NO CURRENT READING")
            return
        numeric = float(value)
        self.value_label.setText(f"{numeric:g}")
        self.bar.setValue(max(0, min(self.scale_max, round(numeric))))
        warning = warning_at is not None and numeric >= warning_at
        color = "#ff5964" if warning else "#54f28b"
        self.value_label.setStyleSheet(
            f"font-size: 30px; font-weight: 700; color: {color}; letter-spacing: 2px;"
        )
        self.source_label.setText("LIVE OCR VALUE")


class RouteSegmentEditor(QGroupBox):
    """Editor for the real W + optional A/D route segment configuration."""

    changed = pyqtSignal(list)

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["STEER (W + A/D)", "DURATION (S)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setMinimumHeight(75)
        self.table.setMaximumHeight(190)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        self.add_button = QPushButton("+ ADD SEGMENT")
        self.remove_button = QPushButton("REMOVE SELECTED")
        self.add_button.clicked.connect(self._add_row)
        self.remove_button.clicked.connect(self._remove_selected)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        layout.addLayout(buttons)
        self.table.itemChanged.connect(lambda _item: self._emit_changed())

    def set_segments(self, segments: list[dict[str, Any]]) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for segment in segments:
            self._insert_row(segment)
        self.table.blockSignals(False)

    def segments(self) -> list[dict[str, Any]]:
        result = []
        for row in range(self.table.rowCount()):
            steer_combo = self.table.cellWidget(row, 0)
            duration_spin = self.table.cellWidget(row, 1)
            if not isinstance(steer_combo, QComboBox) or not isinstance(duration_spin, QDoubleSpinBox):
                continue
            steer = {"STRAIGHT": "", "LEFT (A)": "A", "RIGHT (D)": "D"}[
                steer_combo.currentText()
            ]
            result.append({"steer": steer, "seconds": round(duration_spin.value(), 3)})
        return result

    def _insert_row(self, segment: dict[str, Any] | None = None) -> None:
        segment = segment or {"steer": "", "seconds": 1.0}
        row = self.table.rowCount()
        self.table.insertRow(row)

        steer_combo = QComboBox()
        steer_combo.addItems(["STRAIGHT", "LEFT (A)", "RIGHT (D)"])
        steer = str(segment.get("steer", "")).upper()
        steer_combo.setCurrentText({"": "STRAIGHT", "A": "LEFT (A)", "D": "RIGHT (D)"}.get(steer, "STRAIGHT"))
        steer_combo.currentTextChanged.connect(lambda _text: self._emit_changed())
        self.table.setCellWidget(row, 0, steer_combo)

        duration_spin = QDoubleSpinBox()
        duration_spin.setRange(0.15, 8.0)
        duration_spin.setSingleStep(0.1)
        duration_spin.setDecimals(2)
        duration_spin.setValue(min(8.0, max(0.15, float(segment.get("seconds", 1.0)))))
        duration_spin.valueChanged.connect(lambda _value: self._emit_changed())
        self.table.setCellWidget(row, 1, duration_spin)

    def _add_row(self) -> None:
        self._insert_row()
        self._emit_changed()

    def _remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.rowCount():
            rows = [self.table.rowCount() - 1]
        for row in rows:
            self.table.removeRow(row)
        if rows:
            self._emit_changed()

    def _emit_changed(self) -> None:
        self.changed.emit(self.segments())
