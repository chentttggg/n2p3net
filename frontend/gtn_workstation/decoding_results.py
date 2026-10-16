from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .theme import COLORS, PLOT_COLORS

TEAL = PLOT_COLORS["teal"]
AMBER = PLOT_COLORS["amber"]
BLUE = PLOT_COLORS["blue"]
MUTED = PLOT_COLORS["text"]


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def recipe_name(record: dict[str, Any]) -> str:
    return str(record.get("name") or record.get("candidate_id") or "未命名方案")


class ResultPlot(QtWidgets.QStackedWidget):
    def __init__(self, *, height: int = 210):
        super().__init__()
        self.setMinimumHeight(height)
        self.empty = QtWidgets.QLabel()
        self.empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty.setWordWrap(True)
        self.empty.setObjectName("muted")
        self.addWidget(self.empty)
        self.plot = pg.PlotWidget(background=PLOT_COLORS["bg"])
        self.plot.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Ignored)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.showGrid(y=True, alpha=0.15)
        for axis in ("left", "bottom"):
            self.plot.getAxis(axis).setTextPen(MUTED)
            self.plot.getAxis(axis).setPen(PLOT_COLORS["line"])
        self.addWidget(self.plot)
        self.clear("暂无结果")

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(400, self.minimumHeight())

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(0, self.minimumHeight())

    def clear(self, message: str) -> None:
        self.plot.clear()
        self.empty.setText(message)
        self.setCurrentWidget(self.empty)

    def reveal(self) -> None:
        self.setCurrentWidget(self.plot)


class CalibrationResults(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.record: dict[str, Any] = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        self.status = QtWidgets.QLabel("尚无校准结果")
        self.status.setObjectName("section_title")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        metrics = QtWidgets.QGridLayout()
        self.metrics: dict[str, QtWidgets.QLabel] = {}
        for column, (key, title) in enumerate((
            ("auc", "内部验证 AUC"), ("training", "实际训练 / 输入闪烁"),
            ("sessions", "校准记录"), ("epochs", "最终训练轮数"),
        )):
            label = QtWidgets.QLabel(title)
            label.setObjectName("muted")
            label.setWordWrap(True)
            value = QtWidgets.QLabel("--")
            value.setStyleSheet(f"font-size: 23px; font-weight: 600; color: {COLORS['teal']};")
            value.setWordWrap(True)
            metrics.addWidget(label, 0, column)
            metrics.addWidget(value, 1, column)
            metrics.setColumnStretch(column, 1)
            self.metrics[key] = value
        layout.addLayout(metrics)
        self.scope = QtWidgets.QLabel()
        self.scope.setWordWrap(True)
        self.scope.setObjectName("muted")
        layout.addWidget(self.scope)
        self.usage = QtWidgets.QProgressBar()
        self.usage.setRange(0, 1000)
        self.usage.setFixedHeight(21)
        layout.addWidget(self.usage)
        curves = QtWidgets.QHBoxLayout()
        heading = QtWidgets.QLabel("训练曲线")
        heading.setObjectName("section_title")
        curves.addWidget(heading)
        curves.addStretch()
        self.curve_choice = QtWidgets.QComboBox()
        self.curve_choice.setMinimumWidth(210)
        self.curve_choice.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.curve_choice.currentIndexChanged.connect(self._show_curve)
        curves.addWidget(self.curve_choice)
        layout.addLayout(curves)
        self.loss = ResultPlot()
        self.loss.plot.setLabel("bottom", "训练轮次", color=MUTED)
        self.loss.plot.setLabel("left", "交叉熵损失", color=MUTED)
        self.loss.plot.addLegend(offset=(12, 10))
        layout.addWidget(self.loss, 1)
        self.comparison_title = QtWidgets.QLabel("候选方案 · 内部验证 AUC")
        self.comparison_title.setObjectName("section_title")
        layout.addWidget(self.comparison_title)
        self.comparison = ResultPlot(height=160)
        self.comparison.setMaximumHeight(190)
        self.comparison.plot.getAxis("bottom").setHeight(48)
        layout.addWidget(self.comparison)
        self.clear()

    def clear(self, message: str = "尚无校准结果") -> None:
        self.record = {}
        self.status.setText(message)
        for value in self.metrics.values():
            value.setText("--")
        self.scope.setText("新测试命中：未评估")
        self.usage.hide()
        self.curve_choice.clear()
        self.curve_choice.hide()
        self.loss.clear("尚无训练曲线")
        self.comparison.clear("尚无内部验证结果")
        self.comparison.hide()
        self.comparison_title.hide()

    def set_summary(self, record: dict[str, Any]) -> None:
        self.clear()
        self.record = record
        selection = record.get("internal_validation") or {}
        selected = selection.get("selected") or {}
        history = selection.get("refit_history") or {}
        recipes = [item for item in selection.get("recipes", []) if isinstance(item, dict)]
        auc = finite_number(selected.get("validation_auc", selection.get("validation_auc")))
        if selection.get("selection_evidence") is False:
            auc = None
        self.metrics["auc"].setText(f"{auc:.3f}" if auc is not None else "未验证")
        n_input = record.get("calibration_trials")
        n_training = record.get("actual_training_trials", n_input)
        self.metrics["training"].setText(f"{n_training if n_training is not None else '--'} / {n_input if n_input is not None else '--'}")
        sessions = record.get("calibration_sessions")
        self.metrics["sessions"].setText(str(len(sessions)) if isinstance(sessions, list) else "--")
        epochs = len(history.get("train_losses") or []) or selected.get("selected_epochs")
        self.metrics["epochs"].setText(str(epochs) if epochs else "不适用" if "lda" in record.get("model_kind", "") else "未记录")
        method = {"target_only_shrinkage_lda": "本人 LDA", "source_initialized_n2p3": "源模型微调",
                  "scratch_n2p3": "本人神经网络"}.get(record.get("model_kind"), record.get("model_kind", "个体模型"))
        self.status.setText(f"{record.get('subject_id', '--')} · {method} · 已保存")
        self.status.setToolTip(str(record.get("model_artifact") or ""))
        validation_sessions = selection.get("validation_sessions") or []
        validation_text = f"内部验证 {len(validation_sessions)} 份记录" if validation_sessions else "内部校准验证" if auc is not None else "无独立内部验证"
        source = " · 模拟数据" if record.get("simulation_status") == "declared_mock" else ""
        criterion = {"legacy_epoch_validation_cross_entropy": " · 按内部验证损失选方",
                     "binary_auc": " · 按内部验证 AUC 选方",
                     "brainsync_operational_r5": " · 按内部 R5 选方"}.get(selection.get("scoring_protocol"), "")
        self.scope.setText(f"{validation_text}{criterion} · 新测试命中：未评估{source}")
        if isinstance(n_input, int) and isinstance(n_training, int) and n_input > 0 and 0 <= n_training <= n_input:
            self.usage.setValue(round(1000 * n_training / n_input))
            self.usage.setFormat(f"输入数据采用 {n_training / n_input:.1%} · 未参与最终训练 {n_input - n_training} 次")
            self.usage.show()
        self.curve_choice.blockSignals(True)
        chosen_index = 0
        for number, item in enumerate(recipes, 1):
            fit = item.get("fit_history") or {}
            if fit.get("train_losses") or fit.get("val_losses"):
                label = f"方案 {number} · 内部训练"
                if recipe_name(item) == recipe_name(selected):
                    label += "（已选）"
                    chosen_index = self.curve_choice.count()
                self.curve_choice.addItem(label, fit)
                self.curve_choice.setItemData(self.curve_choice.count() - 1, recipe_name(item), QtCore.Qt.ItemDataRole.ToolTipRole)
        if history.get("train_losses"):
            self.curve_choice.addItem("最终模型 · 全量校准训练", history)
        self.curve_choice.blockSignals(False)
        self.curve_choice.setCurrentIndex(chosen_index)
        self.curve_choice.setVisible(self.curve_choice.count() > 1)
        self._show_curve()
        if not self.curve_choice.count():
            self.loss.clear("LDA 无逐轮训练曲线" if "lda" in record.get("model_kind", "") else "此模型未保存逐轮训练曲线")
        scored = [(number, item, finite_number(item.get("validation_auc"))) for number, item in enumerate(recipes, 1)]
        scored = [(number, item, value) for number, item, value in scored if value is not None]
        if scored:
            plot = self.comparison.plot
            ticks = []
            for index, (number, item, value) in enumerate(scored):
                chosen = recipe_name(item) == recipe_name(selected)
                plot.addItem(pg.BarGraphItem(x=[index], height=[value], width=0.55, brush=TEAL if chosen else BLUE, pen=None))
                ticks.append((index, f"方案 {number}{' · 已选' if chosen else ''}\n{value:.3f}"))
            plot.getAxis("bottom").setTicks([ticks])
            plot.setYRange(0, 1, padding=0.02)
            plot.setXRange(-0.65, max(0.65, len(scored) - 0.35), padding=0)
            plot.addItem(pg.InfiniteLine(pos=0.5, angle=0, pen=pg.mkPen(MUTED, style=QtCore.Qt.PenStyle.DashLine)))
            self.comparison.reveal()
            self.comparison.show()
            self.comparison_title.show()

    def _show_curve(self, _index: int = 0) -> None:
        self.loss.clear("此模型未保存逐轮训练曲线")
        history = self.curve_choice.currentData() or {}
        present = False
        for key, name, color in (("train_losses", "训练损失", TEAL), ("val_losses", "内部验证损失", AMBER)):
            values = history.get(key) or []
            points = [finite_number(value) for value in values]
            if not any(value is not None for value in points):
                continue
            ys = np.asarray([np.nan if value is None else value for value in points])
            self.loss.plot.plot(np.arange(1, len(ys) + 1), ys, pen=pg.mkPen(color, width=2), name=name,
                                symbol="o" if len(ys) == 1 else None, connect="finite")
            present = True
        if present:
            self.loss.plot.enableAutoRange()
            self.loss.reveal()


class EvidenceResults(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QtWidgets.QLabel("九候选证据分数")
        title.setObjectName("section_title")
        title.setToolTip("平均二分类证据，数值不是命中概率")
        layout.addWidget(title)
        self.chart = ResultPlot(height=190)
        self.chart.plot.getAxis("bottom").setHeight(48)
        self.chart.plot.getAxis("bottom").setTicks([[(digit, str(digit)) for digit in range(1, 10)]])
        self.chart.plot.setLabel("left", "证据分数", color=MUTED)
        self.chart.plot.setXRange(0.4, 9.6, padding=0)
        layout.addWidget(self.chart, 1)
        stages = QtWidgets.QHBoxLayout()
        self.stages: list[QtWidgets.QLabel] = []
        for _level in range(1, 6):
            label = QtWidgets.QLabel()
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setMinimumHeight(54)
            label.setStyleSheet(f"border-top: 2px solid {COLORS['line']}; padding: 6px; font-size: 15px;")
            stages.addWidget(label, 1)
            self.stages.append(label)
        layout.addLayout(stages)
        self.availability = QtWidgets.QLabel()
        self.availability.setWordWrap(True)
        self.availability.setObjectName("muted")
        layout.addWidget(self.availability)
        self.coverage = QtWidgets.QLabel()
        self.coverage.setWordWrap(True)
        self.coverage.setObjectName("muted")
        layout.addWidget(self.coverage)
        self.clear()

    def clear(self) -> None:
        self.chart.clear("等待测试结果")
        self.coverage.clear()
        self.availability.clear()
        for level, label in enumerate(self.stages, 1):
            label.setText(f"R{level}\n--")
            label.setToolTip("")

    def set_summary(self, record: dict[str, Any]) -> None:
        self.clear()
        if record.get("filter_phase") == "zero":
            self.availability.setText("R1–R5 为完整采集结束后的各轮证据回顾。")
        scores = record.get("candidate_scores") or {}
        if isinstance(scores, list):
            scores = {str(index + 1): value for index, value in enumerate(scores)}
        prediction = record.get("prediction")
        values = []
        ticks = []
        for digit in range(1, 10):
            value = finite_number(scores.get(str(digit)))
            ticks.append((digit, f"{digit}\n{value:.3f}" if value is not None else f"{digit}\n--"))
            if value is None:
                continue
            values.append(value)
            self.chart.plot.addItem(pg.BarGraphItem(x=[digit], height=[value], width=0.62,
                brush=TEAL if digit == prediction else BLUE, pen=None))
        if values:
            self.chart.plot.getAxis("bottom").setTicks([ticks])
            self.chart.plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(MUTED)))
            low, high = min(0, *values), max(0, *values)
            self.chart.plot.setYRange(low, high if high > low else low + 1, padding=0.12)
            self.chart.reveal()
        else:
            self.chart.clear("无可用候选证据")
        rows = record.get("by_repetition") or []
        if isinstance(rows, dict):
            rows = [{"repetition": key, **value} for key, value in rows.items() if isinstance(value, dict)]
        for row in rows:
            try:
                index = int(row.get("repetition", row.get("repetitions"))) - 1
            except (TypeError, ValueError):
                continue
            if not 0 <= index < 5:
                continue
            value = row.get("prediction")
            self.stages[index].setText(f"R{index + 1}\n{value if type(value) is int else '拒判'}")
            self.stages[index].setToolTip(reason_text(row.get("reason")))
        scheduled = record.get("scheduled_trials")
        usable = record.get("evidence_trials")
        if type(scheduled) is int and type(usable) is int:
            self.coverage.setText(f"有效证据 {usable} / {scheduled} 次闪烁 · 不可用 {scheduled - usable} 次")


def reason_text(reason: Any) -> str:
    return {
        "incomplete_candidate_coverage": "候选覆盖不完整，无法形成有效判定",
        "tie": "最高证据并列，无法形成唯一判定",
        "continuous_qc_no_usable_epoch": "信号质量不足，没有可用证据",
    }.get(str(reason), str(reason or ""))
