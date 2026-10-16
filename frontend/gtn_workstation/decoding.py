from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from PySide6 import QtCore, QtWidgets

from .backend import CalibrationSettings, ModelTask
from .decoding_results import CalibrationResults, EvidenceResults, reason_text
from .models import ChannelMontage
from .storage import (
    atomic_write_json,
    load_inference_input_session,
    load_model_input_session,
)
from .theme import asset_icon


@dataclass(frozen=True)
class SessionEntry:
    path: Path
    session_id: str
    subject: str
    started: str
    target: int | None
    flashes: int
    calibration_ready: bool
    inference_ready: bool
    status: str
    detail: str
    simulation_status: str = "not_declared"


def session_entry(path: Path) -> SessionEntry:
    payload = json.loads((path / "session.json").read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError("session.json must be an object")
    experiment = payload.get("experiment")
    experiment = experiment if isinstance(experiment, dict) else {}
    target = payload.get("target_label")
    target = target if isinstance(target, dict) else {}
    target_value = target.get("thought_digit")
    if target.get("status") != "confirmed" or type(target_value) is not int or not 1 <= target_value <= 9:
        target_value = None
    recording = payload.get("recording")
    recording = recording if isinstance(recording, dict) else {}
    timeline = recording.get("timeline")
    timeline = timeline if isinstance(timeline, dict) else {}
    quality = payload.get("quality")
    quality = quality if isinstance(quality, dict) else {}
    continuity = quality.get("eeg_continuity")
    continuity = continuity if isinstance(continuity, dict) else {}
    ready = (
        payload.get("status") == "completed"
        and isinstance(payload.get("experiment"), dict)
        and isinstance(experiment.get("subject_id"), str)
        and bool(experiment.get("subject_id", "").strip())
        and recording.get("stage") == "bids_raw"
        and timeline.get("status") == "finalized"
        and continuity.get("passed") is True
    )
    detail = (
        str(payload.get("status", "unknown"))
        if isinstance(payload.get("experiment"), dict) else "实验元数据无效"
    )
    if ready:
        try:
            load_inference_input_session(path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            ready = False
            detail = str(exc)
    try:
        montage = ChannelMontage.from_dict(payload.get("montage") or {})
        montage.validate()
        if tuple(montage.eeg) != tuple(ChannelMontage().eeg):
            raise ValueError("通道布局与当前七 EEG / EOG 配置不同")
    except (TypeError, ValueError) as exc:
        ready = False
        detail = str(exc)
    calibration_ready = False
    if ready and target_value is not None:
        try:
            load_model_input_session(path)
            calibration_ready = True
        except (OSError, ValueError, TypeError, KeyError) as exc:
            detail = str(exc)
    if calibration_ready:
        state = "已确认"
    elif ready:
        state = "待确认"
    else:
        state = "不可用"
    return SessionEntry(
        path=path.resolve(),
        session_id=str(payload.get("session_id") or path.name),
        subject=str(experiment.get("subject_id") or payload.get("subject_id") or ""),
        started=str(payload.get("started_utc") or ""),
        target=target_value,
        flashes=int(experiment.get("blocks", 0)) * int(experiment.get("digits_per_block", 0)),
        calibration_ready=calibration_ready,
        inference_ready=ready,
        status=state,
        detail=detail,
        simulation_status="declared_mock" if isinstance(payload.get("device"), dict)
        and payload["device"].get("transport") == "mock" else "not_declared",
    )


def scan_sessions(root: Path) -> list[SessionEntry]:
    entries = []
    if not root.is_dir():
        return entries
    for manifest in root.rglob("session.json"):
        try:
            entries.append(session_entry(manifest.parent))
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            continue
    return sorted(entries, key=lambda entry: (entry.started, entry.session_id))


class DecodingPanel(QtWidgets.QWidget):
    advanced_requested = QtCore.Signal()
    record_requested = QtCore.Signal(str)
    calibration_record_requested = QtCore.Signal(str, int)
    confirmation_requested = QtCore.Signal(str)
    message = QtCore.Signal(str)
    busy_changed = QtCore.Signal()

    def __init__(self, app_root: Path, runner: Any, parent=None, *, mock: bool | None = None):
        super().__init__(parent)
        self.app_root = app_root
        self.runner = runner
        self.settings_path = app_root / "config" / "decoding.json"
        self.entries: list[SessionEntry] = []
        self._job: dict[str, Any] | None = None
        self._allowed = True
        self._device_ready = False
        self._mock_mode = mock
        self._last_result: dict[str, Any] | None = None
        self._last_result_dir: Path | None = None
        self._model_subject = ""
        self._model_summary: dict[str, Any] = {}
        self._available_epoch_qc_path = ""
        self._model_simulation_status: str | None = None
        self._browse_buttons: list[QtWidgets.QToolButton] = []
        self._settings = self._load_settings()
        self._bundled_source = app_root / "models" / "source" / "bnci_cz6_seed20260906_A01.pt"
        self._build()
        self.runner.progress_changed.connect(self.progress.setValue)
        self.runner.output_line.connect(self._output_line)
        self.runner.finished.connect(self._finished)
        self.runner.state_changed.connect(lambda _state: QtCore.QTimer.singleShot(0, self.sync_buttons))
        self.refresh_sessions()
        if Path(self.model_path.text().strip()).is_file():
            QtCore.QTimer.singleShot(0, self.inspect_current_model)

    def _load_settings(self) -> dict[str, Any]:
        try:
            value = json.loads(self.settings_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_settings(self) -> None:
        value = {
            "schema": "brainsync-decoding-ui/1",
            "backend_root": self.backend_root.text().strip(),
            "source_checkpoint": self.source_path.text().strip(),
            "model_artifact": self.model_path.text().strip(),
            "method": self.method.currentData(),
            "device": self.compute_device.currentData(),
            "reference_channel": self.reference.currentData(),
            "reference_by_method": dict(self._reference_by_method),
            "ocular_correction": self.ocular.isChecked(),
            "eog_reference_policy": self.eog_reference.currentData(),
            "epoch_qc": self.epoch_qc.isChecked(),
            "epoch_qc_model": self.epoch_qc_model.text().strip(),
            "continuous_qc_model": self.continuous_qc.text().strip(),
            "data_root": self.data_root.text().strip(),
        }
        try:
            atomic_write_json(self.settings_path, value)
        except OSError as exc:
            self.message.emit(f"模型设置保存失败: {exc}")

    def _file_button(self, callback, tooltip: str) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setIcon(asset_icon("data_viewer"))
        button.setToolTip(tooltip)
        button.setFixedSize(34, 32)
        button.clicked.connect(callback)
        self._browse_buttons.append(button)
        return button

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.settings_dialog = QtWidgets.QDialog(self)
        self.settings_dialog.setWindowTitle("校准与数据设置")
        self.settings_dialog.resize(710, 500)
        settings_layout = QtWidgets.QVBoxLayout(self.settings_dialog)
        self.backend_settings = QtWidgets.QWidget()
        engine = QtWidgets.QHBoxLayout(self.backend_settings)
        engine.setContentsMargins(0, 0, 0, 0)
        engine.addWidget(QtWidgets.QLabel("算法项目"))
        default_backend = self.app_root.parents[1] / "n2p3-net"
        self.backend_root = QtWidgets.QLineEdit(str(self._settings.get("backend_root", default_backend)))
        engine.addWidget(self.backend_root, 1)
        engine.addWidget(self._file_button(self._choose_backend, "选择算法项目目录"))
        settings_layout.addWidget(self.backend_settings)

        data_row = QtWidgets.QHBoxLayout()
        data_row.addWidget(QtWidgets.QLabel("数据目录"))
        configured_data = Path(str(self._settings.get("data_root", self.app_root / "data")))
        self.data_root = QtWidgets.QLineEdit(str(configured_data if configured_data.is_dir() else self.app_root / "data"))
        self.data_root.editingFinished.connect(self.refresh_sessions)
        data_row.addWidget(self.data_root, 1)
        data_row.addWidget(self._file_button(self._choose_data_root, "选择校准和测试记录目录"))
        settings_layout.addLayout(data_row)

        model_row = QtWidgets.QHBoxLayout()
        self.model_path = QtWidgets.QLineEdit(str(self._settings.get("model_artifact", "")), self.settings_dialog)
        self.model_path.setPlaceholderText("尚未选择个体模型")
        self.model_path.hide()
        self.model_path.textChanged.connect(self._model_changed)
        self.model_info = QtWidgets.QLabel("尚未校准")
        self.model_info.setObjectName("muted")
        self.model_info.setWordWrap(True)
        self.model_info.setMinimumWidth(180)
        model_row.addWidget(self.model_info, 1)
        self.load_model = QtWidgets.QPushButton("加载模型")
        self.load_model.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon))
        self.load_model.clicked.connect(self._choose_model)
        model_row.addWidget(self.load_model)
        self.export_model = self._file_button(self._export_model, "另存个体模型")
        self.export_model.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton))
        model_row.addWidget(self.export_model)
        settings_button = QtWidgets.QToolButton()
        settings_button.setIcon(asset_icon("settings"))
        settings_button.setFixedSize(34, 34)
        settings_button.setToolTip("校准设置与研究工具")
        settings_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QtWidgets.QMenu(settings_button)
        menu.addAction("校准与数据设置", self.settings_dialog.show)
        menu.addAction("研究算法任务", self.advanced_requested.emit)
        settings_button.setMenu(menu)
        model_row.addWidget(settings_button)
        layout.addLayout(model_row)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        calibration = QtWidgets.QWidget()
        cal = QtWidgets.QVBoxLayout(calibration)
        options = QtWidgets.QHBoxLayout()
        options.addWidget(QtWidgets.QLabel("受试者"))
        self.subject = QtWidgets.QComboBox()
        self.subject.setEditable(True)
        self.subject.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.subject.setMinimumWidth(130)
        self.subject.currentTextChanged.connect(self._render_sessions)
        options.addWidget(self.subject)
        self.method = QtWidgets.QComboBox()
        self.method.addItem("本人 LDA", "lda")
        self.method.addItem("源模型微调", "source")
        default_method = "lda"
        self.method.setCurrentIndex(max(0, self.method.findData(self._settings.get("method", default_method))))
        self.method.currentIndexChanged.connect(self.sync_buttons)
        research_form = QtWidgets.QFormLayout()
        research_form.addRow("校准方法", self.method)
        self.compute_device = QtWidgets.QComboBox()
        self.compute_device.addItem("CPU", "cpu")
        self.compute_device.addItem("CUDA GPU", "cuda")
        self.compute_device.addItem("自动", "auto")
        self.compute_device.setCurrentIndex(max(0, self.compute_device.findData(self._settings.get("device", "cpu"))))
        research_form.addRow("计算设备", self.compute_device)
        self.reference = QtWidgets.QComboBox()
        self.reference.addItem("原生 REF（左耳垂）", "")
        self.reference.addItem("Cz 差分（六导）", "Cz")
        self._reference_by_method = {"lda": "", "source": "Cz" if self._bundled_source.is_file() else ""}
        saved_references = self._settings.get("reference_by_method", {})
        if isinstance(saved_references, dict):
            self._reference_by_method.update({k: v for k, v in saved_references.items()
                                             if k in self._reference_by_method and v in {"", "Cz"}})
        if not saved_references and self._settings.get("reference_channel") in {"", "Cz"}:
            self._reference_by_method[self.method.currentData()] = self._settings["reference_channel"]
        self.reference.setCurrentIndex(max(0, self.reference.findData(self._reference_by_method[self.method.currentData()])))
        self.reference.currentIndexChanged.connect(self._remember_reference)
        self.method.currentIndexChanged.connect(self._restore_method_reference)
        research_form.addRow("信号参考", self.reference)
        self.ocular = QtWidgets.QCheckBox("眼电回归")
        self.ocular.setChecked(bool(self._settings.get("ocular_correction", True)))
        self.ocular.toggled.connect(self.sync_buttons)
        research_form.addRow("眼电处理", self.ocular)
        self.eog_reference = QtWidgets.QComboBox()
        self.eog_reference.addItem("随 EEG 重参考", "shared_measured_reference")
        self.eog_reference.addItem("保留采集电压", "preserve_acquired_eog")
        self.eog_reference.setCurrentIndex(max(0, self.eog_reference.findData(
            self._settings.get("eog_reference_policy", "shared_measured_reference"))))
        research_form.addRow("EOG 参考", self.eog_reference)
        self.source_filter = QtWidgets.QComboBox()
        self.source_filter.addItem("全部来源", "all")
        self.source_filter.addItem("未标模拟", "not_declared")
        self.source_filter.addItem("模拟记录", "declared_mock")
        self.source_filter.setCurrentIndex(0 if self._mock_mode is None else 2 if self._mock_mode else 1)
        self.source_filter.currentIndexChanged.connect(self._render_sessions)
        research_form.addRow("记录来源", self.source_filter)
        settings_layout.addLayout(research_form)
        self.recipe_info = QtWidgets.QLabel()
        self.recipe_info.setObjectName("muted")
        self.recipe_info.setWordWrap(True)
        options.addWidget(self.recipe_info, 1)
        options.addStretch()
        refresh = QtWidgets.QToolButton()
        refresh.setIcon(asset_icon("refresh"))
        refresh.setFixedSize(34, 34)
        refresh.setToolTip("刷新采集记录")
        refresh.clicked.connect(self.refresh_sessions)
        options.addWidget(refresh)
        cal.addLayout(options)
        self.source_settings = QtWidgets.QWidget()
        source_row = QtWidgets.QHBoxLayout(self.source_settings)
        source_row.setContentsMargins(0, 0, 0, 0)
        source_row.addWidget(QtWidgets.QLabel("公开源模型"))
        self.source_path = QtWidgets.QLineEdit(str(self._settings.get("source_checkpoint", self._bundled_source if self._bundled_source.is_file() else "")))
        source_row.addWidget(self.source_path, 1)
        self.source_browse = self._file_button(self._choose_source, "选择公开 ERP 源模型")
        source_row.addWidget(self.source_browse)
        settings_layout.addWidget(self.source_settings)

        quality_row = QtWidgets.QHBoxLayout()
        self.epoch_qc = QtWidgets.QCheckBox("校准内信号证据 QC")
        self.epoch_qc.setToolTip("需要与当前滤波匹配、包含边界评估和原始恒值检测的连续 QC 模型")
        self.epoch_qc.setChecked(bool(self._settings.get("epoch_qc", False)))
        quality_row.addWidget(self.epoch_qc)
        settings_layout.addLayout(quality_row)
        continuous_row = QtWidgets.QHBoxLayout()
        continuous_row.addWidget(QtWidgets.QLabel("连续 QC 模型"))
        self.continuous_qc = QtWidgets.QLineEdit(str(self._settings.get("continuous_qc_model", "")))
        continuous_row.addWidget(self.continuous_qc, 1)
        continuous_row.addWidget(self._file_button(self._choose_continuous_qc, "选择已拟合的连续 QC 模型"))
        settings_layout.addLayout(continuous_row)

        frozen_quality_row = QtWidgets.QHBoxLayout()
        frozen_quality_row.addWidget(QtWidgets.QLabel("冻结 epoch QC 模型"))
        self.epoch_qc_model = QtWidgets.QLineEdit(str(self._settings.get("epoch_qc_model", "")))
        frozen_quality_row.addWidget(self.epoch_qc_model, 1)
        frozen_quality_row.addWidget(self._file_button(self._choose_epoch_qc_model, "选择已拟合的 epoch QC 模型"))
        self.epoch_qc_model.textChanged.connect(self._sync_epoch_qc_mode)
        self.epoch_qc.toggled.connect(lambda checked: self.epoch_qc_model.clear() if checked else None)
        self._sync_epoch_qc_mode(self.epoch_qc_model.text())
        settings_layout.addLayout(frozen_quality_row)
        settings_layout.addStretch()
        close_settings = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        close_settings.button(QtWidgets.QDialogButtonBox.StandardButton.Close).setText("关闭")
        close_settings.rejected.connect(self.settings_dialog.close)
        self.settings_dialog.finished.connect(lambda _code: (self._save_settings(), self.sync_buttons()))
        settings_layout.addWidget(close_settings)

        self.sessions = QtWidgets.QTableWidget(0, 7)
        self.sessions.setHorizontalHeaderLabels(["校准", "采集时间", "目标", "闪烁数", "来源", "状态", "记录"])
        self.sessions.verticalHeader().hide()
        self.sessions.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sessions.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.sessions.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.sessions.setColumnHidden(6, True)
        self.sessions.setAlternatingRowColors(True)
        self.sessions.setShowGrid(False)
        self.sessions.verticalHeader().setDefaultSectionSize(36)
        self.sessions.setMinimumHeight(190)
        self.sessions.setColumnWidth(0, 48)
        self.sessions.setColumnWidth(1, 168)
        self.sessions.setColumnWidth(2, 52)
        self.sessions.setColumnWidth(3, 68)
        self.sessions.setColumnWidth(4, 76)
        self.sessions.setColumnWidth(5, 76)
        self.sessions.itemChanged.connect(self._update_budget)
        self.sessions.itemSelectionChanged.connect(self.sync_buttons)
        cal.addWidget(self.sessions, 1)
        self.budget = QtWidgets.QLabel()
        self.budget.setWordWrap(True)
        cal.addWidget(self.budget)
        actions = QtWidgets.QHBoxLayout()
        self.select_all = QtWidgets.QPushButton("选择可用记录")
        self.select_all.clicked.connect(self._select_available)
        actions.addWidget(self.select_all)
        self.confirm_session = QtWidgets.QPushButton("确认目标")
        self.confirm_session.clicked.connect(self._confirm_selected)
        actions.addWidget(self.confirm_session)
        actions.addStretch()
        cal.addLayout(actions)
        capture = QtWidgets.QHBoxLayout()
        self.calibration_target = QtWidgets.QComboBox()
        for digit in range(1, 10):
            self.calibration_target.addItem(f"目标 {digit}", digit)
        capture.addWidget(self.calibration_target)
        self.record_calibration = QtWidgets.QPushButton("采集校准")
        self.record_calibration.clicked.connect(lambda: self.calibration_record_requested.emit(
            self.subject.currentText().strip(), int(self.calibration_target.currentData())
        ))
        capture.addWidget(self.record_calibration)
        capture.addStretch()
        self.calibrate = QtWidgets.QPushButton("校准并保存")
        self.calibrate.setObjectName("primary")
        self.calibrate.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton))
        self.calibrate.clicked.connect(self.start_calibration)
        capture.addWidget(self.calibrate)
        cal.addLayout(capture)
        self.calibration_hint = QtWidgets.QLabel()
        self.calibration_hint.setWordWrap(True)
        self.calibration_hint.setObjectName("muted")
        cal.addWidget(self.calibration_hint)
        self.tabs.addTab(calibration, "本人校准")

        inference = QtWidgets.QWidget()
        infer = QtWidgets.QVBoxLayout(inference)
        record_row = QtWidgets.QHBoxLayout()
        self.test_session = QtWidgets.QComboBox()
        self.test_session.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.test_session.setMinimumContentsLength(20)
        self.test_session.currentIndexChanged.connect(self.sync_buttons)
        record_row.addWidget(self.test_session, 1)
        self.infer = QtWidgets.QPushButton("判定所选记录")
        self.infer.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
        self.infer.clicked.connect(self._infer_selected)
        record_row.addWidget(self.infer)
        self.new_test = QtWidgets.QPushButton("开始新测试")
        self.new_test.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
        self.new_test.setObjectName("primary")
        self.new_test.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.new_test.clicked.connect(self._record_test)
        record_row.addStretch()
        record_row.addWidget(self.new_test)
        infer.addLayout(record_row)
        self.prediction = QtWidgets.QLabel("--")
        self.prediction.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.prediction.setStyleSheet("font-size: 42px; font-weight: 700; padding: 6px;")
        self.prediction.setFixedHeight(60)
        infer.addWidget(self.prediction)
        self.result_state = QtWidgets.QLabel("等待测试记录")
        self.result_state.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.result_state.setWordWrap(True)
        self.result_state.setMinimumHeight(25)
        infer.addWidget(self.result_state)
        self.evidence = EvidenceResults()
        infer.addWidget(self.evidence, 1)
        self.feedback = QtWidgets.QLabel("")
        self.feedback.setWordWrap(True)
        infer.addWidget(self.feedback)
        self.tabs.addTab(inference, "猜数字")
        self.calibration_results = CalibrationResults()
        self.report_scroll = QtWidgets.QScrollArea()
        self.report_scroll.setWidgetResizable(True)
        self.report_scroll.setWidget(self.calibration_results)
        self.tabs.addTab(self.report_scroll, "校准结果")

        status = QtWidgets.QHBoxLayout()
        self.state = QtWidgets.QLabel("就绪")
        self.state.setWordWrap(True)
        self.state.setMaximumWidth(420)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.stop = QtWidgets.QToolButton()
        self.stop.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaStop))
        self.stop.setToolTip("停止模型任务")
        self.stop.clicked.connect(self.runner.stop)
        status.addWidget(self.state)
        status.addWidget(self.progress, 1)
        status.addWidget(self.stop)
        self.log_toggle = QtWidgets.QToolButton()
        self.log_toggle.setIcon(asset_icon("data_viewer"))
        self.log_toggle.setToolTip("运行日志")
        self.log_toggle.setCheckable(True)
        status.addWidget(self.log_toggle)
        layout.addLayout(status)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMinimumHeight(35)
        self.log.setMaximumHeight(75)
        layout.addWidget(self.log)
        self.log.hide()
        self.log_toggle.toggled.connect(self.log.setVisible)
        self._model_changed()

    def _export_model(self) -> None:
        source = Path(self.model_path.text().strip())
        if not source.is_file():
            return
        destination, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "另存个体模型", str(source.with_name(f"{self._model_subject or 'personal'}_model.pt")), "模型 (*.pt)"
        )
        if destination:
            try:
                if Path(destination).resolve() != source.resolve():
                    shutil.copy2(source, destination)
                self.state.setText("个体模型已另存")
            except OSError as exc:
                self.state.setText("个体模型另存失败")
                self.message.emit(str(exc))

    def _choose_backend(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "算法项目目录", self.backend_root.text())
        if path:
            self.backend_root.setText(path)
            self._save_settings()

    def _choose_source(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "公开 ERP 源模型", self.backend_root.text(), "模型 (*.pt *.pth)")
        if path:
            self.source_path.setText(path)
            self.method.setCurrentIndex(self.method.findData("source"))
            self._save_settings()

    def _choose_data_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "记录目录", self.data_root.text())
        if path:
            self.set_data_root(Path(path))

    def set_data_root(self, path: Path) -> None:
        self.data_root.setText(str(path.resolve()))
        self.refresh_sessions()
        self._save_settings()

    def _choose_model(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "个体模型", str(self.app_root / "models"), "模型 (*.pt *.pth)")
        if path:
            self.model_path.setText(path)
            self.inspect_current_model()

    def inspect_current_model(self) -> None:
        path = Path(self.model_path.text().strip()).expanduser()
        if path.is_file():
            self._start_job("inspect", artifact=path)

    def _apply_model_summary(self, record: dict[str, Any]) -> None:
        self._model_summary = record
        self._model_subject = str(record.get("subject_id", ""))
        self._available_epoch_qc_path = str(record.get("epoch_qc_model_path") or "")
        self._model_simulation_status = record.get("simulation_status")
        method = {"target_only_shrinkage_lda": "本人 LDA", "source_initialized_n2p3": "源模型微调",
                  "scratch_n2p3": "本人神经网络"}.get(
            record.get("model_kind"), str(record.get("model_kind", "--"))
        )
        quality_method = {
            "personal_epoch_signal_evidence": "信号证据 QC",
            "fold_local_ptp_cv": "历史相对幅度 QC",
        }.get(record.get("epoch_qc_method"))
        quality = f" · {quality_method}" if quality_method else ""
        self.model_info.setText(
            f"{self._model_subject} · {method} · 实际训练 {record.get('actual_training_trials', record.get('calibration_trials', '--'))}"
            f" / 校准输入 {record.get('calibration_trials', '--')} 次闪烁{quality}"
        )
        self.model_info.setToolTip(self.model_path.text())
        self.calibration_results.set_summary(record)
        self._render_test_sessions()
        self.sync_buttons()

    def _model_changed(self, _text: str = "") -> None:
        if not hasattr(self, "model_info"):
            return
        self._model_subject = ""
        self._model_summary = {}
        self._available_epoch_qc_path = ""
        self._model_simulation_status = None
        path = Path(self.model_path.text().strip()).expanduser()
        self.model_info.setText("模型待校验" if path.is_file() else "尚无个体模型")
        self.calibration_results.clear("模型待校验" if path.is_file() else "尚无校准结果")
        self._clear_prediction()
        self._render_test_sessions()
        self.sync_buttons()

    def refresh_sessions(self) -> None:
        selected_subject = self.subject.currentText()
        self.entries = scan_sessions(Path(self.data_root.text().strip()).expanduser())
        subjects = sorted({entry.subject for entry in self.entries if entry.subject})
        self.subject.blockSignals(True)
        self.subject.clear()
        self.subject.addItems(subjects)
        index = self.subject.findText(selected_subject)
        if index >= 0:
            self.subject.setCurrentIndex(index)
        elif selected_subject or not subjects:
            self.subject.setEditText(selected_subject or "P001")
        self.subject.blockSignals(False)
        self._render_sessions()

    def _render_sessions(self, _subject: str = "") -> None:
        checked = {entry.path for entry in self.selected_calibration()}
        self.sessions.blockSignals(True)
        self.sessions.setRowCount(0)
        for entry in reversed(self.entries):
            if entry.subject != self.subject.currentText():
                continue
            if self.source_filter.currentData() not in {"all", entry.simulation_status}:
                continue
            row = self.sessions.rowCount()
            self.sessions.insertRow(row)
            check = QtWidgets.QTableWidgetItem()
            check.setData(QtCore.Qt.ItemDataRole.UserRole, entry)
            check.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            if entry.calibration_ready:
                check.setFlags(check.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                check.setCheckState(QtCore.Qt.CheckState.Checked if entry.path in checked else QtCore.Qt.CheckState.Unchecked)
            self.sessions.setItem(row, 0, check)
            try:
                started = datetime.fromisoformat(entry.started.replace("Z", "+00:00")).astimezone().strftime("%m-%d %H:%M:%S")
            except ValueError:
                started = entry.started[:19].replace("T", " ")
            values = (started, str(entry.target or "--"), str(entry.flashes),
                      "模拟" if entry.simulation_status == "declared_mock" else "未标模拟",
                      entry.status, entry.session_id)
            for column, value in enumerate(values, start=1):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(entry.detail if column == 5 else str(entry.path))
                self.sessions.setItem(row, column, item)
        self.sessions.blockSignals(False)
        self._render_test_sessions()
        self._update_budget()

    def _render_test_sessions(self) -> None:
        current = self.test_session.currentData()
        self.test_session.blockSignals(True)
        self.test_session.clear()
        used = self._model_summary.get("calibration_sessions") or []
        used_ids = {str(item.get("session_id")) if isinstance(item, dict) else str(item) for item in used}
        ready_after = self._model_summary.get("calibration_available_timestamp_s")
        for entry in reversed(self.entries):
            if not entry.inference_ready or entry.flashes != 45 or entry.subject != self._model_subject:
                continue
            if entry.session_id in used_ids or entry.simulation_status != self._model_simulation_status:
                continue
            if isinstance(ready_after, (int, float)):
                try:
                    if datetime.fromisoformat(entry.started.replace("Z", "+00:00")).timestamp() <= ready_after:
                        continue
                except ValueError:
                    continue
            self.test_session.addItem(entry.session_id, str(entry.path))
        index = self.test_session.findData(current)
        self.test_session.setCurrentIndex(index if index >= 0 else 0)
        self.test_session.setPlaceholderText("暂无匹配的新测试记录")
        self.test_session.blockSignals(False)

    def selected_calibration(self) -> list[SessionEntry]:
        return [
            self.sessions.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            for row in range(self.sessions.rowCount())
            if self.sessions.item(row, 0).checkState() == QtCore.Qt.CheckState.Checked
        ]

    def _select_available(self) -> None:
        selected = self.selected_calibration()
        available = [self.sessions.item(row, 0) for row in range(self.sessions.rowCount())
                     if self.sessions.item(row, 0).flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable]
        state = QtCore.Qt.CheckState.Unchecked if len(selected) == len(available) else QtCore.Qt.CheckState.Checked
        self.sessions.blockSignals(True)
        for item in available:
            item.setCheckState(state)
        self.sessions.blockSignals(False)
        self._update_budget()

    def _update_budget(self, _item=None) -> None:
        entries = self.selected_calibration()
        total = sum(entry.flashes for entry in entries)
        digits = sorted({entry.target for entry in entries if entry.target is not None})
        self.budget.setText(
            f"{len(entries)} 份校准记录 · {total} 次闪烁 · 目标覆盖 {len(digits)}/9 · "
            + (", ".join(map(str, digits)) if digits else "--")
        )
        available_count = sum(
            bool(self.sessions.item(row, 0).flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            for row in range(self.sessions.rowCount())
        )
        self.select_all.setText("取消全选" if entries and len(entries) == available_count else "选择可用记录")
        self.sync_buttons()

    def _confirm_selected(self) -> None:
        row = self.sessions.currentRow()
        if row < 0:
            return
        entry = self.sessions.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
        if entry.inference_ready:
            self.confirmation_requested.emit(str(entry.path))

    def set_runtime_state(self, *, allowed: bool, device_ready: bool) -> None:
        self._allowed = allowed
        self._device_ready = device_ready
        self.sync_buttons()

    def _remember_reference(self) -> None:
        self._reference_by_method[self.method.currentData()] = self.reference.currentData()

    def _restore_method_reference(self) -> None:
        selected = self._reference_by_method[self.method.currentData()]
        self.reference.setCurrentIndex(max(0, self.reference.findData(selected)))

    def sync_buttons(self) -> None:
        if not hasattr(self, "calibrate"):
            return
        busy = bool(self.runner.running)
        allowed = self._allowed and not busy and self._job is None
        has_model = Path(self.model_path.text().strip()).is_file()
        source = self.method.currentData() == "source"
        self.source_settings.setVisible(source)
        recipe = "源模型微调" if source else "本人 LDA"
        quality = " · 冻结质量模型" if self.epoch_qc_model.text().strip() or self.continuous_qc.text().strip() else " · 局部 QC" if self.epoch_qc.isChecked() else ""
        self.recipe_info.setText(recipe + " · 双向滤波" + quality)
        self.recipe_info.setToolTip(f"{self.reference.currentText()} · 眼电回归{'开启' if self.ocular.isChecked() else '关闭'}")
        self.source_path.setEnabled(allowed and source)
        self.source_browse.setEnabled(allowed and source)
        count = len(self.selected_calibration())
        required = 2 if source else 1
        source_ready = not source or Path(self.source_path.text().strip()).is_file()
        self.calibrate.setEnabled(allowed and count >= required and source_ready)
        hint = "" if count >= required else f"尚需选择 {required - count} 份已确认校准记录"
        if not source_ready:
            hint = "公开源模型不可用，请在校准设置中检查"
        self.calibration_hint.setText(hint)
        self.infer.setEnabled(allowed and has_model and bool(self._model_subject) and bool(self.test_session.currentData()))
        matching_mode = self._mock_mode is None or self._model_simulation_status == (
            "declared_mock" if self._mock_mode else "not_declared"
        )
        self.new_test.setEnabled(allowed and self._device_ready and has_model and bool(self._model_subject) and matching_mode)
        self.new_test.setToolTip("设备未就绪" if not self._device_ready else "模型与当前采集模式不一致" if not matching_mode else "采集一次新的 45 闪烁测试")
        self.test_session.setVisible(self.test_session.count() > 0)
        self.infer.setVisible(self.test_session.count() > 0)
        self.record_calibration.setEnabled(allowed and self._device_ready and bool(self.subject.currentText().strip()))
        row = self.sessions.currentRow()
        entry = self.sessions.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole) if row >= 0 and self.sessions.item(row, 0) else None
        self.confirm_session.setEnabled(allowed and entry is not None and entry.inference_ready and entry.target is None)
        self.stop.setEnabled(busy)
        for button in self._browse_buttons:
            button.setEnabled(allowed)
        self.source_browse.setEnabled(allowed and source)
        self.load_model.setEnabled(allowed)
        self.export_model.setEnabled(allowed and has_model and bool(self._model_subject))
        for widget in (self.subject, self.method, self.compute_device, self.reference, self.ocular, self.sessions, self.select_all,
                       self.backend_root, self.data_root, self.model_path, self.source_filter,
                       self.epoch_qc, self.epoch_qc_model, self.continuous_qc, self.calibration_target, self.test_session):
            widget.setEnabled(allowed)
        self.eog_reference.setEnabled(allowed and self.ocular.isChecked())

    def start_calibration(self) -> None:
        entries = self.selected_calibration()
        if not entries:
            return
        settings = CalibrationSettings(
            source_checkpoint=self.source_path.text().strip() if self.method.currentData() == "source" else "",
            reference_channel=str(self.reference.currentData() or ""),
            ocular_correction=self.ocular.isChecked(),
            eog_reference_policy=str(self.eog_reference.currentData()),
            epoch_qc=self.epoch_qc.isChecked(),
            epoch_qc_model=self.epoch_qc_model.text().strip(),
            continuous_qc_model=self.continuous_qc.text().strip(),
        )
        if self.method.currentData() == "source" and not settings.source_checkpoint:
            self.state.setText("请先选择源模型")
            return
        self._start_job("calibrate", sessions=tuple(entry.path for entry in entries), settings=settings)

    def _choose_continuous_qc(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "连续 QC 模型", self.continuous_qc.text(), "JSON (*.json)")
        if path:
            self.continuous_qc.setText(path)
            self._save_settings()

    def _sync_epoch_qc_mode(self, value: str) -> None:
        if value.strip():
            self.epoch_qc.setChecked(False)

    def _choose_epoch_qc_model(self) -> None:
        initial = self.epoch_qc_model.text() or self._available_epoch_qc_path
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "冻结 epoch QC 模型", initial, "JSON (*.json)")
        if path:
            self.epoch_qc_model.setText(path)
            self._save_settings()

    def _infer_selected(self) -> None:
        value = self.test_session.currentData()
        if value:
            self.start_inference(Path(value))

    def start_inference(self, session: Path, *, request_confirmation: bool = False) -> bool:
        if self.runner.running or not self._allowed or self._job is not None:
            return False
        model = Path(self.model_path.text().strip()).expanduser()
        if not model.is_file():
            self.state.setText("请先校准或加载个体模型")
            return False
        self.tabs.setCurrentIndex(1)
        self._clear_prediction()
        self.result_state.setText("正在判定")
        return self._start_job(
            "infer", artifact=model, sessions=(session,),
            session=session,
            request_confirmation=request_confirmation,
        )

    def _clear_prediction(self) -> None:
        self.prediction.setText("--")
        self.result_state.setText("等待测试记录")
        self.result_state.setToolTip("")
        self.feedback.clear()
        self._last_result = None
        self._last_result_dir = None
        self.evidence.clear()

    def _record_test(self) -> None:
        self.record_requested.emit(self._model_subject or self.subject.currentText())

    def _start_job(self, task: str, *, artifact: Path | None = None,
                   sessions: tuple[Path, ...] = (), settings: CalibrationSettings | None = None,
                   session: Path | None = None, request_confirmation: bool = False) -> bool:
        if self.runner.running or not self._allowed or self._job is not None:
            return False
        output = self.app_root / "models" / "runs" / f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{task}_{uuid4().hex[:8]}"
        try:
            request = ModelTask.create(
                Path(self.backend_root.text().strip()), task, output, artifact=artifact,
                sessions=sessions, settings=settings, device=str(self.compute_device.currentData()),
            )
            self._save_settings()
            self._job = {"task": task, "output": output, "session": session, "request_confirmation": request_confirmation, "request": request}
            self.progress.setValue(0)
            self.log.clear()
            self.state.setText({"calibrate": "校准中", "infer": "推理中", "inspect": "加载模型"}[task])
            if task == "calibrate":
                self.calibration_results.clear("正在校准")
                self.tabs.setCurrentWidget(self.report_scroll)
            self.runner.start_backend(request.program, list(request.arguments), task=task, output_dir=output)
            self.busy_changed.emit()
            self.sync_buttons()
            return True
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError) as exc:
            self._job = None
            self.state.setText(str(exc))
            if task == "infer":
                self.result_state.setText(f"本轮未形成有效判定：{exc}")
                self.result_state.setToolTip(str(exc))
            elif task == "calibrate":
                self.calibration_results.clear("校准未完成")
            elif task == "inspect":
                self._model_changed()
                self.model_info.setText("模型校验失败")
            self.log.appendPlainText(str(exc))
            self.message.emit(str(exc))
            self.sync_buttons()
            return False

    def _output_line(self, line: str) -> None:
        if self._job is not None:
            self.log.appendPlainText(line)
            stages = {"validate_recordings": "检查校准与记录合同", "fit_calibration_eog_regression": "拟合校准眼电处理",
                      "fit_target_only_shrinkage_lda": "拟合本人 LDA", "refit_full_authorized_calibration": "训练最终个体模型",
                      "predict_unlabelled_recording": "计算新测试证据"}
            if line.strip() in stages:
                self.state.setText(stages[line.strip()])
            elif line.startswith("select_"):
                self.state.setText("比较校准候选方案")

    def _finished(self, ok: bool, message: str) -> None:
        job = self._job
        if job is None:
            return
        self._job = None
        self.state.setText("已完成" if ok else "任务未完成")
        self.message.emit(message)
        if ok:
            self.progress.setValue(100)
            try:
                summary = job["request"].read_result()
                if job["task"] in {"calibrate", "inspect"}:
                    artifact = Path(str(summary["model_artifact"]))
                    if not artifact.is_file():
                        raise ValueError("校准任务未生成个体模型")
                    self.model_path.setText(str(artifact))
                    self._model_changed()
                    self._apply_model_summary(summary)
                    self._save_settings()
                    if job["task"] == "calibrate":
                        self.tabs.setCurrentWidget(self.report_scroll)
                else:
                    self._render_result(summary)
                    self._last_result = summary
                    self._last_result_dir = job["output"]
                    if job["request_confirmation"] and job["session"] is not None:
                        QtCore.QTimer.singleShot(0, lambda path=str(job["session"]): self.confirmation_requested.emit(path))
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                self.state.setText("结果不可用")
                self.log.appendPlainText(str(exc))
                self.log_toggle.setChecked(True)
                if job["task"] == "inspect":
                    self._model_changed()
                elif job["task"] == "calibrate":
                    self._model_changed()
                    self.calibration_results.clear("校准结果不可用")
                else:
                    self._clear_prediction()
                    self.result_state.setText(f"推理结果不可用：{exc}")
                    self.result_state.setToolTip(str(exc))
                    if job["request_confirmation"]:
                        self._retain_failed_inference(job)
        elif job["task"] == "infer":
            failure = message
            try:
                report = json.loads((job["output"] / "summary.json").read_text(encoding="utf-8"))
                failure = str(report.get("error", {}).get("message") or failure)
            except (OSError, ValueError, TypeError, AttributeError):
                pass
            self.result_state.setText(f"本轮未形成有效判定：{failure}")
            self.result_state.setToolTip(failure)
            self.log.appendPlainText(failure)
            self._retain_failed_inference(job)
        elif job["task"] == "inspect":
            self._model_changed()
            self.model_info.setText("模型校验失败")
        if not ok:
            self.log.appendPlainText(message)
            self.log_toggle.setChecked(True)
            if job["task"] == "calibrate":
                self.calibration_results.clear("校准未完成")
        QtCore.QTimer.singleShot(0, self.sync_buttons)
        self.busy_changed.emit()

    def _retain_failed_inference(self, job: dict[str, Any]) -> None:
        if job["session"] is None:
            return
        request = job["request"]
        self._last_result = {"session_id": request.session_ids[0], "prediction": None, "status": "failed",
                             "session_dirs": [str(path) for path in request.sessions],
                             "subject_id": request.subject_id, "request_id": request.request_id}
        self._last_result_dir = job["output"]
        if job["request_confirmation"]:
            QtCore.QTimer.singleShot(0, lambda path=str(job["session"]): self.confirmation_requested.emit(path))

    def _render_result(self, summary: dict[str, Any]) -> None:
        prediction = summary.get("prediction")
        self.prediction.setText(str(prediction) if type(prediction) is int else "未判定")
        reason = reason_text(summary.get("reason"))
        self.result_state.setText("本轮判定" if type(prediction) is int else reason or "本轮未形成有效判定")
        self.result_state.setToolTip(reason)
        self.evidence.set_summary(summary)

    def record_feedback(self, session_dir: Path, target: int) -> None:
        if self._last_result is None or self._last_result_dir is None:
            return
        session_id = self._last_result.get("session_id")
        try:
            session_dir = session_dir.resolve()
            if self._last_result.get("session_dirs") != [str(session_dir)]:
                return
            payload = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
            if str(payload.get("session_id") or session_dir.name) != str(session_id):
                return
            label = payload.get("target_label")
            if (type(target) is not int or not 1 <= target <= 9 or not isinstance(label, dict)
                    or label.get("status") != "confirmed" or label.get("thought_digit") != target
                    or payload.get("experiment", {}).get("subject_id") != self._last_result.get("subject_id")):
                raise ValueError("反馈必须匹配原推理会话及已保存的确认目标")
            prediction = self._last_result.get("prediction")
            correct = prediction == target if type(prediction) is int else None
            record = {"schema": "brainsync-prediction-feedback/1", "session_id": session_id,
                      "session_dir": str(session_dir), "subject_id": self._last_result.get("subject_id"),
                      "request_id": self._last_result.get("request_id"),
                      "prediction": prediction, "target": target, "correct": correct,
                      "result_path": str(self._last_result_dir / "summary.json")}
            atomic_write_json(self._last_result_dir / "feedback.json", record)
            outcome = "未形成有效判定" if correct is None else "命中" if correct else "未命中"
            self.feedback.setText(f"真实目标 {target} · {outcome}")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self.message.emit(f"结果反馈写入失败: {exc}")
