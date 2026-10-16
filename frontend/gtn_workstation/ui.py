from __future__ import annotations

import asyncio
import atexit
import json
import logging
import platform
import secrets
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .acquisition_provenance import acquisition_runtime_record
from .decoding import DecodingPanel
from .device_state import DeviceState
from .models import (
    BRAIN_SYNC_STORAGE_CONTRACT,
    DEFAULT_MODEL_PARAMETER_OPTION,
    DEFAULT_SAMPLE_RATE_HZ,
    DIGIT_VOCAB,
    IMPEDANCE_PASS_THRESHOLD_KOHM,
    MODEL_TASK_OPTIONS,
    ChannelMontage,
    ExperimentConfig,
    ModelParameterOption,
    ModelTaskConfig,
    default_channel_positions_m,
    edf_birthdate,
    parse_model_parameter_options,
    utc_now,
)
from .quality_reports import (
    QualityReportController,
    QualityReportDialog,
    read_report_index,
)
from .recording import RecordingSession
from .runtime import (
    ExperimentController,
    ImpedanceOperationError,
    ModelJobRunner,
    create_device_adapter,
)
from .signal_processing import (
    FilterConfigurationError,
    RealtimeEEGFilter,
    RealtimeFilterConfig,
    validate_realtime_filter_config,
)
from .storage import (
    CONFIRMATION_PENDING,
    SessionStore,
    atomic_write_json,
    find_latest_model_input_session,
    load_model_input_session,
)
from .theme import (
    APP_QSS,
    COLORS,
    PLOT_COLORS,
    application_stylesheet,
    asset_icon,
    tool_button,
    workspace_palette,
)
from .transport import GAINS, TRANSPORTS, ConnectionPreferences, ConnectionProfile, transport_spec

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - the SDK environment normally includes pyqtgraph
    pg = None


EEG_DATA_DELAY_WARNING_SECONDS = 12.0
EEG_DATA_DELAY_LOG_INTERVAL_SECONDS = 30.0
STREAM_RECOVERY_COOLDOWN_SECONDS = 60.0
STARTUP_CONNECTION_TIMEOUT_SECONDS = 20.0
WAVEFORM_DISPLAY_SECONDS = 2.0
PLOT_REFRESH_INTERVAL_MS = 16
DIGIT_FONT_SIZE_PX = 260
MAX_ANALYSIS_EVENT_ROWS = 1000
RECORDING_FILE_READY_TIMEOUT_SECONDS = 10.0
RECORDING_FILE_READY_POLL_SECONDS = 0.1
RECORDING_STOP_TIMEOUT_SECONDS = 15.0
EXPERIMENT_FINISH_TIMEOUT_SECONDS = 60.0


class Panel(QtWidgets.QFrame):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("panel")


class MetricCard(Panel):
    def __init__(self, title: str, value: str = "--", accent: str = COLORS["teal"]):
        super().__init__()
        self.setObjectName("metric")
        self.setMinimumHeight(86)
        self.setMinimumWidth(0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("metric_title")
        self.title.setWordWrap(True)
        self.value = QtWidgets.QLabel(value)
        self.value.setStyleSheet(f"font-size: 22px; font-weight: 600; color: {accent};")
        self.value.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.value)

    def set_value(self, value: str) -> None:
        self.value.setText(str(value))
        self.value.setToolTip(str(value))


class StimulusWindow(QtWidgets.QWidget):
    stop_requested = QtCore.Signal()
    pause_requested = QtCore.Signal()
    resume_requested = QtCore.Signal()
    save_requested = QtCore.Signal()
    discard_requested = QtCore.Signal()

    def __init__(self, screen_index: int = 1):
        super().__init__()
        self.setWindowTitle("BrainSync Stimulus")
        self.setWindowFlags(QtCore.Qt.WindowType.Window | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setStyleSheet("background: #05080b; color: #f7fbff;")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._allow_close = False
        self._stop_emitted = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel("")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setStyleSheet(f"font-size: {DIGIT_FONT_SIZE_PX}px; font-weight: 300; color: #f7fbff;")
        layout.addWidget(self.label)
        self.pause_menu = QtWidgets.QWidget()
        menu = QtWidgets.QVBoxLayout(self.pause_menu)
        menu.setContentsMargins(40, 40, 40, 40)
        menu.addStretch()
        title = QtWidgets.QLabel("测试已暂停\n请休息")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 36px;")
        menu.addWidget(title)
        for text, callback in (
            ("继续测试", self.resume_requested.emit),
            ("终止测试并保存", self.save_requested.emit),
            ("废弃本次测试", self.discard_requested.emit),
        ):
            button = QtWidgets.QPushButton(text)
            button.setMinimumHeight(56)
            button.setFixedWidth(320)
            button.setStyleSheet("font-size: 24px; padding: 12px; background: #292e31; border: 1px solid #737b80;")
            button.clicked.connect(callback)
            menu.addWidget(button, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        menu.addStretch()
        self.pause_menu.hide()
        layout.addWidget(self.pause_menu)
        screens = QtWidgets.QApplication.screens()
        if not 0 <= screen_index < len(screens):
            raise ValueError(f"显示器编号 {screen_index} 不存在，当前共有 {len(screens)} 个显示器。")
        self.target_screen = screens[screen_index]
        self.setGeometry(self.target_screen.geometry())

    def present(self, digit: int | None) -> None:
        self.present_text("" if digit is None else str(digit))

    def present_text(self, text: str) -> None:
        self.label.setStyleSheet(
            f"font-size: {DIGIT_FONT_SIZE_PX}px; font-weight: 300; color: #f7fbff;"
            if len(text) <= 2
            else "font-size: 42px; font-weight: 500; color: #f7fbff;"
        )
        self.label.setText(text)
        # repaint() pushes the paint synchronously. Do NOT call
        # QApplication.processEvents() here: under qasync it re-enters the
        # Qt event loop from inside a running asyncio task and can re-enter
        # experiment code mid-trial.
        self.label.repaint()

    def show_pause_menu(self, visible: bool) -> None:
        self.label.setVisible(not visible)
        self.pause_menu.setVisible(visible)
        self.pause_menu.setEnabled(True)

    def allow_close(self) -> None:
        self._allow_close = True

    def _request_stop(self) -> None:
        if self._stop_emitted:
            return
        self._stop_emitted = True
        self.stop_requested.emit()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            if not event.isAutoRepeat() and not self.pause_menu.isVisible():
                self.pause_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._allow_close:
            event.accept()
            return
        event.ignore()
        self.pause_requested.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, mock: bool = False, *, app_root: Path | None = None, device=None,
                 transport: str | None = None, address: str | None = None):
        super().__init__()
        self.mock = mock
        self.setWindowTitle("BrainSync GTN Workstation" + (" - 模拟模式（非设备数据）" if mock else ""))
        self.resize(1440, 900)
        self.setMinimumSize(1160, 720)
        application = QtWidgets.QApplication.instance()
        if application is not None and application.style().objectName().lower() != "fusion":
            application.setStyle("Fusion")
        self.setPalette(workspace_palette())
        self.setStyleSheet(APP_QSS)

        if app_root is not None:
            self.app_root = app_root.resolve()
        elif getattr(sys, "frozen", False):
            self.app_root = Path(sys.executable).resolve().parent
        else:
            self.app_root = Path(__file__).resolve().parent.parent
        self.store = SessionStore(self.app_root / "data")
        self.montage_path = self.app_root / "channel_assignments.json"
        self.startup_warnings: list[str] = []
        self.connection_settings_path = self.app_root / "device_connection.json"
        self.connection_preferences = ConnectionPreferences()
        if self.connection_settings_path.exists():
            try:
                self.connection_preferences = ConnectionPreferences.from_dict(
                    json.loads(self.connection_settings_path.read_text(encoding="utf-8")),
                )
            except (OSError, ValueError) as exc:
                self.startup_warnings.append(f"连接配置加载失败，已使用默认配置: {exc}")
        if transport is not None:
            self.connection_preferences.transport = transport_spec(transport).key
        if address is not None:
            self.connection_preferences.profiles[self.connection_preferences.transport].address = address.strip()
        self._displayed_transport: str | None = None
        try:
            self.montage = ChannelMontage.from_dict(json.loads(self.montage_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self.montage = ChannelMontage()
            self.startup_warnings.append(f"通道配置加载失败，已使用默认配置: {exc}")
        self.channel_labels = list(self.montage.eeg)
        self.device = device if device is not None else create_device_adapter(mock)
        # Last-resort data safety nets: if any shutdown path hangs or the
        # process is killed, the in-memory callback recording is still exported
        # synchronously (pure file I/O, no event loop required).
        application = QtWidgets.QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self._emergency_persist_recording)
        atexit.register(self._emergency_persist_recording)
        self.recording = RecordingSession(self.device, self.store, mock=mock, log=self._append_log, quality=self._session_quality_payload)
        self.experiment = ExperimentController(self.store, recording_support=self.recording.wait_for_support)
        self.model_runner = ModelJobRunner(self.store)
        self.quality_runner = ModelJobRunner(self.store)
        self.quality_reports = QualityReportController(
            self.quality_runner, self._quality_request,
            can_start=lambda: not self._acquisition_active() and not self.model_runner.running and not self._shutdown_in_progress,
            parent=self,
        )
        self._last_quality_session: Path | None = None
        self._quality_dialog: QualityReportDialog | None = None
        self.stimulus_window: StimulusWindow | None = None
        self.current_page = "overview"
        self._wave_sample_rate_hz = float(DEFAULT_SAMPLE_RATE_HZ)
        self.waveform_history_samples = int(self._wave_sample_rate_hz * WAVEFORM_DISPLAY_SECONDS)
        self.waveforms = [deque(maxlen=self.waveform_history_samples) for _ in self.channel_labels]
        self._wave_dirty = True
        self.eeg_filter = RealtimeEEGFilter(
            sample_rate_hz=float(DEFAULT_SAMPLE_RATE_HZ),
            n_channels=len(self.channel_labels),
            channel_types=self.montage.channel_types,
        )
        self.filter_config_dirty = False
        self.wave_plots: list[Any] = []
        self.wave_curves: list[Any] = []
        self.wave_track_labels: list[Any] = []
        self.latest_impedance = [float("nan")] * 8
        self.last_trial = "--"
        self._close_task: asyncio.Task | None = None
        self._close_ready = False
        self._shutdown_in_progress = False
        self._shutdown_task: asyncio.Task | None = None
        self._connection_task_handle: asyncio.Task | None = None
        self._connection_cancelable = False
        self._device_config_task_handle: asyncio.Task | None = None
        self._stream_task_handle: asyncio.Task | None = None
        self._experiment_prepare_task: asyncio.Task | None = None
        self._experiment_prepare_active = False
        self._impedance_task_handle: asyncio.Task | None = None
        self._impedance_close_task: asyncio.Task | None = None
        self._impedance_operation_pending = False
        self._impedance_cancel_requested = False
        self._experiment_finish_task: asyncio.Task | None = None
        self._thought_digit_task: asyncio.Task | None = None
        self.impedance_completed_monotonic: float | None = None
        self.impedance_completed_utc: str | None = None
        self.impedance_passed = False
        self._device_monitor_task: asyncio.Task | None = None
        self._last_device_timeout_log = 0.0
        self._eeg_data_delay_active = False
        self._last_stream_recovery_monotonic = 0.0
        self._recording_max_eeg_gap_seconds = 0.0
        self._recording_eeg_gap_events = 0
        self.model_session_dir: Path | None = None
        self._model_parameter_options: tuple[ModelParameterOption, ...] = ()
        self.preview_window: StimulusWindow | None = None
        self.preview_timer: QtCore.QTimer | None = None
        self._thought_digit_dialog: QtWidgets.QInputDialog | None = None
        self._decoding_test_requested = False
        self._calibration_target_requested: int | None = None

        self._build_shell()
        self._build_pages()
        self._connect_signals()
        self._apply_ui_density()
        self._set_page("device")
        self._refresh_model_session()
        self._sync_action_state()
        for entry in reversed(self.decoding_panel.entries):
            if read_report_index(entry.path):
                self._last_quality_session = entry.path
                self._on_quality_report_changed(str(entry.path))
                break
        for warning in self.startup_warnings:
            self._append_log(warning)
        for incomplete in SessionStore.incomplete_sessions(self.store.root):
            self._append_log(f"发现未正常结束会话: {incomplete}")
        if not self.mock:
            QtCore.QTimer.singleShot(0, self._start_startup_connection)

    def _channel_display_name(self, index: int) -> str:
        label = self.channel_labels[index]
        return f"EOG ({label})" if self.montage.channel_types[index].casefold() == "eog" and label != "EOG" else label

    @staticmethod
    def _channel_slot_name(index: int) -> str:
        """Show the UI slot and the SDK's zero-based hardware index together."""
        return f"CH{index + 1} / ADC{index}"

    def _build_shell(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(206)
        side = QtWidgets.QVBoxLayout(sidebar)
        side.setContentsMargins(16, 28, 16, 20)
        side.setSpacing(6)
        brand = QtWidgets.QLabel("BrainSync")
        brand.setObjectName("brand")
        side.addWidget(brand)
        eyebrow = QtWidgets.QLabel("GTN / P300 WORKSTATION")
        eyebrow.setObjectName("eyebrow")
        side.addWidget(eyebrow)
        side.addSpacing(26)
        nav_icons = {"overview": "layout", "device": "channel", "experiment": "experimenting",
                     "model": "fft", "analysis": "data_viewer"}
        self.nav_buttons: dict[str, QtWidgets.QPushButton] = {}
        for key, text in (
            ("overview", "总览"),
            ("device", "设备与质量"),
            ("experiment", "GTN 实验"),
            ("model", "校准与推理"),
            ("analysis", "数据分析"),
        ):
            button = QtWidgets.QPushButton(text)
            button.setObjectName("nav")
            button.setIcon(asset_icon(nav_icons[key], "#a9c3c9"))
            button.setIconSize(QtCore.QSize(19, 19))
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, name=key: self._set_page(name))
            self.nav_buttons[key] = button
            side.addWidget(button)
        side.addStretch()
        self.side_mode = QtWidgets.QLabel("模拟模式" if self.mock else "设备模式")
        self.side_mode.setObjectName("mode")
        side.addWidget(self.side_mode)
        outer.addWidget(sidebar)

        content = QtWidgets.QWidget()
        content.setObjectName("workspace")
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(28, 24, 28, 20)
        content_layout.setSpacing(18)

        topbar = QtWidgets.QFrame()
        topbar.setObjectName("topbar")
        top_layout = QtWidgets.QHBoxLayout(topbar)
        top_layout.setContentsMargins(0, 0, 0, 14)
        self.page_title = QtWidgets.QLabel("总览")
        self.page_title.setObjectName("page_title")
        top_layout.addWidget(self.page_title)
        top_layout.addStretch()
        self.display_settings = QtWidgets.QDialog(self)
        self.display_settings.setWindowTitle("显示设置")
        self.display_settings.setMinimumWidth(330)
        display_form = QtWidgets.QFormLayout(self.display_settings)
        display_form.setContentsMargins(20, 20, 20, 20)
        self.ui_font_size_combo = QtWidgets.QComboBox()
        for label, size in (("小", 11), ("标准", 13), ("大", 15)):
            self.ui_font_size_combo.addItem(label, size)
        self.ui_font_size_combo.setCurrentText("标准")
        self.ui_font_size_combo.currentIndexChanged.connect(self._apply_ui_density)
        display_form.addRow("字号", self.ui_font_size_combo)
        self.ui_spacing_combo = QtWidgets.QComboBox()
        for label, spacing in (("紧凑", 8), ("标准", 14), ("舒展", 20)):
            self.ui_spacing_combo.addItem(label, spacing)
        self.ui_spacing_combo.setCurrentText("紧凑")
        self.ui_spacing_combo.currentIndexChanged.connect(self._apply_ui_density)
        display_form.addRow("间距", self.ui_spacing_combo)
        display_close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        display_close.button(QtWidgets.QDialogButtonBox.StandardButton.Close).setText("关闭")
        display_close.rejected.connect(self.display_settings.close)
        display_form.addRow(display_close)
        self.session_label = QtWidgets.QLabel("未开始会话")
        self.session_label.setObjectName("muted")
        self.session_label.setMaximumWidth(340)
        self.session_label.setWordWrap(True)
        top_layout.addWidget(self.session_label)
        self.quality_report_button = QtWidgets.QPushButton("查看质量报告")
        self.quality_report_button.setIcon(asset_icon("data_viewer"))
        self.quality_report_button.clicked.connect(lambda: self._show_quality_report(self._last_quality_session))
        self.quality_report_button.hide()
        top_layout.addWidget(self.quality_report_button)
        top_layout.addSpacing(18)
        self.status_label = QtWidgets.QLabel("未连接")
        self.status_label.setObjectName("status")
        top_layout.addWidget(self.status_label)
        top_layout.addWidget(tool_button("settings", "显示设置", self.display_settings.show))
        content_layout.addWidget(topbar)

        self.pages = QtWidgets.QStackedWidget()
        content_layout.addWidget(self.pages, 1)
        outer.addWidget(content, 1)

    def _build_pages(self) -> None:
        self.overview_page = self._build_overview_page()
        self.device_page = self._build_device_page()
        self.experiment_page = self._build_experiment_page()
        self.model_page = self._build_model_page()
        self.analysis_page = self._build_analysis_page()
        for name in ("overview_page", "device_page", "experiment_page", "analysis_page"):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(getattr(self, name))
            setattr(self, name, scroll)
        for page in (self.overview_page, self.device_page, self.experiment_page, self.model_page, self.analysis_page):
            self.pages.addWidget(page)

    def _build_overview_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        intro = QtWidgets.QLabel("工作站状态")
        intro.setObjectName("section_title")
        layout.addWidget(intro)

        cards = QtWidgets.QHBoxLayout()
        self.card_device = MetricCard("设备状态", "未连接", COLORS["amber"])
        self.card_stream = MetricCard("EEG 流", "未启动", COLORS["blue"])
        self.card_impedance = MetricCard("阻抗质量", "待检测", COLORS["amber"])
        self.card_session = MetricCard("当前会话", "--", COLORS["teal"])
        for card in (self.card_device, self.card_stream, self.card_impedance, self.card_session):
            cards.addWidget(card)
        layout.addLayout(cards)

        body = QtWidgets.QHBoxLayout()
        left = Panel()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(18, 18, 18, 18)
        title = QtWidgets.QLabel("工作流程")
        title.setObjectName("section_title")
        left_layout.addWidget(title)
        for name, label, icon in (("device", "设备与质量", "channel"),
                                  ("experiment", "GTN 实验", "experimenting"),
                                  ("model", "校准与推理", "fft"),
                                  ("analysis", "数据分析", "data_viewer")):
            action = QtWidgets.QPushButton(label)
            action.setObjectName("workflow_action")
            action.setIcon(asset_icon(icon, COLORS["teal"]))
            action.setIconSize(QtCore.QSize(22, 22))
            action.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            action.clicked.connect(lambda _checked=False, page_name=name: self._set_page(page_name))
            left_layout.addWidget(action)
        left_layout.addStretch()
        body.addWidget(left, 1)

        right = Panel()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(18, 18, 18, 18)
        title = QtWidgets.QLabel("运行日志")
        title.setObjectName("section_title")
        right_layout.addWidget(title)
        self.overview_log = QtWidgets.QPlainTextEdit()
        self.overview_log.setReadOnly(True)
        self.overview_log.setMaximumBlockCount(2000)
        self.overview_log.setPlaceholderText("暂无运行记录")
        right_layout.addWidget(self.overview_log)
        body.addWidget(right, 1)
        layout.addLayout(body, 1)
        return page

    def _build_device_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        controls = QtWidgets.QGroupBox("设备连接")
        grid = QtWidgets.QGridLayout(controls)
        grid.addWidget(QtWidgets.QLabel("连接方式"), 0, 0)
        self.transport_combo = QtWidgets.QComboBox()
        for spec in TRANSPORTS:
            self.transport_combo.addItem(spec.label, spec.key)
        self.transport_combo.setCurrentIndex(self.transport_combo.findData(self.connection_preferences.transport))
        grid.addWidget(self.transport_combo, 0, 1)
        self.port_label = QtWidgets.QLabel("串口（可选）")
        grid.addWidget(self.port_label, 0, 2)
        self.port_edit = QtWidgets.QLineEdit()
        self.port_edit.setPlaceholderText("留空：SDK 自动发现")
        grid.addWidget(self.port_edit, 0, 3)
        self.connect_button = QtWidgets.QPushButton("连接设备")
        self.connect_button.setObjectName("primary")
        self.connect_button.setIcon(asset_icon("connect_type", "#ffffff"))
        self.connect_button.clicked.connect(self._toggle_connection)
        grid.addWidget(self.connect_button, 0, 4)
        grid.addWidget(QtWidgets.QLabel("采样率"), 1, 0)
        self.sample_rate_label = QtWidgets.QLabel(f"{DEFAULT_SAMPLE_RATE_HZ} Hz（固定）")
        grid.addWidget(self.sample_rate_label, 1, 1)
        grid.addWidget(QtWidgets.QLabel("增益"), 1, 2)
        self.gain_combo = QtWidgets.QComboBox()
        self.gain_combo.addItems([str(gain) for gain in GAINS])
        self._on_transport_changed()
        self.transport_combo.currentIndexChanged.connect(self._on_transport_changed)
        self.port_edit.editingFinished.connect(self._save_device_settings)
        self.gain_combo.currentTextChanged.connect(self._save_device_settings)
        self.gain_combo.setMaximumWidth(160)
        grid.addWidget(self.gain_combo, 1, 3)
        grid.setAlignment(self.gain_combo, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.apply_config_button = QtWidgets.QPushButton("应用配置")
        self.apply_config_button.setIcon(asset_icon("settings"))
        self.apply_config_button.clicked.connect(self._apply_device_config)
        grid.addWidget(self.apply_config_button, 1, 4)
        self.stream_button = QtWidgets.QPushButton("启动 EEG 流")
        self.stream_button.setObjectName("primary")
        self.stream_button.setIcon(asset_icon("preview", "#ffffff"))
        self.stream_button.clicked.connect(self._toggle_stream)
        grid.addWidget(self.stream_button, 2, 0, 1, 2)
        self.impedance_open_button = QtWidgets.QPushButton("阻抗检测")
        self.impedance_open_button.setIcon(asset_icon("imp_channel"))
        self.impedance_open_button.clicked.connect(self._show_impedance_dialog)
        grid.addWidget(self.impedance_open_button, 2, 2)
        self.loss_label = QtWidgets.QLabel("丢包率 0.00%")
        self.loss_label.setObjectName("muted")
        grid.addWidget(self.loss_label, 2, 3)
        self.firmware_label = QtWidgets.QLabel("固件 --")
        self.firmware_label.setObjectName("muted")
        grid.addWidget(self.firmware_label, 2, 4)
        layout.addWidget(controls)

        filter_box = QtWidgets.QGroupBox("实时 EEG 滤波 · 仅预览波形")
        filter_layout = QtWidgets.QGridLayout(filter_box)
        filter_layout.setContentsMargins(12, 12, 12, 10)
        self.bandpass_check = QtWidgets.QCheckBox("带通")
        self.bandpass_check.setChecked(True)
        self.bandpass_low_spin = QtWidgets.QDoubleSpinBox()
        self.bandpass_low_spin.setRange(0.1, 1000.0)
        self.bandpass_low_spin.setDecimals(1)
        self.bandpass_low_spin.setSingleStep(0.5)
        self.bandpass_low_spin.setValue(1.0)
        self.bandpass_low_spin.setSuffix(" Hz")
        self.bandpass_high_spin = QtWidgets.QDoubleSpinBox()
        self.bandpass_high_spin.setRange(0.1, 1000.0)
        self.bandpass_high_spin.setDecimals(1)
        self.bandpass_high_spin.setSingleStep(1.0)
        self.bandpass_high_spin.setValue(40.0)
        self.bandpass_high_spin.setSuffix(" Hz")
        self.notch_check = QtWidgets.QCheckBox("工频陷波")
        self.notch_check.setChecked(True)
        self.notch_combo = QtWidgets.QComboBox()
        self.notch_combo.addItem("50 Hz", 50.0)
        self.notch_combo.addItem("60 Hz", 60.0)
        self.filter_apply_button = QtWidgets.QPushButton("应用")
        self.filter_apply_button.setObjectName("primary")
        self.filter_apply_button.clicked.connect(self._apply_filter_settings)
        self.filter_state = QtWidgets.QLabel()
        self.filter_state.setObjectName("muted")
        filter_layout.addWidget(self.bandpass_check, 0, 0)
        filter_layout.addWidget(QtWidgets.QLabel("下限"), 0, 1)
        filter_layout.addWidget(self.bandpass_low_spin, 0, 2)
        filter_layout.addWidget(QtWidgets.QLabel("上限"), 0, 3)
        filter_layout.addWidget(self.bandpass_high_spin, 0, 4)
        filter_layout.addWidget(self.notch_check, 0, 5)
        filter_layout.addWidget(self.notch_combo, 0, 6)
        filter_layout.addWidget(self.filter_apply_button, 0, 7)
        filter_layout.addWidget(self.filter_state, 1, 0, 1, 8)
        filter_layout.setColumnStretch(8, 1)
        layout.addWidget(filter_box)
        self.bandpass_check.toggled.connect(self._mark_filter_dirty)
        self.bandpass_low_spin.valueChanged.connect(self._mark_filter_dirty)
        self.bandpass_high_spin.valueChanged.connect(self._mark_filter_dirty)
        self.notch_check.toggled.connect(self._mark_filter_dirty)
        self.notch_combo.currentIndexChanged.connect(self._mark_filter_dirty)
        self._update_filter_state()

        upper = QtWidgets.QHBoxLayout()
        wave_panel = QtWidgets.QGroupBox("实时 EEG · μV")
        wave_layout = QtWidgets.QVBoxLayout(wave_panel)
        if pg is not None:
            self.plot = None
            scale_row = QtWidgets.QHBoxLayout()
            scale_row.addWidget(QtWidgets.QLabel("统一幅度"))
            self.wave_scale_combo = QtWidgets.QComboBox()
            for scale_uv in (25, 50, 100, 150, 300, 600):
                self.wave_scale_combo.addItem(f"±{scale_uv} μV", float(scale_uv))
            self.wave_scale_combo.setCurrentText("±100 μV")
            self.wave_scale_combo.currentIndexChanged.connect(self._apply_wave_scale)
            scale_row.addWidget(self.wave_scale_combo)
            scale_row.addStretch()
            wave_layout.addLayout(scale_row)
            self.wave_graphics = pg.GraphicsLayoutWidget()
            self.wave_graphics.setBackground(PLOT_COLORS["bg"])
            self.wave_graphics.setMinimumHeight(240)
            for index, label in enumerate(self.channel_labels):
                track_label = self.wave_graphics.addLabel(
                    f"{self._channel_slot_name(index)} · {self._channel_display_name(index)}",
                    row=index, col=0, justify="right", color=PLOT_COLORS["text"]
                )
                plot = self.wave_graphics.addPlot(row=index, col=1)
                plot.showGrid(x=False, y=True, alpha=0.15)
                plot.setMouseEnabled(x=False, y=False)
                plot.hideButtons()
                plot.hideAxis("left")
                plot.hideAxis("bottom")
                plot.enableAutoRange(axis="y", enable=False)
                # The selected range is a reference scale, not a hard display
                # limit. Let excursions draw across adjacent channel tracks.
                plot.getViewBox().setFlag(
                    QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape,
                    False,
                )
                curve = plot.plot(
                    pen=pg.mkPen(
                        PLOT_COLORS["amber"] if self.montage.channel_types[index].casefold() == "eog" else PLOT_COLORS["teal"],
                        width=1.1,
                    ),
                    antialias=True,
                )
                curve.setClipToView(False)
                curve.setDownsampling(auto=False)
                self.wave_track_labels.append(track_label)
                self.wave_plots.append(plot)
                self.wave_curves.append(curve)
            wave_layout.addWidget(self.wave_graphics)
            self._apply_wave_scale()
            self._apply_wave_x_range()
        else:
            self.plot = None
            self.wave_graphics = None
            self.wave_scale_combo = None
            fallback = QtWidgets.QLabel("pyqtgraph 未安装，实时波形不可用")
            fallback.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            fallback.setObjectName("muted")
            wave_layout.addWidget(fallback)
        upper.addWidget(wave_panel, 2)

        channels = QtWidgets.QGroupBox("通道映射")
        channels.setMaximumWidth(340)
        channel_layout = QtWidgets.QVBoxLayout(channels)
        self.channel_table = QtWidgets.QTableWidget(8, 4)
        self.channel_table.setHorizontalHeaderLabels(["通道", "物理位置", "类型", "状态"])
        self.channel_table.verticalHeader().setVisible(False)
        self.channel_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.channel_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.channel_table.setAlternatingRowColors(True)
        self.channel_table.setShowGrid(False)
        self.channel_table.setColumnWidth(0, 104)
        self.channel_table.setColumnWidth(1, 64)
        self.channel_table.setColumnWidth(2, 48)
        self.channel_table.setColumnWidth(3, 62)
        self.channel_table.verticalHeader().setDefaultSectionSize(27)
        for index, name in enumerate(self.channel_labels):
            self.channel_table.setItem(index, 0, QtWidgets.QTableWidgetItem(self._channel_slot_name(index)))
            self.channel_table.setItem(index, 1, QtWidgets.QTableWidgetItem(name))
            self.channel_table.setItem(
                index,
                2,
                QtWidgets.QTableWidgetItem(self.montage.channel_types[index].upper()),
            )
            self.channel_table.setItem(
                index,
                3,
                QtWidgets.QTableWidgetItem("待连接" if self.montage.active_mask & (1 << index) else "未启用"),
            )
        channel_layout.addWidget(self.channel_table)
        montage_actions = QtWidgets.QHBoxLayout()
        self.montage_path_label = QtWidgets.QLabel(str(self.montage_path))
        self.montage_path_label.setObjectName("muted")
        self.montage_path_label.hide()
        montage_actions.addWidget(self.montage_path_label, 1)
        self.montage_edit_button = QtWidgets.QPushButton("编辑通道映射")
        self.montage_edit_button.setIcon(asset_icon("edit"))
        self.montage_edit_button.setToolTip(str(self.montage_path))
        self.montage_edit_button.clicked.connect(self._edit_montage)
        montage_actions.addWidget(self.montage_edit_button)
        channel_layout.addLayout(montage_actions)
        upper.addWidget(channels, 1)
        layout.addLayout(upper, 1)
        self._build_impedance_dialog()
        return page

    def _build_impedance_dialog(self) -> None:
        self.impedance_dialog = QtWidgets.QDialog(self)
        self.impedance_dialog.setWindowTitle("阻抗检测 · 模拟数据" if self.mock else "阻抗检测 · 设备测量")
        self.impedance_dialog.setMinimumSize(620, 380)
        self.impedance_dialog.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        layout = QtWidgets.QVBoxLayout(self.impedance_dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        if self.mock:
            source_notice = QtWidgets.QLabel("模拟数据：不读取设备，不反映电极接触阻抗。")
            source_notice.setWordWrap(True)
            source_notice.setStyleSheet(f"color: {COLORS['amber']}; font-weight: 700;")
            layout.addWidget(source_notice)
        header = QtWidgets.QHBoxLayout()
        self.impedance_state = QtWidgets.QLabel("尚未检测")
        self.impedance_state.setWordWrap(True)
        self.impedance_state.setObjectName("status")
        header.addWidget(self.impedance_state)
        header.addStretch()
        self.impedance_start = QtWidgets.QPushButton("开始检测")
        self.impedance_start.setObjectName("primary")
        self.impedance_start.clicked.connect(self._start_impedance)
        header.addWidget(self.impedance_start)
        close_button = QtWidgets.QPushButton("关闭")
        close_button.clicked.connect(self.impedance_dialog.close)
        header.addWidget(close_button)
        self.impedance_dialog.finished.connect(self._on_impedance_dialog_closed)
        layout.addLayout(header)

        self.impedance_error = QtWidgets.QLabel()
        self.impedance_error.setWordWrap(True)
        self.impedance_error.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.impedance_error.setStyleSheet(f"color: {COLORS['amber']};")
        self.impedance_error.hide()
        layout.addWidget(self.impedance_error)
        cards_layout = QtWidgets.QGridLayout()
        cards_layout.setSpacing(10)
        self.impedance_cards: list[MetricCard] = []
        for index, channel in enumerate(self.channel_labels):
            card = MetricCard(
                f"{self._channel_slot_name(index)} · {self._channel_display_name(index)}",
                "--",
                COLORS["amber"] if self.montage.channel_types[index].casefold() == "eog" else COLORS["muted"],
            )
            card.setMinimumWidth(140)
            card.setMinimumHeight(76)
            self.impedance_cards.append(card)
            cards_layout.addWidget(card, index // 4, index % 4)
        layout.addLayout(cards_layout)
        layout.addStretch()
        self._reset_impedance_display()

    def _show_impedance_dialog(self) -> None:
        self.impedance_dialog.show()
        self.impedance_dialog.raise_()
        self.impedance_dialog.activateWindow()
        if not self._impedance_active():
            self._start_impedance()

    def _on_impedance_dialog_closed(self, _result: int) -> None:
        task = self._impedance_task_handle
        if task is None or task.done():
            return
        if self._impedance_close_task is None or self._impedance_close_task.done():
            self._impedance_close_task = asyncio.create_task(self._close_impedance_task())

    async def _close_impedance_task(self) -> None:
        try:
            await self._cancel_impedance_detection()
        finally:
            self._impedance_close_task = None

    def _build_experiment_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        form_box = QtWidgets.QWidget()
        form_layout = QtWidgets.QVBoxLayout(form_box)
        form_layout.setContentsMargins(0, 0, 0, 0)
        self.experiment_tabs = QtWidgets.QTabWidget()
        form_layout.addWidget(self.experiment_tabs, 1)
        defaults = ExperimentConfig()
        self.subject_edit = QtWidgets.QLineEdit(defaults.subject_id)
        self.age_edit = QtWidgets.QLineEdit()
        self.age_edit.setPlaceholderText("可选")
        self.birthdate_edit = QtWidgets.QLineEdit()
        self.birthdate_edit.setPlaceholderText("可选，YYYY-MM-DD")
        self.sex_combo = QtWidgets.QComboBox()
        self.sex_combo.addItems(["unknown", "M", "F"])
        self.blocks_spin = QtWidgets.QSpinBox()
        self.blocks_spin.setRange(1, 20)
        self.blocks_spin.setValue(defaults.blocks)
        self.digits_per_block_spin = QtWidgets.QSpinBox()
        self.digits_per_block_spin.setRange(len(DIGIT_VOCAB), 297)
        self.digits_per_block_spin.setSingleStep(len(DIGIT_VOCAB))
        self.digits_per_block_spin.setValue(defaults.digits_per_block)
        self.stimulus_spin = QtWidgets.QSpinBox()
        self.stimulus_spin.setRange(50, 2000)
        self.stimulus_spin.setValue(defaults.stimulus_ms)
        self.soa_spin = QtWidgets.QSpinBox()
        self.soa_spin.setRange(50, 5000)
        self.soa_spin.setValue(defaults.soa_ms)
        self.rest_spin = QtWidgets.QSpinBox()
        self.rest_spin.setRange(0, 600)
        self.rest_spin.setValue(defaults.rest_sec)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 99999999)
        self.seed_spin.setValue(secrets.randbelow(100_000_000))
        self.auto_seed_check = QtWidgets.QCheckBox("每次自动生成随机种子")
        self.auto_seed_check.setChecked(True)
        self.screen_spin = QtWidgets.QSpinBox()
        screen_count = len(QtWidgets.QApplication.screens())
        self.screen_spin.setRange(0, max(0, screen_count - 1))
        self.screen_spin.setValue(1 if screen_count > 1 else 0)
        self.screen_info = QtWidgets.QLabel(f"当前检测到 {screen_count} 个显示器")
        self.screen_info.setObjectName("muted")
        self.save_dir_edit = QtWidgets.QLineEdit("data")
        self.model_profile_edit = QtWidgets.QLineEdit("未选择模型")
        self.model_profile_edit.setReadOnly(True)
        self.instruction_edit = QtWidgets.QLineEdit(
            "请注视屏幕中央的数字。看到目标数字时保持专注，不要移动身体。"
        )
        self.notes_edit = QtWidgets.QLineEdit()
        self.notes_edit.setPlaceholderText("可选，如：采集中有交谈 / 隔壁使用微波炉等环境情况")
        field_groups = (
            ("实验设置", (("受试者编号", self.subject_edit), ("被试屏幕", self.screen_spin),
                           ("Block 数", self.blocks_spin), ("每 Block 闪烁数", self.digits_per_block_spin),
                           ("数字呈现 (ms)", self.stimulus_spin), ("起始间隔 SOA (ms)", self.soa_spin),
                           ("Block 休息 (s)", self.rest_spin), ("随机种子", self.seed_spin))),
            ("受试者与记录", (("年龄", self.age_edit), ("出生日期", self.birthdate_edit),
                              ("性别", self.sex_combo), ("保存目录", self.save_dir_edit),
                              ("实验说明", self.instruction_edit), ("采集备注", self.notes_edit))),
        )
        for title, fields in field_groups:
            tab = QtWidgets.QWidget()
            form = QtWidgets.QGridLayout(tab)
            form.setContentsMargins(4, 18, 4, 8)
            form.setHorizontalSpacing(20)
            form.setVerticalSpacing(10)
            for index, (label, widget) in enumerate(fields):
                row, column = divmod(index, 2)
                caption = QtWidgets.QLabel(label)
                caption.setObjectName("muted")
                caption.setWordWrap(True)
                form.addWidget(caption, row * 2, column)
                form.addWidget(widget, row * 2 + 1, column)
                form.setColumnStretch(column, 1)
            if title == "实验设置":
                form.addWidget(self.auto_seed_check, 8, 0, 1, 2)
            form.setRowStretch(10, 1)
            self.experiment_tabs.addTab(tab, title)
        self.model_profile_edit.setParent(form_box)
        self.model_profile_edit.hide()
        self.screen_spin.setToolTip(self.screen_info.text())
        self.experiment_start = QtWidgets.QPushButton("开始实验")
        self.experiment_start.setObjectName("primary")
        self.experiment_start.setIcon(asset_icon("start_experiment", "#ffffff"))
        self.experiment_start.clicked.connect(self._start_experiment)
        self.experiment_stop = QtWidgets.QPushButton("停止实验")
        self.experiment_stop.setObjectName("danger")
        self.experiment_stop.setIcon(asset_icon("stop_experiment", COLORS["red"]))
        self.experiment_stop.setEnabled(False)
        self.experiment_stop.clicked.connect(self._stop_experiment)
        actions = QtWidgets.QHBoxLayout()
        self.practice_button = QtWidgets.QPushButton("练习预览")
        self.practice_button.setIcon(asset_icon("stim_preview"))
        self.practice_button.clicked.connect(self._start_practice)
        actions.addWidget(self.practice_button)
        actions.addWidget(self.experiment_start)
        actions.addWidget(self.experiment_stop)
        form_layout.addLayout(actions)
        layout.addWidget(form_box, 1)

        preview = QtWidgets.QGroupBox("协议预览")
        preview_layout = QtWidgets.QVBoxLayout(preview)
        self.protocol_label = QtWidgets.QLabel("1  2  3  4  5  6  7  8  9")
        self.protocol_label.setStyleSheet(f"font-size: 24px; color: {COLORS['teal']}; padding: 14px 0; letter-spacing: 0px;")
        self.protocol_label.setWordWrap(True)
        preview_layout.addWidget(self.protocol_label)
        preview_layout.addWidget(self.screen_info)
        self.timing_info = QtWidgets.QLabel(
            "时序标记：软件重绘；无硬件触发。时序质量控制与眼动伪迹删除由算法后处理。"
        )
        self.timing_info.setObjectName("muted")
        self.timing_info.setWordWrap(True)
        preview_layout.addWidget(self.timing_info)
        self.experiment_state = QtWidgets.QLabel("准备就绪")
        self.experiment_state.setObjectName("status")
        preview_layout.addWidget(self.experiment_state)
        self.trial_label = QtWidgets.QLabel("当前 trial --")
        self.trial_label.setObjectName("muted")
        preview_layout.addWidget(self.trial_label)
        self.experiment_progress = QtWidgets.QProgressBar()
        self.experiment_progress.setValue(0)
        preview_layout.addWidget(self.experiment_progress)
        preview_layout.addStretch()
        self.protocol_log = QtWidgets.QPlainTextEdit()
        self.protocol_log.setReadOnly(True)
        self.protocol_log.setMaximumBlockCount(2000)
        preview_layout.addWidget(self.protocol_log, 1)
        layout.addWidget(preview, 1)
        return page

    def _build_model_page(self) -> QtWidgets.QWidget:
        advanced = self._build_advanced_model_page()
        self.decoding_panel = DecodingPanel(self.app_root, self.model_runner, self, mock=self.mock)
        self.decoding_panel.record_requested.connect(self._start_decoding_test)
        self.decoding_panel.calibration_record_requested.connect(self._start_calibration_record)
        self.decoding_panel.confirmation_requested.connect(self._request_target_confirmation)
        self.decoding_panel.message.connect(self._append_log)
        self.decoding_panel.busy_changed.connect(
            lambda: QtCore.QTimer.singleShot(0, self._sync_action_state)
        )
        page = QtWidgets.QTabWidget()
        page.tabBar().hide()
        self.decoding_scroll = QtWidgets.QScrollArea()
        self.decoding_scroll.setWidgetResizable(True)
        self.decoding_scroll.setWidget(self.decoding_panel)
        page.addTab(self.decoding_scroll, "个体模型")
        page.addTab(advanced, "高级任务")
        self.decoding_panel.advanced_requested.connect(lambda: page.setCurrentIndex(1))
        return page

    def _build_advanced_model_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        back = QtWidgets.QPushButton("返回校准与推理")
        back.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowBack))
        back.clicked.connect(lambda: self.model_page.setCurrentIndex(0))
        layout.addWidget(back, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        config_box = QtWidgets.QGroupBox("算法任务接口")
        form = QtWidgets.QGridLayout(config_box)
        self.task_combo = QtWidgets.QComboBox()
        for option in MODEL_TASK_OPTIONS:
            self.task_combo.addItem(option.label, option.key)
            self.task_combo.setItemData(
                self.task_combo.count() - 1,
                option.description,
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        self.task_combo.currentIndexChanged.connect(self._refresh_parameter_menu)
        self.manifest_edit = QtWidgets.QLineEdit()
        self.python_edit = QtWidgets.QLineEdit()
        self.python_edit.setPlaceholderText("留空：当前 Python")
        self.entrypoint_edit = QtWidgets.QLineEdit()
        self.entrypoint_edit.setPlaceholderText("脚本路径或 Python module，例如 run_model.py")
        self.parameter_combo = QtWidgets.QComboBox()
        self.parameter_combo.setToolTip(
            "参数方案由模型 manifest 提供；未提供时使用模型入口自己的默认参数。"
        )
        self.output_dir_edit = QtWidgets.QLineEdit("models/runs")
        self.model_session_edit = QtWidgets.QLineEdit()
        self.model_session_edit.setReadOnly(True)
        self.model_session_edit.setPlaceholderText("自动寻找最近的可分析会话")
        self.model_session_edit.setToolTip(
            "算法自动使用 data 目录中最近一个已完成、已确认默想数字且通过质量检查的会话。"
        )
        self.dry_run_check = QtWidgets.QCheckBox("模拟任务")
        self.dry_run_check.setChecked(self.mock)
        form.addWidget(QtWidgets.QLabel("模型操作"), 0, 0)
        form.addWidget(self.task_combo, 0, 1)
        form.addWidget(QtWidgets.QLabel("模型 manifest"), 1, 0)
        form.addWidget(self.manifest_edit, 1, 1)
        browse_manifest = QtWidgets.QPushButton("选择")
        browse_manifest.clicked.connect(self._choose_manifest)
        form.addWidget(browse_manifest, 1, 2)
        form.addWidget(QtWidgets.QLabel("Python"), 2, 0)
        form.addWidget(self.python_edit, 2, 1)
        browse_python = QtWidgets.QPushButton("选择")
        browse_python.clicked.connect(self._choose_python)
        form.addWidget(browse_python, 2, 2)
        form.addWidget(QtWidgets.QLabel("Entrypoint"), 3, 0)
        form.addWidget(self.entrypoint_edit, 3, 1, 1, 2)
        form.addWidget(QtWidgets.QLabel("参数方案"), 4, 0)
        form.addWidget(self.parameter_combo, 4, 1, 1, 2)
        form.addWidget(QtWidgets.QLabel("输出目录"), 5, 0)
        form.addWidget(self.output_dir_edit, 5, 1, 1, 2)
        form.addWidget(QtWidgets.QLabel("数据会话（自动）"), 6, 0)
        form.addWidget(self.model_session_edit, 6, 1, 1, 2)
        form.addWidget(self.dry_run_check, 7, 1)
        self.model_start = QtWidgets.QPushButton("开始任务")
        self.model_start.setObjectName("primary")
        self.model_start.clicked.connect(self._start_model_task)
        self.model_stop = QtWidgets.QPushButton("停止任务")
        self.model_stop.setObjectName("danger")
        self.model_stop.setEnabled(False)
        self.model_stop.clicked.connect(self.model_runner.stop)
        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(self.model_start)
        actions.addWidget(self.model_stop)
        form.addLayout(actions, 8, 0, 1, 3)
        layout.addWidget(config_box)
        self._refresh_parameter_menu()

        status_row = QtWidgets.QHBoxLayout()
        self.model_state = QtWidgets.QLabel("待配置")
        self.model_state.setObjectName("status")
        self.model_progress = QtWidgets.QProgressBar()
        self.model_progress.setValue(0)
        status_row.addWidget(self.model_state)
        status_row.addWidget(self.model_progress, 1)
        layout.addLayout(status_row)
        self.model_log = QtWidgets.QPlainTextEdit()
        self.model_log.setReadOnly(True)
        self.model_log.setMaximumBlockCount(2000)
        self.model_log.setPlaceholderText("算法进程输出")
        layout.addWidget(self.model_log, 1)
        return page

    def _build_analysis_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        controls = QtWidgets.QHBoxLayout()
        self.analysis_path = QtWidgets.QLineEdit()
        self.analysis_path.setPlaceholderText("选择包含 session.json 的会话目录")
        choose = QtWidgets.QPushButton("选择目录")
        choose.clicked.connect(self._choose_analysis_dir)
        load = QtWidgets.QPushButton("加载")
        load.setObjectName("primary")
        load.clicked.connect(self._load_analysis)
        controls.addWidget(self.analysis_path, 1)
        controls.addWidget(choose)
        controls.addWidget(load)
        self.analysis_open_button = QtWidgets.QPushButton("打开目录")
        self.analysis_open_button.clicked.connect(self._open_analysis_dir)
        controls.addWidget(self.analysis_open_button)
        self.analysis_quality_button = QtWidgets.QPushButton("质量报告")
        self.analysis_quality_button.setIcon(asset_icon("data_viewer"))
        self.analysis_quality_button.setToolTip("查看或生成所选会话的数据质量报告")
        self.analysis_quality_button.clicked.connect(lambda: self._show_quality_report(Path(self.analysis_path.text())))
        controls.addWidget(self.analysis_quality_button)
        self.analysis_recover_button = QtWidgets.QPushButton("恢复未完成操作")
        self.analysis_recover_button.setEnabled(False)
        self.analysis_recover_button.clicked.connect(self._recover_analysis)
        controls.addWidget(self.analysis_recover_button)
        layout.addLayout(controls)
        self.analysis_summary = QtWidgets.QLabel("尚未加载会话")
        self.analysis_summary.setObjectName("muted")
        self.analysis_summary.setWordWrap(True)
        layout.addWidget(self.analysis_summary)
        self.analysis_table = QtWidgets.QTableWidget(0, 3)
        self.analysis_table.setHorizontalHeaderLabels(["时间", "事件", "内容"])
        self.analysis_table.horizontalHeader().setStretchLastSection(True)
        self.analysis_table.verticalHeader().setVisible(False)
        self.analysis_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.analysis_table.setAlternatingRowColors(True)
        self.analysis_table.setShowGrid(False)
        self.analysis_table.setColumnWidth(0, 184)
        self.analysis_table.setColumnWidth(1, 170)
        self.analysis_table.verticalHeader().setDefaultSectionSize(34)
        self.analysis_table.setMaximumHeight(999999)
        layout.addWidget(self.analysis_table, 1)
        return page

    def _connect_signals(self) -> None:
        self.device.signals.eeg_batch.connect(self._on_eeg_batch)
        self.device.signals.impedance.connect(self._on_impedance)
        self.device.signals.state_changed.connect(self._on_device_state)
        self.device.signals.error.connect(self._on_device_error)
        self.device.signals.acquisition_fault.connect(self._on_acquisition_fault)
        self.experiment.present_digit.connect(self._present_digit)
        self.experiment.present_text.connect(self._present_text)
        self.experiment.stimulus_event.connect(self._on_stimulus_event)
        self.experiment.trial_changed.connect(self._on_trial)
        self.experiment.progress_changed.connect(self._on_experiment_progress)
        self.experiment.state_changed.connect(self._on_experiment_state)
        self.experiment.paused.connect(self._on_experiment_paused)
        self.experiment.resuming.connect(self._on_experiment_resuming)
        self.experiment.message.connect(self._append_log)
        self.experiment.finished.connect(self._on_experiment_finished)
        self.experiment.practice_finished.connect(self._on_practice_finished)
        self.model_runner.progress_changed.connect(self.model_progress.setValue)
        self.model_runner.output_line.connect(self.model_log.appendPlainText)
        self.model_runner.state_changed.connect(self._on_model_state)
        self.model_runner.finished.connect(self._on_model_finished)
        self.quality_reports.changed.connect(self._on_quality_report_changed)
        self.quality_reports.message.connect(self._append_log)
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self._refresh_plot)
        self.plot_timer.start(PLOT_REFRESH_INTERVAL_MS)
        self.device_monitor_timer = QtCore.QTimer(self)
        self.device_monitor_timer.timeout.connect(self._monitor_device)
        self.device_monitor_timer.start(1000)

    def _set_page(self, name: str) -> None:
        pages = {"overview": self.overview_page, "device": self.device_page, "experiment": self.experiment_page, "model": self.model_page, "analysis": self.analysis_page}
        self.current_page = name
        self.pages.setCurrentWidget(pages[name])
        titles = {"overview": "总览", "device": "设备与质量", "experiment": "GTN 实验", "model": "校准与推理", "analysis": "数据分析"}
        self.page_title.setText(titles[name])
        for key, button in self.nav_buttons.items():
            button.setChecked(key == name)

    def _apply_ui_density(self) -> None:
        font_size = int(self.ui_font_size_combo.currentData())
        spacing = int(self.ui_spacing_combo.currentData())
        self.setStyleSheet(application_stylesheet(font_size))
        for layout in self.findChildren(QtWidgets.QLayout):
            if layout.property("baseSpacing") is None:
                layout.setProperty("baseSpacing", max(0, layout.spacing()))
            base = int(layout.property("baseSpacing"))
            if base:
                layout.setSpacing(round(base * spacing / 8))

    def _acquisition_active(self) -> bool:
        return bool(
            self.device.state is DeviceState.RECORDING
            or self.experiment.running
            or self._experiment_prepare_active
            or self.recording.busy
            or (self._experiment_finish_task is not None and not self._experiment_finish_task.done())
        )

    def _impedance_active(self) -> bool:
        return bool(
            self._impedance_operation_pending
            or self.device.state is DeviceState.IMPEDANCE
            or self.device.recovery_may_have_impedance
        )

    def _device_transition_active(self) -> bool:
        return any(task is not None and not task.done() for task in (
            self._connection_task_handle, self._device_config_task_handle, self._stream_task_handle))

    def _sync_action_state(self) -> None:
        device_transition = self._device_transition_active() or self.device.connection_pending
        connection_editable = (
            self.device.state is DeviceState.DISCONNECTED and not device_transition and not self._shutdown_in_progress
        )
        self.transport_combo.setEnabled(connection_editable)
        self.port_edit.setEnabled(connection_editable)
        config_editable = (self.device.state in {DeviceState.DISCONNECTED, DeviceState.READY}
                           and not device_transition and not self._acquisition_active()
                           and not self._impedance_active() and not self._shutdown_in_progress)
        self.gain_combo.setEnabled(config_editable)
        if self._shutdown_in_progress:
            for control in (self.connect_button, self.apply_config_button, self.stream_button,
                            self.impedance_start, self.impedance_open_button, self.montage_edit_button,
                            self.experiment_start, self.practice_button, self.experiment_stop,
                            self.model_start, self.model_stop):
                control.setEnabled(False)
            self.decoding_panel.set_runtime_state(allowed=False, device_ready=False)
            return
        state = self.device.state
        connected = state is not DeviceState.DISCONNECTED
        ready = state is DeviceState.READY
        previewing = state is DeviceState.PREVIEWING
        recovering_preview = (
            state is DeviceState.RECOVERY_REQUIRED
            and self.device.recovery_may_have_preview
        )
        busy = self._acquisition_active()
        impedance_busy = self._impedance_active()
        connection_task = self._connection_task_handle is not None and not self._connection_task_handle.done()
        config_task = self._device_config_task_handle is not None and not self._device_config_task_handle.done()
        recovering_impedance = self.device.recovery_may_have_impedance
        impedance_running = state is DeviceState.IMPEDANCE
        impedance_pending = self._impedance_operation_pending
        experiment_running = self.experiment.running or self._experiment_prepare_active
        model_running = self.model_runner.running
        confirmation_running = (
            self._thought_digit_task is not None and not self._thought_digit_task.done()
        )
        # Disconnect remains available during impedance cleanup so an
        # experimenter always has a recovery path after an SDK error.
        self.connect_button.setEnabled(
            (connection_task and self._connection_cancelable)
            or (not device_transition and not (connected and busy))
        )
        self.apply_config_button.setEnabled(connected and ready and not device_transition and not busy and not impedance_busy)
        self.stream_button.setEnabled(
            connected
            and not busy
            and not device_transition
            and (previewing or recovering_preview or (ready and not impedance_busy))
        )
        self.impedance_start.setText(
            "停止检测" if impedance_running else ("恢复阻抗资源" if recovering_impedance else "开始检测")
        )
        self.impedance_start.setEnabled(
            connected
            and not busy
            and not device_transition
            and (
                impedance_running
                or (recovering_impedance and not impedance_pending)
                or (ready and not impedance_busy and not config_task and not impedance_pending)
            )
        )
        self.impedance_open_button.setEnabled(
            connected
            and not busy
            and not device_transition
            and (
                impedance_running
                or (recovering_impedance and not impedance_pending)
                or (ready and not impedance_busy and not config_task and not impedance_pending)
            )
        )
        self.montage_edit_button.setEnabled(
            ready and not busy and not impedance_busy and not device_transition
        )
        self.experiment_start.setEnabled(
            ready
            and not busy
            and not device_transition
            and not experiment_running
            and not impedance_busy
            and not model_running
            and not confirmation_running
        )
        self.practice_button.setEnabled(
            not device_transition
            and not busy
            and not experiment_running
            and not impedance_busy
            and not model_running
            and not confirmation_running
        )
        self.experiment_stop.setEnabled(self.experiment.running or self._experiment_prepare_active)
        self.model_start.setEnabled(
            not device_transition and not model_running and not busy and not impedance_busy and not confirmation_running
        )
        self.model_stop.setEnabled(model_running)
        if hasattr(self, "decoding_panel"):
            self.decoding_panel.set_runtime_state(
                allowed=not device_transition and not busy and not impedance_busy and not confirmation_running,
                device_ready=ready and not experiment_running,
            )
        quality_reports = getattr(self, "quality_reports", None)
        if quality_reports is not None:
            quality_reports.pump()

    def _on_transport_changed(self) -> None:
        switching = self._displayed_transport is not None
        if switching:
            self._remember_device_fields()
        spec = transport_spec(self.transport_combo.currentData())
        self.connection_preferences.transport = spec.key
        self._displayed_transport = spec.key
        profile = self.connection_preferences.profiles[spec.key]
        self.port_label.setText(spec.address_label)
        self.port_edit.setPlaceholderText(spec.address_hint)
        self.port_edit.setText(profile.address)
        # A switch changes several fields atomically; intermediate combo values
        # must not overwrite the stored configuration of the new transport.
        with QtCore.QSignalBlocker(self.gain_combo):
            self.gain_combo.setCurrentText(str(profile.gain))
        if switching:
            self._save_device_settings()

    def _remember_device_fields(self) -> None:
        self.connection_preferences.profiles[self._displayed_transport] = ConnectionProfile(
            self.port_edit.text().strip(), int(self.gain_combo.currentText()),
        )

    def _save_device_settings(self) -> None:
        self._remember_device_fields()
        try:
            atomic_write_json(self.connection_settings_path, self.connection_preferences.to_dict())
        except OSError as exc:
            self._append_log(f"连接配置保存失败: {exc}")

    def _monitor_device(self) -> None:
        if self._shutdown_in_progress:
            return
        if self._device_monitor_task is None or self._device_monitor_task.done():
            self._device_monitor_task = asyncio.create_task(self._monitor_device_async())

    async def _monitor_device_async(self) -> None:
        # Periodic crash-recovery checkpoint: append recent BLE samples to the
        # .part stream so a force kill loses at most a few seconds of data.
        # getattr keeps this safe for partially constructed windows (tests).
        device = getattr(self, "device", None)
        if device is not None:
            if device.recording_paused:
                self._eeg_data_delay_active = False
                return
            device.checkpoint_recording()
        if self.recording.finishing:
            self._eeg_data_delay_active = False
            return
        if self.device.state is DeviceState.DISCONNECTED:
            self._eeg_data_delay_active = False
            return
        if self.device.state in {DeviceState.PREVIEWING, DeviceState.RECORDING}:
            try:
                await self.device.loss_stats()
            except Exception as exc:
                self._append_log(f"EEG 丢包统计读取失败（采集未停止）: {exc}")
                return

            started = self.device.snapshot.stream_started_monotonic
            # Prefer the recorder-append timestamp: SDK counters keep rising
            # even when the callback->EDF path dies, which would silently
            # suppress this warning (the P004-P006 failure mode). While a
            # local recorder is active, SDK-refreshed timestamps are ignored
            # entirely; a None append timestamp means no data ever arrived.
            last = getattr(self.device, "recording_last_sample_monotonic", None)
            recorder_active = getattr(self.device, "recording_sample_count", None) is not None
            if not recorder_active and last is None:
                last = self.device.snapshot.last_eeg_monotonic
            now = time.monotonic()
            last_update = last if last is not None else started
            if last_update is None:
                return
            delay_seconds = now - last_update
            if delay_seconds <= EEG_DATA_DELAY_WARNING_SECONDS:
                return

            was_delayed = self._eeg_data_delay_active
            self._eeg_data_delay_active = True
            if self.store.active:
                self._recording_max_eeg_gap_seconds = max(
                    self._recording_max_eeg_gap_seconds,
                    delay_seconds,
                )
                if not was_delayed:
                    self._recording_eeg_gap_events += 1
                    self.store.update_manifest(quality=self._session_quality_payload())
            self.status_label.setText(f"EEG 数据延迟 · {delay_seconds:.0f} 秒")
            self.status_label.setStyleSheet(f"color: {COLORS['amber']}; font-weight: 700;")
            # Auto-recovery: P004-P006 showed the SDK can keep counting
            # packets while its native callback dispatch silently dies. The
            # only Python-side remedy is unsubscribe + re-subscribe, tried at
            # most once per cooldown while the stall persists.
            if (
                self.device.state is DeviceState.RECORDING
                and getattr(self.device, "recording_sample_count", None) is not None
                and now - self._last_stream_recovery_monotonic
                >= STREAM_RECOVERY_COOLDOWN_SECONDS
            ):
                self._last_stream_recovery_monotonic = now
                self._append_log(
                    f"EEG 数据已中断 {delay_seconds:.0f} 秒，尝试自动重订阅恢复…"
                )
                try:
                    recovered = await self.device.attempt_stream_recovery()
                except Exception as recovery_exc:
                    recovered = False
                    self._append_log(f"断流恢复尝试异常: {recovery_exc}")
                if self.store.active:
                    try:
                        self.store.event(
                            "stream_recovery_attempt",
                            {
                                "delay_seconds": round(delay_seconds, 3),
                                "recovered": bool(recovered),
                            },
                        )
                    except Exception as event_exc:
                        self._append_log(f"恢复事件写入失败: {event_exc}")
                if recovered:
                    self._append_log("EEG 重订阅完成，等待数据流恢复…")
                else:
                    self._append_log("EEG 重订阅未成功；请检查设备连接与电量。")
            if now - self._last_device_timeout_log >= EEG_DATA_DELAY_LOG_INTERVAL_SECONDS:
                self._last_device_timeout_log = now
                self._append_log(
                    f"EEG 数据已超过 {EEG_DATA_DELAY_WARNING_SECONDS:.0f} 秒未更新；仅告警，不会自动停止实验或录制。"
                )
                if self.store.active:
                    try:
                        self.store.event(
                            "eeg_data_delay",
                            {"delay_seconds": round(delay_seconds, 3), "action": "warning_only"},
                        )
                    except Exception as event_exc:
                        self._append_log(f"EEG 延迟事件写入失败: {event_exc}")
        else:
            self._eeg_data_delay_active = False

    def _session_quality_payload(self) -> dict[str, Any]:
        continuity_passed = self._recording_max_eeg_gap_seconds <= EEG_DATA_DELAY_WARNING_SECONDS
        clock = getattr(self.device, "recording_clock_record", None)
        if clock is not None:
            continuity_passed = continuity_passed and not clock["integrity_failed"]
            continuity_passed = continuity_passed and clock["sequence_continuity_verified"]
        return {
            "impedance": {
                "measurement_source": "simulated" if self.mock else "device_sdk",
                "values_kohm": [value if np.isfinite(value) else None for value in self.latest_impedance],
                "threshold_kohm": IMPEDANCE_PASS_THRESHOLD_KOHM,
                "active_mask": self.montage.active_mask,
                "active_channels": list(self.montage.active_indices),
                "passed": self.impedance_passed,
                "required_for_experiment": False,
                "completed_utc": self.impedance_completed_utc,
            },
            "eeg_continuity": {
                "warning_threshold_seconds": EEG_DATA_DELAY_WARNING_SECONDS,
                "max_gap_seconds": round(self._recording_max_eeg_gap_seconds, 3),
                "gap_event_count": self._recording_eeg_gap_events,
                "passed": continuity_passed,
            },
            "recording_integrity": {
                # Samples that actually reached the EDF buffer vs the count
                # the SDK reports having received. A large gap means the
                # callback->recorder path dropped data (P004-P006 failure).
                "samples_recorded": clock["stored_samples"] if clock is not None else self.device.recording_sample_count,
                "packets_reported_by_sdk": self.device.snapshot.packets,
            },
            "sample_clock": clock,
            "stimulus_timing": {
                "source": "software_repaint",
                "hardware_trigger_available": False,
                "software_repaint_is_optical_onset": False,
                "posthoc_algorithm_qc_required": True,
                "ocular_artifact_rejection": "centered_eog_regression",
                "channel_types": list(self.montage.channel_types),
                "eeg_channel_indices": list(self.montage.eeg_indices),
                "eog_channel_indices": list(self.montage.eog_indices),
                "eog_trace_policy": "retain_raw_for_quality_and_regression",
            },
        }

    def _toggle_connection(self) -> None:
        if self._shutdown_in_progress:
            return
        if self._connection_task_handle is not None and not self._connection_task_handle.done():
            if self._connection_cancelable:
                self._connection_cancelable = False
                self.connect_button.setEnabled(False)
                self._connection_task_handle.cancel()
                self._append_log("正在取消连接并释放设备...")
            return
        if self._device_transition_active():
            self._append_log("设备操作尚未完成，不能切换连接")
            return
        if self.device.state is not DeviceState.DISCONNECTED and self._acquisition_active():
            self._append_log("实验或录制进行中不能断开设备")
            return
        self.connect_button.setEnabled(False)
        self._connection_task_handle = asyncio.create_task(self._connection_task())

    def _start_startup_connection(self) -> None:
        if self._shutdown_in_progress:
            return
        if self.device.state is not DeviceState.DISCONNECTED:
            return
        if self._connection_task_handle is not None and not self._connection_task_handle.done():
            return
        self._connection_task_handle = asyncio.create_task(
            self._connection_task(startup_scan=True)
        )
        self._sync_action_state()

    async def _connection_task(self, startup_scan: bool = False) -> None:
        if self.device.state is not DeviceState.DISCONNECTED:
            self._connection_cancelable = False
            self.connect_button.setEnabled(False)
            try:
                await self._cancel_impedance_detection()
                await self.device.disconnect()
                self._append_log("设备已断开")
            except Exception as exc:
                self._append_log(f"断开失败: {exc}")
            finally:
                self._connection_task_handle = None
                self._sync_action_state()
            return
        self.connect_button.setEnabled(False)
        self.connect_button.setText("取消连接")
        self._connection_cancelable = True
        transport = str(self.transport_combo.currentData() or "serial")
        spec = transport_spec(transport)
        self._save_device_settings()
        self._sync_action_state()
        self.status_label.setText(
            f"{spec.label} 连接中（{STARTUP_CONNECTION_TIMEOUT_SECONDS:.0f} 秒）..."
            if startup_scan
            else "连接中..."
        )
        try:
            connect_request = self.device.connect(
                self.port_edit.text().strip() or None,
                transport=transport,
            )
            if startup_scan:
                await asyncio.wait_for(connect_request, STARTUP_CONNECTION_TIMEOUT_SECONDS)
            else:
                await connect_request
            self.status_label.setText("设备已连接，正在应用采集参数...")
            if await self._apply_device_config_async():
                self._append_log("设备连接成功，采集参数已应用")
            else:
                self._append_log(
                    "设备连接成功，但采集参数未应用；连接已保留，请检查日志后重试。"
                )
        except asyncio.CancelledError:
            try:
                await self.device.disconnect()
            except Exception as cleanup_exc:
                self._append_log(f"取消连接后的清理失败: {cleanup_exc}")
            raise
        except asyncio.TimeoutError:
            self._append_log(
                f"{spec.label} 自动连接超时（{STARTUP_CONNECTION_TIMEOUT_SECONDS:.0f} 秒），未连接设备"
            )
            self.status_label.setText(f"{spec.label} 连接超时")
            try:
                await self.device.disconnect()
            except Exception as cleanup_exc:
                self._append_log(f"连接超时后的清理失败: {cleanup_exc}")
        except Exception as exc:
            self._append_log(f"连接失败: {exc}")
            self.status_label.setText("连接失败")
            try:
                await self.device.disconnect()
            except Exception as cleanup_exc:
                self._append_log(f"连接失败后的清理失败: {cleanup_exc}")
        finally:
            self._connection_cancelable = False
            self._connection_task_handle = None
            self._sync_action_state()

    def _apply_device_config(self) -> None:
        if self._shutdown_in_progress:
            return
        if self._device_config_task_handle is not None and not self._device_config_task_handle.done():
            return
        if self._acquisition_active():
            self._append_log("实验或录制进行中不能修改设备参数")
            return
        if self._device_transition_active() or self._impedance_active():
            self._append_log("设备操作尚未完成，不能修改设备参数")
            return
        self.apply_config_button.setEnabled(False)
        self._device_config_task_handle = asyncio.create_task(self._apply_device_config_async())
        self._sync_action_state()

    async def _apply_device_config_async(self) -> bool:
        try:
            sample_rate = DEFAULT_SAMPLE_RATE_HZ
            filter_config = self._read_filter_config()
            validate_realtime_filter_config(sample_rate, filter_config)
            await self.device.configure(sample_rate, int(self.gain_combo.currentText()))
            actual_rate = float(self.device.snapshot.sample_rate_hz)
            self.eeg_filter.configure(actual_rate, filter_config)
            self._set_wave_sample_rate(actual_rate)
            self.filter_config_dirty = False
            self._clear_waveforms()
            self._update_filter_state()
            verification_labels = {
                "readback_confirmed": "采样率读回已确认，增益写入已接受",
                "write_acknowledged_readback_unavailable": "写入已接受，BLE 读回不可用",
                "mock": "模拟配置",
            }
            verification = verification_labels.get(
                self.device.snapshot.configuration_verification,
                self.device.snapshot.configuration_verification,
            )
            self._append_log(f"采集参数已应用 · {verification}")
            return True
        except Exception as exc:
            self._append_log(f"应用参数失败: {exc}")
            return False
        finally:
            self._device_config_task_handle = None
            self._sync_action_state()

    def _read_filter_config(self) -> RealtimeFilterConfig:
        return RealtimeFilterConfig(
            bandpass_enabled=self.bandpass_check.isChecked(),
            bandpass_low_hz=float(self.bandpass_low_spin.value()),
            bandpass_high_hz=float(self.bandpass_high_spin.value()),
            notch_enabled=self.notch_check.isChecked(),
            notch_hz=float(self.notch_combo.currentData()),
        )

    def _mark_filter_dirty(self) -> None:
        self.filter_config_dirty = True
        if hasattr(self, "filter_state"):
            self.filter_state.setText("滤波参数已修改，请点击“应用”")

    def _apply_filter_settings(self) -> bool:
        if self._acquisition_active():
            self._append_log("实验或录制进行中不能修改实时滤波")
            return False
        try:
            config = self._read_filter_config()
            sample_rate = float(self.device.snapshot.sample_rate_hz)
            validate_realtime_filter_config(sample_rate, config)
            self.eeg_filter.configure(sample_rate, config)
        except (FilterConfigurationError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "实时滤波参数不可用", str(exc))
            return False
        self.filter_config_dirty = False
        self._clear_waveforms()
        self._update_filter_state()
        self._append_log("实时预览滤波已应用")
        return True

    def _update_filter_state(self) -> None:
        if not self.eeg_filter.available:
            self.filter_state.setText("滤波不可用：当前 Python 环境缺少 scipy")
            return
        config = self.eeg_filter.config
        details: list[str] = []
        if config.bandpass_enabled:
            details.append(f"带通 {config.bandpass_low_hz:g}-{config.bandpass_high_hz:g} Hz")
        if config.notch_enabled:
            details.append(f"工频 {config.notch_hz:g} Hz 陷波")
        if self.montage.eog_indices and config.bandpass_enabled:
            details.append("EOG 0.1-10 Hz（原始 EOG 保留用于校正）")
        self.filter_state.setText("显示滤波：" + (" · ".join(details) if details else "关闭"))

    def _clear_waveforms(self) -> None:
        self.eeg_filter.reset()
        for waveform in self.waveforms:
            waveform.clear()
        self._wave_dirty = True

    def _set_wave_sample_rate(self, sample_rate_hz: float) -> None:
        self._wave_sample_rate_hz = float(sample_rate_hz)
        self.waveform_history_samples = max(1, round(self._wave_sample_rate_hz * WAVEFORM_DISPLAY_SECONDS))
        self.waveforms = [deque(maxlen=self.waveform_history_samples) for _ in self.channel_labels]
        self._clear_waveforms()
        self._apply_wave_x_range()

    def _update_wave_plot_labels(self) -> None:
        for index, track_label in enumerate(self.wave_track_labels):
            track_label.setText(f"{self._channel_slot_name(index)} · {self._channel_display_name(index)}")

    def _apply_wave_scale(self) -> None:
        if self.wave_scale_combo is None:
            return
        scale_uv = float(self.wave_scale_combo.currentData())
        for plot in self.wave_plots:
            plot.setYRange(-scale_uv, scale_uv, padding=0.0)

    def _apply_wave_x_range(self) -> None:
        for plot in self.wave_plots:
            plot.setLimits(
                xMin=0,
                xMax=self.waveform_history_samples,
                minXRange=self.waveform_history_samples,
                maxXRange=self.waveform_history_samples,
            )
            plot.setXRange(0, self.waveform_history_samples, padding=0.0)

    def _toggle_stream(self) -> None:
        if self._shutdown_in_progress:
            return
        if self._stream_task_handle is not None and not self._stream_task_handle.done():
            return
        if self._acquisition_active() or self._device_transition_active() or self._impedance_active():
            self._append_log("当前操作尚未完成，不能切换 EEG 预览")
            return
        if self.device.state not in {DeviceState.READY, DeviceState.PREVIEWING, DeviceState.RECOVERY_REQUIRED}:
            self._append_log("当前设备状态不能切换 EEG 预览")
            return
        if self.device.state is DeviceState.READY and self.filter_config_dirty:
            if not self._apply_filter_settings():
                return
        self.stream_button.setEnabled(False)
        self._stream_task_handle = asyncio.create_task(self._stream_task())
        self._sync_action_state()

    async def _stream_task(self) -> None:
        if self.device.state is DeviceState.DISCONNECTED:
            self._append_log("请先连接设备")
            return
        try:
            if self.device.state is DeviceState.PREVIEWING or (
                self.device.state is DeviceState.RECOVERY_REQUIRED and self.device.recovery_may_have_preview
            ):
                await self.device.stop_stream()
                self._clear_waveforms()
                self._append_log("EEG 流已停止")
            else:
                actual_rate = float(self.device.snapshot.sample_rate_hz)
                self.eeg_filter.configure(actual_rate, self.eeg_filter.config)
                self._set_wave_sample_rate(actual_rate)
                await self.device.start_stream()
                self._append_log("EEG 流已启动")
        except Exception as exc:
            self._append_log(f"EEG 流操作失败: {exc}")
        finally:
            self._stream_task_handle = None
            self._sync_action_state()

    def _on_eeg_batch(self, data: np.ndarray) -> None:
        # Recording owns the SDK transfer but has no live preview plot. Ignore
        # delayed callbacks after preview shutdown so old samples cannot appear
        # as a new waveform window.
        if self.device.state is not DeviceState.PREVIEWING:
            return
        if data.ndim != 2:
            return
        try:
            data = self.eeg_filter.process(data)
        except Exception as exc:
            self._append_log(f"实时滤波失败，已显示原始波形: {exc}")
        if data.shape[1] < len(self.channel_labels):
            self._append_log(f"EEG 波形通道数不足: {data.shape[1]}")
            return
        visible = np.asarray(data[:, :len(self.channel_labels)], dtype=np.float64)
        for channel, waveform in enumerate(self.waveforms):
            waveform.extend(visible[:, channel].tolist())
        self._wave_dirty = True
        if self._eeg_data_delay_active:
            self._eeg_data_delay_active = False
            self._last_stream_recovery_monotonic = 0.0
            self._append_log("EEG 数据已恢复更新")
            self._on_device_state(self.device.snapshot)

    def _refresh_plot(self) -> None:
        if not self.wave_curves:
            return
        if not self._wave_dirty:
            return
        for index, curve in enumerate(self.wave_curves):
            values = np.asarray(self.waveforms[index], dtype=np.float64)
            start = self.waveform_history_samples - len(values)
            curve.setData(np.arange(start, self.waveform_history_samples, dtype=np.float64), values)
        self._wave_dirty = False

    def _on_impedance(self, values: list[float]) -> None:
        self.latest_impedance = []
        for value in values[:8]:
            try:
                self.latest_impedance.append(float(value))
            except (TypeError, ValueError):
                self.latest_impedance.append(float("nan"))
        self.latest_impedance.extend([float("nan")] * (8 - len(self.latest_impedance)))
        for index, card in enumerate(self.impedance_cards):
            if not self.montage.active_mask & (1 << index):
                card.title.setText(f"{self._channel_slot_name(index)} · {self._channel_display_name(index)} · 未启用")
                card.set_value("--")
                card.value.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {COLORS['muted']};")
                continue
            value = self.latest_impedance[index]
            quality, color = ("模拟", COLORS["amber"]) if self.mock else self._impedance_quality(value)
            card.set_value(f"{value:.1f} kΩ" if np.isfinite(value) else "--")
            card.title.setText(f"{self._channel_slot_name(index)} · {self._channel_display_name(index)} · {quality}")
            card.value.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {color};")
        if self.mock:
            self.impedance_passed = False
            self.card_impedance.set_value("模拟数据")
            self.impedance_state.setText("模拟输出 · 非接触质量判定")
            return
        active_values = [self.latest_impedance[index] for index in self.montage.active_indices]
        valid_values = [value for value in active_values if np.isfinite(value) and value >= 0]
        passed = sum(
            np.isfinite(value) and 0 <= value <= IMPEDANCE_PASS_THRESHOLD_KOHM
            for value in active_values
        )
        active_count = self.montage.active_count
        complete = len(valid_values) == active_count
        self.impedance_passed = complete and passed == active_count
        self.card_impedance.set_value(
            f"{passed}/{active_count} 通过"
            if complete
            else f"{passed}/{active_count} · 数据不完整"
        )
        if complete:
            self.impedance_state.setText("实时合格" if self.impedance_passed else "实时未通过")
        else:
            self.impedance_state.setText("检测中 · 等待完整数据")

    def _reset_impedance_display(self, state: str = "尚未检测") -> None:
        if hasattr(self, "impedance_error"):
            self.impedance_error.clear()
            self.impedance_error.hide()
        self.latest_impedance = [float("nan")] * 8
        self.impedance_passed = False
        self.impedance_completed_monotonic = None
        self.impedance_completed_utc = None
        if hasattr(self, "impedance_state"):
            self.impedance_state.setText(state)
        if not hasattr(self, "impedance_cards"):
            return
        for index, card in enumerate(self.impedance_cards):
            enabled = bool(self.montage.active_mask & (1 << index))
            card.title.setText(
                f"{self._channel_slot_name(index)} · {self._channel_display_name(index)} · "
                f"{'待检测' if enabled else '未启用'}"
            )
            card.set_value("--")
            card.value.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {COLORS['muted']};")
        if hasattr(self, "card_impedance"):
            self.card_impedance.set_value("待检测")

    def _impedance_quality(self, value: float) -> tuple[str, str]:
        if np.isfinite(value) and value < 0:
            return "无效", COLORS["red"]
        if not np.isfinite(value):
            return "待检测", COLORS["muted"]
        if value <= IMPEDANCE_PASS_THRESHOLD_KOHM * 0.5:
            return "良好", COLORS["green"]
        if value <= IMPEDANCE_PASS_THRESHOLD_KOHM:
            return "合格", COLORS["amber"]
        return "偏高", COLORS["red"]

    def _impedance_result_complete(self) -> bool:
        return bool(
            len(self.latest_impedance) == 8
            and all(np.isfinite(self.latest_impedance[index]) and self.latest_impedance[index] >= 0
                    for index in self.montage.active_indices)
        )

    def _record_impedance_completion(self, prefix: str) -> None:
        self.impedance_completed_monotonic = time.monotonic()
        self.impedance_completed_utc = utc_now()
        if self.mock:
            self.impedance_state.setText(f"{prefix} · 模拟数据，非设备测量")
        else:
            self.impedance_state.setText(f"{prefix} · {'可开始' if self.impedance_passed else '未通过'}")

    def _on_device_state(self, snapshot: Any) -> None:
        state = self.device.state
        connected = state is not DeviceState.DISCONNECTED
        state_labels = {
            DeviceState.DISCONNECTED: "未连接",
            DeviceState.READY: "设备就绪",
            DeviceState.PREVIEWING: "EEG 预览中",
            DeviceState.RECORDING: "EDF 录制中",
            DeviceState.IMPEDANCE: "阻抗检测中",
            DeviceState.RECOVERY_REQUIRED: "需要恢复",
        }
        state_colors = {
            DeviceState.DISCONNECTED: COLORS["amber"],
            DeviceState.READY: COLORS["green"],
            DeviceState.PREVIEWING: COLORS["blue"],
            DeviceState.RECORDING: COLORS["red"],
            DeviceState.IMPEDANCE: COLORS["amber"],
            DeviceState.RECOVERY_REQUIRED: COLORS["red"],
        }
        self.status_label.setText(state_labels[state])
        self.status_label.setStyleSheet(f"color: {state_colors[state]}; font-weight: 700;")
        self.connect_button.setText("断开设备" if connected else "连接设备")
        if state is DeviceState.PREVIEWING or (
            state is DeviceState.RECOVERY_REQUIRED and self.device.recovery_may_have_preview
        ):
            self.stream_button.setText("停止 EEG 流")
        elif state is DeviceState.RECOVERY_REQUIRED:
            self.stream_button.setText("需要恢复")
        else:
            self.stream_button.setText("启动 EEG 流")
        self.firmware_label.setText(f"固件 {snapshot.firmware}")
        self.loss_label.setText(f"丢包率 {snapshot.loss_rate:.2f}%")
        self.card_device.set_value("已连接" if connected else "未连接")
        self.card_stream.set_value(state_labels[state])
        if state is not DeviceState.PREVIEWING and any(self.waveforms):
            self._clear_waveforms()
        if connected:
            for row in range(8):
                self.channel_table.item(row, 3).setText(
                    "在线" if self.montage.active_mask & (1 << row) else "未启用"
                )
        else:
            for row in range(8):
                self.channel_table.item(row, 3).setText(
                    "待连接" if self.montage.active_mask & (1 << row) else "未启用"
                )
            self._clear_waveforms()
            self._reset_impedance_display()
        if state is DeviceState.RECOVERY_REQUIRED:
            self._append_log(f"设备需要恢复: {self.device.recovery_reason}")
        if not connected and self._acquisition_active():
            self._append_log("设备已断开，正在中止当前实验")
            if self.store.active:
                try:
                    self.store.event("device_disconnected", {"reason": "device_state_changed"})
                except Exception as event_exc:
                    self._append_log(f"设备断开事件写入失败: {event_exc}")
            if self._experiment_prepare_active:
                self._experiment_prepare_task.cancel()
            self.experiment.stop()
        self._sync_action_state()

    def _start_impedance(self) -> None:
        if self._shutdown_in_progress:
            return
        task = self._impedance_task_handle
        if task is not None and not task.done():
            if self._impedance_cancel_requested:
                return
            self._impedance_cancel_requested = True
            self.impedance_state.setText("停止检测中...")
            self.impedance_start.setEnabled(False)
            self._impedance_operation_pending = True
            task.cancel()
            return
        if self._device_transition_active() or self._acquisition_active():
            self._append_log("设备操作或采集收尾尚未完成，不能启动阻抗检测")
            return
        if self.device.state is DeviceState.DISCONNECTED:
            self._append_log("请先连接设备")
            return
        if self.device.state is DeviceState.RECOVERY_REQUIRED and self.device.recovery_may_have_impedance:
            self.impedance_state.setText("恢复检测资源中...")
            self.impedance_start.setEnabled(False)
            self._impedance_operation_pending = True
            self._impedance_task_handle = asyncio.create_task(self._retry_impedance_cleanup())
            return
        if self.experiment.running or self._experiment_prepare_active:
            self._append_log("实验进行中不能切换到阻抗检测模式")
            return
        if self.device.state is not DeviceState.READY:
            self._append_log("设备未就绪；请完成当前操作或断开设备恢复后再进行阻抗检测")
            return
        self._reset_impedance_display("检测中...")
        self._impedance_cancel_requested = False
        self.impedance_start.setText("停止检测")
        self._impedance_operation_pending = True
        self._impedance_task_handle = asyncio.create_task(self._run_impedance_task())

    async def _retry_impedance_cleanup(self) -> None:
        try:
            await self.device.stop_impedance()
            self.impedance_state.setText("资源已恢复 · 可重新检测")
            self.impedance_error.clear()
            self.impedance_error.hide()
            self._append_log("阻抗检测资源已恢复")
        except Exception as exc:
            self.impedance_state.setText("恢复失败 · 请断开设备")
            self.impedance_error.setText(str(exc))
            self.impedance_error.show()
            self._append_log(f"阻抗检测资源恢复失败: {exc}")
        finally:
            self.impedance_start.setText("开始检测")
            self._impedance_operation_pending = False
            self._impedance_task_handle = None
            self._sync_action_state()

    async def _run_impedance_task(self) -> None:
        try:
            # The adapter owns continuous transfer until stop or dialog close.
            await self.device.run_impedance()
            if self._impedance_result_complete():
                self._record_impedance_completion("检测完成")
            self._append_log("阻抗检测完成")
        except asyncio.CancelledError:
            if self._impedance_result_complete():
                self._record_impedance_completion("检测已停止")
                self._append_log("阻抗检测已停止，保留最后一次完整判定")
            else:
                self.impedance_state.setText("检测已停止 · 数据不完整")
            raise
        except ImpedanceOperationError as exc:
            state_by_stage = {
                "data": "检测失败 · 未收到数据",
                "cleanup": "检测完成 · 清理失败",
            }
            stage = ImpedanceOperationError._STAGE_LABELS.get(exc.stage, exc.stage)
            self.impedance_state.setText(state_by_stage.get(exc.stage, f"检测失败 · {stage}"))
            self.impedance_error.setText(str(exc))
            self.impedance_error.show()
            self._append_log(str(exc))
        except Exception as exc:
            self._append_log(f"阻抗检测失败: {exc}")
            self.impedance_state.setText("检测失败")
            self.impedance_error.setText(str(exc) or type(exc).__name__)
            self.impedance_error.show()
        finally:
            if self.device.recovery_may_have_impedance:
                self.impedance_state.setText(self.impedance_state.text() + " · 设备需恢复")
                self.impedance_error.setText(
                    self.impedance_error.text() + "\n" + self.device.recovery_reason
                    + "\n可重试恢复阻抗资源，或断开设备后重连。"
                )
                self.impedance_error.show()
            self.impedance_start.setText("开始检测")
            self._impedance_operation_pending = False
            self._impedance_task_handle = None
            self._impedance_cancel_requested = False
            self._sync_action_state()

    def _start_experiment(self) -> None:
        if self._shutdown_in_progress:
            return
        if self._acquisition_active():
            return
        if self._thought_digit_task is not None and not self._thought_digit_task.done():
            self._append_log("请先完成上一份记录的目标确认")
            return
        if self._device_transition_active():
            self._append_log("设备连接或采集参数配置尚未完成")
            return
        if self.model_runner.running:
            self._append_log("算法任务运行中不能开始正式实验")
            self._set_page("model")
            return
        if self.device.state is DeviceState.DISCONNECTED:
            self._append_log("实验开始前请先连接设备")
            self._set_page("device")
            return
        if self.device.state is not DeviceState.READY:
            self._append_log("设备未就绪；请先结束当前操作或断开设备恢复")
            self._set_page("device")
            return
        if self._impedance_active():
            self._append_log("请等待阻抗检测结束后再开始实验")
            self._set_page("device")
            return
        config = self._read_experiment_config()
        validation_error = self._validate_experiment_config(config)
        if validation_error:
            QtWidgets.QMessageBox.warning(self, "实验参数不可用", validation_error)
            return
        expected = self._expected_duration_seconds(config)
        answer = QtWidgets.QMessageBox.question(
            self,
            "确认开始实验",
            f"受试者: {config.subject_id}\n"
            f"总试次: {config.blocks * config.digits_per_block}\n预计时长: {expected / 60:.1f} 分钟\n"
            f"保存目录: {config.save_dir}\n"
            "默想数字将在实验结束后由实验员确认并写入会话数据。\n"
            "注意：当前刺激标记来自软件重绘，未连接硬件触发。"
            "请在算法后处理阶段完成时序质量控制与眼动伪迹删除。\n\n"
            "确认开始？",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        # A modal Qt dialog keeps dispatching callbacks while the operator decides.
        if (self._shutdown_in_progress or self._acquisition_active() or self._device_transition_active()
                or self._impedance_active() or self.model_runner.running or self.device.state is not DeviceState.READY
                or (self._thought_digit_task is not None and not self._thought_digit_task.done())):
            self._append_log("确认期间工作站状态已改变，请重新开始实验")
            self._sync_action_state()
            return
        self.experiment_start.setEnabled(False)
        self.practice_button.setEnabled(False)
        self._experiment_prepare_active = True
        self._experiment_prepare_task = asyncio.create_task(self._prepare_and_start_experiment(config))
        self._experiment_prepare_task.add_done_callback(self._experiment_prepare_done)
        self._sync_action_state()

    def _experiment_prepare_done(self, task: asyncio.Task) -> None:
        if self._experiment_prepare_task is task:
            self._experiment_prepare_task = None
            self._experiment_prepare_active = False
        if task.cancelled():
            self._calibration_target_requested = None
            self._decoding_test_requested = False
        else:
            error = task.exception()
            if error is not None:
                self._calibration_target_requested = None
                self._decoding_test_requested = False
                self._append_log(f"实验准备异常: {error}")
        self._sync_action_state()

    def _start_decoding_test(self, subject: str) -> None:
        if self._acquisition_active() or self.model_runner.running:
            return
        if not subject:
            self._append_log("个体模型未绑定可用受试者，请先选择受试者")
            return
        self.subject_edit.setText(subject)
        self.blocks_spin.setValue(1)
        self.digits_per_block_spin.setValue(45)
        self.instruction_edit.setText("请选择一个数字并默数它的出现次数。")
        self._calibration_target_requested = None
        self._set_page("experiment")
        self._start_experiment()
        self._decoding_test_requested = self._experiment_prepare_active

    def _start_calibration_record(self, subject: str, target: int) -> None:
        if self._acquisition_active() or self.model_runner.running:
            return
        if not subject:
            self._append_log("请填写受试者编号")
            return
        self.subject_edit.setText(subject)
        self.blocks_spin.setValue(10)
        self.digits_per_block_spin.setValue(45)
        self.instruction_edit.setText(f"本轮目标 {target}。请默数该数字的出现次数。")
        self._decoding_test_requested = False
        self._set_page("experiment")
        self._start_experiment()
        self._calibration_target_requested = target if self._experiment_prepare_active else None

    def _run_recorded_test(self, session_dir: Path) -> None:
        if self._shutdown_in_progress:
            return
        self._set_page("model")
        self.model_page.setCurrentIndex(0)
        self.decoding_panel.set_data_root(session_dir.parent)
        self._sync_action_state()
        started = self.decoding_panel.start_inference(session_dir, request_confirmation=True)
        if not started:
            self._request_target_confirmation(str(session_dir))

    def _request_target_confirmation(self, session_dir: str) -> None:
        if self._shutdown_in_progress:
            return
        if self._thought_digit_task is not None and not self._thought_digit_task.done():
            if self._thought_digit_dialog is not None:
                self._thought_digit_dialog.raise_()
            return
        task = asyncio.create_task(self._confirm_thought_digit_async(Path(session_dir)))
        self._thought_digit_task = task
        task.add_done_callback(self._thought_digit_done)
        self._sync_action_state()

    def _read_experiment_config(self) -> ExperimentConfig:
        if self.auto_seed_check.isChecked():
            self.seed_spin.setValue(secrets.randbelow(100_000_000))
        return ExperimentConfig(
            subject_id=self.subject_edit.text().strip() or "P001",
            age=self.age_edit.text().strip(),
            birthdate=self.birthdate_edit.text().strip(),
            sex=self.sex_combo.currentText(),
            blocks=self.blocks_spin.value(),
            digits_per_block=self.digits_per_block_spin.value(),
            stimulus_ms=self.stimulus_spin.value(),
            soa_ms=self.soa_spin.value(),
            rest_sec=self.rest_spin.value(),
            random_seed=self.seed_spin.value(),
            screen_index=self.screen_spin.value(),
            save_dir=self.save_dir_edit.text().strip() or "data",
            model_profile=self.model_profile_edit.text(),
            instruction_text=self.instruction_edit.text().strip(),
            notes=self.notes_edit.text().strip(),
        )

    def _expected_duration_seconds(self, config: ExperimentConfig) -> float:
        trials = config.blocks * config.digits_per_block
        return 3.0 + trials * config.soa_ms / 1000.0 + max(0, config.blocks - 1) * config.rest_sec

    def _validate_experiment_config(self, config: ExperimentConfig) -> str | None:
        try:
            config.validate_protocol()
        except ValueError as exc:
            return str(exc)
        try:
            edf_birthdate(config.birthdate)
        except ValueError as exc:
            return str(exc)
        if config.age and (not config.age.isdigit() or not 0 <= int(config.age) <= 120):
            return "年龄必须为空，或为 0 到 120 之间的整数。"
        screen_count = len(QtWidgets.QApplication.screens())
        if not 0 <= config.screen_index < screen_count:
            return f"被试屏幕编号无效，当前只有 {screen_count} 个显示器。"
        try:
            self.montage.validate()
        except ValueError as exc:
            return f"通道映射无效: {exc}"
        save_root = Path(config.save_dir).expanduser()
        if not save_root.is_absolute():
            save_root = self.app_root / save_root
        try:
            save_root.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(save_root).free < 100 * 1024 * 1024:
                return "保存目录可用磁盘空间少于 100 MB。"
        except OSError as exc:
            return f"保存目录不可写: {exc}"
        return None

    def _start_practice(self) -> None:
        if self._shutdown_in_progress:
            return
        if self._acquisition_active():
            self._append_log("请等待采集与记录收尾完成")
            return
        if self._thought_digit_task is not None and not self._thought_digit_task.done():
            self._append_log("请先完成上一份记录的目标确认")
            return
        if self.model_runner.running:
            self._append_log("算法任务运行中不能开始练习预览")
            self._set_page("model")
            return
        if self._acquisition_active() or self._impedance_active() or self.experiment.running:
            self._append_log("当前已有采集或实验任务，不能开始练习")
            return
        config = self._read_experiment_config()
        try:
            screen_count = len(QtWidgets.QApplication.screens())
            if not 0 <= config.screen_index < screen_count:
                raise ValueError(f"被试屏幕编号无效，当前只有 {screen_count} 个显示器。")
            self.stimulus_window = StimulusWindow(config.screen_index)
            self.stimulus_window.stop_requested.connect(self._on_stimulus_stop_requested)
            self._wire_pause_controls()
            self.stimulus_window.showFullScreen()
            self.experiment_start.setEnabled(False)
            self.practice_button.setEnabled(False)
            self.experiment_stop.setEnabled(True)
            self.experiment.start_practice(config)
            self._append_log("练习预览已开始，不会写入 EEG 录制文件")
        except Exception as exc:
            self._append_log(f"练习启动失败: {exc}")
            self._dispose_stimulus_window()
            self.experiment_start.setEnabled(True)
            self.practice_button.setEnabled(True)

    async def _prepare_and_start_experiment(self, config: ExperimentConfig) -> None:
        try:
            self._recording_max_eeg_gap_seconds = 0.0
            self._recording_eeg_gap_events = 0
            runtime_info = {
                "python": sys.version,
                "platform": platform.platform(),
                "application": "BrainSync GTN Workstation",
                "timing_source": "software_repaint",
                "timing_note": "软件重绘时间戳用于通用记录；未接入外部光电或硬件触发。",
                "hardware_trigger_connected": False,
                "posthoc_algorithm_qc_required": True,
                "ocular_artifact_rejection": "optional_calibration_regression_in_model_backend",
                "ocular_correction_applied": False,
                "simulation_status": "declared_mock" if self.mock else "not_declared",
                "channel_types": list(self.montage.channel_types),
                "eeg_channel_indices": list(self.montage.eeg_indices),
                "eog_channel_indices": list(self.montage.eog_indices),
                "eog_trace_policy": "retain_raw_for_quality_and_regression",
                "live_preview_filter": self.eeg_filter.config.to_dict(),
                "display": [
                    {
                        "index": index,
                        "name": screen.name(),
                        "geometry": [screen.geometry().x(), screen.geometry().y(), screen.geometry().width(), screen.geometry().height()],
                        "refresh_rate_hz": screen.refreshRate(),
                    }
                    for index, screen in enumerate(QtWidgets.QApplication.screens())
                ],
            }
            if self.device.sdk is not None:
                runtime_info["sdk_version"] = str(getattr(self.device.sdk, "__version__", "unknown"))
            runtime_info["acquisition_source"] = acquisition_runtime_record(self.device.sdk)
            session_dir = await self.recording.prepare(
                config, self.montage, sample_rate=DEFAULT_SAMPLE_RATE_HZ,
                gain=int(self.gain_combo.currentText()),
                model_info={"profile": config.model_profile, "manifest": self.manifest_edit.text().strip()},
                runtime_info=runtime_info, requested_target=self._calibration_target_requested,
                impedance=self.latest_impedance,
            )
            self._calibration_target_requested = None
            self.decoding_panel.set_data_root(session_dir.parent)
            self.session_label.setText(self.store.session_id)
            self.card_session.set_value(self.store.session_id)
            self.stimulus_window = StimulusWindow(config.screen_index)
            self.stimulus_window.stop_requested.connect(self._on_stimulus_stop_requested)
            self._wire_pause_controls()
            self.stimulus_window.showFullScreen()
            self.experiment_start.setEnabled(False)
            self.experiment_stop.setEnabled(True)
            self.experiment.start(config)
            self._append_log(f"实验已开始: {self.store.session_id}")
        except BaseException as error:
            self._calibration_target_requested = None
            self._decoding_test_requested = False
            self._dispose_stimulus_window()
            await self.recording.finish("aborted" if isinstance(error, asyncio.CancelledError) else "error")
            self.session_label.setText("未开始会话")
            self.card_session.set_value("--")
            self._refresh_model_session()
            self._append_log(f"实验准备未完成: {error}")
            if isinstance(error, asyncio.CancelledError):
                raise
        finally:
            self._experiment_prepare_active = False
            self._experiment_prepare_task = None
            self._sync_action_state()


    def _stop_experiment(self) -> None:
        if self._experiment_prepare_active and self._experiment_prepare_task is not None:
            self._experiment_prepare_task.cancel()
            self._append_log("正在取消实验准备并保存已采集数据")
            return
        self.experiment.stop()
        if self.store.active:
            self._schedule_experiment_finish("aborted")
        self._append_log("正在停止实验和 EDF 录制...")


    def _emergency_persist_recording(self) -> None:
        """Synchronous last-resort save for atexit / aboutToQuit. Never raises."""

        device = getattr(self, "device", None)
        if device is None:
            return
        try:
            path = device.emergency_save_recording()
        except Exception:
            return
        if path:
            print(f"[emergency-save] EDF 录制已强制保存: {path}", file=sys.stderr)
            try:
                self._append_log(f"紧急保存 EDF 录制: {path}")
            except Exception:
                pass

    def _on_stimulus_stop_requested(self) -> None:
        if self._experiment_prepare_active and self._experiment_prepare_task is not None:
            self._experiment_prepare_task.cancel()
            self._append_log("已取消实验准备")
            return
        if self.experiment.running:
            self.experiment.stop()
            if self.store.active:
                self._schedule_experiment_finish("aborted")
            self._append_log("收到刺激窗口紧急停止请求")

    def _wire_pause_controls(self) -> None:
        window = self.stimulus_window
        window.pause_requested.connect(self.experiment.pause)
        window.resume_requested.connect(self.experiment.resume)
        window.save_requested.connect(lambda: self._end_from_pause(discard=False))
        window.discard_requested.connect(lambda: self._end_from_pause(discard=True))

    def _on_experiment_paused(self) -> None:
        try:
            self.recording.pause()
            if self.stimulus_window is not None:
                self.stimulus_window.show_pause_menu(True)
            self._append_log("测试已暂停，EDF 写入已暂停")
        except Exception as error:
            self.recording.fault(str(error))
            self._append_log(f"暂停失败，停止并保存当前记录: {error}")
            self.experiment.stop()

    def _on_experiment_resuming(self) -> None:
        try:
            self.recording.resume()
            if self.stimulus_window is not None:
                self.stimulus_window.show_pause_menu(False)
            self._append_log("EDF 写入已恢复，开始继续前提示倒计时")
        except Exception as error:
            self.recording.fault(str(error))
            self._append_log(f"继续失败，停止并保存当前记录: {error}")
            self.experiment.stop()

    def _end_from_pause(self, *, discard: bool) -> None:
        if not self.experiment.is_paused:
            return
        self.recording.discard_requested = discard
        if self.store.active:
            self.store.event("experiment_disposition", {"action": "discard" if discard else "save_partial"})
        if self.stimulus_window is not None:
            self.stimulus_window.pause_menu.setEnabled(False)
        self.experiment.stop()

    def _dispose_stimulus_window(self) -> None:
        """Make a full-screen stimulus surface disappear before asynchronous cleanup."""

        window = self.stimulus_window
        self.stimulus_window = None
        if window is None:
            return
        window.allow_close()
        window.hide()
        window.close()
        window.deleteLater()

    def _begin_recording_cleanup(self) -> None:
        """Pause continuity warnings while the recording is being closed."""

        self._eeg_data_delay_active = False

    def _present_digit(self, digit: int | None) -> None:
        if self.stimulus_window is not None:
            self.stimulus_window.present(digit)

    def _present_text(self, text: str) -> None:
        if self.stimulus_window is not None:
            self.stimulus_window.present_text(text)



    def _on_stimulus_event(self, event: dict[str, Any]) -> None:
        try:
            self.recording.marker(event)
        except Exception as error:
            self.recording.fault(str(error))
            self._append_log(f"刺激事件写入失败，正在停止实验: {error}")
            self.experiment.stop()





    def _on_trial(self, data: dict[str, Any]) -> None:
        self.last_trial = f"B{data['block']} · trial {data['trial']} · 数字 {data['digit']}"
        self.trial_label.setText(self.last_trial)

    def _on_experiment_progress(self, completed: int, total: int) -> None:
        self.experiment_progress.setMaximum(max(1, total))
        self.experiment_progress.setValue(completed)

    def _on_experiment_state(self, state: str) -> None:
        labels = {
            "paused": "测试已暂停", "resuming": "继续前准备",
            "running": "实验进行中", "practice": "练习进行中", "rest": "Block 休息",
            "completed": "实验完成", "aborted": "实验已停止", "error": "实验异常",
        }
        self.experiment_state.setText(labels.get(state, state))
        self._sync_action_state()

    def _on_practice_finished(self, state: str) -> None:
        self._dispose_stimulus_window()
        self.experiment_start.setEnabled(True)
        self.practice_button.setEnabled(True)
        self.experiment_stop.setEnabled(False)
        self._append_log(f"练习结束: {state}")

    def _on_experiment_finished(self, state: str) -> None:
        self._schedule_experiment_finish(state)

    def _schedule_experiment_finish(self, state: str) -> None:
        self._begin_recording_cleanup()
        self._dispose_stimulus_window()
        self.experiment_state.setText("正在停止 EDF 录制")
        self.experiment_stop.setEnabled(False)
        if self._experiment_finish_task is not None and not self._experiment_finish_task.done():
            return
        self._experiment_finish_task = asyncio.create_task(self._finish_experiment(state))
        add_done_callback = getattr(self._experiment_finish_task, "add_done_callback", None)
        if callable(add_done_callback):
            add_done_callback(self._experiment_finish_done)

    def _experiment_finish_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            self._append_log("实验收尾任务被取消；关闭程序前将执行录制兜底清理")
            return
        try:
            task.result()
        except Exception as exc:
            self._append_log(f"实验收尾任务异常: {exc}")

    async def _confirm_thought_digit_async(self, session_dir: Path) -> None:
        if not session_dir.is_dir():
            return
        dialog = QtWidgets.QInputDialog(self)
        self._thought_digit_dialog = dialog
        dialog.setWindowTitle("确认默想数字")
        dialog.setLabelText("请填写受试者在本次实验中默想的数字（1-9）：")
        dialog.setInputMode(QtWidgets.QInputDialog.InputMode.IntInput)
        dialog.setIntRange(1, 9)
        try:
            saved = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
            requested_target = (saved.get("calibration") or {}).get("requested_target", 1)
            dialog.setIntValue(requested_target if type(requested_target) is int and 1 <= requested_target <= 9 else 1)
        except (OSError, ValueError, AttributeError):
            dialog.setIntValue(1)
        dialog.setOkButtonText("确认写入")
        dialog.setCancelButtonText("暂不确认")
        result = asyncio.get_running_loop().create_future()

        def resolve(value: int | None) -> None:
            if not result.done():
                result.set_result(value)

        dialog.accepted.connect(lambda: resolve(dialog.intValue()))
        dialog.rejected.connect(lambda: resolve(None))
        dialog.open()
        try:
            thought_digit = await result
            if thought_digit is not None:
                self.store.confirm_thought_digit_for_session(session_dir, thought_digit)
                self._append_log(f"已确认默想数字 {thought_digit}，会话标签已写入")
                self.decoding_panel.record_feedback(session_dir, thought_digit)
            else:
                self.store.confirm_thought_digit_for_session(session_dir, None)
                self._append_log("未确认默想数字；记录不能用于有监督校准")
        finally:
            if self._thought_digit_dialog is dialog:
                self._thought_digit_dialog = None
            dialog.deleteLater()
            self._refresh_model_session()
            self.decoding_panel.refresh_sessions()




    async def _finish_experiment(self, state: str) -> None:
        decoding_test = self._decoding_test_requested
        self._decoding_test_requested = False
        try:
            await self.experiment.wait()
            outcome = await self.recording.finish(state)
            self._dispose_stimulus_window()
            self._refresh_model_session()
            self._append_log(f"实验结束: {outcome.status}")
            self.experiment_state.setText({"completed": "采集已完成", "aborted": "采集已中止",
                                           "discarded": "本次记录已废弃", "recording_error": "采集保存异常"}.get(outcome.status, outcome.status))
            self.session_label.setText("未开始会话")
            self.card_session.set_value("--")
            if not self._shutdown_in_progress and outcome.session_dir is not None:
                if outcome.status in {"completed", "aborted"}:
                    if decoding_test and outcome.status == "completed":
                        QtCore.QTimer.singleShot(0, lambda path=outcome.session_dir: self._run_recorded_test(path))
                    else:
                        self._thought_digit_task = asyncio.create_task(self._confirm_thought_digit_async(outcome.session_dir))
                        self._thought_digit_task.add_done_callback(self._thought_digit_done)
                if outcome.status != "discarded":
                    self._queue_quality_report(outcome.session_dir, inference=decoding_test and outcome.status == "completed")
        finally:
            self._experiment_finish_task = None
            self._sync_action_state()

    def _thought_digit_done(self, task: asyncio.Task) -> None:
        if self._thought_digit_task is task:
            self._thought_digit_task = None
        if task.cancelled():
            self._sync_action_state()
            return
        try:
            task.result()
        except Exception as exc:
            self._append_log(f"默想数字确认失败: {exc}")
        finally:
            self._sync_action_state()

    def _quality_request(self, _session: Path, *, inference=False) -> dict:
        panel = self.decoding_panel
        arguments = []
        if inference:
            arguments.extend(["--model-artifact", panel.model_path.text().strip()])
        else:
            if panel.reference.currentData():
                arguments.extend(["--reference-channel", str(panel.reference.currentData())])
            else:
                arguments.extend(["--reference-mode", "native"])
            for option, field in (("--continuous-qc-model", panel.continuous_qc),
                                  ("--epoch-qc-model", panel.epoch_qc_model)):
                if field.text().strip():
                    arguments.extend([option, field.text().strip()])
        return {"backend_root": panel.backend_root.text().strip(), "arguments": arguments}

    def _queue_quality_report(self, session: Path, *, inference=False) -> None:
        controller = getattr(self, "quality_reports", None)
        if controller is None:
            return
        self._last_quality_session = Path(session).resolve()
        controller.enqueue(self._last_quality_session, self._quality_request(session, inference=inference))
        self._on_quality_report_changed(str(self._last_quality_session))

    def _on_quality_report_changed(self, session: str) -> None:
        if self._last_quality_session != Path(session).resolve():
            return
        record = self.quality_reports.status(self._last_quality_session)
        self.quality_report_button.setVisible(bool(record))
        self.quality_report_button.setText("质量报告生成中" if record.get("status") in {"queued", "running"} else "查看质量报告")
        self.quality_report_button.setToolTip(str(record.get("message") or "查看本次采集的数据质量报告"))

    def _show_quality_report(self, session: Path | None) -> None:
        if session is None or not (session / "session.json").is_file():
            self._append_log("请先选择已保存的采集会话")
            return
        session = session.resolve()
        if not self.quality_reports.status(session):
            self._queue_quality_report(session)
        if self._quality_dialog is not None and self._quality_dialog.session != session:
            self._quality_dialog.close()
            self._quality_dialog = None
        if self._quality_dialog is None:
            dialog = QualityReportDialog(session, self.quality_reports, self)
            self._quality_dialog = dialog
            dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
            dialog.finished.connect(lambda _result: setattr(self, "_quality_dialog", None))
        self._quality_dialog.show()
        self._quality_dialog.raise_()
        self._quality_dialog.activateWindow()

    def _refresh_model_session(self) -> Path | None:
        """Select the newest session that passes the algorithm input contract."""

        session_dir = find_latest_model_input_session(self.store.root)
        self.model_session_dir = session_dir
        if session_dir is None:
            self.model_session_edit.clear()
            self.model_session_edit.setPlaceholderText("暂无可分析会话")
            self.model_session_edit.setToolTip(
                "完成实验并确认默想数字后，系统会自动选择可分析会话。"
            )
        else:
            self.model_session_edit.setText(session_dir.name)
            self.model_session_edit.setToolTip(str(session_dir))
        return session_dir

    def _refresh_parameter_menu(self, _index: int = -1) -> None:
        """Show manifest parameter choices for the selected model operation."""

        if not hasattr(self, "parameter_combo"):
            return
        task = str(self.task_combo.currentData() or "")
        options = tuple(
            option
            for option in self._model_parameter_options
            if not option.tasks or task in option.tasks
        ) or (DEFAULT_MODEL_PARAMETER_OPTION,)
        selected = self.parameter_combo.currentData()
        selected_key = selected.key if isinstance(selected, ModelParameterOption) else None
        self.parameter_combo.blockSignals(True)
        try:
            self.parameter_combo.clear()
            for option in options:
                self.parameter_combo.addItem(option.label, option)
                self.parameter_combo.setItemData(
                    self.parameter_combo.count() - 1,
                    option.description,
                    QtCore.Qt.ItemDataRole.ToolTipRole,
                )
            selected_index = next(
                (
                    index
                    for index, option in enumerate(options)
                    if option.key == selected_key
                ),
                0,
            )
            self.parameter_combo.setCurrentIndex(selected_index)
        finally:
            self.parameter_combo.blockSignals(False)

    def _start_model_task(self) -> None:
        if self._shutdown_in_progress:
            return
        if self._acquisition_active() or self._impedance_active():
            self._append_log("实验、录制或阻抗检测进行中不能启动算法任务")
            return
        session_dir = self._refresh_model_session()
        if session_dir is not None:
            session_dir = session_dir.resolve()
            try:
                load_model_input_session(session_dir)
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "目标会话不可分析", str(exc))
                return
        elif not self.dry_run_check.isChecked():
            QtWidgets.QMessageBox.warning(
                self,
                "暂无可分析会话",
                "请先完成一次实验并确认默想数字。系统会自动选择通过质量检查的会话。",
            )
            return
        selected_parameters = self.parameter_combo.currentData()
        arguments = (
            selected_parameters.arguments
            if isinstance(selected_parameters, ModelParameterOption)
            else ""
        )
        config = ModelTaskConfig(
            task=str(self.task_combo.currentData() or "train"),
            model_manifest=self.manifest_edit.text().strip(),
            python_executable=self.python_edit.text().strip(),
            entrypoint=self.entrypoint_edit.text().strip(),
            arguments=arguments,
            output_dir=self.output_dir_edit.text().strip() or "models/runs",
            dry_run=self.dry_run_check.isChecked(),
        )
        self.model_start.setEnabled(False)
        self.model_stop.setEnabled(True)
        self.model_progress.setValue(0)
        self.model_log.clear()
        self.model_runner.start(config, session_dir, self.mock, base_dir=self.app_root)

    def _on_model_finished(self, ok: bool, message: str) -> None:
        self._append_log(message)
        # Reconcile view permissions on the next Qt turn after the terminal receipt.
        QtCore.QTimer.singleShot(0, self._sync_action_state)

    def _on_model_state(self, state: str) -> None:
        self.model_state.setText(state)
        self._sync_action_state()

    def _choose_manifest(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择模型 manifest", str(self.app_root), "JSON (*.json);;所有文件 (*)")
        if path:
            self._load_manifest(Path(path))

    def _load_manifest(self, path: Path) -> None:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("manifest 顶层必须是 JSON 对象")
            if manifest.get("schema") != "brainsync-model-manifest/1":
                raise ValueError("manifest schema 不受支持")
            entrypoint = str(manifest.get("entrypoint", "")).strip()
            if not entrypoint:
                raise ValueError("manifest 缺少 entrypoint")
            if entrypoint.lower().endswith(".py") or "/" in entrypoint or "\\" in entrypoint:
                entrypoint_path = Path(entrypoint)
                if not entrypoint_path.is_absolute():
                    entrypoint_path = path.parent / entrypoint_path
                entrypoint = str(entrypoint_path.resolve())
            python_executable = str(manifest.get("python_executable", "")).strip()
            if python_executable:
                python_path = Path(python_executable).expanduser()
                if not python_path.is_absolute():
                    python_path = path.parent / python_path
                python_path = python_path.resolve()
                if not python_path.is_file():
                    raise ValueError(f"manifest 的 Python 解释器不存在: {python_path}")
                python_executable = str(python_path)
            tasks = [str(task) for task in manifest.get("tasks", [])]
            parameter_options = parse_model_parameter_options(manifest.get("parameter_options"))
            self.manifest_edit.setText(str(path))
            self.model_profile_edit.setText(str(manifest.get("name") or path.stem))
            self.entrypoint_edit.setText(entrypoint)
            self.python_edit.setText(python_executable)
            if tasks and self.task_combo.currentData() not in tasks:
                index = self.task_combo.findData(tasks[0])
                if index >= 0:
                    self.task_combo.setCurrentIndex(index)
            self._model_parameter_options = parameter_options
            self._refresh_parameter_menu()
            self._append_log(f"已加载模型 manifest: {manifest.get('name', path.stem)}")
        except Exception as exc:
            self._append_log(f"模型 manifest 加载失败: {exc}")

    def _edit_montage(self) -> None:
        if (self._shutdown_in_progress or self._device_transition_active() or self._acquisition_active()
                or self._impedance_active() or self.device.state is not DeviceState.READY):
            self._append_log("EEG 流、实验、录制或阻抗检测进行中不能编辑通道映射")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("编辑通道映射")
        dialog.resize(520, 460)
        layout = QtWidgets.QVBoxLayout(dialog)
        table = QtWidgets.QTableWidget(8, 3)
        table.setHorizontalHeaderLabels(["启用", "物理位置", "类型"])
        table.horizontalHeader().setStretchLastSection(True)
        for index, label in enumerate(self.montage.eeg):
            enabled = QtWidgets.QTableWidgetItem()
            enabled.setCheckState(
                QtCore.Qt.CheckState.Checked
                if self.montage.active_mask & (1 << index)
                else QtCore.Qt.CheckState.Unchecked
            )
            table.setItem(index, 0, enabled)
            table.setItem(index, 1, QtWidgets.QTableWidgetItem(label))
            role = QtWidgets.QTableWidgetItem(self.montage.channel_types[index].upper())
            role.setFlags(role.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            table.setItem(index, 2, role)
        layout.addWidget(table)
        form = QtWidgets.QFormLayout()
        ref_edit = QtWidgets.QLineEdit(self.montage.ref_label)
        gnd_edit = QtWidgets.QLineEdit(self.montage.gnd_label)
        form.addRow("REF", ref_edit)
        form.addRow("GND", gnd_edit)
        layout.addLayout(form)
        hint = QtWidgets.QLabel("物理电极标签必须与实际线束位置一致；保存后用于 EDF 元数据。")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        if (self._shutdown_in_progress or self._device_transition_active() or self._acquisition_active()
                or self._impedance_active() or self.device.state is not DeviceState.READY):
            self._append_log("编辑期间设备状态已改变，通道配置未保存")
            return
        labels = tuple(table.item(index, 1).text().strip() for index in range(8))
        active_mask = sum(
            (1 << index)
            for index in range(8)
            if table.item(index, 0).checkState() == QtCore.Qt.CheckState.Checked
        )
        try:
            positions = (
                self.montage.channel_positions_m
                if labels == self.montage.eeg
                else default_channel_positions_m(labels)
            )
            reference = ref_edit.text().strip()
            derivation = {}
            for index in self.montage.eog_indices:
                declaration = self.montage.ocular_derivation.get(self.montage.eeg[index])
                if declaration is None:
                    continue
                declaration = dict(declaration)
                if reference != self.montage.ref_label and declaration["kind"] == "single_ended":
                    if declaration["reference"] != self.montage.ref_label:
                        raise ValueError("Single-ended EOG reference is not bound to the current montage REF.")
                    declaration["reference"] = reference
                derivation[labels[index]] = declaration
            montage = ChannelMontage(
                labels,
                reference,
                gnd_edit.text().strip(),
                active_mask,
                self.montage.stim,
                positions,
                self.montage.channel_types,
                eog_derivation=derivation,
            )
            montage.validate()
            atomic_write_json(self.montage_path, montage.to_dict())
            self.montage = montage
            self.channel_labels = list(montage.eeg)
            self.eeg_filter.set_channel_types(montage.channel_types)
            for index, label in enumerate(self.channel_labels):
                self.channel_table.item(index, 1).setText(label)
                self.channel_table.item(index, 2).setText(montage.channel_types[index].upper())
            self._update_wave_plot_labels()
            self._clear_waveforms()
            self._reset_impedance_display()
            self._on_device_state(self.device.snapshot)
            self.montage_path_label.setText(str(self.montage_path))
            self._append_log("通道映射已保存")
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "通道映射无效", str(exc))

    def _choose_python(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 Python", str(self.app_root), "Python (*.exe);;所有文件 (*)")
        if path:
            self.python_edit.setText(path)

    def _choose_analysis_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择会话目录", str(self.store.root))
        if path:
            self.analysis_path.setText(path)

    def _open_analysis_dir(self) -> None:
        path = Path(self.analysis_path.text().strip()).expanduser().resolve()
        if path.is_dir():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
        else:
            self.analysis_summary.setText("目录不存在")

    def _recover_analysis(self) -> None:
        path = Path(self.analysis_path.text().strip()).expanduser().resolve()
        try:
            SessionStore.recover_incomplete(path)
            self._load_analysis()
            self._append_log(f"会话恢复已完成: {path}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.analysis_summary.setText(f"恢复失败: {exc}")

    def _load_analysis(self) -> None:
        self.analysis_recover_button.setEnabled(False)
        self.analysis_table.setRowCount(0)
        path = Path(self.analysis_path.text().strip()).expanduser().resolve()
        session_file = path / "session.json"
        events_file = path / "events" / "events.jsonl"
        if not session_file.exists():
            self.analysis_summary.setText("未找到 session.json")
            return
        try:
            session = json.loads(session_file.read_text(encoding="utf-8"))
            if not isinstance(session, dict):
                raise ValueError("session.json 顶层必须是对象")
            rows: deque[dict[str, Any]] = deque(maxlen=MAX_ANALYSIS_EVENT_ROWS)
            event_count = 0
            invalid_rows = 0
            if events_file.exists():
                with events_file.open(encoding="utf-8") as fp:
                    for line in fp:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                if not isinstance(item, dict):
                                    invalid_rows += 1
                                    continue
                                event_count += 1
                                rows.append(item)
                            except json.JSONDecodeError:
                                invalid_rows += 1
            self.analysis_table.setRowCount(0)
            for item in rows:
                row = self.analysis_table.rowCount()
                self.analysis_table.insertRow(row)
                self.analysis_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(item.get("timestamp_utc", ""))))
                self.analysis_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(item.get("event", ""))))
                self.analysis_table.setItem(row, 2, QtWidgets.QTableWidgetItem(json.dumps(item.get("payload", {}), ensure_ascii=False)))
                for column in range(3):
                    cell = self.analysis_table.item(row, column)
                    cell.setToolTip(cell.text())
            experiment = session.get("experiment", {})
            if not isinstance(experiment, dict):
                experiment = {}
            recording = session.get("recording", {})
            if not isinstance(recording, dict):
                recording = {}
            timeline = recording.get("timeline", {})
            if not isinstance(timeline, dict):
                timeline = {}
            quality = session.get("quality", {})
            continuity = quality.get("eeg_continuity", {}) if isinstance(quality, dict) else {}
            if not isinstance(continuity, dict):
                continuity = {}
            recording_state = (
                "BIDS raw，待算法预处理"
                if recording.get("stage") == BRAIN_SYNC_STORAGE_CONTRACT.raw_stage
                else "原始录制未完成"
            )
            recording_path = recording.get("path", "--")
            timeline_state = timeline.get("status", "--")
            continuity_state = "通过" if continuity.get("passed") is True else "未通过"
            target_label = session.get("target_label", {})
            if not isinstance(target_label, dict):
                target_label = {}
            target_state = "已确认" if target_label.get("status") == "confirmed" else "未确认"
            target_digit = target_label.get("thought_digit", "--")
            model_tasks = path / "analysis" / "model_tasks.jsonl"
            model_task_count = 0
            invalid_model_rows = 0
            if model_tasks.exists():
                with model_tasks.open(encoding="utf-8") as fp:
                    for line in fp:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                            if isinstance(item, dict):
                                model_task_count += 1
                            else:
                                invalid_model_rows += 1
                        except json.JSONDecodeError:
                            invalid_model_rows += 1
            self.analysis_summary.setText(
                f"会话 {session.get('session_id', '--')} · 状态 {session.get('status', 'unknown')} · "
                f"被试 {experiment.get('subject_id', '--')} · 事件 {event_count} 条（显示最近 {len(rows)} 条） · "
                f"EEG {recording_path} · 时间轴 {timeline_state} · 连续性 {continuity_state} · "
                f"默想数字 {target_digit}（{target_state}） · {recording_state} · 模型记录 {model_task_count} 条"
                + (f" · 忽略损坏事件行 {invalid_rows} 条" if invalid_rows else "")
                + (f" · 忽略损坏模型行 {invalid_model_rows} 条" if invalid_model_rows else "")
            )
            self.analysis_recover_button.setEnabled(
                session.get("status") == "running" or (path / CONFIRMATION_PENDING).exists())
        except Exception as exc:
            self.analysis_summary.setText(f"加载失败: {exc}")

    def _append_log(self, message: str) -> None:
        logging.getLogger(__name__).info("%s", message)
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        if hasattr(self, "overview_log"):
            self.overview_log.appendPlainText(line)
        if hasattr(self, "protocol_log"):
            self.protocol_log.appendPlainText(line)

    def _on_device_error(self, message: str) -> None:
        """Log device errors AND persist them, so post-mortem analysis can
        distinguish e.g. a packet-conversion failure from a BLE stall."""

        self._append_log(message)
        if self.store.active:
            try:
                self.store.event("device_error", {"error": str(message)})
            except Exception:
                pass

    def _on_acquisition_fault(self, message: str) -> None:
        self.recording.fault(str(message))
        if self.store.active:
            try:
                self.store.event("recording_integrity_failure", {"error": str(message)})
            finally:
                self.experiment.stop()
                self._schedule_experiment_finish("recording_error")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._close_ready:
            event.accept()
            return
        event.ignore()
        self._shutdown_in_progress = True
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(self._shutdown_and_close())

    async def _cancel_impedance_detection(self) -> None:
        task = self._impedance_task_handle
        if task is not None and not task.done():
            self._impedance_operation_pending = True
            if not self._impedance_cancel_requested:
                self._impedance_cancel_requested = True
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._impedance_task_handle = None
        if self.device.state is DeviceState.IMPEDANCE or self.device.recovery_may_have_impedance:
            try:
                await self.device.stop_impedance()
            except Exception as exc:
                self._append_log(f"停止阻抗检测失败: {exc}")
        self._impedance_operation_pending = False

    async def _shutdown_and_close(self) -> None:
        await self.shutdown()
        self.close()

    async def shutdown(self) -> None:
        """Own every asynchronous task until it is safe to close the Qt loop."""
        self._shutdown_in_progress = True
        if self._shutdown_task is None:
            self.device_monitor_timer.stop()
            self.plot_timer.stop()
            if self.preview_timer is not None:
                self.preview_timer.stop()
            self._sync_action_state()
            self._shutdown_task = asyncio.create_task(self._shutdown())
        await asyncio.shield(self._shutdown_task)

    async def _shutdown(self) -> None:
        if self._thought_digit_dialog is not None:
            self._thought_digit_dialog.reject()
        for task in (
            self._connection_task_handle,
            self._device_config_task_handle,
            self._stream_task_handle,
            self._device_monitor_task,
        ):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (
                self._connection_task_handle,
                self._device_config_task_handle,
                self._stream_task_handle,
                self._device_monitor_task,
            ) if task is not None),
            return_exceptions=True,
        )
        if self._experiment_prepare_task is not None and not self._experiment_prepare_task.done():
            self._experiment_prepare_task.cancel()
            await asyncio.gather(self._experiment_prepare_task, return_exceptions=True)
        if self.experiment.running:
            self.experiment.stop()
            await self.experiment.wait()
        if self._experiment_finish_task is not None and not self._experiment_finish_task.done():
            # A normal finish owns the same device resources that shutdown
            # must release. Let it finish first so closing the main window
            # cannot discard an in-memory callback recording.
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._experiment_finish_task),
                    timeout=EXPERIMENT_FINISH_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                self._append_log("实验收尾超时，开始执行关闭时录制兜底清理")
                self._experiment_finish_task.cancel()
                await asyncio.gather(self._experiment_finish_task, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._append_log(f"实验收尾异常，开始执行关闭时录制兜底清理: {exc}")
        thought_digit_task = getattr(self, "_thought_digit_task", None)
        if thought_digit_task is not None and not thought_digit_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(thought_digit_task),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                thought_digit_task.cancel()
                await asyncio.gather(thought_digit_task, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._append_log(f"关闭时默想数字确认失败: {exc}")
        await self._cancel_impedance_detection()
        quality_reports = getattr(self, "quality_reports", None)
        if quality_reports is not None:
            await quality_reports.shutdown()
        await self.model_runner.shutdown()
        await self.recording.finish("aborted")
        self._experiment_finish_task = None
        self._dispose_stimulus_window()
        await self.device.wait_for_connection_cleanup()
        try:
            await self.device.disconnect()
        except Exception as exc:
            self._append_log(f"关闭时断开设备失败: {exc}")
        self._close_ready = True
        atexit.unregister(self._emergency_persist_recording)
