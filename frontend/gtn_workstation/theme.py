from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

COLORS = {
    "bg": "#f3f6f7", "panel": "#ffffff", "panel_alt": "#edf2f3",
    "line": "#d4dfe2", "muted": "#5c6e77", "text": "#24343c",
    "teal": "#087f72", "blue": "#326fb2", "amber": "#95611d",
    "red": "#be414b", "green": "#197d55", "sidebar": "#19282c",
}
PLOT_COLORS = {
    "bg": "#19282c", "line": "#405157", "text": "#b7c9cc",
    "teal": "#64dec4", "blue": "#83b6eb", "amber": "#edc47f",
}
ICON_ROOT = Path(__file__).resolve().parents[1] / "resources" / "icons"


@lru_cache(maxsize=64)
def asset_icon(name: str, color: str = COLORS["muted"]) -> QtGui.QIcon:
    original = QtGui.QIcon(str(ICON_ROOT / f"{name}.svg"))
    icon = QtGui.QIcon()
    for state, tint in ((QtGui.QIcon.State.Off, color), (QtGui.QIcon.State.On, "#75e4cd")):
        for size in (16, 20, 24, 32):
            pixmap = original.pixmap(size * 2, size * 2)
            painter = QtGui.QPainter(pixmap)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceIn)
            painter.fillRect(pixmap.rect(), QtGui.QColor(tint))
            painter.end()
            pixmap.setDevicePixelRatio(2)
            icon.addPixmap(pixmap, QtGui.QIcon.Mode.Normal, state)
    return icon


def tool_button(name: str, tooltip: str, callback) -> QtWidgets.QToolButton:
    button = QtWidgets.QToolButton()
    button.setIcon(asset_icon(name))
    button.setIconSize(QtCore.QSize(18, 18))
    button.setToolTip(tooltip)
    button.setAccessibleName(tooltip)
    button.setFixedSize(34, 34)
    button.clicked.connect(callback)
    return button


def workspace_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    roles = {"Window": COLORS["bg"], "WindowText": COLORS["text"], "Base": "#ffffff",
             "AlternateBase": "#f4f8f9", "Text": COLORS["text"], "Button": "#ffffff",
             "ButtonText": COLORS["text"], "Highlight": "#dcefe9", "HighlightedText": "#174f45",
             "PlaceholderText": "#83939a", "Light": "#ffffff", "Mid": "#b4c5ca", "Dark": "#5c6e77"}
    for role, color in roles.items():
        palette.setColor(getattr(QtGui.QPalette.ColorRole, role), QtGui.QColor(color))
    return palette


def application_stylesheet(font_size: int = 13) -> str:
    return f"""
QMainWindow, QWidget {{
    background: {COLORS['bg']}; color: {COLORS['text']};
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Alibaba PuHuiTi 2.0", "Segoe UI";
    font-size: {font_size}px; letter-spacing: 0px;
}}
QLabel {{ background: transparent; border: none; }}
QWidget#workspace {{ background: {COLORS['bg']}; }}
QFrame#sidebar {{ background: {COLORS['sidebar']}; border-right: 1px solid #26393e; }}
QFrame#sidebar QLabel {{ color: #adc1c6; }}
QFrame#sidebar QLabel#brand {{ color: #f5fbfc; font-size: 24px; font-weight: 600; }}
QFrame#sidebar QLabel#eyebrow {{ color: #90a8b0; font-size: 11px; }}
QFrame#sidebar QLabel#mode {{ color: #b8ddd6; border-top: 1px solid #34464b; padding: 14px 0 0 0; }}
QFrame#topbar {{ background: transparent; border-bottom: 1px solid {COLORS['line']}; }}
QLabel#page_title {{ font-size: 23px; font-weight: 600; color: #22373d; }}
QLabel#section_title {{ font-size: 15px; font-weight: 600; color: #294149; }}
QLabel#muted {{ color: {COLORS['muted']}; }}
QLabel#status {{ background: #e9eeee; border-radius: 4px; padding: 5px 10px; color: {COLORS['amber']}; font-weight: 600; }}
QFrame#panel {{ background: transparent; border: none; }}
QFrame#metric {{ background: white; border: 1px solid {COLORS['line']}; border-radius: 6px; }}
QFrame#metric QLabel#metric_title {{ font-size: 12px; color: {COLORS['muted']}; }}
QGroupBox {{
    background: transparent; border: none; border-top: 1px solid {COLORS['line']};
    margin-top: 9px; padding: 8px 0 0 0; font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 0; padding: 0 10px 0 0; color: #304951; }}
QPushButton {{
    background: white; color: {COLORS['text']}; border: 1px solid #cbd8dc;
    border-radius: 5px; padding: 7px 12px; min-height: 18px;
}}
QPushButton:hover {{ background: #edf7f4; border-color: #55a79a; }}
QPushButton:pressed {{ background: #dceee8; }}
QPushButton:focus, QToolButton:focus {{ border: 1px solid {COLORS['teal']}; }}
QPushButton:disabled {{ background: #e9eef0; border-color: #d9e2e5; color: #8b9a9f; }}
QPushButton#primary {{ background: {COLORS['teal']}; color: white; border-color: {COLORS['teal']}; font-weight: 600; }}
QPushButton#primary:hover {{ background: #096b61; border-color: #096b61; }}
QPushButton#primary:disabled {{ background: #dce7e5; color: #839a95; border-color: #dce7e5; }}
QPushButton#danger {{ background: #fff5f4; border-color: #e1b3b4; color: #ac3f49; }}
QPushButton#danger:disabled {{ background: #f0ebec; border-color: #e6dddd; color: #a4979a; }}
QPushButton#nav {{
    text-align: left; padding: 11px 14px; border: 1px solid transparent;
    background: transparent; color: #b2c5ca; border-radius: 5px; min-height: 22px;
}}
QPushButton#nav:hover {{ background: #263b40; color: white; }}
QPushButton#nav:checked {{ background: #284b4b; color: #95edd8; border-color: #3c6861; font-weight: 600; }}
QPushButton#workflow_action {{
    text-align: left; background: transparent; border: none; border-bottom: 1px solid {COLORS['line']};
    border-radius: 0; padding: 16px 8px; font-size: 14px;
}}
QPushButton#workflow_action:hover {{ background: #e6efed; color: {COLORS['teal']}; }}
QToolButton {{ background: white; border: 1px solid #cedcdf; border-radius: 5px; padding: 4px; }}
QToolButton:hover, QToolButton:checked {{ background: #e3f1ed; border-color: #80b9ad; }}
QToolButton:disabled {{ background: #edf1f2; border-color: #dce4e6; }}
QToolButton::menu-indicator {{ image: none; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
    background: white; color: {COLORS['text']}; border: 1px solid #cedbdf;
    border-radius: 4px; padding: 6px 8px; selection-background-color: #c7e9de; selection-color: #18483e;
}}
QLineEdit:read-only {{ background: #eef3f4; color: #60737b; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: #8b9a9f; background: #eaf0f1; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {{ border-color: {COLORS['teal']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url("{(ICON_ROOT / 'theme/chevron-down.png').as_posix()}"); width: 10px; height: 10px; }}
QSpinBox, QDoubleSpinBox {{ padding-right: 24px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right;
    width: 21px; background: #edf3f5; border-left: 1px solid #d5e1e5; border-top-right-radius: 4px; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right;
    width: 21px; background: #edf3f5; border-left: 1px solid #d5e1e5; border-bottom-right-radius: 4px; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{(ICON_ROOT / 'theme/chevron-up.png').as_posix()}"); width: 9px; height: 9px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{(ICON_ROOT / 'theme/chevron-down.png').as_posix()}"); width: 9px; height: 9px; }}
QComboBox QAbstractItemView {{ background: white; color: {COLORS['text']}; selection-background-color: #d9eee7; outline: none; padding: 4px; }}
QCheckBox {{ spacing: 7px; background: transparent; }}
QCheckBox::indicator, QTableWidget::indicator {{ width: 16px; height: 16px; border: 1px solid #b7cbd0; background: white; border-radius: 3px; }}
QCheckBox::indicator:checked, QTableWidget::indicator:checked {{ background: {COLORS['teal']}; border-color: {COLORS['teal']}; image: url("{(ICON_ROOT / 'theme/check.png').as_posix()}"); }}
QCheckBox::indicator:disabled {{ background: #dde7e8; border-color: #cad9dc; }}
QCheckBox:disabled {{ color: #8b9a9f; }}
QProgressBar {{ background: #e3ebed; border: none; border-radius: 4px; text-align: center; color: #29423e; height: 16px; }}
QProgressBar::chunk {{ background: #8cd6bc; border-radius: 4px; }}
QTableWidget {{ background: white; alternate-background-color: #f4f8f9; color: {COLORS['text']};
    gridline-color: #e2eaed; border: 1px solid {COLORS['line']}; border-radius: 5px;
    selection-background-color: #dcefe9; selection-color: #174f45; outline: none; }}
QTableWidget::item {{ padding: 4px 6px; border-bottom: 1px solid #edf2f4; }}
QHeaderView::section {{ background: #eaf1f3; color: #506872; border: none; border-bottom: 1px solid #d5e1e5; padding: 8px 6px; font-weight: 600; }}
QTabWidget::pane {{ border: none; border-top: 1px solid {COLORS['line']}; top: -1px; }}
QTabBar::tab {{ background: transparent; color: #5d737c; padding: 10px 18px; margin-right: 12px; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {COLORS['teal']}; border-bottom: 2px solid {COLORS['teal']}; font-weight: 600; }}
QTabBar::tab:hover {{ background: #e8f1ef; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QScrollBar:vertical {{ width: 8px; background: transparent; margin: 2px; }}
QScrollBar:horizontal {{ height: 8px; background: transparent; margin: 2px; }}
QScrollBar::handle {{ background: #b5c6cb; border-radius: 3px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: #8ba9ae; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QMenu {{ background: white; color: {COLORS['text']}; border: 1px solid {COLORS['line']}; padding: 5px; }}
QMenu::item {{ padding: 8px 24px 8px 12px; border-radius: 3px; }}
QMenu::item:selected {{ background: #dfeee9; color: #125f51; }}
QToolTip {{ background: #20373e; color: #f4fbfc; border: none; padding: 6px; }}
"""


APP_QSS = application_stylesheet()
