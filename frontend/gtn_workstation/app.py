from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


class RuntimeDependencyError(RuntimeError):
    """Raised when the selected Python runtime cannot load the UI dependencies."""


_FROZEN_DLL_HANDLES: list[object] = []


def _prepare_frozen_dll_paths() -> None:
    """Register bundled native dependency directories before importing Qt.

    PySide6's own runtime hook covers normal layouts, but a shortcut can launch
    the onedir executable with an unrelated working directory. Registering the
    sibling DLL directories here makes QtCore and shiboken6 resolution depend on
    the executable layout rather than the caller's PATH.
    """

    if not getattr(sys, "frozen", False):
        return
    # A frozen release must not inherit another Python/Qt installation from a
    # terminal, IDE, or user-level launcher. Those variables can redirect
    # embedded imports and produce a misleading QtCore DLL error.
    for variable in (
        "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "QT_PLUGIN_PATH",
        "QML2_IMPORT_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH",
    ):
        os.environ.pop(variable, None)
    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent / "_internal"))
    directories = (
        base,
        base / "shiboken6",
        base / "PySide6",
        base / "numpy.libs",
        base / "scipy.libs",
    )
    for directory in directories:
        if not directory.is_dir():
            continue
        value = str(directory)
        if hasattr(os, "add_dll_directory"):
            _FROZEN_DLL_HANDLES.append(os.add_dll_directory(value))
        if value not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = value + os.pathsep + os.environ.get("PATH", "")


def application_root() -> Path:
    """Return the durable data/config root for source and frozen launches."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def configure_diagnostic_logging(directory: Path | None = None) -> Path:
    directory = directory or Path(__file__).resolve().parent.parent / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "workstation.log"
    logger = logging.getLogger("gtn_workstation")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return path


def _declared_python_minor(candidate: Path) -> tuple[int, int] | None:
    """Read the interpreter version a bundled/virtualenv runtime was built for."""
    pyvenv_cfg = candidate.parent.parent / "pyvenv.cfg"
    if pyvenv_cfg.is_file():
        text = pyvenv_cfg.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"version_info\s*=\s*(\d+)\.(\d+)", text)
        if match:
            return int(match.group(1)), int(match.group(2))

    if candidate.name == "_internal":
        for dll in candidate.glob("python3*.dll"):
            match = re.fullmatch(r"python(\d)(\d+)\.dll", dll.name.lower())
            if match:
                return int(match.group(1)), int(match.group(2))
    return None


def _runtime_path_is_compatible(candidate: Path) -> bool:
    declared = _declared_python_minor(candidate)
    return declared is None or declared == sys.version_info[:2]


def _resolved_sys_path_entry(entry: object) -> Path | None:
    if not isinstance(entry, str) or not entry:
        return None
    try:
        return Path(entry).expanduser().resolve()
    except (OSError, RuntimeError):
        return None


def _remove_runtime_path(candidate: Path) -> None:
    """Remove a known incompatible SDK path that was injected before startup."""
    for entry in list(sys.path):
        if _resolved_sys_path_entry(entry) == candidate:
            sys.path.remove(entry)


def configure_runtime_paths(explicit: str | None = None) -> list[Path]:
    """Use the selected interpreter; external SDK locations require an explicit override."""
    value = explicit or os.environ.get("BRAIN_SYNC_SDK_SITE_PACKAGES", "").strip()
    if not value:
        return []
    candidate = Path(value).expanduser().resolve()
    if not candidate.is_dir():
        raise RuntimeDependencyError(f"显式 SDK 依赖目录不存在: {candidate}")
    if not _runtime_path_is_compatible(candidate):
        _remove_runtime_path(candidate)
        raise RuntimeDependencyError(f"显式 SDK 依赖目录与当前 Python 不兼容: {candidate}")
    _remove_runtime_path(candidate)
    sys.path.insert(0, str(candidate))
    logging.getLogger("gtn_workstation").warning("Explicit SDK environment override: %s", candidate)
    return [candidate]


def build_parser() -> argparse.ArgumentParser:
    from .transport import TRANSPORTS

    parser = argparse.ArgumentParser(description="BrainSync GTN/P300 research workstation")
    parser.add_argument("--mock", action="store_true", help="使用模拟设备，不连接硬件")
    parser.add_argument("--fullscreen", action="store_true", help="主控制台全屏显示")
    parser.add_argument("--screen", type=int, default=None, help="默认被试刺激屏幕编号")
    parser.add_argument("--transport", choices=[spec.key for spec in TRANSPORTS], help="覆盖保存的连接方式")
    parser.add_argument("--address", help="串口 COM 名称或 BLE 设备名称；留空自动发现")
    parser.add_argument(
        "--sdk-site-packages",
        default=None,
        help="指定包含 brainsync_sdk、PySide6 和 qasync 的 site-packages 目录",
    )
    return parser


def load_application_font() -> None:
    from PySide6 import QtGui

    font_path = Path(__file__).resolve().parent.parent / "resources" / "fonts" / "AlibabaPuHuiTi-2-45-Light.ttf"
    if font_path.exists():
        QtGui.QFontDatabase.addApplicationFont(str(font_path))
    if os.name == "nt":
        system_font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
        if system_font.is_file():
            QtGui.QFontDatabase.addApplicationFont(str(system_font))


def _load_runtime_dependencies(require_sdk: bool) -> tuple[object, object]:
    try:
        _prepare_frozen_dll_paths()
        import numpy  # noqa: F401
        import qasync
        from PySide6 import QtWidgets
        if require_sdk:
            import brainsync_sdk  # noqa: F401
        return QtWidgets, qasync
    except (ImportError, OSError) as exc:
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeDependencyError(
            "当前 Python 运行时与 BrainSync SDK 依赖不兼容或未安装。"
            f"当前解释器: {sys.executable} (Python {version})。"
            "请在工作站目录执行 uv sync --locked，"
            "再用本目录 .venv/Scripts/python.exe 运行 run_gtn_workstation.py。"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = configure_diagnostic_logging()
    logger = logging.getLogger("gtn_workstation")
    logger.info("Starting workstation mock=%s python=%s", args.mock, sys.executable)
    configure_runtime_paths(args.sdk_site_packages)

    QtWidgets, qasync = _load_runtime_dependencies(require_sdk=not args.mock)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("BrainSync GTN Workstation")
    app.setOrganizationName("BrainSync")
    load_application_font()
    from .ui import MainWindow

    window = MainWindow(
        mock=args.mock,
        transport=args.transport,
        address=args.address,
    )
    window._append_log(f"运行日志: {log_path}")
    if args.screen is not None:
        window.screen_spin.setValue(max(0, args.screen))
    if args.fullscreen:
        window.showFullScreen()
    else:
        window.show()

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    def report_async_error(_loop, context):
        error = context.get("exception")
        logger.error("Async error: %s", context.get("message"), exc_info=(type(error), error, error.__traceback__) if error else None)

    loop.set_exception_handler(report_async_error)
    with loop:
        try:
            loop.run_forever()
        finally:
            # app.quit() and startup/runtime errors can bypass closeEvent.
            # IOCP subprocess handles must be reaped while the loop can run.
            loop.run_until_complete(window.shutdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
