from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PySide6 import QtCore, QtGui, QtWidgets

from .storage import atomic_write_json
from .theme import asset_icon

REPORT_SCHEMA = "n2p3_recording_quality_report/1"
INDEX_SCHEMA = "brainsync-quality-report-job/1"


def report_index_path(session: Path) -> Path:
    return session / "analysis" / "quality_report.json"


def read_report_index(session: Path) -> dict:
    try:
        record = json.loads(report_index_path(session).read_text(encoding="utf-8"))
        return record if isinstance(record, dict) and record.get("schema") == INDEX_SCHEMA else {}
    except (OSError, ValueError):
        return {}


def report_file(session: Path, record: dict) -> Path:
    base = (session / "analysis" / "quality_reports").resolve()
    path = (session / str(record.get("report", ""))).resolve()
    if not path.is_relative_to(base) or path.name != "report.json":
        raise ValueError("质量报告路径超出当前会话")
    return path


def _declared_session_id(session: Path) -> str | None:
    try:
        record = json.loads((session / "session.json").read_text(encoding="utf-8-sig"))
        identity = record.get("session_id") if isinstance(record, dict) else None
        return identity if isinstance(identity, str) and identity.strip() else None
    except (OSError, ValueError):
        return None


def _validate_report(report, session: Path, expected_session_id: str | None = None) -> None:
    if (not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA
            or Path(report["session_dir"]).resolve() != session):
        raise ValueError("质量报告与请求会话不一致")
    identity = _declared_session_id(session)
    if expected_session_id is not None and identity != expected_session_id:
        raise ValueError("报告任务期间会话身份已改变")
    if report.get("status") not in {"completed", "attention", "not_evaluable"}:
        raise ValueError("质量报告状态无效")
    # Contract failures deliberately produce a diagnostic report whose ID may
    # be the directory name because the backend could not admit the manifest.
    if report["status"] != "not_evaluable" and identity is not None and report.get("session_id") != identity:
        raise ValueError("质量报告与录制会话身份不一致")


class QualityReportController(QtCore.QObject):
    changed = QtCore.Signal(str)
    message = QtCore.Signal(str)

    def __init__(self, runner, request_factory, can_start=lambda: True, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.request_factory = request_factory
        self.can_start = can_start
        self.queue = deque()
        self.active = None
        self.closed = False
        self.runner.progress_changed.connect(self._progress)
        self.runner.output_line.connect(self._output)
        self.runner.finished.connect(self._finished)

    def status(self, session: Path) -> dict:
        session = session.resolve()
        if self.active is not None and self.active["session"] == session:
            return self.active["record"]
        record = read_report_index(session)
        pending = self.active is not None and self.active["session"] == session or any(job["session"] == session for job in self.queue)
        if record.get("status") in {"queued", "running"} and not pending:
            return {**record, "status": "interrupted", "message": "上次报告任务已中断，可重新生成"}
        return record

    def _retry_request(self, session: Path, previous: dict) -> dict:
        if "request" in previous:
            return previous["request"]
        request = self.request_factory(session)
        if not isinstance(request, dict):
            raise ValueError("质量报告请求必须是对象")
        request = dict(request)
        # Earlier indexes omitted the command, but completed backend reports
        # retain their processing options. Replay those instead of current UI settings.
        try:
            report = json.loads(report_file(session, previous).read_text(encoding="utf-8"))
            if (not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA
                    or Path(report["session_dir"]).resolve() != session):
                return request
            saved = report.get("request")
            if isinstance(saved, dict) and saved:
                arguments = []
                for name in ("model_artifact", "continuous_qc_model", "epoch_qc_model", "ocular_model",
                             "preprocessing_config", "reference_channel", "reference_mode"):
                    if saved.get(name) is not None:
                        arguments.extend([f"--{name.replace('_', '-')}", str(saved[name])])
                request["arguments"] = arguments
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return request

    def enqueue(self, session: Path, request=None, *, force=False) -> None:
        session = session.resolve()
        if self.closed or not (session / "session.json").is_file():
            return
        if self.active is not None and self.active["session"] == session or any(job["session"] == session for job in self.queue):
            return
        previous = self.status(session)
        if previous.get("status") == "completed" and not force:
            try:
                if report_file(session, previous).is_file():
                    self.changed.emit(str(session))
                    return
            except ValueError:
                pass
        output = session / "analysis" / "quality_reports" / f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        record = {"schema": INDEX_SCHEMA, "status": "queued", "progress": 0,
                  "message": "等待生成数据质量报告", "report": (output / "report.json").relative_to(session).as_posix(),
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "session_id": _declared_session_id(session)}
        job = {"session": session, "output": output, "record": record}
        try:
            request = request if request is not None else self._retry_request(session, previous)
            record["request"] = request
            if not isinstance(request, dict):
                raise ValueError("质量报告请求必须是对象")
            arguments = request.get("arguments", [])
            if not isinstance(arguments, (list, tuple)) or any(not isinstance(item, str) for item in arguments):
                raise ValueError("质量报告 arguments 必须是字符串列表")
            request = {**request, "arguments": list(arguments)}
            record["request"] = job["request"] = request
        except (OSError, ValueError, RuntimeError, TypeError, KeyError, AttributeError) as error:
            record.update(status="failed", message=f"质量报告请求准备失败：{error}")
            self.message.emit(record["message"])
            try:
                self._publish(job)
            except (OSError, ValueError, TypeError) as publish_error:
                self.message.emit(f"质量报告失败状态保存失败: {publish_error}")
            self.changed.emit(str(session))
            return
        try:
            self._publish(job)
        except (OSError, ValueError, TypeError) as error:
            self.message.emit(f"质量报告任务保存失败: {error}")
            return
        self.queue.append(job)
        self.changed.emit(str(session))
        QtCore.QTimer.singleShot(0, self.pump)

    def _publish(self, job) -> None:
        atomic_write_json(report_index_path(job["session"]), job["record"])

    def pump(self) -> None:
        if self.closed or self.active is not None or self.runner.running or not self.queue or not self.can_start():
            return
        job = self.queue.popleft()
        self.active = job
        try:
            request = job["request"]
            backend = Path(request["backend_root"]).expanduser().resolve()
            script = backend / "experiments" / "report_brainsync_quality.py"
            if not script.is_file():
                raise ValueError("算法项目中未找到数据质量报告入口")
            arguments = [str(script), "--session-dir", str(job["session"]), "--output-dir", str(job["output"]),
                         *request.get("arguments", [])]
            job["record"].update(status="running", message="正在检查录制数据")
            self._publish(job)
            self.runner.start_backend(str(backend / ".venv" / "Scripts" / "python.exe"), arguments,
                                      task="quality", output_dir=job["output"])
            self.changed.emit(str(job["session"]))
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError) as error:
            self._finished(False, str(error))

    def _progress(self, value) -> None:
        if self.active is not None:
            self.active["record"]["progress"] = value
            self.changed.emit(str(self.active["session"]))

    def _output(self, line) -> None:
        if self.active is not None and line.strip() and not line.startswith("PROGRESS"):
            self.active["last_output"] = line.strip()
            self.active["record"]["message"] = line.strip()
            self.changed.emit(str(self.active["session"]))

    def _finished(self, ok, message) -> None:
        job = self.active
        if job is None:
            return
        self.active = None
        try:
            if ok:
                report = json.loads((job["output"] / "report.json").read_text(encoding="utf-8"))
                _validate_report(report, job["session"], job["record"].get("session_id"))
                job["record"].update(status="completed", progress=100, message=report["headline"],
                                     report_status=report["status"])
            else:
                if job.get("last_output") and not self.runner.cancel_requested:
                    message = f"{message}：{job['last_output']}"
                job["record"].update(status="interrupted" if self.runner.cancel_requested else "failed",
                                     message=message)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            job["record"].update(status="failed", message=str(error))
        try:
            self._publish(job)
        except OSError as error:
            self.message.emit(f"质量报告状态保存失败: {error}")
        self.changed.emit(str(job["session"]))
        QtCore.QTimer.singleShot(0, self.pump)

    def cancel(self, session: Path) -> None:
        session = session.resolve()
        if self.active is not None and self.active["session"] == session:
            self.runner.stop()
        for job in tuple(self.queue):
            if job["session"] == session:
                self.queue.remove(job)
                job["record"].update(status="interrupted", message="报告任务已取消")
                try:
                    self._publish(job)
                except OSError as error:
                    self.message.emit(f"质量报告取消状态保存失败: {error}")
                self.changed.emit(str(session))

    async def shutdown(self) -> None:
        self.closed = True
        try:
            for job in tuple(self.queue):
                self.cancel(job["session"])
        finally:
            await self.runner.shutdown()


class ReportImage(QtWidgets.QLabel):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.original = QtGui.QPixmap(str(path))
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(1)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        if self.original.isNull():
            self.setText("图表文件不可用")
            self.setFixedHeight(44)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.original.isNull() and self.width() > 1:
            image = self.original.scaledToWidth(min(self.width(), 1200), QtCore.Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(image)
            if self.height() != image.height():
                self.setFixedHeight(image.height())


class QualityReportDialog(QtWidgets.QDialog):
    def __init__(self, session: Path, controller: QualityReportController, parent=None):
        super().__init__(parent)
        self.session = session.resolve()
        self.controller = controller
        self.loaded_path = None
        self.setWindowTitle("数据质量报告")
        self.resize(1000, 780)
        self.setMinimumSize(600, 440)
        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        self.headline = QtWidgets.QLabel("数据质量报告")
        self.headline.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.headline.setObjectName("section_title")
        self.headline.setWordWrap(True)
        header.addWidget(self.headline, 1)
        self.browser = QtWidgets.QToolButton()
        self.browser.setIcon(asset_icon("data_viewer"))
        self.browser.setToolTip("在浏览器中打开报告")
        self.browser.clicked.connect(self._open_browser)
        header.addWidget(self.browser)
        self.retry = QtWidgets.QPushButton("重新生成")
        self.retry.setIcon(asset_icon("refresh"))
        self.retry.clicked.connect(lambda: controller.enqueue(self.session, force=True))
        header.addWidget(self.retry)
        self.stop = QtWidgets.QToolButton()
        self.stop.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaStop))
        self.stop.setToolTip("取消报告生成")
        self.stop.clicked.connect(lambda: controller.cancel(self.session))
        header.addWidget(self.stop)
        layout.addLayout(header)
        self.progress = QtWidgets.QProgressBar()
        layout.addWidget(self.progress)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll, 1)
        self.content = QtWidgets.QWidget()
        self.body = QtWidgets.QVBoxLayout(self.content)
        self.body.setSpacing(14)
        self.scroll.setWidget(self.content)
        controller.changed.connect(self._refresh)
        self._refresh(str(self.session))

    def _refresh(self, session):
        if Path(session).resolve() != self.session:
            return
        record = self.controller.status(self.session)
        if self.controller.active is not None and self.controller.active["session"] == self.session:
            record = self.controller.active["record"]
        busy = record.get("status") in {"queued", "running"}
        self.retry.setEnabled(not busy)
        self.stop.setEnabled(busy)
        self.progress.setVisible(busy)
        self.progress.setValue(int(record.get("progress", 0)))
        self.headline.setText(str(record.get("message") or "暂无质量报告"))
        self.browser.setEnabled(False)
        if record.get("status") != "completed":
            if self.loaded_path is not None:
                self._clear_body()
                self.loaded_path = None
            return
        try:
            path = report_file(self.session, record)
            if self.loaded_path == path:
                self.browser.setEnabled(path.with_name("report.html").is_file())
                return
            report = json.loads(path.read_text(encoding="utf-8"))
            _validate_report(report, self.session, record.get("session_id"))
            self._clear_body()
            created = datetime.fromisoformat(report["created_utc"].replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            self._text(f"{report['session_id']} · {created}", muted=True)
            for note in report.get("notes", []):
                self._text(note, muted=True)
            for title, fields in report.get("sections", []):
                self._text(title, heading=True)
                band = QtWidgets.QWidget()
                grid = QtWidgets.QGridLayout(band)
                grid.setContentsMargins(0, 0, 0, 0)
                for index, (name, value) in enumerate(fields):
                    label = QtWidgets.QLabel(f"{name}：{value}")
                    label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
                    label.setWordWrap(True)
                    grid.addWidget(label, index // 2, index % 2)
                grid.setColumnStretch(0, 1)
                grid.setColumnStretch(1, 1)
                self.body.addWidget(band)
            for chart in report.get("plots", []):
                image = (path.parent / chart["file"]).resolve()
                if not image.is_relative_to(path.parent) or image.suffix.lower() != ".png":
                    raise ValueError("报告图片路径无效")
                self._text(chart["title"], heading=True)
                self._text(chart.get("caption", ""), muted=True)
                self.body.addWidget(ReportImage(image))
            self.body.addStretch()
            self.loaded_path = path
            self.browser.setEnabled(path.with_name("report.html").is_file())
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            self._clear_body()
            self.loaded_path = None
            self.headline.setText(f"报告读取失败：{error}")

    def _text(self, text, *, heading=False, muted=False):
        label = QtWidgets.QLabel(str(text))
        label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setObjectName("section_title" if heading else "muted" if muted else "")
        self.body.addWidget(label)

    def _clear_body(self):
        while self.body.count():
            item = self.body.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def _open_browser(self):
        if self.loaded_path is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.loaded_path.with_name("report.html"))))
