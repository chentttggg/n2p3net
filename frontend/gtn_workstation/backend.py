"""Model commands and receipts at the workstation/backend process seam."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

MODEL_RESULT_SCHEMA = "n2p3_brainsync_model_task/2"


def _file(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"文件不存在: {path}")
    return path


@dataclass(frozen=True)
class CalibrationSettings:
    source_checkpoint: str = ""
    reference_channel: str = ""
    ocular_correction: bool = False
    eog_reference_policy: str = "shared_measured_reference"
    epoch_qc: bool = False
    epoch_qc_model: str = ""
    continuous_qc_model: str = ""

    def arguments(self) -> list[str]:
        arguments = []
        for option, value in (("--source-checkpoint", self.source_checkpoint),
                              ("--epoch-qc-model", self.epoch_qc_model),
                              ("--continuous-qc-model", self.continuous_qc_model)):
            if value:
                try:
                    path = _file(value)
                except ValueError as error:
                    label = {"--source-checkpoint": "源模型", "--epoch-qc-model": "epoch QC",
                             "--continuous-qc-model": "连续 QC"}[option]
                    raise ValueError(f"{label} 模型文件不存在: {value}") from error
                arguments.extend([option, str(path)])
        if self.epoch_qc and self.epoch_qc_model:
            raise ValueError("拟合 epoch QC 与冻结 epoch QC 模型互斥")
        personal_quality = self.epoch_qc
        if self.epoch_qc_model:
            record = json.loads(_file(self.epoch_qc_model).read_text(encoding="utf-8"))
            model = record.get("model", {}) if isinstance(record, dict) else {}
            personal_quality = isinstance(model, dict) and model.get("method") == "personal_epoch_signal_evidence"
        if personal_quality and not self.continuous_qc_model:
            raise ValueError("信号证据 QC 需要匹配当前滤波的连续 QC 模型（包含恒值检测与边界评估）")
        if self.reference_channel:
            arguments.extend(["--reference-channel", self.reference_channel])
        else:
            arguments.extend(["--reference-mode", "native"])
        if self.ocular_correction:
            arguments.extend(["--ocular-correction", "--eog-reference-policy", self.eog_reference_policy])
        if self.epoch_qc:
            arguments.append("--epoch-qc")
        return arguments


@dataclass(frozen=True)
class ModelTask:
    """Bind a command to its request, session identities and model bytes.

    The runner transports the command. Only read_result admits its output for
    display; a successful process exit is not a successful model receipt.
    """

    task: str
    request_id: str
    program: str
    arguments: tuple[str, ...]
    output: Path
    artifact: Path
    artifact_sha256: str | None
    sessions: tuple[Path, ...]
    session_ids: tuple[str, ...]
    subject_id: str | None
    simulation_status: str | None

    @classmethod
    def create(cls, backend: Path, task: str, output: Path, *, artifact: Path | None = None,
               sessions: tuple[Path, ...] = (), settings: CalibrationSettings | None = None,
               device: str = "cpu") -> ModelTask:
        if task not in {"calibrate", "inspect", "infer"}:
            raise ValueError("不支持的模型任务")
        backend, output = backend.expanduser().resolve(), output.resolve()
        program = _file(backend / ".venv/Scripts/python.exe")
        script = _file(backend / "experiments/brainsync_model_task.py")
        if task == "calibrate":
            artifact = output / "personal_model.pt"
        elif artifact is None:
            raise ValueError("请先选择个体模型")
        else:
            artifact = _file(artifact)
        sessions = tuple(path.resolve() for path in sessions)
        if len(set(sessions)) != len(sessions) or (task == "infer" and len(sessions) != 1):
            raise ValueError("模型任务会话数量无效")
        if (task == "inspect" and sessions) or (task == "calibrate" and not sessions):
            raise ValueError("模型任务会话范围无效")
        identities = []
        for path in sessions:
            record = json.loads((path / "session.json").read_text(encoding="utf-8"))
            identity = (record["session_id"], record["experiment"]["subject_id"],
                        "declared_mock" if record.get("device", {}).get("transport") == "mock" else "not_declared")
            if any(not isinstance(value, str) or not value.strip() for value in identity):
                raise ValueError("会话缺少明确身份")
            identities.append(identity)
        if identities and len({identity[1:] for identity in identities}) != 1:
            raise ValueError("模型任务要求同一受试者和采集模式")
        request_id = uuid4().hex
        arguments = [str(script), "--task", task, "--request-id", request_id,
                     "--model-artifact", str(artifact), "--output-dir", str(output), "--device", device]
        for path in sessions:
            arguments.extend(["--session-dir", str(path)])
        if task == "calibrate":
            arguments.extend((settings or CalibrationSettings()).arguments())
        elif task == "infer":
            arguments.extend(["--repetitions", "5"])
        digest = None if task == "calibrate" else hashlib.sha256(artifact.read_bytes()).hexdigest()
        return cls(task, request_id, str(program), tuple(arguments), output, artifact, digest, sessions,
                   tuple(identity[0] for identity in identities),
                   identities[0][1] if identities else None, identities[0][2] if identities else None)

    def read_result(self) -> dict:
        result = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result.get("schema") != MODEL_RESULT_SCHEMA:
            raise ValueError("模型结果 schema 不受支持")
        if result.get("task") != self.task or result.get("request_id") != self.request_id:
            raise ValueError("模型任务结果与请求不一致")
        statuses = {"predicted", "abstained"} if self.task == "infer" else {"completed"}
        if result.get("status") not in statuses:
            raise ValueError("模型结果状态不可用")
        if result.get("session_dirs") != [str(path) for path in self.sessions]:
            raise ValueError("模型结果与请求会话路径不一致")
        if _file(result["model_artifact"]) != self.artifact:
            raise ValueError("模型结果与请求模型路径不一致")
        digest = hashlib.sha256(self.artifact.read_bytes()).hexdigest()
        if result.get("artifact_sha256") != digest or self.artifact_sha256 not in {None, digest}:
            raise ValueError("模型文件在任务期间改变或摘要不一致")
        if not isinstance(result.get("subject_id"), str) or not result["subject_id"].strip():
            raise ValueError("模型结果缺少受试者身份")
        if self.subject_id is not None and result["subject_id"] != self.subject_id:
            raise ValueError("模型结果与请求受试者不一致")
        if self.simulation_status is not None and result.get("simulation_status") != self.simulation_status:
            raise ValueError("模型结果与请求采集模式不一致")
        if self.task == "calibrate":
            saved_sessions = result.get("calibration_sessions")
            if (not isinstance(saved_sessions, list)
                    or sorted(row["session_id"] for row in saved_sessions) != sorted(self.session_ids)):
                raise ValueError("模型结果的校准范围与请求不一致")
        if self.task == "infer":
            if result.get("session_id") != self.session_ids[0]:
                raise ValueError("推理结果与请求会话身份不一致")
            if result.get("target_labels_used") is not False or result.get("mode") != "completed_recording_inference":
                raise ValueError("推理结果不是无标签完整记录推理")
            prediction = result.get("prediction")
            if result["status"] == "predicted":
                if type(prediction) is not int or not 1 <= prediction <= 9:
                    raise ValueError("推理数字无效")
            elif prediction is not None:
                raise ValueError("拒判结果不能包含预测数字")
        return result
