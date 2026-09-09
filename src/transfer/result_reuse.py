"""Worker-local reuse of identical evaluation treatments and their saved models."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from research.contracts import semantic_sha256

RESULT_INTEGRITY_SCHEMA = "n2p3_result_integrity/1"


class ResultIntegrityError(RuntimeError):
    """A broken result artifact must abort reuse, not become a subject failure."""


def _read_bytes(path: str | Path) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as error:
        raise ResultIntegrityError(f"result integrity: cannot read {path}") from error


def read_verified_bytes(path: str | Path, expected_sha256: str) -> bytes:
    """Verify and consume the same bytes even if the path is replaced afterwards."""
    content = _read_bytes(path)
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ResultIntegrityError(f"result integrity: artifact changed: {path}")
    return content


def seal_result_record(record: Mapping, *, files: Mapping[str, Path] | None = None) -> dict:
    """Bind a finalized record to its companion files before publishing it."""
    payload = {key: copy.deepcopy(value) for key, value in record.items() if key != "result_integrity"}
    manifest = {}
    for role, path in (files or {}).items():
        path = Path(path)
        content = _read_bytes(path)
        manifest[role] = {"name": path.name, "sha256": hashlib.sha256(content).hexdigest(),
                          "byte_size": len(content)}
    return {**payload, "result_integrity": {
        "schema": RESULT_INTEGRITY_SCHEMA, "files": manifest,
        "record_sha256": semantic_sha256({"record": payload, "files": manifest}),
    }}


def validate_result_record(record: Mapping) -> dict:
    integrity = record.get("result_integrity") if isinstance(record, Mapping) else None
    if not isinstance(integrity, dict) or integrity.get("schema") != RESULT_INTEGRITY_SCHEMA:
        raise ResultIntegrityError("unverified result: reuse requires an original result integrity declaration")
    files = integrity.get("files")
    if not isinstance(files, dict):
        raise ResultIntegrityError("result integrity: invalid artifact manifest")
    payload = {key: value for key, value in record.items() if key != "result_integrity"}
    try:
        actual = semantic_sha256({"record": payload, "files": files})
    except (TypeError, ValueError) as error:
        raise ResultIntegrityError("result integrity: invalid record values") from error
    if actual != integrity.get("record_sha256"):
        raise ResultIntegrityError("result integrity: finalized record changed")
    return dict(record)


def read_result_record(path: str | Path, *, expected_sha256: str | None = None) -> dict:
    content = _read_bytes(path) if expected_sha256 is None else read_verified_bytes(path, expected_sha256)
    return _decode_result_record(content)


def _decode_result_record(content: bytes) -> dict:
    try:
        record = json.loads(content)
    except (ValueError, UnicodeError) as error:
        raise ResultIntegrityError("result integrity: invalid JSON record") from error
    return validate_result_record(record)


def _read_artifact_bytes(record: Mapping, directory: str | Path) -> dict[str, bytes]:
    record = validate_result_record(record)
    content = {}
    for role, item in record["result_integrity"]["files"].items():
        name = item.get("name") if isinstance(item, Mapping) else None
        if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
            raise ResultIntegrityError("result integrity: invalid companion filename")
        content[role] = read_verified_bytes(Path(directory) / name, item.get("sha256"))
        if len(content[role]) != item.get("byte_size"):
            raise ResultIntegrityError("result integrity: companion byte size changed")
    return content


def _decode_predictions(content: Mapping[str, bytes]) -> dict:
    try:
        with np.load(io.BytesIO(content["prediction"]), allow_pickle=False) as archive:
            predictions = {name: archive[name].copy() for name in archive.files}
        scores = predictions["scores"]
        if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
            raise ValueError("predictions must contain a nonempty finite score vector")
        if any(predictions[name].shape != scores.shape for name in ("y", "test_rows") if name in predictions):
            raise ValueError("prediction rows and labels must align with scores")
    except Exception as error:
        raise ResultIntegrityError("result integrity: invalid prediction payload") from error
    return predictions


def read_result_artifacts(record: Mapping, directory: str | Path) -> tuple[dict, dict | None]:
    """Load predictions and the optional model from their verified byte snapshots."""
    content = _read_artifact_bytes(record, directory)
    predictions = _decode_predictions(content)
    try:
        model = (None if "model" not in content else
                 torch.load(io.BytesIO(content["model"]), map_location="cpu", weights_only=False))
        if model is not None and not isinstance(model, dict):
            raise ValueError("model payload must be a mapping")
    except Exception as error:
        raise ResultIntegrityError("result integrity: invalid model payload") from error
    return predictions, model


def target_result_key(*, dataset_sha256, participant_key, split, architecture, treatment,
                      batch_size, source_state, snapshot_sha256, device):
    return semantic_sha256({
        "schema": "n2p3_target_result_key/1", "dataset_sha256": dataset_sha256,
        "participant_key": participant_key, "split": split, "architecture": architecture,
        "treatment": treatment, "batch_size": batch_size, "source_state": source_state,
        "snapshot_sha256": snapshot_sha256,
        "runtime": {"device": str(device), "torch": torch.__version__,
                    "dtype": str(torch.get_default_dtype()), "threads": torch.get_num_threads(),
                    "deterministic": torch.are_deterministic_algorithms_enabled(),
                    "matmul_precision": torch.get_float32_matmul_precision(),
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                    "matmul_tf32": torch.backends.cuda.matmul.allow_tf32},
    })


class TargetResultReuse:
    """Bound to one worker's target-cache/participant shard; failures are not cached."""

    def __init__(self):
        self._results = {}

    def get(self, key):
        value = self._results.get(key)
        if value is None:
            return None
        cell = read_result_record(value["cell_path"], expected_sha256=value["cell_sha256"])
        predictions, model = read_result_artifacts(cell, value["directory"])
        files = cell["result_integrity"]["files"]
        return {
            "scores": predictions["scores"], "fit_history": cell.get("fit_history"),
            "model_payload": model, "model_sha256": files.get("model", {}).get("sha256"),
            "origin_prediction": str(value["directory"] / files["prediction"]["name"]),
            "origin_contract_digest": cell["evaluation_contract_digest"],
            "origin_result_integrity": cell["result_integrity"],
        }

    def put(self, key, *, scores, cell, prediction_path, model_path=None):
        if cell["status"] != "completed":
            return
        prediction = Path(prediction_path).resolve()
        cell_path = prediction.with_suffix(".json")
        cell_content = _read_bytes(cell_path)
        saved = _decode_result_record(cell_content)
        if saved != cell:
            raise ResultIntegrityError("result integrity: cache cell differs from its published record")
        predictions = _decode_predictions(_read_artifact_bytes(saved, prediction.parent))
        files = saved["result_integrity"]["files"]
        if (files["prediction"]["name"] != prediction.name
                or (model_path is None) != ("model" not in files)
                or (model_path is not None and Path(model_path).resolve() != prediction.parent / files["model"]["name"])
                or not np.array_equal(predictions["scores"], scores)):
            raise ResultIntegrityError("result integrity: cache inputs differ from published artifacts")
        self._results[key] = {
            "directory": prediction.parent, "cell_path": cell_path,
            "cell_sha256": hashlib.sha256(cell_content).hexdigest(),
        }
