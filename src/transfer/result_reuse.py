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


_COMPLETED_REQUEST_FIELDS = {
    "public_target": frozenset({
        "task_manifest", "target_study", "target_protocol", "target_cache_sha256",
        "checkpoint_sha256", "source_seed", "evidence_scope", "evaluation_snapshot",
        "source_study_membership", "target_study_membership", "environment",
    }),
    "personal": frozenset({
        "schema", "task", "recipe", "source_snapshot", "checkpoint_sha256",
        "environment", "fit_performed",
    }),
}


def admit_completed_result(path: str | Path, *, kind: str, expected: Mapping) -> dict:
    """Admit a complete experiment against its producer's current task declaration.

    Read and decode the original sealed bytes, including every declared cell,
    model and prediction. Failure cells and BrainSync unavailable events retain
    their original meaning. Missing integrity declarations are never repaired.
    """
    if kind not in _COMPLETED_REQUEST_FIELDS:
        raise ValueError(f"unknown completed result kind: {kind}")
    missing = _COMPLETED_REQUEST_FIELDS[kind] - set(expected)
    if missing:
        raise ValueError(f"completed result admission requires the full task declaration: {sorted(missing)}")
    path = Path(path)
    try:
        record = read_result_record(path)
        if record.get("status") != "completed":
            raise ResultIntegrityError("result integrity: experiment is not completed")
        for key, value in expected.items():
            if key not in record or semantic_sha256(record[key]) != semantic_sha256(value):
                raise ResultIntegrityError(f"result integrity: completed experiment differs in {key}")
        if kind == "public_target":
            _admit_public_result(record, path.parent)
        else:
            _admit_personal_result(record, path.parent)
        return record
    except ResultIntegrityError:
        raise
    except Exception as error:
        raise ResultIntegrityError("result integrity: malformed completed experiment") from error


def _admit_public_result(record: Mapping, directory: Path) -> None:
    if record["schema"] != "n2p3_target_stage_run/1":
        raise ResultIntegrityError("result integrity: unsupported public target result schema")
    _read_artifact_bytes(record, directory)
    manifest = record["task_manifest"]
    if json.loads(_read_bytes(directory / "tasks.json")) != manifest:
        raise ResultIntegrityError("result integrity: task manifest disagrees with the completed run")
    tasks, cells = manifest["tasks"], record["cells"]
    if len(cells) != len(tasks):
        raise ResultIntegrityError("result integrity: completed run has missing or extra cells")
    completed = 0
    for index, (task, cell) in enumerate(zip(tasks, cells, strict=True)):
        if read_result_record(directory / f"{index:05d}.json") != cell:
            raise ResultIntegrityError("result integrity: cell disagrees with its completed run")
        for key, value in task.items():
            actual = cell.get("requested_recipe", cell.get(key)) if key == "recipe" else cell.get(key)
            if key not in cell or semantic_sha256(actual) != semantic_sha256(value):
                raise ResultIntegrityError(f"result integrity: cell differs from its task in {key}")
        if semantic_sha256(cell["evaluation_contract"]) != cell["evaluation_contract_digest"]:
            raise ResultIntegrityError("result integrity: cell evaluation contract digest differs")
        files = cell["result_integrity"]["files"]
        if cell["status"] == "fit_failed":
            if files or not cell.get("failure"):
                raise ResultIntegrityError("result integrity: failed cell has inconsistent evidence")
            continue
        if cell["status"] != "completed":
            raise ResultIntegrityError("result integrity: unfinished cell in completed run")
        completed += 1
        roles = {"prediction"} if cell.get("recipe") is None else {"prediction", "model"}
        if set(files) != roles:
            raise ResultIntegrityError("result integrity: completed cell lacks its required artifacts")
        predictions, model = read_result_artifacts(cell, directory)
        scope = manifest["scopes"][task["scope_id"]]
        inner = manifest["stage"] == "B0"
        for field, scope_field in (
            ("train_rows", "inner_training_rows" if inner else "calibration_rows"),
            ("test_rows", "inner_validation_rows" if inner else "test_rows"),
        ):
            rows = predictions[field]
            if rows.ndim != 1 or rows.dtype.kind not in "iu" or not np.array_equal(rows, scope[scope_field]):
                raise ResultIntegrityError("result integrity: prediction rows differ from the declared scope")
        if predictions["y"].shape != predictions["scores"].shape or not np.isin(predictions["y"], (0, 1)).all():
            raise ResultIntegrityError("result integrity: invalid binary prediction labels")
        if model is not None and (
            model["recipe"] != cell["recipe"] or model["fit_history"] != cell["fit_history"]
            or _tensor_state_digest(model["state"]) != cell["fitted_state_sha256"]
        ):
            raise ResultIntegrityError("result integrity: saved model differs from its cell")
    if (type(record["completed_cells"]) is not int or type(record["failed_cells"]) is not int
            or record["completed_cells"] != completed or record["failed_cells"] != len(cells) - completed):
        raise ResultIntegrityError("result integrity: completed/failed cell counts do not reconcile")


def _admit_personal_result(record: Mapping, directory: Path) -> None:
    if record["schema"] != "n2p3_personal_matrix_result/1":
        raise ResultIntegrityError("result integrity: unsupported personal matrix result schema")
    predictions = record["predictions"]
    sessions = [item["session"] for item in predictions]
    declared_sessions = record["task"]["evaluation_sessions"]
    if (not sessions or len(set(sessions)) != len(sessions)
            or sessions != sorted(declared_sessions)):
        raise ResultIntegrityError("result integrity: missing or duplicate prediction sessions")
    fitted = record["task"]["budget"] > 0
    if record["fit_performed"] is not fitted:
        raise ResultIntegrityError("result integrity: fit declaration differs from calibration budget")
    roles = {f"prediction:{session}" for session in sessions} | ({"model"} if fitted else set())
    files = record["result_integrity"]["files"]
    if set(files) != roles or len({item["name"] for item in files.values()}) != len(files):
        raise ResultIntegrityError("result integrity: personal result lacks its required artifacts")
    content = _read_artifact_bytes(record, directory)
    seen_events = set()
    for item in predictions:
        role = f"prediction:{item['session']}"
        if item["prediction_sha256"] != files[role]["sha256"]:
            raise ResultIntegrityError("result integrity: personal prediction digest differs")
        with np.load(io.BytesIO(content[role]), allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        fields = ("scores", "raw_scores", "y", "event_ids", "candidate_ids", "repetition_indices", "group_ids")
        size = item["requested_events"]
        if type(size) is not int or size < 1 or any(arrays[name].shape != (size,) for name in fields):
            raise ResultIntegrityError("result integrity: personal prediction rows do not cover the requested events")
        scores, raw_scores = arrays["scores"], arrays["raw_scores"]
        if any(values.dtype.kind not in "fiu" or np.isinf(values).any() for values in (scores, raw_scores)):
            raise ResultIntegrityError("result integrity: personal scores may be unavailable, but cannot be infinite")
        if not np.isin(arrays["y"], (0, 1)).all():
            raise ResultIntegrityError("result integrity: invalid personal prediction labels")
        if arrays["repetition_indices"].dtype.kind not in "iu" or (arrays["repetition_indices"] < 0).any():
            raise ResultIntegrityError("result integrity: invalid personal repetition indices")
        if any(not value for field in ("candidate_ids", "group_ids") for value in arrays[field].astype(str)):
            raise ResultIntegrityError("result integrity: missing personal candidate/group identities")
        event_ids = arrays["event_ids"].astype(str).tolist()
        if any(not event for event in event_ids) or len(set(event_ids)) != size or seen_events.intersection(event_ids):
            raise ResultIntegrityError("result integrity: missing or repeated personal event identities")
        seen_events.update(event_ids)
        scoring_ids = record["task"]["variant"].get("scoring_event_ids")
        expected_scores = raw_scores if scoring_ids is None else np.where(np.isin(event_ids, scoring_ids), raw_scores, np.nan)
        if not np.array_equal(scores, expected_scores, equal_nan=True):
            raise ResultIntegrityError("result integrity: personal scoring view differs from its declared events")
    if not fitted:
        if record["recipe"] is not None or record["target_updates"] != 0 or "model_sha256" in record:
            raise ResultIntegrityError("result integrity: zero calibration cannot carry a fitted target model")
        return
    if record["model_sha256"] != files["model"]["sha256"] or record["model_reload_equal"] is not True:
        raise ResultIntegrityError("result integrity: personal model declaration differs")
    from transfer.calibrated import load_calibrated_predictor

    predictor = load_calibrated_predictor(io.BytesIO(content["model"]))
    metadata = predictor.metadata
    for key, expected in (
        ("matrix_task", record["task"]), ("experiment_source_snapshot", record["source_snapshot"]),
        ("recipe", record["recipe"]), ("fit_history", record["fit_history"]),
        ("calibration_trials", record["actual_training_rows"]),
    ):
        if metadata.get(key) != expected:
            raise ResultIntegrityError(f"result integrity: personal model differs in {key}")
    if metadata["selection"]["plan_sha256"] != record["plan_sha256"]:
        raise ResultIntegrityError("result integrity: personal model lost its original execution plan")
    if record["task"]["initialization"] == "source" and (
        metadata["source_checkpoint_sha256"] != record["checkpoint_sha256"]
        or semantic_sha256(metadata["source_training_contract"]) != record["source_training_contract_digest"]
    ):
        raise ResultIntegrityError("result integrity: personal source checkpoint/contract differs")


def _tensor_state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def target_result_key(*, dataset_sha256, participant_key, split, architecture, treatment,
                      batch_size, source_state, snapshot_sha256, device):
    treatment = copy.deepcopy(treatment)
    recipe = treatment.get("recipe")
    if recipe is not None and recipe.get("initialization") == "waveform_lda":
        recipe.pop("seed", None)
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
