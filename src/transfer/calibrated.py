"""Participant-bound calibration artifacts for later, unlabelled EEG decisions."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data.contract import assert_causal_p300_input_contract
from data.epochs import EpochDataset, materialize_dataset_identity
from data.exclusions import assert_origin_keys_allowed
from data.ocular import validate_eog_calibration_budget
from models.n2p3net import N2P3ArchitectureConfig, N2P3Net
from research.contracts import semantic_sha256
from transfer.binary import (
    fit_shrinkage_feature_model,
    plain_trunk_configuration,
    target_shrinkage_features,
)
from transfer.checkpoint import (
    CHECKPOINT_SCHEMA,
    checkpoint_classifier_is_trained,
    checkpoint_input_stats,
    checkpoint_training_contract,
    load_n2p3_trunk_checkpoint,
    predict_n2p3_checkpoint,
)
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig

CALIBRATED_PREDICTOR_SCHEMA = "n2p3_calibrated_predictor/1"


@dataclass(frozen=True)
class CalibrationRecipe:
    name: str
    epochs: int
    learning_rate: float

    def __post_init__(self) -> None:
        if not self.name.strip() or self.epochs < 1 or not np.isfinite(self.learning_rate):
            raise ValueError("calibration recipe requires a name and positive finite budget")
        if self.learning_rate <= 0:
            raise ValueError("calibration learning rate must be positive")


DEFAULT_CALIBRATION_RECIPES = (
    CalibrationRecipe("full_fine_low", 15, 1e-4),
    CalibrationRecipe("full_fine_high", 60, 1e-3),
)


def _json_record(value: object) -> dict[str, object]:
    return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))


def _input_contract(dataset: EpochDataset) -> dict[str, object]:
    reference = dataset.provenance.get("source_reference")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("calibrated predictors require an explicit physical reference")
    return _json_record({
        "channel_names": list(dataset.channel_names),
        "channel_positions_m": np.asarray(dataset.channel_positions_m).tolist(),
        "preprocessing": asdict(dataset.preprocessing),
        "source_reference": reference,
        "n_channels": dataset.n_channels,
        "n_times": dataset.n_times,
        "simulation_status": dataset.provenance.get("simulation_status", "not_declared"),
        "ocular_artifact_processing": dataset.provenance.get(
            "ocular_artifact_processing", {"status": "not_applied"}
        ),
    })


def _participant(dataset: EpochDataset) -> tuple[str, str]:
    subjects = tuple(np.unique(np.asarray(dataset.subject_ids).astype(str)))
    if len(subjects) != 1:
        raise ValueError("a personalized model requires exactly one participant")
    identity = materialize_dataset_identity(dataset)
    return subjects[0], identity.record_for(subjects[0]).authority_key("source_or_global")


def _observed(dataset: EpochDataset) -> np.ndarray:
    return (
        np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2]).copy()
        if dataset.trial_channel_mask is None
        else np.asarray(dataset.trial_channel_mask, dtype=bool)
    )


def _session_scope(dataset: EpochDataset, *, calibration: bool) -> list[dict[str, object]]:
    fields = ["session", "session_start_timestamp_s"]
    if calibration:
        fields += ["session_end_timestamp_s", "calibration_available_timestamp_s"]
    missing = set(fields) - set(dataset.metadata.columns)
    if missing:
        raise ValueError(f"session timing metadata missing: {sorted(missing)}")
    result = []
    for session, rows in dataset.metadata.groupby("session", sort=False):
        row: dict[str, object] = {"session_id": str(session), "n_trials": len(rows)}
        for field in fields[1:]:
            values = rows[field].to_numpy(dtype=float)
            if not np.isfinite(values).all() or len(np.unique(values)) != 1:
                raise ValueError(f"session {session}: inconsistent finite {field} required")
            row[field] = float(values[0])
        if calibration and not (
            row["session_start_timestamp_s"] < row["session_end_timestamp_s"]
            <= row["calibration_available_timestamp_s"]
        ):
            raise ValueError("calibration availability must follow the complete recording")
        result.append(row)
    return sorted(result, key=lambda item: (item["session_start_timestamp_s"], item["session_id"]))


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _calibration_data_digest(dataset: EpochDataset) -> str:
    digest = hashlib.sha256()
    for value in (np.asarray(dataset.X, dtype=np.float32),
                  np.asarray(dataset.y, dtype=np.int64), _observed(dataset)):
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(json.dumps(dataset.metadata["session"].astype(str).tolist()).encode("utf-8"))
    return digest.hexdigest()


class CalibratedPredictor:
    """Frozen binary evidence model with an explicit physical input contract."""

    def __init__(
        self, metadata: Mapping[str, object], state: Mapping[str, torch.Tensor], *, device: str = "cpu"
    ) -> None:
        self.metadata = _json_record(metadata)
        self.state = {name: value.detach().cpu().clone() for name, value in state.items()}
        self.device = torch.device(device)
        contract = self.metadata["input_contract"]
        if not isinstance(contract, dict) or not self.metadata.get("participant_key"):
            raise ValueError("calibrated model lacks participant and input contracts")
        self.trunk = None
        if self.metadata["model_kind"] == "source_initialized_n2p3":
            assert_origin_keys_allowed(
                self.metadata.get("source_training_contract", {}).get("training_participant_keys", [])
            )
            configuration = N2P3ArchitectureConfig(**self.metadata["architecture_configuration"])
            self.trunk = N2P3Net(
                contract["n_channels"], n_times=contract["n_times"],
                sfreq=contract["preprocessing"]["sfreq"],
                tmin_s=contract["preprocessing"]["tmin_ms"] / 1000.0,
                pooling_mode=self.metadata["pooling_mode"], **configuration.model_kwargs(),
            )
            if _json_record(self.trunk.architecture_record()) != self.metadata["architecture"]:
                raise ValueError("calibrated architecture does not reconstruct exactly")
            self.trunk.load_state_dict(
                {name.removeprefix("trunk."): value for name, value in self.state.items()
                 if name.startswith("trunk.")}, strict=True,
            )
            self.trunk.eval()
            if self.state["input_mean"].shape != (1, contract["n_channels"], 1) or (
                self.state["input_std"].shape != self.state["input_mean"].shape
                or not bool((self.state["input_std"] > 0).all())
            ):
                raise ValueError("calibrated input statistics have invalid shape or scale")
        elif self.metadata["model_kind"] != "target_only_shrinkage_lda":
            raise ValueError("unsupported calibrated predictor kind")
        else:
            feature_shape = self.state["coefficient"].shape
            if len(feature_shape) != 1 or self.state["feature_mean"].shape != feature_shape or (
                self.state["feature_scale"].shape != feature_shape
                or not bool((self.state["feature_scale"] > 0).all())
                or self.state["intercept"].numel() != 1
            ):
                raise ValueError("calibrated LDA feature state has invalid shape or scale")
        if any(not bool(torch.isfinite(value).all()) for value in self.state.values()):
            raise ValueError("calibrated predictor contains nonfinite state")

    def predict_logit(self, dataset: EpochDataset) -> np.ndarray:
        dataset.validate(require_labels=False)
        if _input_contract(dataset) != self.metadata["input_contract"]:
            raise ValueError(
                "model input channels, reference, ocular processing, preprocessing or simulation status differ"
            )
        _, participant = _participant(dataset)
        if participant != self.metadata["participant_key"]:
            raise ValueError("personalized model belongs to a different participant")
        observed = _observed(dataset)
        if self.trunk is not None:
            return predict_n2p3_checkpoint(
                self.trunk, dataset.X,
                input_stats=(self.state["input_mean"].numpy(), self.state["input_std"].numpy()),
                trial_channel_mask=observed, device=self.device,
            )
        if not bool(observed.all()):
            raise ValueError("target-only LDA requires every selected EEG channel to be usable")
        features = target_shrinkage_features(dataset.X, sfreq=dataset.preprocessing.sfreq)
        standardized = (features - self.state["feature_mean"].numpy()) / self.state["feature_scale"].numpy()
        return np.asarray(
            standardized @ self.state["coefficient"].numpy() + self.state["intercept"].item(),
            dtype=np.float64,
        )

    def validate_inference_scope(self, dataset: EpochDataset) -> None:
        sessions = _session_scope(dataset, calibration=False)
        calibration_ids = {row["session_id"] for row in self.metadata["calibration_sessions"]}
        if any(row["session_id"] in calibration_ids for row in sessions):
            raise ValueError("calibration recordings cannot be reused as independent inference")
        if any(row["session_start_timestamp_s"] <= self.metadata["calibration_available_timestamp_s"]
               for row in sessions):
            raise ValueError("inference recording starts before calibration labels were available")


def fit_calibrated_predictor(
    dataset: EpochDataset, *, source_checkpoint: str | Path | None = None,
    recipes: Sequence[CalibrationRecipe] = DEFAULT_CALIBRATION_RECIPES,
    device: str = "cpu", seed: int = 20260906, batch_size: int = 64,
    input_request: Mapping[str, object] | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> CalibratedPredictor:
    """Choose source adaptation only inside calibration, then refit its full scope."""
    dataset.validate(require_labels=True)
    assert_causal_p300_input_contract(dataset.preprocessing)
    subject, participant = _participant(dataset)
    sessions = _session_scope(dataset, calibration=True)
    if not sessions:
        raise ValueError("calibration requires at least one labelled recording session")
    if source_checkpoint is not None and len(sessions) < 2:
        raise ValueError("source recipe selection requires at least two separate labelled recording sessions")
    session_ids = dataset.metadata["session"].astype(str).to_numpy()
    for session in sessions:
        labels = dataset.y[session_ids == session["session_id"]]
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("every calibration session must retain target and non-target trials")
    ocular = dataset.provenance.get("ocular_artifact_processing", {"status": "not_applied"})
    validate_eog_calibration_budget(ocular, {participant: tuple(row["session_id"] for row in sessions)})
    ocular_record = ocular.get("model") if ocular.get("status") == "applied" else None
    reference_record = dataset.provenance.get("domain_adapter", {}).get("reference", {})
    inferred_reference = (
        ocular_record.get("reference_channel") if ocular_record is not None
        else reference_record.get("channel") if reference_record.get("mode") == "measured_channel"
        else None
    )
    request = {"reference_channel": inferred_reference, "ocular_model": ocular_record,
               "channel_names": list(dataset.channel_names)}
    if input_request is not None and any(request.get(key) != value for key, value in input_request.items()):
        raise ValueError("raw input request disagrees with the actual calibrated preprocessing")
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "participant_key": participant, "subject_id": subject, "calibration_sessions": sessions,
        "calibration_available_timestamp_s": max(row["calibration_available_timestamp_s"] for row in sessions),
        "calibration_trials": dataset.n_epochs,
        "calibration_positives": int(dataset.y.sum()),
        "calibration_data_sha256": _calibration_data_digest(dataset),
        "input_contract": _input_contract(dataset), "input_request": request,
        "candidate_vocabulary": list(range(1, 10)),
        "evidence_scope": "personalized_development_not_independent_performance_confirmation",
        "seed": seed,
    }
    notify = progress or (lambda _percent, _message: None)
    if source_checkpoint is None:
        if not bool(_observed(dataset).all()):
            raise ValueError("LDA fallback needs all selected EEG channels; correct channel QC first")
        notify(45, "fit_target_only_shrinkage_lda")
        features = target_shrinkage_features(dataset.X, sfreq=dataset.preprocessing.sfreq)
        fitted = fit_shrinkage_feature_model(features, dataset.y)
        scaler, lda = fitted.steps[0][1], fitted.steps[1][1]
        metadata.update(model_kind="target_only_shrinkage_lda", source_pretraining_used=False,
                        selection={"mode": "fixed_fallback_no_recipe_selection", "validation_auc": None})
        state = {
            "feature_mean": torch.as_tensor(scaler.mean_, dtype=torch.float64),
            "feature_scale": torch.as_tensor(scaler.scale_, dtype=torch.float64),
            "coefficient": torch.as_tensor(lda.coef_[0], dtype=torch.float64),
            "intercept": torch.as_tensor(lda.intercept_[0], dtype=torch.float64),
        }
        return CalibratedPredictor(metadata, state, device=device)

    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("source checkpoint must use the current audited transfer schema")
    if not checkpoint_classifier_is_trained(payload):
        raise ValueError("source checkpoint needs a supervised source classifier")
    try:
        source, _ = load_n2p3_trunk_checkpoint(payload, dataset, target_subject=subject)
    except ValueError as error:
        raise ValueError(
            f"source checkpoint is incompatible: {error} Select a checkpoint with the same "
            "EEG order/reference, or explicitly choose the required measured reference."
        ) from error
    configuration = plain_trunk_configuration(source)
    source_mean, source_std = checkpoint_input_stats(payload, dataset.n_channels)
    recipes = tuple(recipes)
    if not recipes or len({recipe.name for recipe in recipes}) != len(recipes):
        raise ValueError("calibration recipes must be a nonempty unique named set")
    validation_count = max(1, int(np.ceil(0.2 * len(sessions))))
    training_sessions, validation_sessions = sessions[:-validation_count], sessions[-validation_count:]
    validation_start = min(row["session_start_timestamp_s"] for row in validation_sessions)
    if max(row["calibration_available_timestamp_s"] for row in training_sessions) > validation_start:
        raise ValueError("inner-training labels were unavailable before validation recording started")
    validate_eog_calibration_budget(
        ocular, {participant: tuple(row["session_id"] for row in training_sessions)}
    )
    train_mask = np.isin(session_ids, [row["session_id"] for row in training_sessions])
    validation_mask = ~train_mask
    recipe_records = []
    for index, recipe in enumerate(recipes):
        notify(15 + int(55 * index / len(recipes)), f"select_{recipe.name}")
        adapter = SubjectAdapter(
            copy.deepcopy(source), device=torch.device(device), config=SubjectAdapterConfig(
                head_kind="full_fine", epochs=recipe.epochs, lr=recipe.learning_rate,
                batch_size=batch_size, seed=seed, val_group_fraction=None,
                refit_full_prefix=False, freeze_batchnorm_stats=True, input_statistics="source",
                input_mean=source_mean, input_std=source_std,
            ),
        )
        adapter.fit(dataset.X, dataset.y, training_mask=train_mask, validation_mask=validation_mask,
                    prefix_mask=np.ones(dataset.n_epochs, dtype=bool), trial_channel_mask=_observed(dataset))
        history = adapter.last_history
        if history["best_epoch"] is None or history["best_val_loss"] is None:
            raise ValueError("internal validation did not select a finite training epoch")
        recipe_records.append({
            **asdict(recipe), "selected_epochs": int(history["best_epoch"]),
            "validation_loss": float(history["best_val_loss"]),
            "validation_auc": float(roc_auc_score(dataset.y[validation_mask], adapter.calibration_logits_)),
        })
    selected = min(recipe_records, key=lambda item: item["validation_loss"])
    notify(78, "refit_full_authorized_calibration")
    adapter = SubjectAdapter(
        copy.deepcopy(source), device=torch.device(device), config=SubjectAdapterConfig(
            head_kind="full_fine", epochs=selected["selected_epochs"], lr=selected["learning_rate"],
            batch_size=batch_size, seed=seed, val_group_fraction=None, refit_full_prefix=False,
            freeze_batchnorm_stats=True, input_statistics="source", input_mean=source_mean, input_std=source_std,
        ),
    )
    adapter.fit(dataset.X, dataset.y, prefix_mask=np.ones(dataset.n_epochs, dtype=bool),
                trial_channel_mask=_observed(dataset))
    metadata.update(
        model_kind="source_initialized_n2p3", source_pretraining_used=True,
        architecture=source.architecture_record(), architecture_configuration=asdict(configuration),
        pooling_mode=source.pooling_mode,
        source_checkpoint_sha256=hashlib.sha256(Path(source_checkpoint).read_bytes()).hexdigest(),
        source_training_contract=checkpoint_training_contract(payload).record(),
        selection={"mode": "chronological_calibration_decisions_then_full_refit",
                   "training_sessions": [row["session_id"] for row in training_sessions],
                   "validation_sessions": [row["session_id"] for row in validation_sessions],
                   "recipes": recipe_records, "selected": selected,
                   "refit_rows": dataset.n_epochs, "fit_scope": adapter.last_history["fit_scope"]},
    )
    state = {f"trunk.{name}": value.detach().cpu() for name, value in adapter.trunk.state_dict().items()}
    state.update(input_mean=torch.as_tensor(source_mean), input_std=torch.as_tensor(source_std))
    return CalibratedPredictor(metadata, state, device=device)


def save_calibrated_predictor(path: str | Path, predictor: CalibratedPredictor) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("choose a new model artifact path; existing models are immutable")
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    payload = {
        "schema": CALIBRATED_PREDICTOR_SCHEMA,
        "metadata": predictor.metadata, "metadata_digest": semantic_sha256(predictor.metadata),
        "state": predictor.state, "state_digest": _state_digest(predictor.state),
    }
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_calibrated_predictor(path: str | Path, *, device: str = "cpu") -> CalibratedPredictor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != CALIBRATED_PREDICTOR_SCHEMA:
        raise ValueError("unsupported calibrated model artifact")
    if payload.get("metadata_digest") != semantic_sha256(payload["metadata"]):
        raise ValueError("calibrated model metadata digest mismatch")
    if payload.get("state_digest") != _state_digest(payload["state"]):
        raise ValueError("calibrated model state digest mismatch")
    return CalibratedPredictor(payload["metadata"], payload["state"], device=device)
