"""Participant-bound calibration artifacts for later, unlabelled EEG decisions."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data.brainsync import (
    apply_brainsync_reference,
    load_brainsync_inference_session,
    resolve_brainsync_reference,
)
from data.brainsync_contract import ValidatedBrainSyncRecording
from data.channel import build_channel_identity, resolve_channel_layout
from data.continuous_qc import (
    continuous_qc_from_record,
    minimum_usable_channel_count,
    validate_continuous_calibration_budget,
)
from data.contract import assert_personal_p300_input_contract
from data.epochs import (
    EpochDataset,
    PreprocessingSpec,
    materialize_dataset_identity,
    preprocessing_execution_binding,
    preprocessing_execution_identity,
)
from data.exclusions import assert_origin_keys_allowed
from data.identity import DatasetIdentityTable
from data.ocular import eog_regression_from_record, validate_eog_calibration_budget
from models.adapters import FeatureDomainTransfer
from models.n2p3net import N2P3ArchitectureConfig, N2P3Net
from research.contracts import semantic_sha256
from train.early_stopping import UnifiedTrainingBudget
from transfer.binary import (
    fit_shrinkage_feature_model,
    plain_trunk_configuration,
    target_shrinkage_features,
)
from transfer.calibration_candidates import (  # noqa: F401
    SELECTOR_MODES,
    CalibrationCandidateFitCache,
    FittedCalibrationCandidate,
    calibration_candidate_recipes,
    calibration_decision_groups,
    fit_calibration_candidate,
    score_calibration_candidate,
    select_calibration_candidate,
    validate_calibration_quality_ingress,
    validate_calibration_raw_fit_scope,
    waveform_lda_recipe,
    waveform_lda_training_selection,
)
from transfer.calibration_split import dataset_calibration_inner_split
from transfer.checkpoint import (
    CHECKPOINT_SCHEMA,
    checkpoint_classifier_is_trained,
    checkpoint_input_stats,
    checkpoint_training_contract,
    load_n2p3_trunk_checkpoint,
    predict_n2p3_checkpoint,
)
from transfer.quality import (
    EpochQualityModel,
    EpochQualityPolicy,
    PersonalEpochRawQualityError,
    epoch_quality_channel_count,
    epoch_quality_from_record,
    validate_epoch_quality_fit_scope,
    validate_personal_epoch_raw_quality,
)
from transfer.recipes import TargetRecipe
from transfer.subject_adapter import SubjectAdapter

CALIBRATED_PREDICTOR_SCHEMA = "n2p3_calibrated_predictor/4"


@dataclass(frozen=True)
class CalibrationRecipe:
    name: str
    epochs: int
    learning_rate: float
    initialization: str = "source"
    input_statistics: str = "target_prefix"
    bn_buffers: str = "online"
    bn_affine: str = "trainable"
    dropout: str = "checkpoint"

    def __post_init__(self) -> None:
        if not self.name.strip() or self.epochs < 1 or not np.isfinite(self.learning_rate):
            raise ValueError("calibration recipe requires a name and positive finite budget")
        if self.learning_rate <= 0:
            raise ValueError("calibration learning rate must be positive")

    def target_recipe(self, *, batch_size: int, seed: int) -> TargetRecipe:
        return TargetRecipe(
            self.name, initialization=self.initialization, input_statistics=self.input_statistics,
            bn_buffers=self.bn_buffers, bn_affine=self.bn_affine, dropout=self.dropout,
            epochs=self.epochs, lr=self.learning_rate, batch_size=batch_size, seed=seed,
            weights="none" if self.initialization == "waveform_lda" else "non_bn",
        )


DEFAULT_CALIBRATION_RECIPES = (
    CalibrationRecipe("full_fine_low", 15, 1e-4),
    CalibrationRecipe("full_fine_high", 60, 1e-3),
)


def _resolve_legacy_calibration_recipes(
    recipes: Sequence[CalibrationRecipe | TargetRecipe], *, batch_size: int, seed: int,
    transfer_policy: TargetRecipe | None = None,
) -> tuple[TargetRecipe, ...]:
    """Resolve the same executable legacy policy before choosing a quality route.

    CalibrationRecipe retains its historical budget-only statistics/BN defaults
    in legacy mode. TargetRecipe is a complete declaration and is never rewritten.
    """
    if transfer_policy is not None:
        raise ValueError("legacy product replay has a fixed statistics/BN policy")
    resolved = []
    for recipe in recipes:
        if isinstance(recipe, CalibrationRecipe):
            if (recipe.initialization != "source" or recipe.bn_affine != "trainable"
                    or recipe.dropout != "checkpoint"
                    or recipe.input_statistics not in {"target_prefix", "source"}
                    or recipe.bn_buffers not in {"online", "source"}):
                raise ValueError("CalibrationRecipe conflicts with the explicit legacy product replay configuration")
            recipe = TargetRecipe(
                recipe.name, epochs=recipe.epochs, lr=recipe.learning_rate,
                input_statistics="source", bn_buffers="source", batch_size=batch_size, seed=seed,
            )
        if not isinstance(recipe, TargetRecipe) or (
            recipe.initialization != "source" or recipe.input_statistics != "source"
            or recipe.bn_buffers != "source" or recipe.bn_affine != "trainable"
            or recipe.dropout != "checkpoint" or recipe.weights != "non_bn"
            or recipe.batch_size != batch_size or recipe.seed != seed
            or recipe.max_updates is not None or recipe.refit_full_prefix
            or recipe.candidate_loss_weight is not None or recipe.source_prior_weight != 0
        ):
            raise ValueError("TargetRecipe conflicts with the explicit legacy product replay configuration")
        resolved.append(recipe)
    if not resolved or len({recipe.name for recipe in resolved}) != len(resolved):
        raise ValueError("legacy calibration recipes must be a nonempty unique named set")
    return tuple(resolved)


def _load_calibration_checkpoint(source):
    # NumPy-written source identity ledgers can contain Unicode scalar keys.
    scalar_constructor = np.str_("").__reduce__()[0]
    with torch.serialization.safe_globals([scalar_constructor, np.dtype, type(np.dtype("U"))]):
        return torch.load(source, map_location="cpu", weights_only=True)


def load_calibration_source_checkpoint(dataset: EpochDataset, source_checkpoint: str | Path | None):
    """Read the common source gate without fitting target statistics or weights."""
    if source_checkpoint is None:
        return None, None, None, {"status": "not_supplied"}
    source_path = Path(source_checkpoint)
    source_hash, payload = None, None
    try:
        source_bytes = source_path.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        payload = _load_calibration_checkpoint(io.BytesIO(source_bytes))
        if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("source checkpoint must use the current audited transfer schema")
        if not checkpoint_classifier_is_trained(payload):
            raise ValueError("source checkpoint needs a supervised source classifier")
        subject, _ = _participant(dataset)
        source, _ = load_n2p3_trunk_checkpoint(payload, dataset, target_subject=subject)
        stats = checkpoint_input_stats(payload, dataset.n_channels)
        return source, payload, stats, {"status": "compatible", "checkpoint_sha256": source_hash}
    except (OSError, EOFError, pickle.UnpicklingError) as error:
        status = "unreadable"
        failure = {"type": type(error).__name__, "message": str(error)}
    except (ValueError, RuntimeError) as error:
        status = "incompatible"
        failure = {"type": type(error).__name__, "message": str(error)}
    return None, payload, None, {"status": status, "checkpoint_sha256": source_hash,
                                **failure}


def _resolve_calibration_batch_rule(selector_mode, batch_size, batch_rule):
    if selector_mode == "legacy_product":
        if batch_size not in {None, 64} or batch_rule is not None:
            raise ValueError("legacy product replay uses fixed batch64 independently of B0")
        return 64, {"schema": "n2p3_legacy_product_batch/1", "batch_size": 64,
                    "selection_evidence": False, "reason": "historical_product_replay_independent_of_B0"}
    from transfer.target_experiments import (
        operator_fixed_target_batch_rule,
        resolve_target_batch_rule,
    )

    if selector_mode == "unified" and batch_rule is None:
        batch_rule = operator_fixed_target_batch_rule(batch_size=batch_size,
            rationale="Explicit unified calibration recipe; no target-score batch selection")
    plan = {} if batch_rule is None else {"batch_rule": dict(batch_rule)}
    return resolve_target_batch_rule(plan, requested_batch=batch_size)


def _calibration_source_failure(gate, selector_mode):
    if gate["status"] == "unreadable":
        return f"source checkpoint cannot be read: {gate['message']}"
    if selector_mode in {"legacy_product", "unified"} and gate["status"] == "incompatible":
        return f"source checkpoint is incompatible: {gate['message']}"
    return None


def _validate_unified_recipe(recipe, source, *, batch_size, seed, stopping):
    if not isinstance(recipe, TargetRecipe):
        raise ValueError("unified calibration requires one explicit TargetRecipe")
    if (recipe.initialization not in {"source", "scratch"} or recipe.weights != "non_bn"
            or recipe.candidate_loss_weight is None or recipe.max_updates is not None or recipe.refit_full_prefix
            or recipe.epochs != stopping.target_epochs
            or recipe.bn_buffers != "calibration" or recipe.bn_affine != "trainable"
            or recipe.dropout != "off" or recipe.batch_size != batch_size or recipe.seed != seed):
        raise ValueError("unified calibration requires a joint full-model recipe with support BN and fixed target budget")
    if recipe.initialization == "source":
        if type(source) is not N2P3Net or not source.observed_channel_norm:
            raise ValueError("unified calibration requires an observed-channel source checkpoint")
    elif recipe.source_prior_weight != 0:
        raise ValueError("scratch calibration cannot claim a source-centered prior")


def _lda_calibration_gate(dataset, training, *, require_all_channels=False):
    requested = int(np.asarray(training).sum())
    try:
        channels, usable, policy = waveform_lda_training_selection(dataset, training)
    except ValueError as error:
        return {"legal": False, "requested_training_rows": requested, "training_rows": 0,
                "observed_channel_indices": [], "observed_channel_count": 0,
                "feature_count": 0, "waveform_training_policy": None, "reason": str(error)}
    reason = None
    if int(usable.sum()) <= 2 or set(np.unique(dataset.y[usable])) != {0, 1} or not len(channels):
        reason = "waveform LDA requires >2 usable rows, both classes and observed channels"
    if require_all_channels and not bool(_observed(dataset)[training].all()):
        reason = "LDA fallback needs all selected EEG channels; correct channel QC first"
    feature_count = (int(target_shrinkage_features(
        np.zeros((1, len(channels), dataset.n_times), dtype=dataset.X.dtype),
        sfreq=dataset.preprocessing.sfreq,
    ).shape[1]) if len(channels) else 0)
    return {"legal": reason is None, "requested_training_rows": requested,
            "training_rows": int(usable.sum()), "observed_channel_indices": channels.tolist(),
            "observed_channel_count": len(channels), "feature_count": feature_count,
            "waveform_training_policy": policy,
            "reason": reason}


def _validate_calibration_preprocessing(dataset, declared_preprocessing):
    if declared_preprocessing is None:
        assert_personal_p300_input_contract(dataset.preprocessing)
        return
    declared_preprocessing.validate()
    if declared_preprocessing.filter_phase == "zero" and declared_preprocessing.zero_phase_edge_policy != "mne_reflect_limited":
        raise ValueError("personal zero-phase calibration requires the declared finite-record boundary policy")
    if dataset.preprocessing != declared_preprocessing:
        raise ValueError("calibration data differ from the explicitly declared preprocessing")


def preflight_calibrated_predictor(
    dataset: EpochDataset, *, source_checkpoint: str | Path | None = None,
    selector_mode: str = "legacy_product", selection_protocol: str = "binary_auc",
    batch_size: int | None = None, seed: int = 20260906, planned_ocular: bool = False,
    batch_rule: Mapping[str, object] | None = None,
    epoch_quality_policy: EpochQualityPolicy | None = None,
    epoch_quality_model: EpochQualityModel | None = None,
    declared_preprocessing: PreprocessingSpec | None = None,
    transfer_policy: TargetRecipe | None = None,
    unified_training_budget: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return actual input gates and candidate scopes with no learned fitting."""
    validate_calibration_quality_ingress(dataset)
    if selector_mode == "unified" and transfer_policy is not None and batch_size is None:
        batch_size = transfer_policy.batch_size
    raw_quality_failure = None
    try:
        validate_epoch_quality_fit_scope(dataset, np.ones(dataset.n_epochs, dtype=bool),
                                         policy=epoch_quality_policy, model=epoch_quality_model)
    except PersonalEpochRawQualityError as error:
        raw_quality_failure = str(error)
    batch_size, applied_batch_rule = _resolve_calibration_batch_rule(selector_mode, batch_size, batch_rule)
    dataset.validate(require_labels=True)
    _validate_calibration_preprocessing(dataset, declared_preprocessing)
    subject, participant = _participant(dataset)
    sessions = _session_scope(dataset, calibration=True)
    source, _, _, gate = load_calibration_source_checkpoint(dataset, source_checkpoint)
    candidates = (
        _resolve_legacy_calibration_recipes(DEFAULT_CALIBRATION_RECIPES, batch_size=batch_size, seed=seed)
        if selector_mode == "legacy_product" else
        calibration_candidate_recipes(selector_mode, batch_size=batch_size, seed=seed,
                                      transfer_policy=transfer_policy)
    )
    fixed_lda = selector_mode == "legacy_product" and source_checkpoint is None
    if fixed_lda:
        candidates = (waveform_lda_recipe(batch_size=batch_size, seed=seed),)
    unified = selector_mode == "unified"
    stopping = UnifiedTrainingBudget(**dict(unified_training_budget or {})) if unified else None
    fixed_fit = fixed_lda or unified
    split = dataset_calibration_inner_split(dataset)
    inner = split.record()
    if len(sessions) < 2:
        inner.update(legal=False, reason="product selection requires separate complete calibration sessions")
    selection_planned = bool(inner["legal"] and not fixed_fit)
    continuous = dataset.provenance.get("continuous_artifact_processing", {})
    local_quality = (continuous.get("status") == "applied"
                     and continuous.get("model", {}).get("spec", {}).get("channel_local") is True)
    strict_legacy_lda = fixed_lda and not local_quality
    lda_gate = _lda_calibration_gate(
        dataset, np.ones(dataset.n_epochs, dtype=bool), require_all_channels=strict_legacy_lda,
    )
    lda_legal = lda_gate["legal"]
    data_failures = [] if raw_quality_failure is None else [raw_quality_failure]
    if unified:
        try:
            _validate_unified_recipe(transfer_policy, source, batch_size=batch_size, seed=seed, stopping=stopping)
            groups, _, _, _ = calibration_decision_groups(dataset, np.ones(dataset.n_epochs, dtype=bool))
            if not groups:
                raise ValueError("unified calibration has no authorized physical decision groups")
        except ValueError as error:
            data_failures.append(str(error))
    if selector_mode == "legacy_product":
        session_ids = dataset.metadata["session"].astype(str).to_numpy()
        if any(set(np.unique(dataset.y[session_ids == record["session_id"]])) != {0, 1} for record in sessions):
            data_failures.append("every legacy calibration session must retain target and non-target trials")
    scoring_gate = {"status": "not_required_without_inner_selection"}
    if selection_planned:
        if selector_mode in {"legacy_product", "unified"}:
            scoring_gate = {"status": "compatible", "protocol": (
                "fixed_full_calibration" if unified else "legacy_epoch_validation_cross_entropy")}
        else:
            try:
                score_calibration_candidate(dataset, np.zeros(int(split.validation_mask.sum())),
                                            split.validation_mask, protocol=selection_protocol)
                scoring_gate = {"status": "compatible", "protocol": selection_protocol}
            except ValueError as error:
                scoring_gate = {"status": "incompatible", "protocol": selection_protocol, "message": str(error)}
                data_failures.append(str(error))
    source_failure = _calibration_source_failure(gate, selector_mode)
    if source_failure is not None:
        data_failures.append(source_failure)
    if selector_mode == "legacy_product" and source_checkpoint is not None and not inner["legal"]:
        data_failures.append(inner["reason"])
    eligible = []
    for recipe in candidates:
        source_eligible = recipe.initialization != "source" or gate["status"] == "compatible"
        train = np.ones(dataset.n_epochs, dtype=bool) if fixed_fit else split.training_mask
        legal = source_eligible and int(train.sum()) > 2 and (selection_planned or fixed_fit)
        fit_gate = {}
        if recipe.initialization == "waveform_lda":
            fit_gate = _lda_calibration_gate(dataset, train, require_all_channels=strict_legacy_lda)
            legal = legal and fit_gate["legal"]
        eligible.append({"candidate_id": recipe.name, "fit_eligible": bool(legal),
                         "training_rows": int(train.sum()), "validation_rows": int(split.validation_mask.sum()) if selection_planned else 0,
                         **{key: value for key, value in fit_gate.items() if key not in {"legal", "reason"}},
                         "reason": None if legal else (fit_gate.get("reason") or "source_gate_or_no_legal_inner_training_scope")})
    fallback_planned = not any(row["fit_eligible"] for row in eligible) and not (fixed_fit or unified)
    if (fallback_planned or fixed_lda) and not lda_legal:
        data_failures.append(lda_gate["reason"])
    ocular_scope = split.training_mask if selection_planned else np.ones(dataset.n_epochs, dtype=bool)
    ocular_ids = set(dataset.metadata.loc[ocular_scope, "session"].astype(str))
    ocular_sessions = tuple(record["session_id"] for record in sessions if record["session_id"] in ocular_ids)
    try:
        validate_calibration_raw_fit_scope(dataset, ocular_scope)
    except ValueError as error:
        data_failures.append(str(error))
    try:
        validate_epoch_quality_fit_scope(dataset, ocular_scope,
                                         policy=epoch_quality_policy, model=epoch_quality_model)
    except ValueError as error:
        if str(error) not in data_failures:
            data_failures.append(str(error))
    return {
        "status": "dry_run", "data_fitted": False, "model_written": False,
        "selector_mode": selector_mode, "selection_protocol": (
            "fixed_full_calibration" if unified else
            "legacy_epoch_validation_cross_entropy" if selector_mode == "legacy_product" else selection_protocol
        ),
        "participant_key": participant, "subject_id": subject,
        "candidate_count": len(candidates), "candidates": [recipe.record() for recipe in candidates],
        "batch_rule": applied_batch_rule,
        "candidate_fit_gates": eligible, "inner_split": inner,
        "inner_training_rows": [] if unified else np.flatnonzero(split.training_mask).tolist(),
        "inner_validation_rows": [] if unified else np.flatnonzero(split.validation_mask).tolist(),
        "inner_selection_planned": selection_planned,
        "full_refit_rows": dataset.n_epochs, "full_calibration_lda_gate": lda_gate,
        "source_checkpoint_supplied": source_checkpoint is not None, "source_checkpoint_gate": gate,
        "scoring_gate": scoring_gate, "execution_ready": not data_failures,
        "execution_blockers": data_failures, "fallback_planned": fallback_planned,
        "fallback": None if unified else "full_calibration_waveform_lda_without_selection_evidence",
        **({"unified_training_budget": stopping.record()} if unified else {}),
        "calibration_sessions": [record["session_id"] for record in sessions],
        "calibration_data_sha256": _calibration_data_digest(dataset), "input_contract": _input_contract(dataset),
        "planned_ocular": {"requested": planned_ocular, "fitted": False,
                           "fit_sessions": list(ocular_sessions), "fit_rows": int(ocular_scope.sum()),
                           "lifecycle": ("freeze_after_full_calibration_fit" if fixed_fit else
                                         "freeze_after_inner_training_fit_through_refit_and_inference")},
        "input_gate_basis": "deterministic_preprocessing_without_planned_eog" if planned_ocular else "actual_deterministic_input",
        "repeat_input_gates_after_planned_eog": planned_ocular,
        "planned_epoch_quality": {
            "requested": epoch_quality_policy is not None or epoch_quality_model is not None,
            "method": (epoch_quality_model.record()["method"] if epoch_quality_model is not None else
                       getattr(epoch_quality_policy, "method", "fold_local_ptp_cv")
                       if epoch_quality_policy is not None else None),
            "lifecycle": "frozen_explicit" if epoch_quality_model is not None else
                         "fit_inside_calibration" if epoch_quality_policy is not None else "not_applied",
            "fitted": False,
        },
        "evidence_scope": "software_plan_no_device_or_accuracy_evidence",
    }


def _json_record(value: object) -> dict[str, object]:
    return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))


def _validate_raw_input_request(request: dict, supplied: Mapping | None) -> None:
    if supplied is not None and any(request.get(key) != value for key, value in supplied.items()):
        raise ValueError("raw input request disagrees with the actual calibrated preprocessing")


def _input_contract(dataset: EpochDataset) -> dict[str, object]:
    reference = dataset.provenance.get("source_reference")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("calibrated predictors require an explicit physical reference")
    return _json_record({
        "channel_names": list(dataset.channel_names),
        "channel_positions_m": np.asarray(dataset.channel_positions_m).tolist(),
        "preprocessing": dataset.preprocessing.record(),
        "source_reference": reference,
        "n_channels": dataset.n_channels,
        "n_times": dataset.n_times,
        "simulation_status": dataset.provenance.get("simulation_status", "not_declared"),
        "ocular_artifact_processing": dataset.provenance.get(
            "ocular_artifact_processing", {"status": "not_applied"}
        ),
        "continuous_artifact_processing": dataset.provenance.get(
            "continuous_artifact_processing", {"status": "not_applied"}),
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
        decision_score = self.metadata.get("decision_score")
        if decision_score is not None and (
            decision_score.get("method") != "membership_llr_sum/1"
            or decision_score.get("prior") != "uniform"
            or decision_score.get("coverage") != "complete_physical_schedule"
            or decision_score.get("probability_calibrated") is not False
            or not isinstance(decision_score.get("temperature"), (int, float))
            or not np.isfinite(decision_score["temperature"]) or decision_score["temperature"] <= 0
        ):
            raise ValueError("calibrated decision-score contract is invalid")
        self.epoch_quality = None
        quality = self.metadata.get("epoch_quality")
        if quality is not None:
            if quality.get("model_digest") != semantic_sha256(quality.get("model")):
                raise ValueError("calibrated epoch QC digest is invalid")
            self.epoch_quality = epoch_quality_from_record(quality["model"])
        contract = self.metadata["input_contract"]
        if not isinstance(contract, dict) or not self.metadata.get("participant_key"):
            raise ValueError("calibrated model lacks participant and input contracts")
        execution = self.metadata.get("preprocessing_execution")
        binding = preprocessing_execution_binding(execution, PreprocessingSpec(**contract["preprocessing"]))
        if binding != self.metadata.get("preprocessing_execution_binding", {}):
            raise ValueError("personal model preprocessing execution binding differs from its evidence")
        if self.epoch_quality is not None:
            fitted = self.epoch_quality
            validate_personal_epoch_raw_quality(contract.get("continuous_artifact_processing", {}), fitted)
            expected = {"participant_key": self.metadata["participant_key"],
                        "source_reference": contract["source_reference"],
                        "channel_names": contract["channel_names"], "preprocessing": contract["preprocessing"]}
            if any(_json_record({"value": fitted.fit_provenance.get(key)}) != _json_record({"value": value})
                   for key, value in expected.items()):
                raise ValueError("calibrated epoch QC participant or input contract differs")
            if epoch_quality_channel_count(fitted) != contract["n_channels"]:
                raise ValueError("calibrated epoch QC channel geometry differs")
            calibration_ids = {row["session_id"] for row in self.metadata["calibration_sessions"]}
            if not set(fitted.fit_provenance.get("session_ids", [])) <= calibration_ids:
                raise ValueError("calibrated epoch QC exceeds the calibration session budget")
        self.trunk = None
        if self.metadata["model_kind"] in {"source_initialized_n2p3", "scratch_n2p3"}:
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
            if self.metadata.get("recipe", {}).get("weights") == "adapter":
                recipe = TargetRecipe(**self.metadata["recipe"])
                self.trunk = FeatureDomainTransfer(self.trunk, config=recipe.target_adapter_config)
                self.trunk.set_target_adapter_enabled(True)
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
            selected = self.state.get("waveform_channels")
            if selected is not None and (
                selected.dtype != torch.int64 or selected.ndim != 1 or not selected.numel()
                or selected.unique().numel() != selected.numel()
                or bool((selected < 0).any()) or bool((selected >= contract["n_channels"]).any())
            ):
                raise ValueError("calibrated LDA channel selection is invalid")
            feature_shape = self.state["coefficient"].shape
            if len(feature_shape) != 1 or self.state["feature_mean"].shape != feature_shape or (
                self.state["feature_scale"].shape != feature_shape
                or not bool((self.state["feature_scale"] > 0).all())
                or self.state["intercept"].numel() != 1
            ):
                raise ValueError("calibrated LDA feature state has invalid shape or scale")
        if any(not bool(torch.isfinite(value).all()) for value in self.state.values()):
            raise ValueError("calibrated predictor contains nonfinite state")

    def load_inference_session(self, session_dir: str | Path) -> EpochDataset:
        """Replay this model's frozen raw input without fitting or predicting.

        Identity, prospective time and simulation admission precede raw QC,
        including when QC leaves no usable epochs. The returned target-free
        dataset has the saved final montage and passed the model input gates.
        """
        request = self.metadata["input_request"]
        contract = self.metadata["input_contract"]
        channels = tuple(request["channel_names"])
        if channels != tuple(contract["channel_names"]):
            raise ValueError("saved raw input request channels differ from the model input contract")
        for name, processing_name in (
            ("ocular_model", "ocular_artifact_processing"),
            ("continuous_qc_model", "continuous_artifact_processing"),
        ):
            processing = contract.get(processing_name, {})
            expected = processing["model"] if processing.get("status") == "applied" else None
            if request.get(name) != expected:
                raise ValueError(f"saved raw input request {name} differs from the model input contract")
        preprocessing = PreprocessingSpec(**contract["preprocessing"])
        preprocessing.validate()
        ocular = eog_regression_from_record(request["ocular_model"]) if request.get("ocular_model") is not None else None
        continuous = continuous_qc_from_record(request["continuous_qc_model"]) if request.get("continuous_qc_model") is not None else None
        reference = resolve_brainsync_reference(
            request.get("reference_channel"), ocular, continuous,
            mode=request.get("reference_mode", "auto"),
        )
        if (reference is not None and contract["source_reference"] != "measured_channel:" + reference
                or reference is None and contract["source_reference"].startswith("measured_channel:")):
            raise ValueError("saved raw input reference differs from the model input contract")
        raw_reference = (
            ocular.reference_channel if ocular is not None else
            continuous.spec.reference_channel if continuous is not None and continuous.spec.channel_local else None
        )
        post_reference = reference is not None and raw_reference != reference
        # Raw EOG/QC can select the final subset after referencing. Post-epoch
        # referencing must retain the measured anchor until it has been used.
        dataset = load_brainsync_inference_session(
            session_dir, preprocessing=preprocessing, ocular_model=ocular,
            continuous_qc_model=continuous,
            channels=None if post_reference else channels,
            recording_validator=lambda recording: self._validate_inference_recording(
                recording, post_reference=post_reference),
        )
        dataset = apply_brainsync_reference(dataset, reference, ocular, channels=channels)
        self.validate_input(dataset)
        self.validate_inference_scope(dataset)
        return dataset

    def _validate_inference_recording(
        self, recording: ValidatedBrainSyncRecording, *, post_reference: bool,
    ) -> None:
        participant = DatasetIdentityTable.from_source_rows(
            [recording.subject_id], ["BrainSync-GTN"]
        ).record_for(recording.subject_id).authority_key("source_or_global")
        if participant != self.metadata["participant_key"]:
            raise ValueError("personalized model belongs to a different participant")
        self._validate_inference_session(recording.session_id, recording.started_timestamp_s)
        device = recording.manifest.get("device")
        transport = device.get("transport") if isinstance(device, dict) else None
        simulation = "declared_mock" if transport == "mock" else "not_declared"
        if simulation != self.metadata["input_contract"]["simulation_status"]:
            raise ValueError("recording simulation status differs from the model input contract")
        references = self.metadata["input_request"].get("acquisition_references", [])
        if references and not any(
            item["label"] == recording.bids.source_reference
            and item["details"] == dict(recording.bids.source_reference_details)
            for item in references
        ):
            raise ValueError("recording acquisition reference differs from the calibrated input")
        contract = self.metadata["input_contract"]
        if post_reference:
            # The projection produces standard coordinates. Its raw-coordinate
            # check depends on observed masks and must remain after raw QC.
            positions = build_channel_identity(contract["channel_names"], allow_missing_positions=False).coords
        else:
            positions = resolve_channel_layout(
                contract["channel_names"], montage=None, allow_missing=False,
                positions_m={name: position for name, kind, position in zip(
                    recording.bids.channel_names, recording.bids.channel_types,
                    recording.bids.channel_positions_m, strict=True,
                ) if kind == "EEG"}, coordinate_source="bids_electrodes_tsv",
            ).positions_m
        if positions.tolist() != contract["channel_positions_m"]:
            raise ValueError("recording channel geometry differs from the model input contract")

    def validate_input(self, dataset: EpochDataset) -> None:
        """Check a recording against this frozen model without evaluating a classifier."""
        dataset.validate(require_labels=False)
        if _input_contract(dataset) != self.metadata["input_contract"]:
            raise ValueError(
                "model input channels, reference, ocular processing, preprocessing or simulation status differ"
            )
        expected_execution = self.metadata.get("preprocessing_execution_binding", {}).get("execution_identity")
        if expected_execution is not None and preprocessing_execution_identity(dataset) != expected_execution:
            raise ValueError("model requires compatible preprocessing execution evidence")
        _, participant = _participant(dataset)
        if participant != self.metadata["participant_key"]:
            raise ValueError("personalized model belongs to a different participant")

    def predict_logit(self, dataset: EpochDataset) -> np.ndarray:
        self.validate_input(dataset)
        observed = _observed(dataset)
        values = dataset.X
        available = np.ones(dataset.n_epochs, dtype=bool)
        if self.epoch_quality is not None:
            if dataset.provenance.get("epoch_artifact_processing", {}).get("status") == "applied":
                raise ValueError("epoch QC must not be applied twice")
            transformed = self.epoch_quality.transform(values, observed)
            values, observed = transformed.X, transformed.trial_channel_mask
            available &= ~transformed.drop_epoch_mask & ~transformed.all_channels_bad
        logits = np.full(dataset.n_epochs, np.nan)
        available &= observed.sum(axis=1) >= minimum_usable_channel_count(
            self.metadata["input_contract"].get("continuous_artifact_processing", {}), dataset.n_channels)
        if self.trunk is not None:
            if not available.any():
                return logits
            logits[available] = predict_n2p3_checkpoint(
                self.trunk, values[available],
                input_stats=(self.state["input_mean"].numpy(), self.state["input_std"].numpy()),
                trial_channel_mask=observed[available], device=self.device,
            )
            return logits
        selected_channels = self.state.get("waveform_channels")
        selected_channels = np.arange(dataset.n_channels) if selected_channels is None else selected_channels.numpy()
        continuous = self.metadata["input_contract"].get("continuous_artifact_processing", {})
        channel_quality_availability = (
            continuous.get("status") == "applied"
            and continuous.get("model", {}).get("spec", {}).get("channel_local") is True
        )
        if self.epoch_quality is not None or channel_quality_availability:
            available &= observed[:, selected_channels].all(axis=1)
        elif not bool(observed[:, selected_channels].all()):
            raise ValueError("target-only LDA requires every selected EEG channel to be usable")
        if not available.any():
            return logits
        features = target_shrinkage_features(values[available][:, selected_channels], sfreq=dataset.preprocessing.sfreq)
        standardized = (features - self.state["feature_mean"].numpy()) / self.state["feature_scale"].numpy()
        logits[available] = np.asarray(
            standardized @ self.state["coefficient"].numpy() + self.state["intercept"].item(),
            dtype=np.float64,
        )
        return logits

    def validate_inference_scope(self, dataset: EpochDataset) -> None:
        for row in _session_scope(dataset, calibration=False):
            self._validate_inference_session(row["session_id"], row["session_start_timestamp_s"])

    def _validate_inference_session(self, session_id: str, started_timestamp_s: float) -> None:
        calibration_ids = {row["session_id"] for row in self.metadata["calibration_sessions"]}
        if session_id in calibration_ids:
            raise ValueError("calibration recordings cannot be reused as independent inference")
        if started_timestamp_s <= self.metadata["calibration_available_timestamp_s"]:
            raise ValueError("inference recording starts before calibration labels were available")


def _reference_request_mode(input_request: Mapping[str, object] | None, reference_channel: str | None) -> str:
    mode = (input_request or {}).get("reference_mode", "auto")
    if mode not in {"auto", "native"}:
        raise ValueError("unknown calibration reference mode")
    if mode == "native" and reference_channel is not None:
        raise ValueError("native acquisition reference conflicts with measured-reference preprocessing")
    return str(mode)


def fit_calibrated_predictor(
    dataset: EpochDataset, *, source_checkpoint: str | Path | None = None,
    recipes: Sequence[CalibrationRecipe | TargetRecipe] = DEFAULT_CALIBRATION_RECIPES,
    device: str = "cpu", seed: int = 20260906, batch_size: int | None = None,
    input_request: Mapping[str, object] | None = None,
    progress: Callable[[int, str], None] | None = None,
    selector_mode: str = "legacy_product", selection_protocol: str = "binary_auc",
    transfer_policy: TargetRecipe | None = None,
    batch_rule: Mapping[str, object] | None = None,
    epoch_quality_policy: EpochQualityPolicy | None = None,
    epoch_quality_model: EpochQualityModel | None = None,
    declared_preprocessing: PreprocessingSpec | None = None,
    unified_training_budget: Mapping[str, object] | None = None,
) -> CalibratedPredictor:
    """Choose source adaptation only inside calibration, then refit its full scope."""
    validate_calibration_quality_ingress(dataset)
    if selector_mode == "unified" and transfer_policy is not None and batch_size is None:
        batch_size = transfer_policy.batch_size
    validate_epoch_quality_fit_scope(dataset, np.ones(dataset.n_epochs, dtype=bool),
                                     policy=epoch_quality_policy, model=epoch_quality_model)
    if selector_mode not in SELECTOR_MODES:
        raise ValueError(f"unknown calibration selector mode: {selector_mode}")
    batch_size, applied_batch_rule = _resolve_calibration_batch_rule(selector_mode, batch_size, batch_rule)
    if selector_mode == "legacy_product":
        recipes = _resolve_legacy_calibration_recipes(
            recipes, batch_size=batch_size, seed=seed, transfer_policy=transfer_policy)
    continuous = dataset.provenance.get("continuous_artifact_processing", {})
    local_quality = continuous.get("status") == "applied" and continuous.get("model", {}).get("spec", {}).get("channel_local") is True
    if selector_mode != "legacy_product" or epoch_quality_policy is not None or epoch_quality_model is not None or local_quality:
        return _fit_candidate_predictor(
            dataset, source_checkpoint=source_checkpoint, recipes=recipes, device=device,
            seed=seed, batch_size=batch_size, input_request=input_request, progress=progress,
            selector_mode=selector_mode, selection_protocol=selection_protocol,
            transfer_policy=transfer_policy,
            batch_rule=applied_batch_rule,
            epoch_quality_policy=epoch_quality_policy,
            epoch_quality_model=epoch_quality_model,
            declared_preprocessing=declared_preprocessing,
            unified_training_budget=unified_training_budget,
        )
    dataset.validate(require_labels=True)
    _validate_calibration_preprocessing(dataset, declared_preprocessing)
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
    continuous_record = _continuous_record(dataset, participant, sessions)
    reference_record = dataset.provenance.get("domain_adapter", {}).get("reference", {})
    inferred_reference = (
        ocular_record.get("reference_channel") if ocular_record is not None
        else reference_record.get("channel") if reference_record.get("mode") == "measured_channel"
        else None
    )
    if continuous_record is not None and continuous_record["spec"].get("reference_channel") is not None:
        inferred_reference = continuous_record["spec"]["reference_channel"]
    request = {"reference_channel": inferred_reference, "ocular_model": ocular_record,
               "channel_names": list(dataset.channel_names)}
    request["acquisition_references"] = copy.deepcopy(dataset.provenance.get("acquisition_references", []))
    request["reference_mode"] = _reference_request_mode(input_request, inferred_reference)
    request["continuous_qc_model"] = continuous_record
    _validate_raw_input_request(request, input_request)
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "participant_key": participant, "subject_id": subject, "calibration_sessions": sessions,
        "calibration_available_timestamp_s": max(row["calibration_available_timestamp_s"] for row in sessions),
        "calibration_trials": dataset.n_epochs,
        "calibration_positives": int(dataset.y.sum()),
        "calibration_data_sha256": _calibration_data_digest(dataset),
        "input_contract": _input_contract(dataset), "input_request": request,
        "preprocessing_execution": copy.deepcopy(dataset.provenance.get("preprocessing_execution")),
        "preprocessing_execution_binding": preprocessing_execution_binding(
            dataset.provenance.get("preprocessing_execution"), dataset.preprocessing),
        "candidate_vocabulary": list(range(1, 10)),
        "evidence_scope": "personalized_development_not_independent_performance_confirmation",
        "seed": seed,
        "batch_rule": applied_batch_rule,
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

    payload = _load_calibration_checkpoint(source_checkpoint)
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
    train_mask = np.isin(session_ids, [row["session_id"] for row in training_sessions])
    validation_mask = ~train_mask
    validate_calibration_raw_fit_scope(dataset, train_mask)
    recipe_records = []
    for index, recipe in enumerate(recipes):
        notify(15 + int(55 * index / len(recipes)), f"select_{recipe.name}")
        adapter = SubjectAdapter(
            copy.deepcopy(source), device=torch.device(device),
            config=recipe.adapter_config(input_mean=source_mean, input_std=source_std),
        )
        adapter.fit(dataset.X, dataset.y, training_mask=train_mask, validation_mask=validation_mask,
                    prefix_mask=np.ones(dataset.n_epochs, dtype=bool), trial_channel_mask=_observed(dataset))
        history = adapter.last_history
        if history["best_epoch"] is None or history["best_val_loss"] is None:
            raise ValueError("internal validation did not select a finite training epoch")
        recipe_records.append({
            **recipe.record(), "learning_rate": recipe.lr,
            "selected_epochs": int(history["best_epoch"]),
            "validation_loss": float(history["best_val_loss"]),
            "validation_auc": float(roc_auc_score(dataset.y[validation_mask], adapter.calibration_logits_)),
            "fit_history": copy.deepcopy(history),
        })
    selected = min(recipe_records, key=lambda item: item["validation_loss"])
    refit_recipe = replace(
        next(recipe for recipe in recipes if recipe.name == selected["name"]),
        epochs=selected["selected_epochs"],
    )
    notify(78, "refit_full_authorized_calibration")
    adapter = SubjectAdapter(
        copy.deepcopy(source), device=torch.device(device),
        config=refit_recipe.adapter_config(input_mean=source_mean, input_std=source_std),
    )
    adapter.fit(dataset.X, dataset.y, prefix_mask=np.ones(dataset.n_epochs, dtype=bool),
                trial_channel_mask=_observed(dataset))
    metadata.update(
        model_kind="source_initialized_n2p3", source_pretraining_used=True,
        architecture=source.architecture_record(), architecture_configuration=asdict(configuration),
        pooling_mode=source.pooling_mode,
        source_checkpoint_sha256=hashlib.sha256(Path(source_checkpoint).read_bytes()).hexdigest(),
        source_training_contract=checkpoint_training_contract(payload).record(),
        recipe=refit_recipe.record(),
        selection={"mode": "chronological_calibration_decisions_then_full_refit",
                   "scoring_protocol": "legacy_epoch_validation_cross_entropy",
                   "training_sessions": [row["session_id"] for row in training_sessions],
                   "validation_sessions": [row["session_id"] for row in validation_sessions],
                   "recipes": recipe_records, "selected": selected,
                   "refit_recipe": refit_recipe.record(),
                   "refit_rows": dataset.n_epochs, "fit_scope": adapter.last_history["fit_scope"],
                   "refit_history": copy.deepcopy(adapter.last_history)},
    )
    state = {f"trunk.{name}": value.detach().cpu() for name, value in adapter.trunk.state_dict().items()}
    fitted_mean, fitted_std = adapter.fitted_input_statistics()
    state.update(input_mean=torch.as_tensor(fitted_mean), input_std=torch.as_tensor(fitted_std))
    metadata["actual_fitted_state_sha256"] = _state_digest(state)
    metadata["selection"]["refit_state_sha256"] = metadata["actual_fitted_state_sha256"]
    return CalibratedPredictor(metadata, state, device=device)


def _fit_candidate_predictor(
    dataset: EpochDataset, *, source_checkpoint, recipes, device, seed, batch_size,
    input_request, progress, selector_mode, selection_protocol, transfer_policy, batch_rule,
    epoch_quality_policy=None,
    epoch_quality_model=None,
    declared_preprocessing=None,
    unified_training_budget=None,
) -> CalibratedPredictor:
    dataset.validate(require_labels=True)
    _validate_calibration_preprocessing(dataset, declared_preprocessing)
    subject, participant = _participant(dataset)
    sessions = _session_scope(dataset, calibration=True)
    if not sessions:
        raise ValueError("calibration requires labelled recording sessions")
    ocular = dataset.provenance.get("ocular_artifact_processing", {"status": "not_applied"})
    validate_eog_calibration_budget(ocular, {participant: tuple(row["session_id"] for row in sessions)})
    ocular_record = ocular.get("model") if ocular.get("status") == "applied" else None
    continuous_record = _continuous_record(dataset, participant, sessions)
    reference = dataset.provenance.get("domain_adapter", {}).get("reference", {})
    reference_channel = ocular_record.get("reference_channel") if ocular_record is not None else (
        reference.get("channel") if reference.get("mode") == "measured_channel" else None
    )
    if continuous_record is not None and continuous_record["spec"].get("reference_channel") is not None:
        reference_channel = continuous_record["spec"]["reference_channel"]
    request = {"reference_channel": reference_channel, "ocular_model": ocular_record,
               "channel_names": list(dataset.channel_names)}
    request["acquisition_references"] = copy.deepcopy(dataset.provenance.get("acquisition_references", []))
    request["reference_mode"] = _reference_request_mode(input_request, reference_channel)
    request["continuous_qc_model"] = continuous_record
    _validate_raw_input_request(request, input_request)
    source, payload, stats, source_gate = load_calibration_source_checkpoint(dataset, source_checkpoint)
    source_failure = _calibration_source_failure(source_gate, selector_mode)
    if source_failure is not None:
        raise ValueError(source_failure)
    if selector_mode == "legacy_product":
        session_ids = dataset.metadata["session"].astype(str).to_numpy()
        if any(set(np.unique(dataset.y[session_ids == row["session_id"]])) != {0, 1} for row in sessions):
            raise ValueError("every calibration session must retain target and non-target trials")
        if source_checkpoint is not None:
            if len(sessions) < 2:
                raise ValueError("source recipe selection requires at least two separate labelled recording sessions")
            split = dataset_calibration_inner_split(dataset)
            if not split.legal:
                raise ValueError(split.reason)
    notify = progress or (lambda _percent, _message: None)
    if selector_mode == "unified":
        budget = UnifiedTrainingBudget(**dict(unified_training_budget or {}))
        _validate_unified_recipe(transfer_policy, source, batch_size=batch_size, seed=seed, stopping=budget)
        if source is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                source = N2P3Net(dataset.n_channels, n_times=dataset.n_times,
                    sfreq=dataset.preprocessing.sfreq, tmin_s=dataset.preprocessing.tmin_ms / 1000,
                    **N2P3ArchitectureConfig.unified().model_kwargs())
        notify(35, "fit_unified_full_calibration")
        fitted = fit_calibration_candidate(dataset, transfer_policy, np.ones(dataset.n_epochs, dtype=bool),
            source_trunk=source, source_input_stats=stats, device=device,
            epoch_quality_policy=epoch_quality_policy, epoch_quality_model=epoch_quality_model)
        selection = {"mode": "fixed_unified_program", "selector_mode": selector_mode,
            "scoring_protocol": "fixed_full_calibration", "selection_evidence": False,
            "unified_training_budget": budget.record(), "selected_epochs": transfer_policy.epochs,
            "fit_execution": {"requested_fits": 1, "fit_calls": 1, "cache_hits": 0},
            "recipes": [], "selected": transfer_policy.name,
            "refit_recipe": fitted.recipe.record(), "refit_rows": dataset.n_epochs,
            "fit_scope": fitted.fit_history["fit_scope"], "refit_history": fitted.fit_history,
            "refit_state_sha256": _state_digest(fitted.state),
        }
    elif selector_mode == "legacy_product" and source_checkpoint is None:
        notify(45, "fit_target_only_shrinkage_lda")
        fitted = fit_calibration_candidate(
            dataset, waveform_lda_recipe(batch_size=batch_size, seed=seed),
            np.ones(dataset.n_epochs, dtype=bool), device=device,
            epoch_quality_policy=epoch_quality_policy, epoch_quality_model=epoch_quality_model,
        )
        selection = {
            "mode": "fixed_fallback_no_recipe_selection", "selector_mode": selector_mode,
            "scoring_protocol": "not_applicable_fixed_recipe", "selection_evidence": False,
            "recipes": [], "selected": None, "validation_auc": None,
            "refit_rows": dataset.n_epochs, "refit_recipe": fitted.recipe.record(),
            "fit_scope": fitted.fit_history["fit_scope"], "refit_history": fitted.fit_history,
            "refit_state_sha256": _state_digest(fitted.state),
            "fit_execution": {"requested_fits": 1, "fit_calls": 1, "cache_hits": 0},
        }
    else:
        candidates = (
            calibration_candidate_recipes(selector_mode, batch_size=batch_size, seed=seed,
                                          transfer_policy=transfer_policy)
            if recipes is DEFAULT_CALIBRATION_RECIPES else
            tuple(recipe if isinstance(recipe, TargetRecipe) else recipe.target_recipe(batch_size=batch_size, seed=seed)
                  for recipe in recipes)
        )
        allowed = {"source", "scratch", "waveform_lda"} if selector_mode == "five_candidates" else (
            {"source"} if selector_mode in {"transfer_only", "legacy_product"} else {"scratch", "waveform_lda"}
        )
        if any(recipe.initialization not in allowed for recipe in candidates):
            raise ValueError("recipe initialization conflicts with the declared selector mode")
        if any(recipe.batch_size != batch_size for recipe in candidates):
            raise ValueError("candidate recipe batch differs from the frozen B0 rule")
        notify(20, "select_inside_authorized_calibration_then_fresh_refit")
        fitted, selection = select_calibration_candidate(
            dataset, candidates, calibration_mask=np.ones(dataset.n_epochs, dtype=bool),
            source_trunk=source, source_input_stats=stats, device=device,
            protocol="binary_auc" if selector_mode == "legacy_product" else selection_protocol,
            selector_mode=selector_mode, allow_within_recording=False,
            epoch_quality_policy=epoch_quality_policy, epoch_quality_model=epoch_quality_model,
        )
    selection["source_checkpoint_gate"] = source_gate
    transferred = fitted.recipe.initialization == "source"
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(), "participant_key": participant,
        "subject_id": subject, "calibration_sessions": sessions,
        "calibration_available_timestamp_s": max(row["calibration_available_timestamp_s"] for row in sessions),
        "calibration_trials": dataset.n_epochs, "calibration_positives": int(dataset.y.sum()),
        "calibration_data_sha256": _calibration_data_digest(dataset),
        "input_contract": _input_contract(dataset), "input_request": request,
        "preprocessing_execution": copy.deepcopy(dataset.provenance.get("preprocessing_execution")),
        "preprocessing_execution_binding": preprocessing_execution_binding(
            dataset.provenance.get("preprocessing_execution"), dataset.preprocessing),
        "candidate_vocabulary": list(range(1, 10)), "seed": fitted.recipe.seed, "batch_rule": batch_rule,
        "evidence_scope": "personalized_development_not_independent_performance_confirmation",
        "model_kind": "source_initialized_n2p3" if transferred else (
            "scratch_n2p3" if fitted.adapter is not None else "target_only_shrinkage_lda"
        ),
        "source_pretraining_used": transferred, "selection": selection,
        "recipe": fitted.recipe.record(), "actual_fitted_state_sha256": selection["refit_state_sha256"],
        "epoch_quality": fitted.fit_history.get("epoch_quality"),
    }
    if fitted.adapter is not None:
        trunk = fitted.adapter.trunk
        metadata.update(architecture=trunk.architecture_record(),
                        architecture_configuration=asdict(plain_trunk_configuration(
                            trunk.core if isinstance(trunk, FeatureDomainTransfer) else trunk)),
                        pooling_mode=trunk.pooling_mode,
                        fitted_input_statistics=fitted.adapter.input_statistics_record())
        if fitted.recipe.candidate_loss_weight is not None:
            metadata["decision_score"] = {"method": "membership_llr_sum/1",
                "temperature": fitted.recipe.candidate_loss_temperature,
                "prior": "uniform", "coverage": "complete_physical_schedule",
                "probability_calibrated": False}
    if transferred:
        metadata.update(source_checkpoint_sha256=source_gate["checkpoint_sha256"],
                        source_training_contract=checkpoint_training_contract(payload).record())
    notify(95, "frozen_calibrated_model_ready")
    return CalibratedPredictor(metadata, fitted.state, device=device)


def _continuous_record(dataset, participant, sessions):
    processing = dataset.provenance.get("continuous_artifact_processing", {"status": "not_applied"})
    validate_continuous_calibration_budget(processing, {participant: tuple(row["session_id"] for row in sessions)})
    return processing.get("model") if processing.get("status") == "applied" else None


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


def load_calibrated_predictor(path: str | Path | BinaryIO, *, device: str = "cpu") -> CalibratedPredictor:
    """Restore a personal model from a path or an already verified byte stream."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") not in {
        CALIBRATED_PREDICTOR_SCHEMA, "n2p3_calibrated_predictor/3", "n2p3_calibrated_predictor/1",
    }:
        raise ValueError("unsupported calibrated model artifact")
    if payload.get("metadata_digest") != semantic_sha256(payload["metadata"]):
        raise ValueError("calibrated model metadata digest mismatch")
    if payload.get("state_digest") != _state_digest(payload["state"]):
        raise ValueError("calibrated model state digest mismatch")
    quality = payload["metadata"].get("epoch_quality")
    if payload["schema"] != CALIBRATED_PREDICTOR_SCHEMA and quality is not None and (
        quality["model"].get("method") != "fold_local_ptp_cv"
    ):
        raise ValueError("legacy calibrated artifact cannot declare a new epoch quality method")
    return CalibratedPredictor(payload["metadata"], payload["state"], device=device)
