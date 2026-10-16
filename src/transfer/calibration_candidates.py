"""Fixed target candidates, calibration-only scores and independent full refits."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data.candidate_task import (
    CandidateTaskContract,
    MembershipDecision,
    validate_candidate_membership_metadata,
)
from data.continuous_qc import minimum_usable_channel_count, validate_continuous_calibration_budget
from data.epochs import EpochDataset, materialize_dataset_identity
from data.ocular import validate_eog_calibration_budget
from models.adapters import FeatureDomainTransfer
from models.candidate_evidence import candidate_evidence_scores
from models.n2p3net import N2P3Net
from research.contracts import semantic_sha256
from research.execution import ExpectedSubjectError
from train.early_stopping import DEFAULT_EARLY_STOP_PATIENCE
from transfer.binary import (
    fit_shrinkage_feature_model,
    plain_trunk_configuration,
    scratch_like_trunk,
    target_shrinkage_features,
)
from transfer.calibration_split import dataset_calibration_inner_split
from transfer.quality import EpochQualityModel, EpochQualityPolicy, validate_epoch_quality_fit_scope
from transfer.recipes import TargetRecipe
from transfer.subject_adapter import SubjectAdapter

SELECTOR_MODES = ("five_candidates", "transfer_only", "no_transfer", "legacy_product", "unified")


def calibration_candidate_recipes(
    selector_mode: str = "five_candidates", *, batch_size: int = 64, seed: int = 20260906,
    transfer_policy: TargetRecipe | None = None,
) -> tuple[TargetRecipe, ...]:
    if selector_mode not in SELECTOR_MODES:
        raise ValueError(f"unknown calibration selector mode: {selector_mode}")
    if selector_mode == "unified":
        if not isinstance(transfer_policy, TargetRecipe):
            raise ValueError("unified calibration requires one explicit TargetRecipe")
        return (transfer_policy,)
    policy = transfer_policy or TargetRecipe("transfer", batch_size=batch_size, seed=seed)
    if selector_mode == "legacy_product":
        policy = replace(policy, input_statistics="source", bn_buffers="source")
        batch_size = 64
    candidates = []
    for initialization in ("source", "scratch"):
        if selector_mode in {"transfer_only", "legacy_product"} and initialization == "scratch":
            continue
        if selector_mode == "no_transfer" and initialization == "source":
            continue
        for budget, epochs, lr in (("low", 15, 1e-4), ("high", 60, 1e-3)):
            candidate = replace(
                policy, name=f"{initialization}_{budget}", initialization=initialization,
                input_statistics="target_prefix" if initialization == "scratch" else policy.input_statistics,
                batch_size=batch_size, seed=seed, epochs=epochs, lr=lr, max_updates=None,
                refit_full_prefix=False,
            )
            candidates.append(candidate)
    if selector_mode in {"five_candidates", "no_transfer"}:
        candidates.append(waveform_lda_recipe(batch_size=batch_size, seed=seed))
    return tuple(candidates)


def waveform_lda_recipe(*, batch_size: int = 64, seed: int = 0) -> TargetRecipe:
    return TargetRecipe(
        "waveform_lda", initialization="waveform_lda", weights="none", bn_buffers="source",
        bn_affine="frozen", dropout="off", epochs=1, batch_size=batch_size, seed=seed,
    )


def _observed(dataset: EpochDataset) -> np.ndarray:
    return np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2]) if dataset.trial_channel_mask is None else (
        np.asarray(dataset.trial_channel_mask, dtype=bool)
    )


def _mask(mask: np.ndarray, n: int) -> np.ndarray:
    result = np.asarray(mask)
    if result.dtype != bool or result.shape != (n,) or not result.any():
        raise ValueError("fit scope requires a nonempty aligned boolean calibration mask")
    return result


def validate_calibration_quality_ingress(dataset: EpochDataset) -> None:
    """Require epochs before fitted epoch-QC selection at public fit boundaries."""
    processing = dataset.provenance.get("epoch_artifact_processing", {})
    if not isinstance(processing, Mapping):
        raise ValueError("epoch QC processing provenance must be a mapping")
    if processing.get("status") == "applied":
        raise ValueError(
            "Personalized calibration received epochs with epoch QC already applied; "
            "use the original pre-epoch-QC epochs and fit quality inside the calibration scope. "
            "Frozen continuous-recovery preprocessing may remain applied."
        )


def waveform_lda_training_selection(
    dataset: EpochDataset, training_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Select the fixed LDA channels and their complete training observations."""
    training = _mask(training_mask, dataset.n_epochs).copy()
    observed = _observed(dataset)
    channels = np.flatnonzero(observed[training].all(axis=0))
    policy = "common_observed_training_channels"
    continuous = dataset.provenance.get("continuous_artifact_processing", {})
    minimum = minimum_usable_channel_count(continuous, dataset.n_channels)
    if len(channels) < minimum and continuous.get("model", {}).get("spec", {}).get("channel_local") is True:
        channels = np.flatnonzero(observed[training].any(axis=0))
        training &= observed[:, channels].all(axis=1)
        policy = "complete_observations_of_fixed_training_channel_union"
    return channels, training, policy


def validate_calibration_raw_fit_scope(dataset: EpochDataset, training_mask: np.ndarray) -> None:
    """Check personal raw-fit authority within complete training recordings.

    Physical recording identity and raw operator support remain loader gates;
    session membership alone does not certify a within-recording split.
    Source checkpoint statistics have their own authority and are not checked here.
    """
    training = _mask(training_mask, dataset.n_epochs)
    ocular = dataset.provenance.get("ocular_artifact_processing", {"status": "not_applied"})
    continuous = dataset.provenance.get("continuous_artifact_processing", {"status": "not_applied"})
    if ocular.get("status") != "applied" and continuous.get("status") != "applied":
        return
    if "session" not in dataset.metadata:
        raise ValueError("session-fitted raw preprocessing requires explicit session identities")
    subjects = np.asarray(dataset.subject_ids).astype(str)
    sessions = dataset.metadata["session"].astype(str).to_numpy()
    identity = materialize_dataset_identity(dataset)
    budget = {identity.record_for(subject).authority_key("source_or_global"):
              tuple(np.unique(sessions[training & (subjects == subject)]))
              for subject in np.unique(subjects[training])}
    validate_eog_calibration_budget(ocular, budget)
    validate_continuous_calibration_budget(continuous, budget)
    for subject in np.unique(subjects[training]):
        selected_sessions = np.unique(sessions[training & (subjects == subject)])
        if ((subjects == subject) & np.isin(sessions, selected_sessions) & ~training).any():
            raise ValueError("session-fitted raw preprocessing cannot certify a partial-recording training scope")


def calibration_decision_groups(dataset: EpochDataset, training_mask: np.ndarray, *, repetitions: int = 5):
    """Build authorized complete requested groups before epoch QC removes rows.

    Direct tasks use nonoverlapping declared repetition windows; row/column
    tasks retain each physical selection. No target outside the row budget is
    read to complete a partial group. Raw/time-scope admission remains upstream.
    """
    training = _mask(training_mask, dataset.n_epochs)
    if dataset.y is None:
        raise ValueError("calibration decisions require authorized epoch labels")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("decision repetition budget must be a positive integer")
    timeline = dataset.event_timeline
    evidence = np.asarray(timeline.evidence_indices, dtype=np.int64)
    present = evidence >= 0
    if not np.array_equal(np.sort(evidence[present]), np.arange(dataset.n_epochs)):
        raise ValueError("calibration decisions require one physical event per epoch row")
    ledger = {"schema": "n2p3_calibration_decisions/1", "available": False,
              "risk": "participant_macro_original_decision_macro", "groups": [], "excluded_groups": []}
    task = dataset.provenance.get("candidate_task_contract")
    task = None if task is None else CandidateTaskContract.from_record(task)
    vocabulary = dataset.provenance.get("candidate_vocabulary")
    if task is None and vocabulary is None:
        from data.brainsync_contract import BRAIN_SYNC_BIDS_CONTRACT, BRAIN_SYNC_SESSION_SCHEMA

        if dataset.provenance.get("session_schema") == BRAIN_SYNC_SESSION_SCHEMA:
            vocabulary = BRAIN_SYNC_BIDS_CONTRACT.candidate_vocabulary
    if task is None and vocabulary is None:
        ledger["reason"] = "membership_unavailable"
        return (), (), np.empty(0), ledger
    if not timeline.complete:
        ledger["reason"] = "incomplete_physical_timeline"
        return (), (), np.empty(0), ledger
    vocabulary = None if vocabulary is None else tuple(str(value) for value in vocabulary)
    subjects = np.asarray(timeline.subject_ids).astype(str)
    group_ids = np.asarray(timeline.group_ids).astype(str)
    repeat = np.asarray(timeline.repetition_indices)
    groups, targets, origins = [], [], []
    for subject, original in dict.fromkeys(zip(subjects, group_ids, strict=True)):
        original_events = (subjects == subject) & (group_ids == original)
        if task is None:
            if np.any(repeat[original_events] < 0):
                raise ValueError("direct decisions require declared physical repetition indices")
            windows = tuple(dict.fromkeys((repeat[original_events] // repetitions).tolist()))
        else:
            windows = (None,)
        for window in windows:
            selected = original_events if window is None else original_events & (repeat // repetitions == window)
            rows = evidence[selected]
            rows = rows[rows >= 0]
            identifier = json.dumps([subject, original, window], separators=(",", ":"))
            if not len(rows) or not training[rows].any():
                ledger["excluded_groups"].append({"decision_id": identifier, "reason": "no_authorized_label_evidence"})
                continue
            if not training[rows].all():
                ledger["excluded_groups"].append({"decision_id": identifier, "reason": "outside_training_budget"})
                continue
            if task is None:
                group = MembershipDecision.from_timeline(timeline, selected, repetitions=repetitions,
                    repetition_start=window * repetitions, candidate_vocabulary=vocabulary)
                retained_events = selected & present
                truths = np.unique(np.asarray(timeline.target_candidate_ids).astype(str)[retained_events])
                if len(truths) != 1 or truths[0] not in group.candidate_vocabulary:
                    raise ValueError("authorized direct decision must have one declared target")
                target = group.candidate_vocabulary.index(str(truths[0]))
                if "block_id" in dataset.metadata and dataset.metadata.iloc[rows]["block_id"].nunique() != 1:
                    raise ValueError("direct decision window crosses physical BrainSync blocks")
            else:
                # Physical schedules, not only surviving metadata rows, own the group.
                group_repetitions = int(np.max(repeat[selected])) + 1
                if "selection_repetitions" in dataset.metadata:
                    declared = dataset.metadata.iloc[rows]["selection_repetitions"].unique()
                    if (len(declared) != 1 or isinstance(declared[0], (bool, np.bool_))
                            or not isinstance(declared[0], (int, np.integer)) or declared[0] < 1):
                        raise ValueError("row/column selection requires one positive declared repetition count")
                    group_repetitions = int(declared[0])
                group = MembershipDecision.from_timeline(timeline, selected, repetitions=group_repetitions,
                                                         task_contract=task)
                if not group.schedule_complete:
                    ledger["excluded_groups"].append({"decision_id": identifier, "reason": "incomplete_requested_schedule"})
                    continue
                metadata = validate_candidate_membership_metadata(dataset.metadata.iloc[rows], task,
                                                                  labels=dataset.y[rows])
                target_pairs = np.unique(np.column_stack((metadata.target_rows, metadata.target_columns)), axis=0)
                if len(target_pairs) != 1:
                    raise ValueError("authorized row/column decision must have one target intersection")
                target = int(target_pairs[0, 0] * task.n_columns + target_pairs[0, 1])
                observed_events = np.flatnonzero(selected & present)
                by_event = {int(row): index for index, row in enumerate(rows)}
                if any(str(metadata.candidate_ids[by_event[int(evidence[event])]]) != str(timeline.candidate_ids[event])
                       for event in observed_events):
                    raise ValueError("row/column epoch membership differs from the physical timeline")
            group.validate_epoch_labels(dataset.y, target)
            if not group.schedule_complete:
                ledger["excluded_groups"].append({"decision_id": identifier, "reason": "incomplete_requested_schedule"})
                continue
            groups.append(group)
            targets.append(target)
            origins.append((subject, original))
            ledger["groups"].append({"decision_id": identifier, "target_candidate": group.candidate_vocabulary[target],
                "scheduled_events": len(group.event_ids), "available_events": int(np.sum(group.evidence_indices >= 0)),
                "membership_sha256": semantic_sha256(group.record())})
    people = set(subject for subject, _ in origins)
    weights = np.asarray([1. / len(people) / len({origin for owner, origin in origins if owner == subject})
                          / origins.count((subject, original)) for subject, original in origins], dtype=float)
    ledger.update(available=bool(groups), requested_groups=len(groups), risk_weights=weights.tolist(),
                  reason=None if groups else "no_complete_decision_inside_training_budget")
    return tuple(groups), tuple(targets), weights, ledger


def remap_decision_groups(groups, source_rows):
    """Map authorized event evidence through an explicit row subset/QC result.

    This mapping does not grant label authority. Build groups against the
    authorized pre-QC scope first; removed evidence then remains requested as -1.
    """
    rows = np.asarray(source_rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or np.any(rows < 0) or len(np.unique(rows)) != len(rows):
        raise ValueError("decision row remapping requires unique original integer row indices")
    mapping = {int(row): index for index, row in enumerate(rows)}
    return tuple(replace(group, evidence_indices=np.asarray([
        mapping.get(int(row), -1) if row >= 0 else -1 for row in group.evidence_indices], dtype=np.int64))
        for group in groups)


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass
class FittedCalibrationCandidate:
    recipe: TargetRecipe
    sfreq: float
    adapter: SubjectAdapter | None
    lda: object | None
    fit_history: dict[str, object]
    waveform_channels: np.ndarray | None = None
    epoch_quality: EpochQualityModel | None = None
    channel_quality_availability: bool = False
    minimum_usable_channels: int = 1

    @property
    def state(self) -> dict[str, torch.Tensor]:
        state = {}
        if self.adapter is not None:
            state = {f"trunk.{key}": value.detach().cpu().clone()
                     for key, value in self.adapter.trunk.state_dict().items()}
            mean, std = self.adapter.fitted_input_statistics()
            state.update(input_mean=torch.as_tensor(mean), input_std=torch.as_tensor(std))
        if self.lda is not None:
            scaler, model = self.lda.steps[0][1], self.lda.steps[1][1]
            state.update(
                feature_mean=torch.as_tensor(scaler.mean_, dtype=torch.float64),
                feature_scale=torch.as_tensor(scaler.scale_, dtype=torch.float64),
                coefficient=torch.as_tensor(model.coef_[0], dtype=torch.float64),
                intercept=torch.as_tensor(model.intercept_[0], dtype=torch.float64),
            )
            if self.waveform_channels is not None:
                state["waveform_channels"] = torch.as_tensor(self.waveform_channels, dtype=torch.int64)
        return state

    def predict_logit(self, X: np.ndarray, trial_channel_mask: np.ndarray | None = None) -> np.ndarray:
        if self.epoch_quality is not None:
            transformed = self.epoch_quality.transform(X, trial_channel_mask)
            available = ~transformed.drop_epoch_mask & ~transformed.all_channels_bad
            available &= transformed.trial_channel_mask.sum(axis=1) >= self.minimum_usable_channels
            if self.adapter is None and self.waveform_channels is not None:
                available &= transformed.trial_channel_mask[:, self.waveform_channels].all(axis=1)
            logits = np.full(len(X), np.nan)
            if available.any():
                logits[available] = replace(self, epoch_quality=None).predict_logit(
                    transformed.X[available], transformed.trial_channel_mask[available])
            return logits
        if self.channel_quality_availability and trial_channel_mask is not None:
            usable = np.asarray(trial_channel_mask).sum(axis=1) >= self.minimum_usable_channels
            if not usable.all():
                logits = np.full(len(X), np.nan)
                if usable.any():
                    logits[usable] = replace(self, minimum_usable_channels=1).predict_logit(
                        X[usable], np.asarray(trial_channel_mask)[usable])
                return logits
        if self.lda is None:
            return self.adapter.predict_logit(X, trial_channel_mask=trial_channel_mask)
        available = np.ones(len(X), dtype=bool)
        if self.adapter is None:
            if trial_channel_mask is not None:
                available &= np.asarray(trial_channel_mask)[:, self.waveform_channels].all(axis=1)
                if not self.channel_quality_availability and not available.all():
                    raise ValueError("raw waveform LDA needs every selected channel to be observed")
            if not available.any():
                return np.full(len(X), np.nan)
            features = target_shrinkage_features(X[available][:, self.waveform_channels], sfreq=self.sfreq)
        else:
            features = self.adapter.predict_features(X, trial_channel_mask=trial_channel_mask)
        logits = np.full(len(X), np.nan)
        logits[available] = np.asarray(self.lda.decision_function(features), dtype=np.float64)
        return logits


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _fit_rows_digest(dataset: EpochDataset, mask: np.ndarray) -> str:
    rows = np.flatnonzero(_mask(mask, dataset.n_epochs))
    identity = materialize_dataset_identity(dataset)
    timeline = dataset.event_timeline
    events = np.isin(timeline.evidence_indices, rows)
    return semantic_sha256({
        "rows": rows.tolist(),
        "participants": [identity.record_for(str(subject)).authority_key("source_or_global")
                         for subject in dataset.subject_ids[rows]],
        "events": {name: np.asarray(getattr(timeline, name))[events].tolist() for name in (
            "event_ids", "evidence_indices", "session_ids", "run_ids", "onset_times_s",
            "evidence_available_times_s",
        )},
        "X": _array_digest(dataset.X[rows]), "y": _array_digest(dataset.y[rows]),
        "observed": _array_digest(_observed(dataset)[rows]),
        "qc_features": None if dataset.qc_features is None else {
            "schema": dataset.qc_features.schema,
            **{name: _array_digest(np.asarray(getattr(dataset.qc_features, name))[rows])
               for name in ("relative_ptp", "channel_std_v", "epoch_scale_v", "observed_mask")},
        },
    })


class CalibrationCandidateFitCache:
    """In-memory exact fits owned by one participant/calibration-budget scope."""

    def __init__(self) -> None:
        self._fits: dict[str, FittedCalibrationCandidate] = {}
        self.fit_calls = 0
        self.cache_hits = 0

    def clear(self) -> None:
        self._fits.clear()

    def record(self) -> dict[str, int]:
        return {"fit_calls": self.fit_calls, "cache_hits": self.cache_hits, "entries": len(self._fits)}

    def fit(
        self, dataset: EpochDataset, recipe: TargetRecipe, training_mask: np.ndarray, *,
        phase: str, source_trunk: N2P3Net | None = None,
        source_input_stats: tuple[np.ndarray, np.ndarray] | None = None,
        device: str = "cpu", validation_mask: np.ndarray | None = None,
        validation_behavior: str = "select_epoch",
        early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE, early_stop_min_delta: float = 1e-6,
        epoch_quality_policy: EpochQualityPolicy | None = None,
        epoch_quality_model: EpochQualityModel | None = None,
    ) -> FittedCalibrationCandidate:
        validate_calibration_quality_ingress(dataset)
        if phase not in {"inner", "full_prefix"}:
            raise ValueError("cached candidate phase must be inner or full_prefix")
        training = _mask(training_mask, dataset.n_epochs)
        validate_epoch_quality_fit_scope(dataset, training, policy=epoch_quality_policy, model=epoch_quality_model)
        validate_calibration_raw_fit_scope(dataset, training)
        decision_key = {}
        if recipe.candidate_loss_weight is not None:
            groups, targets, weights, _ = calibration_decision_groups(dataset, training)
            decision_key = {"decision_scope": {"groups": [group.record() for group in groups],
                "targets": targets, "weights": weights.tolist()}}
        key = semantic_sha256({
            "schema": "n2p3_calibration_fit_key/1", "phase": phase,
            "recipe": recipe.record(), "training": _fit_rows_digest(dataset, training),
            "validation": None if validation_mask is None else _fit_rows_digest(dataset, validation_mask),
            "validation_behavior": validation_behavior,
            "early_stop_patience": early_stop_patience, "early_stop_min_delta": early_stop_min_delta,
            "preprocessing": dataset.preprocessing.record(), "channel_names": dataset.channel_names,
            "channel_positions": _array_digest(dataset.channel_positions_m),
            "ocular_processing": dataset.provenance.get("ocular_artifact_processing"),
            "continuous_processing": dataset.provenance.get("continuous_artifact_processing"),
            "epoch_quality_policy": None if epoch_quality_policy is None else asdict(epoch_quality_policy),
            "epoch_quality_model_digest": None if epoch_quality_model is None else semantic_sha256(epoch_quality_model.record()),
            **decision_key,
            "source": None if source_trunk is None else {
                "architecture": asdict(plain_trunk_configuration(source_trunk)),
                "state": _state_digest(dict(source_trunk.state_dict())),
            },
            "source_input_statistics": None if source_input_stats is None else [
                _array_digest(value) for value in source_input_stats],
            "runtime": {"device": str(torch.device(device)), "torch": torch.__version__,
                "dtype": str(torch.get_default_dtype()), "threads": torch.get_num_threads(),
                "deterministic": torch.are_deterministic_algorithms_enabled(),
                "matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic},
        })
        reused = key in self._fits
        if reused:
            self.cache_hits += 1
            fitted = copy.deepcopy(self._fits[key])
        else:
            self.fit_calls += 1
            fitted = fit_calibration_candidate(dataset, recipe, training,
                source_trunk=source_trunk, source_input_stats=source_input_stats, device=device,
                validation_mask=validation_mask, validation_behavior=validation_behavior,
                early_stop_patience=early_stop_patience, early_stop_min_delta=early_stop_min_delta,
                epoch_quality_policy=epoch_quality_policy, epoch_quality_model=epoch_quality_model)
            fitted.fit_history["fit_reuse"] = {
                "fit_key_sha256": key, "phase": phase, "origin_fit_id": self.fit_calls,
                "cache_hit": False, "fit_executed": True,
            }
            self._fits[key] = copy.deepcopy(fitted)
        fitted.fit_history["fit_reuse"].update(cache_hit=reused, fit_executed=not reused)
        return fitted


def fit_calibration_candidate(
    dataset: EpochDataset, recipe: TargetRecipe, training_mask: np.ndarray, *,
    source_trunk: N2P3Net | None = None,
    source_input_stats: tuple[np.ndarray, np.ndarray] | None = None, device: str = "cpu",
    validation_mask: np.ndarray | None = None,
    validation_behavior: str = "select_epoch",
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE, early_stop_min_delta: float = 1e-6,
    epoch_quality_policy: EpochQualityPolicy | None = None,
    epoch_quality_model: EpochQualityModel | None = None,
) -> FittedCalibrationCandidate:
    """Fit declared rows; validation either replays selection or records fixed-budget diagnostics."""
    validate_calibration_quality_ingress(dataset)
    validate_epoch_quality_fit_scope(dataset, training_mask, policy=epoch_quality_policy, model=epoch_quality_model)
    return _fit_calibration_candidate(
        dataset, recipe, training_mask, source_trunk=source_trunk,
        source_input_stats=source_input_stats, device=device, validation_mask=validation_mask,
        validation_behavior=validation_behavior, early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta, epoch_quality_policy=epoch_quality_policy,
        epoch_quality_model=epoch_quality_model,
    )


def _fit_calibration_candidate(
    dataset: EpochDataset, recipe: TargetRecipe, training_mask: np.ndarray, *,
    source_trunk: N2P3Net | None = None,
    source_input_stats: tuple[np.ndarray, np.ndarray] | None = None, device: str = "cpu",
    validation_mask: np.ndarray | None = None,
    validation_behavior: str = "select_epoch",
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE, early_stop_min_delta: float = 1e-6,
    epoch_quality_policy: EpochQualityPolicy | None = None,
    epoch_quality_model: EpochQualityModel | None = None,
    _decision_scope=None,
) -> FittedCalibrationCandidate:
    """Internal fitting also accepts the current call's quality-transformed rows."""
    if validation_behavior not in {"select_epoch", "select_epoch_full_trajectory", "diagnostics_only"}:
        raise ValueError("unsupported candidate validation_behavior")
    training = _mask(training_mask, dataset.n_epochs)
    validate_calibration_raw_fit_scope(dataset, training)
    if recipe.candidate_loss_weight is not None and _decision_scope is None:
        _decision_scope = calibration_decision_groups(dataset, training)
        if not _decision_scope[0]:
            raise ValueError(f"decision recipe has no authorized complete group: {_decision_scope[3]['reason']}")
    if epoch_quality_policy is not None or epoch_quality_model is not None:
        from transfer.quality import apply_epoch_quality, fit_epoch_quality
        quality = (fit_epoch_quality(dataset, training, epoch_quality_policy)
                   if epoch_quality_model is None else copy.deepcopy(epoch_quality_model))
        cleaned = apply_epoch_quality(dataset, quality)
        positions = np.flatnonzero(cleaned.event_timeline.evidence_indices >= 0)
        positions = positions[np.argsort(cleaned.event_timeline.evidence_indices[positions])]
        original_rows = dataset.event_timeline.evidence_indices[positions]
        cleaned_scope = None
        if _decision_scope is not None:
            groups, targets, weights, ledger = _decision_scope
            cleaned_scope = (remap_decision_groups(groups, original_rows), targets, weights, ledger)
        fitted = _fit_calibration_candidate(cleaned, recipe, training[original_rows],
            source_trunk=source_trunk, source_input_stats=source_input_stats, device=device,
            validation_mask=None if validation_mask is None else np.asarray(validation_mask)[original_rows],
            validation_behavior=validation_behavior, early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta, _decision_scope=cleaned_scope)
        fitted.epoch_quality = quality
        fitted.fit_history["epoch_quality"] = {"model": quality.record(),
            "model_digest": semantic_sha256(quality.record()),
            "lifecycle": "fitted_in_scope" if epoch_quality_model is None else "frozen_explicit",
            "requested_training_rows": int(training.sum()),
            "retained_training_rows": int(training[original_rows].sum())}
        return fitted
    labels = np.asarray(dataset.y)[training]
    if set(np.unique(labels)) != {0, 1} or len(labels) <= 2:
        raise ValueError("candidate fitting requires more than two rows and both binary classes")
    observed = _observed(dataset)
    adapter = None
    fitted_lda = None
    waveform_channels = None
    history = {"fit_scope": {"training_rows": int(training.sum()), "validation_rows": 0,
                             "prefix_mask_explicit": True}}
    training_digest = hashlib.sha256()
    for array in (dataset.X[training], labels, observed[training]):
        training_digest.update(np.ascontiguousarray(array).tobytes())
    requested_training_digest = training_digest.hexdigest()
    if recipe.initialization != "waveform_lda":
        if source_trunk is None and recipe.initialization != "scratch":
            raise ValueError("source initialized candidate has no compatible source checkpoint")
        if recipe.initialization == "scratch":
            if source_trunk is None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(recipe.seed)
                    trunk = N2P3Net(
                        dataset.n_channels, n_times=dataset.n_times, sfreq=dataset.preprocessing.sfreq,
                        tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
                    )
            else:
                trunk = scratch_like_trunk(source_trunk, seed=recipe.seed)
            stats = (None, None)
        else:
            trunk = copy.deepcopy(source_trunk)
            stats = source_input_stats or (None, None)
        if recipe.weights == "adapter":
            if type(trunk) is not N2P3Net or trunk.splitbn_config is not None:
                raise ValueError("target residual construction requires a plain source core")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(recipe.seed)
                trunk = FeatureDomainTransfer(trunk, config=recipe.target_adapter_config)
        initial_state_sha256 = _state_digest(dict(trunk.state_dict()))
        adapter = SubjectAdapter(
            trunk, device=torch.device(device),
            config=replace(
                replace(recipe, refit_full_prefix=False).adapter_config(input_mean=stats[0], input_std=stats[1]),
                validation_behavior=validation_behavior,
                early_stop_patience=early_stop_patience, early_stop_min_delta=early_stop_min_delta,
            ),
        )
        scope = training
        if validation_mask is not None:
            validation = _mask(validation_mask, dataset.n_epochs)
            if (training & validation).any():
                raise ValueError("candidate training and validation must be disjoint")
            scope = training | validation
        decision_kwargs = {}
        if _decision_scope is not None:
            groups, targets, weights, ledger = _decision_scope
            decision_kwargs = {"decision_groups": remap_decision_groups(groups, np.flatnonzero(scope)),
                               "decision_targets": targets, "decision_weights": weights}
        if validation_mask is None:
            adapter.fit(dataset.X[training], labels, prefix_mask=np.ones(int(training.sum()), dtype=bool),
                        trial_channel_mask=observed[training], **decision_kwargs)
        else:
            adapter.fit(dataset.X[scope], dataset.y[scope], training_mask=training[scope],
                        validation_mask=validation[scope], prefix_mask=np.ones(int(scope.sum()), dtype=bool),
                        trial_channel_mask=observed[scope], **decision_kwargs)
        history = copy.deepcopy(adapter.last_history)
        history["initial_state_sha256"] = initial_state_sha256
        if _decision_scope is not None:
            history["decision_scope"] = {**copy.deepcopy(ledger),
                "post_quality_groups": [group.record() for group in groups],
                "post_quality_complete_groups": sum(group.schedule_complete and bool((group.evidence_indices >= 0).all())
                                                     for group in groups)}
    if recipe.initialization in {"waveform_lda", "source_feature_lda"}:
        if adapter is None:
            waveform_channels, lda_training, waveform_policy = waveform_lda_training_selection(dataset, training)
            if waveform_policy == "complete_observations_of_fixed_training_channel_union":
                if np.sum(lda_training) <= 2 or set(np.unique(dataset.y[lda_training])) != {0, 1}:
                    raise ValueError("channel-local waveform LDA lacks complete observed training rows with both classes")
                history["fit_scope"]["requested_training_rows"] = int(training.sum())
                history["fit_scope"]["training_rows"] = int(lda_training.sum())
                history["waveform_training_policy"] = waveform_policy
            if not len(waveform_channels):
                raise ValueError("raw waveform LDA requires channels observed on every training row")
            features = target_shrinkage_features(
                dataset.X[lda_training][:, waveform_channels], sfreq=dataset.preprocessing.sfreq,
            )
            labels = dataset.y[lda_training]
            training_digest = hashlib.sha256()
            for array in (
                dataset.X[lda_training][:, waveform_channels], labels,
                observed[lda_training][:, waveform_channels],
            ):
                training_digest.update(np.ascontiguousarray(array).tobytes())
            history["waveform_channel_indices"] = waveform_channels.tolist()
        else:
            features = adapter.predict_features(dataset.X[training], trial_channel_mask=observed[training])
        fitted_lda = fit_shrinkage_feature_model(features, labels)
    history["requested_training_data_sha256"] = requested_training_digest
    history["training_data_sha256"] = training_digest.hexdigest()
    continuous = dataset.provenance.get("continuous_artifact_processing", {})
    channel_quality_availability = (
        continuous.get("status") == "applied"
        and continuous.get("model", {}).get("spec", {}).get("channel_local") is True
    )
    return FittedCalibrationCandidate(
        recipe, dataset.preprocessing.sfreq, adapter, fitted_lda, history, waveform_channels,
        channel_quality_availability=channel_quality_availability,
        minimum_usable_channels=minimum_usable_channel_count(continuous, dataset.n_channels),
    )


def score_calibration_candidate(
    dataset: EpochDataset, logits: np.ndarray, validation_mask: np.ndarray, *,
    protocol: str = "binary_auc",
) -> dict[str, object]:
    validation = _mask(validation_mask, dataset.n_epochs)
    values = np.asarray(logits, dtype=float)
    if values.shape == (dataset.n_epochs,):
        values = values[validation]
    if values.shape != (int(validation.sum()),) or np.isinf(values).any():
        raise ValueError("candidate predictions must align with validation rows; NaN means unavailable")
    usable = np.isfinite(values)
    auc = (float(roc_auc_score(dataset.y[validation][usable], values[usable]))
           if len(np.unique(dataset.y[validation][usable])) == 2 else None)
    result = {"protocol": protocol, "validation_auc": auc, "primary_score": auc}
    result.update(requested_validation_rows=int(validation.sum()), usable_validation_rows=int(usable.sum()),
                  unavailable_validation_rows=int((~usable).sum()))
    if protocol == "binary_auc":
        if auc is None:
            raise ValueError("quality-filtered validation lacks both classes for AUC")
        return result
    if protocol != "brainsync_operational_r5":
        raise ValueError("unknown calibration scoring protocol")
    if "block_id" not in dataset.metadata or "session" not in dataset.metadata:
        raise ValueError("BrainSync operational R5 requires a real 45-flash block ledger")
    timeline = dataset.event_timeline
    evidence = np.asarray(timeline.evidence_indices, dtype=int)
    event_sessions = np.asarray(timeline.session_ids).astype(str)
    selected_sessions = tuple(np.unique(dataset.metadata.loc[validation, "session"].astype(str)))
    by_row = np.full(dataset.n_epochs, np.nan)
    by_row[validation] = values
    session_scores = []
    block_records = []
    for session in selected_sessions:
        rows = np.flatnonzero(event_sessions == session)
        if len(rows) % 45 or not len(rows):
            raise ValueError("BrainSync operational R5 requires exactly 45 scheduled flashes per block")
        hits = []
        for start in range(0, len(rows), 45):
            block = rows[start:start + 45]
            codes = np.asarray(timeline.candidate_ids)[block].astype(int)
            if any(int(np.sum(codes == code)) != 5 for code in range(1, 10)):
                raise ValueError("BrainSync operational R5 requires all nine candidates five times per 45-flash block")
            targets = np.unique(np.asarray(timeline.target_candidate_ids)[block].astype(int))
            if len(targets) != 1 or targets[0] not in range(1, 10):
                raise ValueError("BrainSync block requires one confirmed target identity")
            retained = evidence[block] >= 0
            block_rows = evidence[block][retained]
            if not validation[block_rows].all():
                raise ValueError("operational R5 validation must retain complete recording blocks")
            block_ids = dataset.metadata.iloc[block_rows]["block_id"].unique()
            if len(block_ids) > 1:
                raise ValueError("45-event ledger disagrees with actual BrainSync block identifiers")
            retained[retained] &= np.isfinite(by_row[block_rows])
            block_rows = evidence[block][retained]
            score = candidate_evidence_scores(
                by_row[block_rows], codes[retained], tuple(range(1, 10)),
                aggregation="mean", missing_candidate_policy="abstain",
            )
            complete = all(score.counts[code] == 5 for code in range(1, 10))
            predicted = score.predicted if complete else None
            hit = int(predicted == int(targets[0]))
            hits.append(hit)
            block_records.append({"session": session, "block_index": start // 45,
                                  "prediction": predicted, "hit": hit, "complete": complete})
        session_scores.append(float(np.mean(hits)))
    result.update(primary_score=float(np.mean(session_scores)),
                  operational_hit_r5=float(np.mean(session_scores)), blocks=block_records,
                  session_count=len(session_scores), failure_denominator="all_scheduled_blocks")
    return result


def select_calibration_candidate(
    dataset: EpochDataset, recipes: Sequence[TargetRecipe], *, calibration_mask: np.ndarray,
    source_trunk: N2P3Net | None = None,
    source_input_stats: tuple[np.ndarray, np.ndarray] | None = None,
    device: str = "cpu", protocol: str = "binary_auc",
    recording_order: Sequence[str] | Mapping | None = None, selector_mode: str = "five_candidates",
    allow_within_recording: bool = True,
    fit_cache: CalibrationCandidateFitCache | None = None,
    epoch_quality_policy: EpochQualityPolicy | None = None,
    epoch_quality_model: EpochQualityModel | None = None,
) -> tuple[FittedCalibrationCandidate, dict[str, object]]:
    validate_calibration_quality_ingress(dataset)
    allowed = _mask(calibration_mask, dataset.n_epochs)
    validate_epoch_quality_fit_scope(dataset, allowed, policy=epoch_quality_policy, model=epoch_quality_model)
    candidates = tuple(recipes)
    if not candidates or len({recipe.name for recipe in candidates}) != len(candidates):
        raise ValueError("candidate recipes must have unique names and be nonempty")
    split = dataset_calibration_inner_split(dataset, allowed, recording_order=recording_order)
    record = {**split.record(), "selector_mode": selector_mode, "scoring_protocol": protocol,
              "selection_evidence": split.legal, "recipes": [], "refit_rows": int(allowed.sum()),
              "tie_rule": "primary_desc_auc_desc_lda_scratch_source_low_high_candidate_id"}
    before = fit_cache.record() if fit_cache is not None else None
    fit_calls = 0
    if selector_mode == "legacy_product":
        record.update(scoring_protocol="legacy_epoch_validation_cross_entropy",
                      tie_rule="minimum_validation_ce_then_predeclared_candidate_order")
    selection_legal = split.legal
    if selection_legal and split.mode == "within_recording_purged_80_20" and (
        not allow_within_recording
        or dataset.provenance.get("ocular_artifact_processing", {}).get("status") == "applied"
        or dataset.provenance.get("continuous_artifact_processing", {}).get("status") == "applied"
    ):
        selection_legal = False
        record.update(selection_evidence=False, legal=False,
                      reason="complete separate calibration sessions are required for this product/EOG selection")
    if selection_legal:
        # Operational protocol validity is a data gate, never a candidate failure.
        score_calibration_candidate(dataset, np.zeros(int(split.validation_mask.sum())),
                                    split.validation_mask, protocol=protocol)
    if selection_legal:
        validate_calibration_raw_fit_scope(dataset, split.training_mask)
        validate_epoch_quality_fit_scope(dataset, split.training_mask, policy=epoch_quality_policy, model=epoch_quality_model)
        for recipe in candidates:
            item = {**recipe.record(), "candidate_id": recipe.name}
            try:
                fit_calls += 1
                fit_function = fit_calibration_candidate if fit_cache is None else fit_cache.fit
                fitted = fit_function(
                    dataset, recipe, split.training_mask, source_trunk=source_trunk,
                    source_input_stats=source_input_stats, device=device,
                    validation_mask=split.validation_mask if selector_mode == "legacy_product" else None,
                    **({"epoch_quality_policy": epoch_quality_policy} if epoch_quality_policy is not None else {}),
                    **({"epoch_quality_model": epoch_quality_model} if epoch_quality_model is not None else {}),
                    **({"phase": "inner"} if fit_cache is not None else {}),
                )
                logits = fitted.predict_logit(dataset.X[split.validation_mask], _observed(dataset)[split.validation_mask])
                item.update(status="completed", **score_calibration_candidate(
                    dataset, logits, split.validation_mask, protocol=protocol),
                    fitted_state_sha256=_state_digest(fitted.state),
                    fit_scope=fitted.fit_history["fit_scope"], fit_history=fitted.fit_history,
                )
                if selector_mode == "legacy_product":
                    item["selected_epochs"] = int(fitted.fit_history["best_epoch"])
                    item["primary_score"] = -float(fitted.fit_history["best_val_loss"])
                del fitted
            except (ValueError, ExpectedSubjectError) as error:
                item.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
            record["recipes"].append(item)
    completed = [item for item in record["recipes"] if item["status"] == "completed"]
    if completed:
        rank = {"waveform_lda": 0, "source_feature_lda": 0, "scratch": 1, "source": 2}
        selected = (
            min(completed, key=lambda item: -item["primary_score"])
            if selector_mode == "legacy_product" else min(completed, key=lambda item: (
                -item["primary_score"], -(item["validation_auc"] if item["validation_auc"] is not None else -1), rank[item["initialization"]],
                item["epochs"], item["candidate_id"],
            ))
        )
        recipe = next(recipe for recipe in candidates if recipe.name == selected["candidate_id"])
        if selector_mode == "legacy_product":
            recipe = replace(recipe, epochs=selected["selected_epochs"])
        record.update(selected=selected, validation_auc=selected["validation_auc"])
    else:
        if selector_mode == "legacy_product":
            failures = "; ".join(item["failure"]["message"] for item in record["recipes"] if item["status"] == "failed")
            raise ValueError(f"legacy source calibration failed: {record['reason'] or failures or 'no valid candidate'}")
        recipe = waveform_lda_recipe(seed=candidates[0].seed, batch_size=candidates[0].batch_size)
        record.update(selection_evidence=False, selected=None, validation_auc=None,
                      fallback="full_calibration_waveform_lda", fallback_reason=record["reason"] or "all_candidates_failed")
    fit_calls += 1
    fit_function = fit_calibration_candidate if fit_cache is None else fit_cache.fit
    fitted = fit_function(
        dataset, recipe, allowed, source_trunk=source_trunk, source_input_stats=source_input_stats, device=device,
        **({"epoch_quality_policy": epoch_quality_policy} if epoch_quality_policy is not None else {}),
        **({"epoch_quality_model": epoch_quality_model} if epoch_quality_model is not None else {}),
        **({"phase": "full_prefix"} if fit_cache is not None else {}),
    )
    record.update(refit_recipe=recipe.record(), fit_scope=fitted.fit_history["fit_scope"],
                  refit_history=fitted.fit_history, refit_state_sha256=_state_digest(fitted.state))
    record["fit_execution"] = {
        "requested_fits": fit_calls,
        "fit_calls": fit_calls if fit_cache is None else fit_cache.fit_calls - before["fit_calls"],
        "cache_hits": 0 if fit_cache is None else fit_cache.cache_hits - before["cache_hits"],
    }
    return fitted, record
