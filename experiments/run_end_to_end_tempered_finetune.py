"""End-to-end GTN fine-tuning for count-corrected all-evidence decisions.

Each run starts from one target-block-excluded v4 checkpoint, visits every legal
source EEG epoch once per fine-tuning epoch, and updates the complete N2P3Net
plus a differentiable nine-candidate decision head.  Target-block labels never
enter training, model selection, calibration, or stopping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.contract import (  # noqa: E402
    GTN_SINGLE_SUBJECT_CAUSAL_DATA_CONTRACT,
    assert_p300_input_contract,
)
from data.epochs import load_epoch_dataset, read_epoch_cache_attestation  # noqa: E402
from train.runtime import (  # noqa: E402
    DEFAULT_FUSED_ADAM,
    resolve_optimizer_execution,
)
from transfer.checkpoint import (  # noqa: E402
    checkpoint_input_stats,
    load_n2p3_trunk_checkpoint,
)

MANIFEST_SCHEMA = "n2p3_end_to_end_tempered_finetune/1"
RESULT_SCHEMA = "n2p3_end_to_end_tempered_finetune_result/1"
ANALYSIS_SCHEMA = "n2p3_end_to_end_tempered_finetune_analysis/1"
CHECKPOINT_SCHEMA = "n2p3_end_to_end_tempered_checkpoint/1"
N_CANDIDATES = 9


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_mapping(path: str | Path, *, label: str) -> dict[str, object]:
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON mapping.")
    return decoded


def read_subjects(path: str | Path) -> tuple[str, ...]:
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, list) or not decoded or not all(
        isinstance(value, str) and value for value in decoded
    ):
        raise ValueError("target-subject file must contain a non-empty JSON string list.")
    subjects = tuple(decoded)
    if len(set(subjects)) != len(subjects):
        raise ValueError("target-subject file contains duplicates.")
    return subjects


def _manifest_tuple(manifest: Mapping[str, object], name: str) -> tuple[int, ...]:
    values = manifest.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"manifest {name} must be a non-empty list.")
    return tuple(int(value) for value in values)


def validate_manifest(
    path: str | Path,
    *,
    kernel: int | None = None,
    seed: int | None = None,
    block: int | None = None,
) -> tuple[dict[str, object], str]:
    manifest = read_json_mapping(path, label="end-to-end manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA!r}.")
    for name, value in (("kernels", kernel), ("seeds", seed), ("blocks", block)):
        allowed = _manifest_tuple(manifest, name)
        if value is not None and value not in allowed:
            raise ValueError(f"{name[:-1]} {value} is outside the frozen manifest.")
    training = manifest.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("manifest lacks a training mapping.")
    if training.get("updates") != ["backbone", "classifier", "decision_head"]:
        raise ValueError("manifest must declare full backbone/classifier/decision-head updates.")
    return manifest, sha256_file(path)


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    subject_id: str
    target_index: int
    epoch_rows: np.ndarray
    candidate_indices: np.ndarray
    occurrence_slots: np.ndarray
    occurrence_fraction: np.ndarray
    listwise_eligible: bool = True


@dataclass(frozen=True)
class PackedGroups:
    local_rows: np.ndarray
    candidate_indices: np.ndarray
    occurrence_slots: np.ndarray
    occurrence_fraction: np.ndarray
    group_indices: np.ndarray
    targets: np.ndarray
    listwise_eligible: np.ndarray
    group_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    shape: tuple[int, int, int]


def source_qc_mask(
    dataset: object,
    *,
    target_subjects: Sequence[str],
    qc_ptp_uv: float,
) -> tuple[np.ndarray, dict[str, object]]:
    subjects = np.asarray(dataset.subject_ids).astype(str)
    source = ~np.isin(subjects, np.asarray(tuple(target_subjects), dtype=str))
    labels = np.asarray(dataset.y, dtype=np.int64)
    before = np.bincount(labels[source], minlength=2).astype(np.int64)
    bad = np.zeros(len(source), dtype=bool)
    if qc_ptp_uv > 0.0:
        threshold_v = float(qc_ptp_uv) * 1e-6
        values = np.asarray(dataset.X, dtype=np.float32)
        bad = (np.ptp(values, axis=2) >= threshold_v).any(axis=1)
    dropped = np.bincount(labels[source & bad], minlength=2).astype(np.int64)
    legal = source & ~bad
    return legal, {
        "source_rows_before_qc": int(source.sum()),
        "source_rows_after_qc": int(legal.sum()),
        "source_label_counts_before_qc": before.tolist(),
        "qc_dropped_source_epochs_by_label": dropped.tolist(),
        "qc_dropped_source_epochs": int((source & bad).sum()),
        "source_label_counts_after_qc": (before - dropped).tolist(),
    }


def build_group_specs(
    dataset: object,
    row_mask: np.ndarray,
    *,
    vocabulary: tuple[str, ...] | None = None,
) -> tuple[tuple[GroupSpec, ...], tuple[str, ...]]:
    """Map every selected EEG row to one intact candidate-decision group."""

    row_mask = np.asarray(row_mask)
    if row_mask.dtype != np.dtype(bool) or row_mask.shape != (len(dataset.X),):
        raise ValueError("row_mask must be boolean and aligned with dataset.X.")
    timeline = dataset.event_timeline
    evidence_indices = np.asarray(timeline.evidence_indices, dtype=np.int64)
    valid_events = np.flatnonzero(evidence_indices >= 0)
    valid_epoch_rows = evidence_indices[valid_events]
    if np.any(valid_epoch_rows >= len(dataset.X)) or len(np.unique(valid_epoch_rows)) != len(
        valid_epoch_rows
    ):
        raise ValueError("event-to-EEG evidence mapping must be in-range and one-to-one.")
    event_by_epoch = np.full(len(dataset.X), -1, dtype=np.int64)
    event_by_epoch[valid_epoch_rows] = valid_events
    selected_epoch_rows = np.flatnonzero(row_mask)
    if np.any(event_by_epoch[selected_epoch_rows] < 0):
        raise ValueError("every selected EEG row must map to one scheduled event.")

    event_rows = event_by_epoch[selected_epoch_rows]
    group_ids = np.asarray(timeline.group_ids).astype(str)
    event_subjects = np.asarray(timeline.subject_ids).astype(str)
    candidates = np.asarray(timeline.candidate_ids).astype(str)
    targets = np.asarray(timeline.target_candidate_ids).astype(str)
    onsets = np.asarray(timeline.onset_times_s, dtype=np.float64)
    if vocabulary is None:
        vocabulary = tuple(sorted(np.unique(candidates[valid_events]).tolist()))
    if len(vocabulary) != N_CANDIDATES or len(set(vocabulary)) != N_CANDIDATES:
        raise ValueError(f"expected exactly {N_CANDIDATES} candidates, got {vocabulary}.")
    candidate_lookup = {candidate: index for index, candidate in enumerate(vocabulary)}

    specs: list[GroupSpec] = []
    for group_id in sorted(np.unique(group_ids[event_rows]).tolist()):
        selected_events = event_rows[group_ids[event_rows] == group_id]
        selected_events = selected_events[np.argsort(onsets[selected_events], kind="stable")]
        subjects = np.unique(event_subjects[selected_events]).tolist()
        truths = np.unique(targets[selected_events]).tolist()
        if len(subjects) != 1 or len(truths) != 1 or truths[0] not in candidate_lookup:
            raise ValueError(f"group {group_id!r} has an invalid subject or target contract.")
        group_candidates = candidates[selected_events]
        unknown = sorted(set(group_candidates.tolist()) - set(vocabulary))
        if unknown:
            raise ValueError(f"group {group_id!r} contains unknown candidates {unknown}.")
        counts = {candidate: int(np.sum(group_candidates == candidate)) for candidate in vocabulary}
        listwise_eligible = min(counts.values()) >= 1

        candidate_indices = np.asarray(
            [candidate_lookup[value] for value in group_candidates], dtype=np.int64
        )
        occurrence_slots = np.empty(len(selected_events), dtype=np.int64)
        occurrence_fraction = np.empty(len(selected_events), dtype=np.float32)
        for candidate_index in candidate_lookup.values():
            positions = np.flatnonzero(candidate_indices == candidate_index)
            occurrence_slots[positions] = np.arange(len(positions), dtype=np.int64)
            denominator = max(len(positions) - 1, 1)
            occurrence_fraction[positions] = np.arange(len(positions)) / denominator
        specs.append(
            GroupSpec(
                group_id=group_id,
                subject_id=str(subjects[0]),
                target_index=candidate_lookup[str(truths[0])],
                epoch_rows=evidence_indices[selected_events].astype(np.int64),
                candidate_indices=candidate_indices,
                occurrence_slots=occurrence_slots,
                occurrence_fraction=occurrence_fraction,
                listwise_eligible=listwise_eligible,
            )
        )

    assigned = np.concatenate([spec.epoch_rows for spec in specs])
    if len(assigned) != len(selected_epoch_rows) or not np.array_equal(
        np.sort(assigned), selected_epoch_rows
    ):
        raise ValueError("group construction did not consume every selected EEG row exactly once.")
    return tuple(specs), vocabulary


def remap_groups_to_local_rows(
    groups: Sequence[GroupSpec], selected_rows: np.ndarray, total_rows: int
) -> tuple[GroupSpec, ...]:
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    local_by_global = np.full(total_rows, -1, dtype=np.int64)
    local_by_global[selected_rows] = np.arange(len(selected_rows), dtype=np.int64)
    output = []
    for group in groups:
        local = local_by_global[group.epoch_rows]
        if np.any(local < 0):
            raise ValueError("group contains an EEG row outside the selected matrix.")
        output.append(
            GroupSpec(
                group_id=group.group_id,
                subject_id=group.subject_id,
                target_index=group.target_index,
                epoch_rows=local,
                candidate_indices=group.candidate_indices,
                occurrence_slots=group.occurrence_slots,
                occurrence_fraction=group.occurrence_fraction,
                listwise_eligible=group.listwise_eligible,
            )
        )
    return tuple(output)


def pack_groups(groups: Sequence[GroupSpec]) -> PackedGroups:
    if not groups:
        raise ValueError("cannot pack an empty group batch.")
    max_occurrences = max(
        int(group.occurrence_slots.max()) + 1 for group in groups
    )
    return PackedGroups(
        local_rows=np.concatenate([group.epoch_rows for group in groups]),
        candidate_indices=np.concatenate([group.candidate_indices for group in groups]),
        occurrence_slots=np.concatenate([group.occurrence_slots for group in groups]),
        occurrence_fraction=np.concatenate([group.occurrence_fraction for group in groups]),
        group_indices=np.concatenate(
            [np.full(len(group.epoch_rows), index, dtype=np.int64) for index, group in enumerate(groups)]
        ),
        targets=np.asarray([group.target_index for group in groups], dtype=np.int64),
        listwise_eligible=np.asarray(
            [group.listwise_eligible for group in groups], dtype=bool
        ),
        group_ids=tuple(group.group_id for group in groups),
        subject_ids=tuple(group.subject_id for group in groups),
        shape=(len(groups), N_CANDIDATES, max_occurrences),
    )


def iter_group_batches(
    groups: Sequence[GroupSpec],
    *,
    groups_per_batch: int,
    rng: np.random.Generator,
) -> Iterable[PackedGroups]:
    if groups_per_batch < 1:
        raise ValueError("groups_per_batch must be positive.")
    order = rng.permutation(len(groups))
    for start in range(0, len(order), groups_per_batch):
        yield pack_groups([groups[index] for index in order[start : start + groups_per_batch]])


def dense_evidence(
    evidence: torch.Tensor,
    packed: PackedGroups,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group_indices = torch.as_tensor(packed.group_indices, device=device)
    candidates = torch.as_tensor(packed.candidate_indices, device=device)
    slots = torch.as_tensor(packed.occurrence_slots, device=device)
    values = evidence.new_zeros(packed.shape)
    values = values.index_put((group_indices, candidates, slots), evidence, accumulate=False)
    mask = torch.zeros(packed.shape, device=device, dtype=torch.bool)
    mask[group_indices, candidates, slots] = True
    occurrence = evidence.new_zeros(packed.shape)
    occurrence = occurrence.index_put(
        (group_indices, candidates, slots),
        torch.as_tensor(packed.occurrence_fraction, device=device, dtype=evidence.dtype),
        accumulate=False,
    )
    return values, mask, occurrence


class EndToEndTemperedDecision(nn.Module):
    """Differentiable all-evidence decision head with label-free trial weights."""

    def __init__(
        self,
        *,
        abs_center: float,
        abs_scale: float,
        score_scale: float,
        initial_count_power: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_count_power < 1.0:
            raise ValueError("initial_count_power must be strictly between zero and one.")
        if abs_scale <= 0.0 or score_scale <= 0.0:
            raise ValueError("feature scales must be positive.")
        self.raw_beta = nn.Parameter(
            torch.tensor(math.log(initial_count_power / (1.0 - initial_count_power)))
        )
        self.weight_abs = nn.Parameter(torch.tensor(0.0))
        self.weight_occurrence = nn.Parameter(torch.tensor(0.0))
        self.raw_gain = nn.Parameter(torch.tensor(float(np.log(np.expm1(1.0)))))
        self.register_buffer("abs_center", torch.tensor(float(abs_center)))
        self.register_buffer("abs_scale", torch.tensor(float(abs_scale)))
        self.register_buffer("score_scale", torch.tensor(float(score_scale)))

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        occurrence: torch.Tensor,
    ) -> torch.Tensor:
        if values.shape != mask.shape or values.shape != occurrence.shape or values.ndim != 3:
            raise ValueError("decision tensors must share shape (groups,candidates,occurrences).")
        mask_float = mask.to(dtype=values.dtype)
        group_count = mask_float.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        group_center = (values * mask_float).sum(dim=(1, 2), keepdim=True) / group_count
        centered = (values - group_center) * mask_float
        normalized_abs = (torch.log1p(centered.abs()) - self.abs_center) / self.abs_scale
        weight_logits = self.weight_abs * normalized_abs + self.weight_occurrence * (
            occurrence - 0.5
        )
        weights = (F.softplus(weight_logits) + 1e-4) * mask_float
        weight_sum = weights.sum(dim=-1)
        squared_weight_sum = weights.square().sum(dim=-1)
        weighted_mean = (weights * (centered / self.score_scale)).sum(dim=-1) / (
            weight_sum.clamp_min(1e-8)
        )
        effective_count = weight_sum.square() / squared_weight_sum.clamp_min(1e-8)
        beta = torch.sigmoid(self.raw_beta)
        gain = F.softplus(self.raw_gain) + 1e-4
        scores = gain * weighted_mean * effective_count.clamp_min(1.0).pow(beta)
        return scores.masked_fill(weight_sum <= 0.0, torch.finfo(scores.dtype).min)

    def parameters_record(self) -> dict[str, float]:
        return {
            "count_power": float(torch.sigmoid(self.raw_beta).detach().cpu()),
            "weight_abs": float(self.weight_abs.detach().cpu()),
            "weight_occurrence": float(self.weight_occurrence.detach().cpu()),
            "gain": float((F.softplus(self.raw_gain) + 1e-4).detach().cpu()),
            "abs_center": float(self.abs_center.detach().cpu()),
            "abs_scale": float(self.abs_scale.detach().cpu()),
            "score_scale": float(self.score_scale.detach().cpu()),
        }


def standardize_selected_rows(
    dataset: object,
    selected_rows: np.ndarray,
    input_stats: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    values = np.asarray(dataset.X[selected_rows], dtype=np.float32)
    if dataset.trial_channel_mask is None:
        mask = np.broadcast_to(np.asarray(dataset.channel_mask, dtype=bool), values.shape[:2])
    else:
        mask = np.asarray(dataset.trial_channel_mask[selected_rows], dtype=bool)
    mean, std = input_stats
    standardized = ((values - mean) / std).astype(np.float32, copy=False)
    np.copyto(standardized, 0.0, where=~mask[:, :, None])
    return np.ascontiguousarray(standardized)


def _group_center_dense(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(dtype=values.dtype)
    center = (values * mask_float).sum(dim=(1, 2), keepdim=True) / mask_float.sum(
        dim=(1, 2), keepdim=True
    ).clamp_min(1.0)
    return (values - center) * mask_float


def initial_decision_statistics(
    model: nn.Module,
    X: torch.Tensor,
    groups: Sequence[GroupSpec],
    *,
    device: torch.device,
    inference_batch_size: int,
) -> dict[str, float]:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(X), inference_batch_size):
            logits = model(X[start : start + inference_batch_size])
            chunks.append((logits[:, 1] - logits[:, 0]).float())
    evidence = torch.cat(chunks)
    packed = pack_groups(groups)
    values, mask, _ = dense_evidence(evidence[torch.as_tensor(packed.local_rows, device=device)], packed, device=device)
    centered = _group_center_dense(values, mask)
    observed = centered[mask]
    log_abs = torch.log1p(observed.abs())
    return {
        "abs_center": float(log_abs.mean().cpu()),
        "abs_scale": float(log_abs.std(unbiased=False).clamp_min(1e-6).cpu()),
        "score_scale": float(observed.square().mean().sqrt().clamp_min(1e-6).cpu()),
    }


def _batch_tensors(
    packed: PackedGroups,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PackedGroups]:
    rows = torch.as_tensor(packed.local_rows, device=device)
    return (
        X[rows],
        y[rows],
        torch.as_tensor(packed.group_indices, device=device),
        torch.as_tensor(packed.candidate_indices, device=device),
        torch.as_tensor(packed.occurrence_slots, device=device),
        packed,
    )


def parameter_delta_record(
    initial_state: Mapping[str, torch.Tensor], model: nn.Module
) -> dict[str, object]:
    squared = 0.0
    changed = 0
    tensors = 0
    by_component: dict[str, float] = {}
    for name, value in model.state_dict().items():
        if name not in initial_state or not torch.is_floating_point(value):
            continue
        tensors += 1
        delta = value.detach().cpu().float() - initial_state[name].float()
        norm = float(torch.linalg.vector_norm(delta))
        squared += norm * norm
        if norm > 0.0:
            changed += 1
        component = "classifier" if name.startswith("classifier.") else "backbone"
        by_component[component] = by_component.get(component, 0.0) + norm * norm
    return {
        "floating_parameter_and_buffer_tensors": tensors,
        "changed_tensors": changed,
        "global_l2": math.sqrt(squared),
        "component_l2": {key: math.sqrt(value) for key, value in by_component.items()},
    }


def fit_end_to_end(
    model: nn.Module,
    decision: EndToEndTemperedDecision,
    X: torch.Tensor,
    y: torch.Tensor,
    groups: Sequence[GroupSpec],
    *,
    device: torch.device,
    epochs: int,
    groups_per_batch: int,
    seed: int,
    backbone_learning_rate: float,
    decision_learning_rate: float,
    weight_decay: float,
    pos_weight: float,
    decision_loss_weight: float,
    compile_mode: str | None,
    fused_adam: bool,
) -> tuple[list[dict[str, float | int]], dict[str, object]]:
    if epochs < 1 or len(groups) < 1:
        raise ValueError("end-to-end training requires positive epochs and source groups.")
    execution = resolve_optimizer_execution(
        device,
        fused_adam=fused_adam,
        compile_mode=compile_mode,
    )
    optimizer_kwargs: dict[str, object] = {"weight_decay": weight_decay}
    if execution.fused_adam:
        optimizer_kwargs["fused"] = True
    if execution.uses_cuda_graphs:
        optimizer_kwargs["capturable"] = True
    optimizer = torch.optim.Adam(
        [
            {"params": list(model.parameters()), "lr": backbone_learning_rate},
            {"params": list(decision.parameters()), "lr": decision_learning_rate},
        ],
        **optimizer_kwargs,
    )
    class_weight = torch.tensor([1.0, pos_weight], device=device)

    def train_step(
        xb: torch.Tensor,
        yb: torch.Tensor,
        group_indices: torch.Tensor,
        candidate_indices: torch.Tensor,
        occurrence_slots: torch.Tensor,
        occurrence_fraction: torch.Tensor,
        targets: torch.Tensor,
        listwise_eligible: torch.Tensor,
        shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        trial_loss = F.cross_entropy(logits, yb, weight=class_weight)
        evidence = logits[:, 1] - logits[:, 0]
        values = evidence.new_zeros(shape).index_put(
            (group_indices, candidate_indices, occurrence_slots), evidence, accumulate=False
        )
        mask = torch.zeros(shape, device=device, dtype=torch.bool)
        mask[group_indices, candidate_indices, occurrence_slots] = True
        occurrence = evidence.new_zeros(shape).index_put(
            (group_indices, candidate_indices, occurrence_slots),
            occurrence_fraction,
            accumulate=False,
        )
        candidate_scores = decision(values, mask, occurrence)
        per_group_listwise = F.cross_entropy(
            candidate_scores, targets, reduction="none"
        )
        eligible_float = listwise_eligible.to(dtype=per_group_listwise.dtype)
        listwise_loss = (per_group_listwise * eligible_float).sum() / eligible_float.sum().clamp_min(
            1.0
        )
        total = trial_loss / math.log(2.0) + decision_loss_weight * (
            listwise_loss / math.log(float(N_CANDIDATES))
        )
        total.backward()
        optimizer.step()
        return total.detach(), trial_loss.detach(), listwise_loss.detach()

    effective_train_step = train_step
    if execution.compile_mode is not None:
        effective_train_step = torch.compile(
            train_step,
            mode=execution.compile_mode,
            fullgraph=False,
        )

    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []
    optimizer_steps = 0
    for epoch in range(epochs):
        model.train()
        decision.train()
        sums = np.zeros(3, dtype=np.float64)
        n_trials = 0
        n_groups = 0
        for packed in iter_group_batches(
            groups, groups_per_batch=groups_per_batch, rng=rng
        ):
            rows = torch.as_tensor(packed.local_rows, device=device)
            group_indices = torch.as_tensor(packed.group_indices, device=device)
            candidate_indices = torch.as_tensor(packed.candidate_indices, device=device)
            occurrence_slots = torch.as_tensor(packed.occurrence_slots, device=device)
            occurrence_fraction = torch.as_tensor(
                packed.occurrence_fraction, device=device, dtype=X.dtype
            )
            targets = torch.as_tensor(packed.targets, device=device)
            listwise_eligible = torch.as_tensor(
                packed.listwise_eligible, device=device
            )
            total, trial, listwise = effective_train_step(
                X[rows],
                y[rows],
                group_indices,
                candidate_indices,
                occurrence_slots,
                occurrence_fraction,
                targets,
                listwise_eligible,
                packed.shape,
            )
            group_count = len(packed.targets)
            sums += np.asarray(
                [float(total.cpu()), float(trial.cpu()), float(listwise.cpu())]
            ) * group_count
            n_trials += len(packed.local_rows)
            n_groups += group_count
            optimizer_steps += 1
        if n_trials != len(X) or n_groups != len(groups):
            raise RuntimeError("an epoch did not consume every legal source row and group.")
        history.append(
            {
                "epoch": epoch + 1,
                "groups": n_groups,
                "trials": n_trials,
                "total_loss": float(sums[0] / n_groups),
                "trial_ce": float(sums[1] / n_groups),
                "listwise_ce": float(sums[2] / n_groups),
            }
        )
        print(json.dumps({"phase": "fine_tune", **history[-1]}), flush=True)
    return history, {**execution.record(), "optimizer_steps": optimizer_steps}


def _unique_argmax(scores: np.ndarray) -> int | None:
    maximum = float(np.max(scores))
    tied = np.flatnonzero(np.isclose(scores, maximum, rtol=1e-12, atol=1e-12))
    return int(tied[0]) if len(tied) == 1 else None


def evaluate_target(
    model: nn.Module,
    decision: EndToEndTemperedDecision,
    X: torch.Tensor,
    y: torch.Tensor,
    groups: Sequence[GroupSpec],
    vocabulary: tuple[str, ...],
    *,
    device: torch.device,
    inference_batch_size: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    model.eval()
    decision.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(X), inference_batch_size):
            logits = model(X[start : start + inference_batch_size])
            chunks.append((logits[:, 1] - logits[:, 0]).float())
    evidence = torch.cat(chunks)
    packed = pack_groups(groups)
    packed_rows = torch.as_tensor(packed.local_rows, device=device)
    values, mask, occurrence = dense_evidence(evidence[packed_rows], packed, device=device)
    centered = _group_center_dense(values, mask)
    candidate_counts = mask.sum(dim=-1)
    fixed_tensor = (centered * mask).sum(dim=-1) / candidate_counts.clamp_min(1)
    fixed_tensor = fixed_tensor.masked_fill(candidate_counts == 0, -torch.inf)
    fixed_scores = fixed_tensor.cpu().numpy()
    with torch.inference_mode():
        learned_scores = decision(values, mask, occurrence).float().cpu().numpy()
    labels = y.detach().cpu().numpy()
    evidence_cpu = evidence.detach().cpu().numpy()

    records: list[dict[str, object]] = []
    for index, group in enumerate(groups):
        fixed_index = _unique_argmax(fixed_scores[index])
        learned_index = _unique_argmax(learned_scores[index])
        counts = np.bincount(group.candidate_indices, minlength=N_CANDIDATES)
        group_labels = labels[group.epoch_rows]
        group_logits = evidence_cpu[group.epoch_rows]
        auc = (
            float(roc_auc_score(group_labels, group_logits))
            if len(np.unique(group_labels)) == 2
            else None
        )
        records.append(
            {
                "subject": group.subject_id,
                "group": group.group_id,
                "truth": vocabulary[group.target_index],
                "candidate_counts": {
                    candidate: int(counts[candidate_index])
                    for candidate_index, candidate in enumerate(vocabulary)
                },
                "binary_auc": auc,
                "fixed_mean": {
                    "prediction": None if fixed_index is None else vocabulary[fixed_index],
                    "hit": bool(fixed_index == group.target_index),
                    "scores": fixed_scores[index].tolist(),
                },
                "learned_tempered": {
                    "prediction": None if learned_index is None else vocabulary[learned_index],
                    "hit": bool(learned_index == group.target_index),
                    "scores": learned_scores[index].tolist(),
                },
            }
        )
    metrics = {
        "requested_subjects": len(groups),
        "fixed_mean_hits": int(sum(record["fixed_mean"]["hit"] for record in records)),
        "fixed_mean_operational_hit": float(
            np.mean([record["fixed_mean"]["hit"] for record in records])
        ),
        "learned_tempered_hits": int(
            sum(record["learned_tempered"]["hit"] for record in records)
        ),
        "learned_tempered_operational_hit": float(
            np.mean([record["learned_tempered"]["hit"] for record in records])
        ),
        "binary_auc_subject_macro": float(
            np.mean([record["binary_auc"] for record in records if record["binary_auc"] is not None])
        ),
    }
    return records, metrics


def _validate_checkpoint_contract(
    payload: Mapping[str, object],
    *,
    cache_sha256: str,
    target_subjects: tuple[str, ...],
    all_subjects: set[str],
    source_rows: int,
    kernel: int,
    seed: int,
    qc_ptp_uv: float,
) -> None:
    if payload.get("source_cache_sha256") != cache_sha256:
        raise ValueError("base checkpoint source-cache SHA-256 mismatch.")
    if set(str(value) for value in payload.get("holdout_subjects", [])) != set(target_subjects):
        raise ValueError("base checkpoint holdout subjects do not equal the target block.")
    if set(str(value) for value in payload.get("training_subjects", [])) != (
        all_subjects - set(target_subjects)
    ):
        raise ValueError("base checkpoint training subjects are not cache-minus-target.")
    if int(payload.get("n_source_epochs_used", -1)) != source_rows:
        raise ValueError("base checkpoint source-row count differs from current QC100 rows.")
    if not np.isclose(float(payload.get("qc_ptp_uv", -1.0)), qc_ptp_uv):
        raise ValueError("base checkpoint QC threshold mismatch.")
    config = payload.get("config")
    architecture = payload.get("architecture")
    if not isinstance(config, Mapping) or int(config.get("seed", -1)) != seed:
        raise ValueError("base checkpoint seed mismatch.")
    if not isinstance(architecture, Mapping) or int(
        architecture.get("st_temporal_kernel_samples", -1)
    ) != kernel:
        raise ValueError("base checkpoint temporal kernel mismatch.")
    if architecture.get("pooling_mode") != "full_unfold":
        raise ValueError("end-to-end comparison requires full_unfold checkpoints.")


def run(args: argparse.Namespace) -> None:
    manifest, manifest_sha = validate_manifest(
        args.manifest, kernel=args.kernel, seed=args.seed, block=args.block
    )
    training = manifest["training"]
    assert isinstance(training, Mapping)
    target_subjects = read_subjects(args.target_subjects_file)
    expected_target_sizes = manifest.get("target_block_sizes")
    if not isinstance(expected_target_sizes, list) or len(target_subjects) != int(
        expected_target_sizes[args.block]
    ):
        raise ValueError("target block size differs from the frozen manifest.")

    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    assert_p300_input_contract(
        dataset.preprocessing, GTN_SINGLE_SUBJECT_CAUSAL_DATA_CONTRACT
    )
    cache_attestation = read_epoch_cache_attestation(args.dataset_cache)
    cache_sha = str(cache_attestation["sha256"])
    expected_cache_sha = manifest.get("source_cache_sha256")
    if cache_sha != expected_cache_sha:
        raise ValueError("dataset cache differs from the frozen end-to-end manifest.")
    dataset_subjects = np.asarray(dataset.subject_ids).astype(str)
    all_subjects = set(dataset_subjects.tolist())
    if not set(target_subjects) <= all_subjects:
        raise ValueError("target block contains subjects absent from the dataset cache.")

    legal_source, qc_record = source_qc_mask(
        dataset,
        target_subjects=target_subjects,
        qc_ptp_uv=float(training["source_qc_ptp_uv"]),
    )
    source_groups, vocabulary = build_group_specs(dataset, legal_source)
    target_mask = np.isin(dataset_subjects, np.asarray(target_subjects, dtype=str))
    target_groups, target_vocabulary = build_group_specs(
        dataset, target_mask, vocabulary=vocabulary
    )
    if target_vocabulary != vocabulary:
        raise AssertionError("source and target candidate vocabularies differ.")
    if len(target_groups) != len(target_subjects) or {
        group.subject_id for group in target_groups
    } != set(target_subjects):
        raise ValueError("target block must map one decision group to each requested subject.")

    model, payload = load_n2p3_trunk_checkpoint(args.base_checkpoint, dataset)
    _validate_checkpoint_contract(
        payload,
        cache_sha256=cache_sha,
        target_subjects=target_subjects,
        all_subjects=all_subjects,
        source_rows=int(legal_source.sum()),
        kernel=args.kernel,
        seed=args.seed,
        qc_ptp_uv=float(training["source_qc_ptp_uv"]),
    )
    input_stats = checkpoint_input_stats(payload, dataset.n_channels, required=True)
    assert input_stats is not None

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    model = model.to(device)
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if torch.is_floating_point(value)
    }

    source_rows = np.flatnonzero(legal_source)
    target_rows = np.flatnonzero(target_mask)
    source_groups = remap_groups_to_local_rows(source_groups, source_rows, len(dataset.X))
    target_groups = remap_groups_to_local_rows(target_groups, target_rows, len(dataset.X))
    source_X = torch.as_tensor(
        standardize_selected_rows(dataset, source_rows, input_stats), device=device
    )
    source_y = torch.as_tensor(np.asarray(dataset.y[source_rows], dtype=np.int64), device=device)
    target_X = torch.as_tensor(
        standardize_selected_rows(dataset, target_rows, input_stats), device=device
    )
    target_y = torch.as_tensor(np.asarray(dataset.y[target_rows], dtype=np.int64), device=device)

    stats = initial_decision_statistics(
        model,
        source_X,
        source_groups,
        device=device,
        inference_batch_size=int(training["inference_batch_size"]),
    )
    decision = EndToEndTemperedDecision(
        **stats,
        initial_count_power=float(training["initial_count_power"]),
    ).to(device)
    compile_mode = training.get("compile_mode")
    compile_mode = None if compile_mode is None else str(compile_mode)
    if args.compile_mode is not None:
        compile_mode = None if args.compile_mode == "none" else args.compile_mode

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    history, runtime = fit_end_to_end(
        model,
        decision,
        source_X,
        source_y,
        source_groups,
        device=device,
        epochs=int(training["epochs"]),
        groups_per_batch=int(training["groups_per_batch"]),
        seed=seed,
        backbone_learning_rate=float(training["backbone_learning_rate"]),
        decision_learning_rate=float(training["decision_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        pos_weight=float(payload["training_pos_weight"]),
        decision_loss_weight=float(training["decision_loss_weight"]),
        compile_mode=compile_mode,
        fused_adam=bool(training.get("fused_adam", DEFAULT_FUSED_ADAM)),
    )
    fit_seconds = time.perf_counter() - started
    delta = parameter_delta_record(initial_state, model)
    if delta["component_l2"].get("backbone", 0.0) <= 0.0 or delta[
        "component_l2"
    ].get("classifier", 0.0) <= 0.0:
        raise RuntimeError("end-to-end training failed to update backbone and classifier.")

    records, metrics = evaluate_target(
        model,
        decision,
        target_X,
        target_y,
        target_groups,
        vocabulary,
        device=device,
        inference_batch_size=int(training["inference_batch_size"]),
    )
    checkpoint_path = Path(args.output_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    fine_tuned_payload = dict(payload)
    fine_tuned_payload.update(
        {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "trunk_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "decision_state_dict": {
                key: value.detach().cpu() for key, value in decision.state_dict().items()
            },
            "end_to_end_finetune": {
                "manifest_sha256": manifest_sha,
                "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
                "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
                "kernel": args.kernel,
                "seed": args.seed,
                "block": args.block,
                "target_subjects": list(target_subjects),
                "source_groups": len(source_groups),
                "source_listwise_eligible_groups": int(
                    sum(group.listwise_eligible for group in source_groups)
                ),
                "source_listwise_ineligible_groups": int(
                    sum(not group.listwise_eligible for group in source_groups)
                ),
                "source_rows": len(source_X),
                "epochs": int(training["epochs"]),
                "every_source_row_per_epoch": True,
                "group_preserving_batches": True,
                "updates": ["backbone", "classifier", "decision_head"],
                "decision_parameters": decision.parameters_record(),
                "parameter_delta": delta,
                "runtime": runtime,
                "fit_seconds": fit_seconds,
            },
        }
    )
    torch.save(fine_tuned_payload, checkpoint_path)

    peak_memory_mb = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    output = {
        "schema": RESULT_SCHEMA,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest_sha,
        "dataset_cache": str(Path(args.dataset_cache).resolve()),
        "dataset_cache_sha256": cache_sha,
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "output_checkpoint": str(checkpoint_path.resolve()),
        "output_checkpoint_sha256": sha256_file(checkpoint_path),
        "kernel": args.kernel,
        "seed": args.seed,
        "block": args.block,
        "target_subjects_file": str(Path(args.target_subjects_file).resolve()),
        "target_subjects": list(target_subjects),
        "candidate_vocabulary": list(vocabulary),
        "training_contract": {
            **dict(training),
            **qc_record,
            "source_subjects": len({group.subject_id for group in source_groups}),
            "source_groups": len(source_groups),
            "source_listwise_eligible_groups": int(
                sum(group.listwise_eligible for group in source_groups)
            ),
            "source_listwise_ineligible_groups": int(
                sum(not group.listwise_eligible for group in source_groups)
            ),
            "every_source_row_per_epoch": True,
            "group_preserving_batches": True,
            "full_eeg_backpropagation": True,
            "target_labels_used_for_training": False,
            "optimizer_steps": runtime["optimizer_steps"],
            "fit_seconds": fit_seconds,
            "peak_cuda_memory_mb": peak_memory_mb,
            "runtime": runtime,
            "parameter_delta": delta,
        },
        "decision_parameters": decision.parameters_record(),
        "history": history,
        "metrics": metrics,
        "subjects": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "phase": "complete",
                "output": str(output_path),
                "checkpoint": str(checkpoint_path),
                "fit_seconds": fit_seconds,
                "source_rows": len(source_X),
                "metrics": metrics,
                "decision_parameters": decision.parameters_record(),
            }
        ),
        flush=True,
    )


def analyze(args: argparse.Namespace) -> None:
    manifest, manifest_sha = validate_manifest(args.manifest)
    kernels = _manifest_tuple(manifest, "kernels")
    seeds = _manifest_tuple(manifest, "seeds")
    blocks = _manifest_tuple(manifest, "blocks")
    result_dir = Path(args.result_dir)
    rows: dict[tuple[int, int, str], dict[str, object]] = {}
    run_metrics: list[dict[str, object]] = []
    for kernel in kernels:
        for seed in seeds:
            seen_subjects: set[str] = set()
            for block in blocks:
                path = result_dir / f"k{kernel}_seed{seed}_blk{block}.json"
                result = read_json_mapping(path, label="end-to-end result")
                if result.get("schema") != RESULT_SCHEMA or result.get(
                    "manifest_sha256"
                ) != manifest_sha:
                    raise ValueError(f"{path} is not bound to the frozen manifest.")
                if (result.get("kernel"), result.get("seed"), result.get("block")) != (
                    kernel,
                    seed,
                    block,
                ):
                    raise ValueError(f"{path} run identity mismatch.")
                training = result.get("training_contract")
                if not isinstance(training, Mapping) or training.get(
                    "full_eeg_backpropagation"
                ) is not True:
                    raise ValueError(f"{path} did not attest end-to-end EEG training.")
                subjects = result.get("subjects")
                if not isinstance(subjects, list):
                    raise ValueError(f"{path} lacks subject records.")
                for record in subjects:
                    if not isinstance(record, dict):
                        raise ValueError(f"{path} has an invalid subject record.")
                    subject = str(record["subject"])
                    if subject in seen_subjects:
                        raise ValueError(f"subject {subject} appears in multiple target blocks.")
                    seen_subjects.add(subject)
                    rows[(kernel, seed, subject)] = record
                run_metrics.append(
                    {
                        "kernel": kernel,
                        "seed": seed,
                        "block": block,
                        "fit_seconds": float(training["fit_seconds"]),
                        **dict(result["metrics"]),
                    }
                )
            if len(seen_subjects) != int(manifest["requested_subjects"]):
                raise ValueError("four blocks do not cover the requested subject denominator.")

    endpoint_metrics: dict[str, object] = {}
    subject_seed_means: dict[tuple[int, str, str], float] = {}
    for kernel in kernels:
        kernel_output: dict[str, object] = {}
        subjects = sorted(
            {subject for (row_kernel, _, subject) in rows if row_kernel == kernel}
        )
        for endpoint in ("fixed_mean", "learned_tempered"):
            per_seed = {}
            for seed in seeds:
                hits = [bool(rows[(kernel, seed, subject)][endpoint]["hit"]) for subject in subjects]
                per_seed[str(seed)] = float(np.mean(hits))
            for subject in subjects:
                subject_seed_means[(kernel, endpoint, subject)] = float(
                    np.mean(
                        [rows[(kernel, seed, subject)][endpoint]["hit"] for seed in seeds]
                    )
                )
            kernel_output[endpoint] = {
                "operational_hit_seed_mean": float(np.mean(list(per_seed.values()))),
                "per_seed": per_seed,
            }
        endpoint_metrics[str(kernel)] = kernel_output

    contrasts = {}
    for kernel in kernels:
        subjects = sorted(
            {subject for (row_kernel, _, subject) in rows if row_kernel == kernel}
        )
        differences = np.asarray(
            [
                subject_seed_means[(kernel, "learned_tempered", subject)]
                - subject_seed_means[(kernel, "fixed_mean", subject)]
                for subject in subjects
            ],
            dtype=float,
        )
        contrasts[str(kernel)] = {
            "learned_minus_fixed_mean": float(differences.mean()),
            "positive_subjects": int(np.sum(differences > 0.0)),
            "negative_subjects": int(np.sum(differences < 0.0)),
            "ties": int(np.sum(differences == 0.0)),
        }
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest_sha,
        "requested_subjects": int(manifest["requested_subjects"]),
        "metrics": endpoint_metrics,
        "contrasts": contrasts,
        "runs": run_metrics,
        "inference_scope": (
            "GTN development cohort; paired subjects across frozen kernels, seeds, and target blocks"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **analysis}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="fine-tune and evaluate one kernel/seed/block")
    run_parser.add_argument("--dataset-cache", required=True)
    run_parser.add_argument("--base-checkpoint", required=True)
    run_parser.add_argument("--target-subjects-file", required=True)
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--kernel", type=int, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--block", type=int, required=True)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "reduce-overhead", "max-autotune"),
        default=None,
        help="Optional runtime override; omission uses the frozen manifest.",
    )
    run_parser.add_argument("--output-checkpoint", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.set_defaults(func=run)

    analyze_parser = subparsers.add_parser("analyze", help="aggregate all frozen runs")
    analyze_parser.add_argument("--result-dir", required=True)
    analyze_parser.add_argument("--manifest", required=True)
    analyze_parser.add_argument("--output", required=True)
    analyze_parser.set_defaults(func=analyze)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
