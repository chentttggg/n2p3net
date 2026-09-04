"""Auditable target-aware source-row sampling.

The sampler is deliberately independent of the model and evaluator.  It may
use only a declared calibration/unlabelled target buffer and returns row
indices plus a provenance record.  A PDS-style greedy repulsion term favours
target-similar rows while avoiding redundant source examples.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

VALID_SELECTION_SCOPES = frozenset({"target_calibration", "target_unlabelled_buffer"})


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PDSSelection:
    """Selected rows and the frozen configuration that produced them."""

    selected_indices: np.ndarray
    source_count: int
    target_count: int
    selection_scope: str
    seed: int
    target_temperature: float
    diversity_weight: float
    bandwidth: float
    source_digest: str
    target_digest: str
    selected_digest: str

    def __post_init__(self) -> None:
        indices = np.asarray(self.selected_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0 or len(np.unique(indices)) != len(indices):
            raise ValueError("selected_indices must be a non-empty unique vector.")
        if self.source_count < 1 or self.target_count < 1:
            raise ValueError("source_count and target_count must be positive.")
        if bool((indices < 0).any()) or bool((indices >= self.source_count).any()):
            raise ValueError("selected_indices are outside the source row range.")
        if self.selection_scope not in VALID_SELECTION_SCOPES:
            raise ValueError("selection_scope must name a calibration or unlabelled target buffer.")
        if not np.isfinite(self.target_temperature) or self.target_temperature <= 0.0:
            raise ValueError("target_temperature must be finite and positive.")
        if not np.isfinite(self.diversity_weight) or self.diversity_weight < 0.0:
            raise ValueError("diversity_weight must be finite and non-negative.")
        if not np.isfinite(self.bandwidth) or self.bandwidth <= 0.0:
            raise ValueError("bandwidth must be finite and positive.")
        for name, digest in (
            ("source_digest", self.source_digest),
            ("target_digest", self.target_digest),
            ("selected_digest", self.selected_digest),
        ):
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest.")
        indices.setflags(write=False)
        object.__setattr__(self, "selected_indices", indices)

    def record(self) -> dict[str, object]:
        return {
            "schema": "n2p3_pds_source_selection/1",
            "selected_indices": self.selected_indices.tolist(),
            "selected_count": int(len(self.selected_indices)),
            "source_count": int(self.source_count),
            "target_count": int(self.target_count),
            "selection_scope": self.selection_scope,
            "seed": int(self.seed),
            "target_temperature": float(self.target_temperature),
            "diversity_weight": float(self.diversity_weight),
            "bandwidth": float(self.bandwidth),
            "source_digest": self.source_digest,
            "target_digest": self.target_digest,
            "selected_digest": self.selected_digest,
        }


def _validate_matrix(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape (N,P), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return array


def select_pds_rows(
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    n_select: int,
    selection_scope: str,
    seed: int = 0,
    target_temperature: float = 1.0,
    diversity_weight: float = 1.0,
    bandwidth: float | None = None,
    labels: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    min_per_label: int = 0,
    min_per_group: int = 0,
) -> PDSSelection:
    """Select target-similar, diverse source rows with explicit quotas.

    ``selection_scope`` intentionally has no test option.  Labels and groups
    are used only to enforce source coverage quotas; they never use target
    correctness or evaluator output.
    """

    source = _validate_matrix(source_features, name="source_features")
    target = _validate_matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target feature dimensions must match.")
    if selection_scope not in VALID_SELECTION_SCOPES:
        raise ValueError(
            "selection_scope must be 'target_calibration' or 'target_unlabelled_buffer'; "
            "target test scope is forbidden."
        )
    if isinstance(n_select, bool) or n_select < 1 or n_select > len(source):
        raise ValueError("n_select must be within the source row count.")
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    if not np.isfinite(target_temperature) or target_temperature <= 0.0:
        raise ValueError("target_temperature must be finite and positive.")
    if not np.isfinite(diversity_weight) or diversity_weight < 0.0:
        raise ValueError("diversity_weight must be finite and non-negative.")
    if isinstance(min_per_label, bool) or min_per_label < 0:
        raise ValueError("min_per_label must be non-negative.")
    if isinstance(min_per_group, bool) or min_per_group < 0:
        raise ValueError("min_per_group must be non-negative.")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != len(source):
            raise ValueError("labels must align with source rows.")
    if groups is not None:
        groups = np.asarray(groups).astype(str)
        if groups.ndim != 1 or len(groups) != len(source) or bool((groups == "").any()):
            raise ValueError("groups must be non-empty strings aligned with source rows.")
    if min_per_label and labels is None:
        raise ValueError("min_per_label requires source labels.")
    if min_per_group and groups is None:
        raise ValueError("min_per_group requires source groups.")

    if bandwidth is None:
        combined = np.concatenate((source, target), axis=0)
        distances = combined[:, None, :] - combined[None, :, :]
        norms = np.sqrt(np.sum(distances * distances, axis=2))
        positive = norms[norms > 0.0]
        bandwidth = float(np.median(positive)) if len(positive) else 1.0
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and positive.")

    prototype = target.mean(axis=0)
    distances = np.sum((source - prototype[None, :]) ** 2, axis=1)
    quality = np.exp(-distances / (2.0 * target_temperature**2))
    kernel = np.exp(
        -np.sum((source[:, None, :] - source[None, :, :]) ** 2, axis=2)
        / (2.0 * bandwidth**2)
    )
    rng = np.random.default_rng(int(seed))
    tie_break = rng.uniform(0.0, np.finfo(np.float64).eps, len(source))

    selected: list[int] = []
    eligible = np.ones(len(source), dtype=bool)

    def select_best(candidates: np.ndarray) -> int:
        if not len(candidates):
            raise ValueError("source quotas cannot be satisfied by available rows.")
        if not selected:
            score = np.log(np.maximum(quality[candidates], 1e-300)) + tie_break[candidates]
        else:
            score = (
                np.log(np.maximum(quality[candidates], 1e-300))
                - diversity_weight * kernel[np.ix_(candidates, np.asarray(selected))].sum(axis=1)
                + tie_break[candidates]
            )
        return int(candidates[int(np.argmax(score))])

    def add(index: int) -> None:
        selected.append(index)
        eligible[index] = False

    if min_per_group:
        for group in sorted(np.unique(groups).tolist()):
            candidates = np.flatnonzero(eligible & (groups == group))
            for _ in range(min_per_group):
                add(select_best(candidates if _ == 0 else np.flatnonzero(eligible & (groups == group))))
    if min_per_label:
        for label in np.unique(labels):
            selected_label_count = sum(labels[index] == label for index in selected)
            for _ in range(max(0, min_per_label - selected_label_count)):
                candidates = np.flatnonzero(eligible & (labels == label))
                add(select_best(candidates))
    if len(selected) > n_select:
        raise ValueError("source quotas require more rows than n_select.")
    while len(selected) < n_select:
        add(select_best(np.flatnonzero(eligible)))

    indices = np.asarray(selected, dtype=np.int64)
    return PDSSelection(
        selected_indices=indices,
        source_count=len(source),
        target_count=len(target),
        selection_scope=selection_scope,
        seed=int(seed),
        target_temperature=float(target_temperature),
        diversity_weight=float(diversity_weight),
        bandwidth=float(bandwidth),
        source_digest=_array_digest(source),
        target_digest=_array_digest(target),
        selected_digest=_array_digest(indices),
    )


__all__ = ["PDSSelection", "VALID_SELECTION_SCOPES", "select_pds_rows"]
