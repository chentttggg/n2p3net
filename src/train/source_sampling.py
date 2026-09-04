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
from pathlib import Path

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
    bandwidth_estimator: str = "all_pairs_median"

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
        if self.bandwidth_estimator not in {
            "all_pairs_median",
            "subsampled_median_512",
            "explicit",
        }:
            raise ValueError(
                "bandwidth_estimator must be all_pairs_median, "
                "subsampled_median_512, or explicit."
            )
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
            "bandwidth_estimator": self.bandwidth_estimator,
            "source_digest": self.source_digest,
            "target_digest": self.target_digest,
            "selected_digest": self.selected_digest,
        }


def array_digest(value: np.ndarray) -> str:
    """Deterministic SHA-256 over a numeric array's dtype, shape, and bytes."""
    return _array_digest(value)


def pds_selection_from_record(
    record: dict[str, object],
    *,
    expected_source_digest: str | None = None,
    expected_source_cache_sha256: str | None = None,
) -> PDSSelection:
    """Reconstruct and validate a PDS selection from its JSON record.

    The record schema, selected-index digest, optional feature-matrix digest,
    and optional source-cache SHA-256 are all checked so a runner can consume
    a selection file without silently trusting a stale or edited artifact.
    """

    if record.get("schema") != "n2p3_pds_source_selection/1":
        raise ValueError("unsupported PDS source-selection record schema.")
    if expected_source_cache_sha256 is not None:
        record_cache_sha256 = record.get("source_cache_sha256")
        if (
            not isinstance(record_cache_sha256, str)
            or len(record_cache_sha256) != 64
            or record_cache_sha256 != expected_source_cache_sha256
        ):
            raise ValueError(
                "PDS source-selection source_cache_sha256 is missing or disagrees "
                "with the loaded source cache."
            )
    try:
        selected_indices = np.asarray(record["selected_indices"], dtype=np.int64)
        selection = PDSSelection(
            selected_indices=selected_indices,
            source_count=int(record["source_count"]),
            target_count=int(record["target_count"]),
            selection_scope=str(record["selection_scope"]),
            seed=int(record["seed"]),
            target_temperature=float(record["target_temperature"]),
            diversity_weight=float(record["diversity_weight"]),
            bandwidth=float(record["bandwidth"]),
            source_digest=str(record["source_digest"]),
            target_digest=str(record["target_digest"]),
            selected_digest=str(record["selected_digest"]),
            bandwidth_estimator=str(
                record.get("bandwidth_estimator", "all_pairs_median")
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid PDS source-selection record: {error}") from error
    if selection.selected_digest != array_digest(selection.selected_indices):
        raise ValueError("PDS source-selection selected_indices digest disagrees with the record.")
    if (
        expected_source_digest is not None
        and selection.source_digest != expected_source_digest
    ):
        raise ValueError("PDS source-selection source_digest disagrees with the expected matrix.")
    return selection


def load_pds_selection_record(path: str | Path) -> tuple[PDSSelection, dict[str, object]]:
    """Load a PDS selection JSON and return ``(selection, raw_record)``."""
    resolved = Path(path).resolve(strict=True)
    record = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"{resolved} does not contain a JSON object.")
    return pds_selection_from_record(record), record


def _validate_matrix(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape (N,P), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return array


def _pairwise_squared_distances(
    left: np.ndarray,
    right: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Return ``(M,N)`` squared Euclidean distances without a ``(M,N,P)`` tensor."""
    output = np.empty((left.shape[0], right.shape[0]), dtype=np.float64)
    right_sq = np.einsum("np,np->n", right, right)
    for start in range(0, left.shape[0], chunk_size):
        block = left[start : start + chunk_size]
        block_sq = np.einsum("np,np->n", block, block)
        squared = block_sq[:, None] + right_sq[None, :] - 2.0 * (block @ right.T)
        np.maximum(squared, 0.0, out=squared)
        output[start : start + chunk_size] = squared
    return output


def _estimate_bandwidth(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, str]:
    """Estimate a kernel bandwidth from median pairwise distances.

    For small pools the exact all-pairs median is used.  For large pools a
    deterministic, evenly spaced 512-row subsample keeps the estimate memory
    bounded at ``512*512`` distances instead of ``N*N``.
    """

    combined = np.concatenate((source, target), axis=0)
    if len(combined) <= 512:
        reference = combined
        estimator = "all_pairs_median"
    else:
        positions = np.linspace(0, len(combined) - 1, 512).round().astype(np.int64)
        reference = combined[positions]
        estimator = "subsampled_median_512"
    squared = _pairwise_squared_distances(reference, reference)
    upper = np.triu_indices(len(reference), k=1)
    positive = np.sqrt(squared[upper])
    positive = positive[positive > 0.0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    return bandwidth, estimator


def _add_rbf_penalty(
    penalty: np.ndarray,
    source: np.ndarray,
    selected_vector: np.ndarray,
    *,
    bandwidth: float,
    source_sq: np.ndarray,
    chunk_size: int = 4096,
) -> None:
    """Accumulate ``exp(-||x_i-x_j||^2/(2 b^2))`` over ``j`` in chunks."""
    selected_sq = float(np.dot(selected_vector, selected_vector))
    scale = -1.0 / (2.0 * bandwidth**2)
    for start in range(0, len(source), chunk_size):
        block = source[start : start + chunk_size]
        squared = source_sq[start : start + chunk_size] + selected_sq - 2.0 * (
            block @ selected_vector
        )
        np.maximum(squared, 0.0, out=squared)
        penalty[start : start + chunk_size] += np.exp(scale * squared)


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

    bandwidth_estimator = "explicit"
    if bandwidth is None:
        bandwidth, bandwidth_estimator = _estimate_bandwidth(source, target)
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and positive.")

    prototype = target.mean(axis=0)
    distances = np.sum((source - prototype[None, :]) ** 2, axis=1)
    quality = np.exp(-distances / (2.0 * target_temperature**2))
    source_sq = np.einsum("np,np->n", source, source)
    rng = np.random.default_rng(int(seed))
    tie_break = rng.uniform(0.0, np.finfo(np.float64).eps, len(source))

    selected: list[int] = []
    eligible = np.ones(len(source), dtype=bool)
    # ``penalty[i]`` holds sum_j K(source[i], source[j]) over previously
    # selected rows.  It is updated incrementally in O(N*P) per added row,
    # which keeps peak memory at O(N) instead of an O(N*N) kernel matrix.
    penalty = np.zeros(len(source), dtype=np.float64)

    def select_best(candidates: np.ndarray) -> int:
        if not len(candidates):
            raise ValueError("source quotas cannot be satisfied by available rows.")
        score = (
            np.log(np.maximum(quality[candidates], 1e-300))
            - diversity_weight * penalty[candidates]
            + tie_break[candidates]
        )
        return int(candidates[int(np.argmax(score))])

    def add(index: int) -> None:
        selected.append(index)
        eligible[index] = False
        if diversity_weight > 0.0:
            _add_rbf_penalty(
                penalty,
                source,
                source[index],
                bandwidth=float(bandwidth),
                source_sq=source_sq,
            )

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
        bandwidth_estimator=bandwidth_estimator,
    )


__all__ = [
    "PDSSelection",
    "VALID_SELECTION_SCOPES",
    "array_digest",
    "load_pds_selection_record",
    "pds_selection_from_record",
    "select_pds_rows",
]
