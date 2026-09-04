"""Auditable source-row sampling.

The module exposes two independent samplers.  ``select_pds_rows`` is the
target-similarity diverse (TSD) greedy sampler already used by the training
pipeline.  ``select_dense_pds_rows`` is a faithful implementation of the
dataset-internal dart-throwing Dense PDS sampler (Flores et al. 2024,
arXiv:2412.17833, Algorithm 1); it samples mingling classes and enforces a
hard disk radius, and deliberately ignores target features beyond validating
the declared calibration scope and binding provenance digests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VALID_SELECTION_SCOPES = frozenset({"target_calibration", "target_unlabelled_buffer"})
SOURCE_SELECTION_SCHEMA = "n2p3_source_selection/2"
LEGACY_SOURCE_SELECTION_SCHEMA = "n2p3_pds_source_selection/1"
VALID_SAMPLER_KINDS = frozenset({"target_similarity_diverse", "dense_pds"})


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _record_integer(value: object, *, name: str) -> int:
    """Coerce a JSON record field to an integer, rejecting non-integral values."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be an integer.")
    integer = int(value)
    if float(value) != integer:
        raise TypeError(f"{name} must be an integer.")
    return integer


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
    sampler_kind: str = "target_similarity_diverse"
    pds_radius: float | None = None
    pds_mingling_mixture: tuple[float, ...] | None = None
    pds_attempts: int = 0
    pds_fallback_count: int = 0

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
        if self.sampler_kind not in VALID_SAMPLER_KINDS:
            raise ValueError(
                f"sampler_kind must be one of {sorted(VALID_SAMPLER_KINDS)!r}."
            )
        if self.sampler_kind == "dense_pds":
            radius = self.pds_radius
            try:
                radius_valid = (
                    radius is not None and np.isfinite(radius) and radius > 0.0
                )
            except (TypeError, ValueError):
                radius_valid = False
            if not radius_valid:
                raise ValueError("dense_pds requires a finite positive pds_radius.")
            mixture = self.pds_mingling_mixture
            try:
                mixture_valid = (
                    isinstance(mixture, tuple)
                    and len(mixture) >= 2
                    and all(np.isfinite(value) and value > 0.0 for value in mixture)
                    and np.isclose(sum(mixture), 1.0, rtol=1e-9, atol=1e-9)
                )
            except (TypeError, ValueError):
                mixture_valid = False
            if not mixture_valid:
                raise ValueError(
                    "dense_pds requires a tuple pds_mingling_mixture with "
                    "positive entries summing to one."
                )
            if (
                isinstance(self.pds_attempts, bool)
                or not isinstance(self.pds_attempts, (int, np.integer))
                or self.pds_attempts < 1
            ):
                raise ValueError("dense_pds pds_attempts must be a positive integer.")
            if (
                isinstance(self.pds_fallback_count, bool)
                or not isinstance(self.pds_fallback_count, (int, np.integer))
                or self.pds_fallback_count < 0
                or self.pds_fallback_count > self.pds_attempts
            ):
                raise ValueError(
                    "pds_fallback_count must be a non-negative integer no larger than pds_attempts."
                )
        else:
            if self.pds_radius is not None or self.pds_mingling_mixture is not None:
                raise ValueError(
                    "TSD selections cannot declare dense-PDS radius/mixture fields."
                )
            if (
                isinstance(self.pds_attempts, bool)
                or not isinstance(self.pds_attempts, (int, np.integer))
                or self.pds_attempts != 0
                or isinstance(self.pds_fallback_count, bool)
                or not isinstance(self.pds_fallback_count, (int, np.integer))
                or self.pds_fallback_count != 0
            ):
                raise ValueError("TSD selections cannot declare dense-PDS attempt counters.")
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
            "schema": SOURCE_SELECTION_SCHEMA,
            "sampler_kind": self.sampler_kind,
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
            "pds_radius": None if self.pds_radius is None else float(self.pds_radius),
            "pds_mingling_mixture": (
                None
                if self.pds_mingling_mixture is None
                else [float(value) for value in self.pds_mingling_mixture]
            ),
            "pds_attempts": int(self.pds_attempts),
            "pds_fallback_count": int(self.pds_fallback_count),
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
    Legacy ``n2p3_pds_source_selection/1`` records reconstruct as TSD and keep
    the additive dense-PDS fields at their neutral ``None``/``0`` values.
    """

    schema = record.get("schema")
    if schema == SOURCE_SELECTION_SCHEMA:
        sampler_kind = str(record.get("sampler_kind", "target_similarity_diverse"))
        try:
            raw_radius = record.get("pds_radius")
            pds_radius = None if raw_radius is None else float(raw_radius)
            raw_mixture = record.get("pds_mingling_mixture")
            if raw_mixture is None:
                pds_mingling_mixture = None
            elif not isinstance(raw_mixture, (list, tuple)):
                raise TypeError("pds_mingling_mixture must be a JSON list")
            else:
                pds_mingling_mixture = tuple(float(value) for value in raw_mixture)
            pds_attempts = _record_integer(
                record.get("pds_attempts", 0), name="pds_attempts"
            )
            pds_fallback_count = _record_integer(
                record.get("pds_fallback_count", 0), name="pds_fallback_count"
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"invalid PDS source-selection dense fields: {error}"
            ) from error
    elif schema == LEGACY_SOURCE_SELECTION_SCHEMA:
        sampler_kind = "target_similarity_diverse"
        pds_radius = None
        pds_mingling_mixture = None
        pds_attempts = 0
        pds_fallback_count = 0
    else:
        raise ValueError("unsupported source-selection record schema.")
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
            sampler_kind=sampler_kind,
            pds_radius=pds_radius,
            pds_mingling_mixture=pds_mingling_mixture,
            pds_attempts=pds_attempts,
            pds_fallback_count=pds_fallback_count,
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


def _validate_selection_inputs(
    source_features: object,
    target_features: object,
    *,
    n_select: int,
    selection_scope: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate the inputs shared by both PDS samplers."""
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
    return source, target, int(seed)


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


def _update_min_distances(
    minimum_distance: np.ndarray,
    source: np.ndarray,
    selected_vector: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> None:
    """Fold Euclidean distances to one selected row into running minima.

    Coordinate deltas are formed first and reduced with ``numpy.hypot`` in
    chunks.  This avoids both the ``||a||^2 + ||b||^2 - 2 a.b`` cancellation
    for large nearby coordinates and the square/square-root underflow for tiny
    coordinates, while keeping peak memory at ``chunk_size * P``.
    """
    for start in range(0, len(source), chunk_size):
        block = source[start : start + chunk_size]
        delta = block - selected_vector[None, :]
        distances = np.hypot.reduce(delta, axis=1)
        np.minimum(
            minimum_distance[start : start + chunk_size],
            distances,
            out=minimum_distance[start : start + chunk_size],
        )


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

    source, target, seed = _validate_selection_inputs(
        source_features,
        target_features,
        n_select=n_select,
        selection_scope=selection_scope,
        seed=seed,
    )
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
    log_quality = -distances / (2.0 * target_temperature**2)
    source_sq = np.einsum("np,np->n", source, source)
    rng = np.random.default_rng(int(seed))
    tie_break = rng.uniform(0.0, 1e-9, len(source))

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
            log_quality[candidates]
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




def select_dense_pds_rows(
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    n_select: int,
    selection_scope: str,
    seed: int = 0,
    labels: np.ndarray | None = None,
    pds_radius: float = 1.0,
    pds_mingling_mixture: tuple[float, ...] | list[float] | np.ndarray | None = None,
    pds_max_attempts: int = 10000,
) -> PDSSelection:
    """Select source rows with Flores et al. Dense PDS dart-throwing.

    The sampler draws a mingling class ``m ~ Cat(pi)`` and then uniformly
    draws one still-unselected row from that class.  A proposal is accepted
    only when its Euclidean distance to every already selected row is strictly
    greater than ``pds_radius``; ``distance <= pds_radius`` rejects.  Class
    draws that find an empty class also consume one attempt.  Target features
    are validated and digested for provenance but never influence this sampler.

    ``pds_mingling_mixture`` is aligned with ``sorted(numpy.unique(labels))``.
    When omitted it defaults to the empirical class frequencies in that order.
    If ``pds_max_attempts`` is exhausted before ``n_select`` rows are chosen,
    remaining slots use the unselected row with the largest minimum distance
    to the selected rows (ties resolved by an RNG permutation).
    """

    source, target, seed = _validate_selection_inputs(
        source_features,
        target_features,
        n_select=n_select,
        selection_scope=selection_scope,
        seed=seed,
    )
    try:
        radius_valid = bool(np.isfinite(pds_radius)) and pds_radius > 0.0
    except (TypeError, ValueError):
        radius_valid = False
    if not radius_valid:
        raise ValueError("pds_radius must be finite and positive.")
    if (
        isinstance(pds_max_attempts, bool)
        or not isinstance(pds_max_attempts, (int, np.integer))
        or pds_max_attempts < 1
    ):
        raise ValueError("pds_max_attempts must be a positive integer.")
    if labels is None:
        raise ValueError("dense_pds requires integer source labels aligned with source rows.")
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(source):
        raise ValueError("labels must align with source rows.")
    if labels.dtype == bool or labels.dtype.kind not in "iuf":
        raise ValueError("labels must be integer mingling classes.")
    if labels.dtype.kind == "f":
        if not np.isfinite(labels).all() or not np.all(labels == np.floor(labels)):
            raise ValueError("labels must be integer mingling classes.")
        if bool((labels < np.iinfo(np.int64).min).any()) or bool(
            (labels > np.iinfo(np.int64).max).any()
        ):
            raise ValueError("labels must fit signed 64-bit integer mingling classes.")
        labels = labels.astype(np.int64)
    elif labels.dtype.kind == "u":
        if bool((labels > np.iinfo(np.int64).max).any()):
            raise ValueError("labels must fit signed 64-bit integer mingling classes.")
        labels = labels.astype(np.int64)
    else:
        labels = labels.astype(np.int64, copy=False)

    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("dense_pds requires at least two mingling classes.")
    if pds_mingling_mixture is None:
        counts = np.asarray(
            [int(np.count_nonzero(labels == label)) for label in classes],
            dtype=np.float64,
        )
        mixture = tuple(float(value) for value in counts / float(len(labels)))
    else:
        if isinstance(pds_mingling_mixture, str):
            raise ValueError("pds_mingling_mixture must be a sequence of weights.")
        try:
            mixture = tuple(float(value) for value in pds_mingling_mixture)
        except (TypeError, ValueError) as error:
            raise ValueError("pds_mingling_mixture must be numeric weights.") from error
        if len(mixture) != len(classes):
            raise ValueError(
                "pds_mingling_mixture must align with sorted unique labels "
                f"({len(classes)} classes)."
            )
        if any(not np.isfinite(value) or value <= 0.0 for value in mixture):
            raise ValueError("pds_mingling_mixture weights must be finite and positive.")
        if not np.isclose(sum(mixture), 1.0, rtol=1e-9, atol=1e-9):
            raise ValueError("pds_mingling_mixture weights must sum to one.")

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    selected_vectors: list[np.ndarray] = []
    eligible = np.ones(len(source), dtype=bool)
    pds_attempts = 0
    pds_fallback_count = 0

    def min_distances(indices: np.ndarray) -> np.ndarray:
        """Chunked min distance to every selected row (O(k*P) per block pair)."""
        output = np.full(len(indices), np.inf, dtype=np.float64)
        if not selected_vectors:
            return output
        selected_block = np.asarray(selected_vectors, dtype=np.float64)
        for selected_start in range(0, len(selected_block), 128):
            selected_slab = selected_block[selected_start : selected_start + 128]
            for index_start in range(0, len(indices), 16):
                index_slab = indices[index_start : index_start + 16]
                delta = source[index_slab, None, :] - selected_slab[None, :, :]
                distances = np.hypot.reduce(delta, axis=2)
                np.minimum(
                    output[index_start : index_start + len(index_slab)],
                    distances.min(axis=1),
                    out=output[index_start : index_start + len(index_slab)],
                )
        return output

    def accept(index: int) -> None:
        selected.append(int(index))
        selected_vectors.append(source[index])
        eligible[index] = False

    while len(selected) < n_select:
        if pds_attempts >= pds_max_attempts:
            candidates = np.flatnonzero(eligible)
            if not len(candidates):
                raise ValueError(
                    "dense_pds cannot select the requested rows from the remaining source pool."
                )
            permuted = rng.permutation(candidates)
            chosen = int(permuted[int(np.argmax(min_distances(permuted)))])
            accept(chosen)
            pds_fallback_count += 1
            continue
        pds_attempts += 1
        mixture_position = int(rng.choice(len(mixture), p=mixture))
        mingling_class = int(classes[mixture_position])
        candidates = np.flatnonzero(eligible & (labels == mingling_class))
        if not len(candidates):
            # An empty mingling class is a degenerate proposal, not a
            # dart-throwing attempt; keep sampling the declared mixture.
            pds_attempts -= 1
            continue
        candidate = int(rng.choice(candidates))
        if float(min_distances(np.asarray([candidate], dtype=np.int64))[0]) > pds_radius:
            accept(candidate)

    indices = np.asarray(selected, dtype=np.int64)
    if pds_fallback_count > pds_attempts:
        pds_attempts = pds_fallback_count
    return PDSSelection(
        selected_indices=indices,
        source_count=len(source),
        target_count=len(target),
        selection_scope=selection_scope,
        seed=int(seed),
        target_temperature=1.0,
        diversity_weight=0.0,
        bandwidth=1.0,
        source_digest=_array_digest(source),
        target_digest=_array_digest(target),
        selected_digest=_array_digest(indices),
        bandwidth_estimator="explicit",
        sampler_kind="dense_pds",
        pds_radius=float(pds_radius),
        pds_mingling_mixture=mixture,
        pds_attempts=pds_attempts,
        pds_fallback_count=pds_fallback_count,
    )

__all__ = [
    "PDSSelection",
    "VALID_SELECTION_SCOPES",
    "array_digest",
    "load_pds_selection_record",
    "pds_selection_from_record",
    "select_dense_pds_rows",
    "select_pds_rows",
]
