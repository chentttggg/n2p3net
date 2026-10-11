"""Source-only auxiliary objectives with explicit row and participant identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SourceObjectiveConfig:
    supcon_weight: float = 0.0
    mmd_weight: float = 0.0
    temperature: float = 0.1
    min_cell_count: int = 2
    feature_location: str = "post_mst"

    def __post_init__(self) -> None:
        for name in ("supcon_weight", "mmd_weight", "temperature"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if self.supcon_weight < 0 or self.mmd_weight < 0:
            raise ValueError("source auxiliary weights must be nonnegative")
        if self.temperature <= 0:
            raise ValueError("contrastive temperature must be finite and positive")
        if type(self.min_cell_count) is not int or self.min_cell_count < 2:
            raise ValueError("MMD min_cell_count must be an integer of at least two")
        if self.feature_location not in {"post_mst", "post_residual"}:
            raise ValueError("unsupported source objective feature location")

    @property
    def enabled(self) -> bool:
        return self.supcon_weight > 0 or self.mmd_weight > 0

    def record(self) -> dict:
        return {
            "schema": "n2p3_source_objective/2",
            **asdict(self),
            "projection": "parameter_free_l2_normalized_flattened_features",
            "loss_reduction": "supcon_weight_times_supcon_plus_mmd_weight_times_mmd",
            "fit_scope": "source_training_rows_only",
            "duplicate_policy": "first_occurrence_per_physical_training_row",
            "positive_policy": "same_class_different_domain_and_participant",
            "negative_policy": "different_class_only",
            "mmd_policy": "class_conditional_domain_pairs",
            "ce_sampling_changed": False,
        }


def source_objective_recipes(
    records: dict | None = None,
) -> dict[str, SourceObjectiveConfig | None]:
    """Predeclared fixed-dose CE/SupCon/MMD source arms, never selected by target scores."""
    records = (
        {"ce": {}, "supcon": {"supcon_weight": 0.05}, "mmd": {"mmd_weight": 0.05}}
        if records is None
        else records
    )
    if not isinstance(records, dict) or not records:
        raise ValueError("source objectives must be a nonempty named mapping")
    resolved = {}
    for name, record in records.items():
        if (
            not isinstance(name, str)
            or not name
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name)
        ):
            raise ValueError(
                "source objective arm names require lowercase letters, digits or underscores"
            )
        if not isinstance(record, dict):
            raise ValueError("source objective recipes must be configuration mappings")
        config = SourceObjectiveConfig(**record)
        resolved[name] = config if config.enabled else None
    if None not in resolved.values():
        raise ValueError("source objective comparisons require a CE control")
    return resolved


def _legacy_source_objective_record(raw: dict) -> dict | None:
    """Validate frozen v1 identity without constructing or relabelling a new recipe."""
    if set(raw) - {"kind", "weight", "temperature", "min_cell_count"}:
        raise ValueError("legacy source objective configuration contains new fields")
    values = {"kind": "ce", "weight": 0.0, "temperature": 0.1, "min_cell_count": 2, **raw}
    kind, weight = values["kind"], values["weight"]
    if kind not in {"ce", "cross_domain_supcon", "conditional_rbf_mmd"}:
        raise ValueError("unsupported legacy source objective")
    if (
        not isinstance(weight, Real)
        or not isfinite(weight)
        or weight < 0
        or (kind == "ce") != (weight == 0)
    ):
        raise ValueError("invalid legacy source auxiliary weight")
    if (
        not isinstance(values["temperature"], Real)
        or not isfinite(values["temperature"])
        or values["temperature"] <= 0
        or type(values["min_cell_count"]) is not int
        or values["min_cell_count"] < 2
    ):
        raise ValueError("invalid legacy source auxiliary configuration")
    if kind == "ce":
        return None
    return {
        "schema": "n2p3_source_objective/1",
        **values,
        "projection": "parameter_free_l2_normalized_flattened_post_mst",
        "fit_scope": "source_training_rows_only",
        "duplicate_policy": "first_occurrence_per_physical_training_row",
        "positive_policy": (
            "same_class_different_domain_and_participant"
            if kind == "cross_domain_supcon"
            else "class_conditional_domain_pairs"
        ),
        "negative_policy": "different_class_only" if kind == "cross_domain_supcon" else None,
        "ce_sampling_changed": False,
    }


def validate_source_objective_contract(optimizer: dict, objective: dict) -> None:
    """Check that a checkpoint's actual auxiliary procedure matches its configuration."""
    config = optimizer.get("refit_config", optimizer.get("config", {}))
    raw = config.get("source_objective")
    declared = objective.get("source_auxiliary")
    if raw is not None and not isinstance(raw, dict):
        raise ValueError("checkpoint source auxiliary configuration must be a mapping")
    legacy = raw is not None and "kind" in raw
    try:
        parsed = None if raw is None or legacy else SourceObjectiveConfig(**raw)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint source auxiliary configuration is invalid") from error
    expected = (
        _legacy_source_objective_record(raw)
        if legacy
        else parsed.record()
        if parsed is not None and parsed.enabled
        else None
    )
    if expected is None:
        if declared is not None:
            raise ValueError(
                "checkpoint source auxiliary is undeclared by its optimizer configuration"
            )
        return
    if not isinstance(declared, dict) or any(declared.get(k) != v for k, v in expected.items()):
        raise ValueError(
            "checkpoint source auxiliary contract differs from its optimizer configuration"
        )
    updates = optimizer.get(
        "actual_updates", optimizer.get("actual_optimizer_updates", config.get("fixed_updates"))
    )
    if declared.get("updates") != updates or type(updates) is not int or updates < 1:
        raise ValueError("checkpoint source auxiliary history has the wrong update scope")
    mean_loss = declared.get("mean_auxiliary_loss")
    if not isinstance(mean_loss, Real) or not isfinite(mean_loss):
        raise ValueError("checkpoint source auxiliary history must be finite")
    if not legacy:
        for field in ("mean_supcon_loss", "mean_mmd_loss"):
            value = declared.get(field)
            if not isinstance(value, Real) or not isfinite(value):
                raise ValueError("checkpoint source auxiliary component history must be finite")
        for field in (
            "unique_rows",
            "active_anchors",
            "positive_pairs",
            "active_mmd_terms",
            "negative_mmd_terms",
            "supcon_inactive_batches",
            "mmd_inactive_batches",
            "inactive_batches",
        ):
            value = declared.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or value < 0
                or value != int(value)
            ):
                raise ValueError(
                    "checkpoint source auxiliary counts must be finite nonnegative integers"
                )
        if (
            any(
                declared[name] > updates
                for name in ("inactive_batches", "supcon_inactive_batches", "mmd_inactive_batches")
            )
            or declared["negative_mmd_terms"] > declared["active_mmd_terms"]
        ):
            raise ValueError("checkpoint source auxiliary counts have an invalid update scope")
        weighted_supcon = parsed.supcon_weight * declared["mean_supcon_loss"]
        weighted_mmd = parsed.mmd_weight * declared["mean_mmd_loss"]
        expected_loss = weighted_supcon + weighted_mmd
        # Components can cancel because the off-diagonal MMD may be negative.
        # Bound accumulation rounding against both component magnitudes.
        tolerance = 1e-7 + 2e-5 * (abs(weighted_supcon) + abs(weighted_mmd))
        if not isfinite(expected_loss) or abs(mean_loss - expected_loss) > tolerance:
            raise ValueError(
                "checkpoint source auxiliary weighted history differs from its components"
            )
        for weight, fields in (
            (
                parsed.supcon_weight,
                ("mean_supcon_loss", "active_anchors", "positive_pairs", "supcon_inactive_batches"),
            ),
            (
                parsed.mmd_weight,
                ("mean_mmd_loss", "active_mmd_terms", "negative_mmd_terms", "mmd_inactive_batches"),
            ),
        ):
            if weight == 0 and any(declared[field] != 0 for field in fields):
                raise ValueError("checkpoint reports activity for a disabled source auxiliary")
        inactive_counts = []
        for weight, inactive, active in (
            (parsed.supcon_weight, "supcon_inactive_batches", "active_anchors"),
            (parsed.mmd_weight, "mmd_inactive_batches", "active_mmd_terms"),
        ):
            if weight > 0:
                inactive_counts.append(declared[inactive])
                if declared[active] < updates - declared[inactive] or (
                    declared[inactive] == updates and declared[active] != 0
                ):
                    raise ValueError(
                        "checkpoint source auxiliary exposure disagrees with its inactive batches"
                    )
        if (
            not max(0, sum(inactive_counts) - (len(inactive_counts) - 1) * updates)
            <= declared["inactive_batches"]
            <= min(inactive_counts)
            or declared["active_anchors"] > declared["unique_rows"]
            or declared["positive_pairs"] < declared["active_anchors"]
        ):
            raise ValueError("checkpoint source auxiliary exposure counts are inconsistent")


class SourceObjective:
    """Validate the source axes once; compute each auxiliary loss on its actual batch.

    Rows index the immutable input training matrix. Repeated sampler draws still
    contribute to CE but appear once in the auxiliary objective. No target data,
    mutable feature queue, learned projection, or random sampling is introduced.
    """

    def __init__(self, config: SourceObjectiveConfig, *, labels, domains, participants, device):
        if not isinstance(config, SourceObjectiveConfig):
            raise TypeError("source objective requires its explicit configuration")
        self.config = config
        labels = np.asarray(labels)
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or set(labels) != {0, 1}:
            raise ValueError("source objective requires both binary classes")
        axes = []
        for name, values in (("domains", domains), ("participants", participants)):
            if not config.enabled:
                axes.append(np.zeros(len(labels), dtype=np.int64))
                continue
            if values is None:
                raise ValueError(f"source objective requires explicit {name}")
            values = np.asarray(values)
            if values.shape != labels.shape or any(not str(value).strip() for value in values):
                raise ValueError(f"source {name} must be nonempty aligned identities")
            vocabulary, codes = np.unique(values.astype(str), return_inverse=True)
            if len(vocabulary) < 2:
                raise ValueError(f"source objective requires multiple {name}")
            axes.append(codes)
        for label in (0, 1) if config.enabled else ():
            if any(len(np.unique(axis[labels == label])) < 2 for axis in axes):
                raise ValueError(
                    "each class needs cross-domain and cross-participant source support"
                )
        self.domain_count = len(np.unique(axes[0]))
        self.labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        self.domains = torch.as_tensor(axes[0], dtype=torch.long, device=device)
        self.participants = torch.as_tensor(axes[1], dtype=torch.long, device=device)

    def loss(
        self, features: torch.Tensor, rows: torch.Tensor, *, feature_location: str
    ) -> tuple[torch.Tensor, dict]:
        """Return the already weighted auxiliary sum from one actual feature pass."""
        if feature_location != self.config.feature_location:
            raise ValueError("source objective feature location differs from the actual forward")
        if features.ndim < 2 or rows.ndim != 1 or len(rows) != len(features):
            raise ValueError("source features and physical row indices must align")
        if rows.dtype != torch.long or rows.device != features.device:
            raise ValueError("source rows must be int64 on the feature device")
        if features.device != self.labels.device or not features.is_floating_point():
            raise ValueError("source features must be floating point on the source identity device")
        flat = features.float().flatten(start_dim=1)
        if flat.shape[1] == 0:
            raise ValueError("source features require a nonempty feature dimension")
        valid = (
            torch.isfinite(flat).all()
            & torch.isfinite(flat.norm(dim=1)).all()
            & ((rows >= 0) & (rows < len(self.labels))).all()
        )
        message = "source features must be finite and physical row indices must be in range"
        if features.is_cuda:
            # One device assertion, before normalization/dedup, preserves the
            # batched kernel's lack of per-cell host synchronization.
            torch._assert_async(valid, message)
        elif not bool(valid):
            raise ValueError(message)
        z = F.normalize(flat, dim=1)
        labels = self.labels.index_select(0, rows)
        domains = self.domains.index_select(0, rows)
        people = self.participants.index_select(0, rows)
        order = torch.arange(len(rows), device=rows.device)
        duplicates = (rows[:, None] == rows[None, :]) & (order[:, None] > order[None, :])
        unique = ~duplicates.any(dim=1)
        zero = flat.reshape(-1)[:0].sum()
        supcon_loss, mmd_loss = zero, zero
        diagnostics = {
            name: zero.detach()
            for name in (
                "active_anchors",
                "positive_pairs",
                "active_mmd_terms",
                "negative_mmd_terms",
                "supcon_inactive_batches",
                "mmd_inactive_batches",
            )
        }
        inactive = torch.ones((), dtype=torch.bool, device=features.device)
        if self.config.supcon_weight > 0:
            supcon_loss, metrics = self._supcon(z, labels, domains, people, unique)
            inactive = inactive & (metrics["inactive_batches"] > 0)
            diagnostics.update(
                {name: value for name, value in metrics.items() if name != "inactive_batches"}
            )
            diagnostics["supcon_inactive_batches"] = metrics["inactive_batches"]
        if self.config.mmd_weight > 0:
            mmd_loss, metrics = self._conditional_mmd(z, labels, domains, unique)
            inactive = inactive & (metrics["inactive_batches"] > 0)
            diagnostics.update(
                {name: value for name, value in metrics.items() if name != "inactive_batches"}
            )
            diagnostics["mmd_inactive_batches"] = metrics["inactive_batches"]
        value = self.config.supcon_weight * supcon_loss + self.config.mmd_weight * mmd_loss
        return value, {
            "auxiliary_loss": value.detach(),
            "supcon_loss": supcon_loss.detach(),
            "mmd_loss": mmd_loss.detach(),
            "unique_rows": unique.sum(),
            "inactive_batches": inactive.to(torch.float32),
            **diagnostics,
        }

    def _supcon(self, z, labels, domains, people, unique):
        same_class = labels[:, None] == labels[None, :]
        pair = unique[:, None] & unique[None, :]
        positive = (
            pair
            & same_class
            & (domains[:, None] != domains[None, :])
            & (people[:, None] != people[None, :])
        )
        # Same-class observations in an ineligible domain/person are neither
        # cross-domain positives nor artificial negative examples.
        candidates = positive | (pair & ~same_class)
        similarity = z @ z.T / self.config.temperature
        masked = similarity.masked_fill(~candidates, -torch.inf)
        masked = torch.where(candidates.any(dim=1, keepdim=True), masked, torch.zeros_like(masked))
        positive_count = positive.sum(dim=1)
        active = positive_count > 0
        per_anchor = torch.logsumexp(masked, dim=1) - (
            (similarity * positive).sum(dim=1) / positive_count.clamp_min(1)
        )
        value = (per_anchor * active).sum() / active.sum().clamp_min(1)
        return value, {
            "active_anchors": active.sum(),
            "positive_pairs": positive.sum(),
            "inactive_batches": (~active.any()).to(dtype=torch.float32),
        }

    def _conditional_mmd(self, z, labels, domains, unique):
        """Same off-diagonal pairwise MMD, without per-cell host synchronization.

        Each domain pair/class retains its own detached median of strictly
        positive pairwise distances. Padding uses infinity only for sorting;
        inactive cells contribute neither a term nor its denominator.
        A same-batch bandwidth estimate is detached, not statistically independent.
        """
        cells = [
            (first, second, label)
            for first in range(self.domain_count)
            for second in range(first + 1, self.domain_count)
            for label in (0, 1)
        ]
        left = torch.stack([unique & (domains == a) & (labels == c) for a, _, c in cells])
        right = torch.stack([unique & (domains == b) & (labels == c) for _, b, c in cells])
        n, m = left.sum(1), right.sum(1)
        active = (n >= self.config.min_cell_count) & (m >= self.config.min_cell_count)
        indices = torch.triu_indices(len(z), len(z), offset=1, device=z.device)
        distances = torch.pdist(z.detach())
        combined = left | right
        valid = combined[:, indices[0]] & combined[:, indices[1]] & (distances > 0)
        counts = valid.sum(1)
        ordered = distances.expand(len(cells), -1).masked_fill(~valid, torch.inf).sort(dim=1).values
        position = ((counts - 1).clamp_min(0) // 2).unsqueeze(1)
        # A singleton batch has no pdist entries; normal source batches have at
        # least two rows, but the loss interface also supports this inactive case.
        padded = torch.cat((ordered, z.new_ones((len(cells), 1))), dim=1)
        median = padded.gather(1, position).squeeze(1)
        sigma = torch.where(counts > 0, median, torch.ones_like(median))
        squared = torch.cdist(z, z).square()
        kernel = torch.exp(-squared.unsqueeze(0) / (2 * sigma.square()[:, None, None]))
        off_diagonal = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
        ll = left[:, :, None] & left[:, None, :] & off_diagonal
        rr = right[:, :, None] & right[:, None, :] & off_diagonal
        lr = left[:, :, None] & right[:, None, :]
        terms = (
            (kernel * ll).sum((1, 2)) / (n * (n - 1)).clamp_min(1)
            + (kernel * rr).sum((1, 2)) / (m * (m - 1)).clamp_min(1)
            - 2 * (kernel * lr).sum((1, 2)) / (n * m).clamp_min(1)
        )
        value = (terms * active).sum() / active.sum().clamp_min(1)
        return value, {
            "active_mmd_terms": active.sum(),
            "negative_mmd_terms": ((terms.detach() < 0) & active).sum(),
            "inactive_batches": (~active.any()).to(torch.float32),
        }

    def record(self, totals: dict[str, torch.Tensor], updates: int) -> dict:
        if type(updates) is not int or updates < 1:
            raise ValueError("source objective history requires a positive integer update count")
        values = {key: float(value.detach().cpu()) for key, value in totals.items()}
        means = {
            "mean_" + name: values.pop(name, 0.0) / updates
            for name in ("auxiliary_loss", "supcon_loss", "mmd_loss")
        }
        return {**self.config.record(), "updates": updates, **means, **values}
