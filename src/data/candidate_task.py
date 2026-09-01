"""Typed contracts for candidate-membership decision datasets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

CANDIDATE_TASK_CONTRACT_SCHEMA = "n2p3_candidate_task_contract/1"


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a JSON array of integers.")
    output: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{name} must contain only integers.")
        output.append(int(item))
    return tuple(output)


def _json_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty JSON mapping.")
    output: dict[str, Any] = {}
    for key, item in value.items():
        text = _nonempty_text(key, f"{name} key")
        if item is None or isinstance(item, (str, bool, int)):
            output[text] = item
        elif isinstance(item, float) and math.isfinite(item):
            output[text] = item
        elif isinstance(item, Mapping):
            output[text] = _json_mapping(item, f"{name}.{text}")
        elif isinstance(item, (list, tuple)):
            output[text] = list(item)
        else:
            raise ValueError(f"{name}.{text} is not JSON-compatible.")
    return output


@dataclass(frozen=True)
class CandidateTaskContract:
    """Dataset-declared semantics for one candidate-membership task.

    Candidate IDs are deliberately independent of acquisition flash codes. For
    a row/column task, rows occupy the first contiguous ID range and columns
    the second. This representation also leaves room for future set-valued
    membership tasks whose acquisition codebooks differ by dataset.
    """

    dataset_id: str
    task_id: str
    population: Mapping[str, Any]
    evidence_scope: Mapping[str, Any]
    membership_kind: str
    grid_shape: tuple[int, int]
    candidate_ids: tuple[int, ...]
    row_candidate_ids: tuple[int, ...]
    column_candidate_ids: tuple[int, ...]
    target_representation: str
    raw_target_label_is_target: Mapping[str, bool] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _nonempty_text(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "task_id", _nonempty_text(self.task_id, "task_id"))
        population = _json_mapping(self.population, "population")
        evidence_scope = _json_mapping(self.evidence_scope, "evidence_scope")
        if not isinstance(population.get("label"), str) or not population["label"].strip():
            raise ValueError("population.label must be a non-empty string.")
        if set(evidence_scope) != {"stage", "product_confirmation"}:
            raise ValueError("evidence_scope must contain exactly stage and product_confirmation.")
        _nonempty_text(evidence_scope["stage"], "evidence_scope.stage")
        if not isinstance(evidence_scope["product_confirmation"], bool):
            raise ValueError("evidence_scope.product_confirmation must be boolean.")
        object.__setattr__(self, "population", population)
        object.__setattr__(self, "evidence_scope", evidence_scope)
        if self.membership_kind != "row_column":
            raise ValueError("Only membership_kind='row_column' is currently supported.")
        if (
            len(self.grid_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in self.grid_shape
            )
            or any(value < 2 for value in self.grid_shape)
        ):
            raise ValueError("grid_shape must contain row and column counts of at least two.")
        n_rows, n_columns = self.grid_shape
        expected_rows = tuple(range(n_rows))
        expected_columns = tuple(range(n_rows, n_rows + n_columns))
        if self.row_candidate_ids != expected_rows:
            raise ValueError("row_candidate_ids must be the canonical zero-based row range.")
        if self.column_candidate_ids != expected_columns:
            raise ValueError("column_candidate_ids must immediately follow the row range.")
        if self.candidate_ids != expected_rows + expected_columns:
            raise ValueError("candidate_ids must be the canonical rows-then-columns range.")
        if self.target_representation != "row_column_intersection":
            raise ValueError(
                "target_representation must be 'row_column_intersection' for row/column tasks."
            )
        raw_codebook = self.raw_target_label_is_target
        if raw_codebook is not None:
            if not isinstance(raw_codebook, Mapping) or not raw_codebook:
                raise ValueError("raw_target_label_is_target must be a non-empty mapping.")
            normalized: dict[str, bool] = {}
            for key, value in raw_codebook.items():
                label = _nonempty_text(key, "raw target label")
                if not isinstance(value, bool):
                    raise ValueError("raw target label codebook values must be boolean.")
                normalized[label] = value
            if set(normalized.values()) != {False, True}:
                raise ValueError("raw target label codebook must define both target classes.")
            object.__setattr__(self, "raw_target_label_is_target", normalized)

    @property
    def n_rows(self) -> int:
        return self.grid_shape[0]

    @property
    def n_columns(self) -> int:
        return self.grid_shape[1]

    def record(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "schema": CANDIDATE_TASK_CONTRACT_SCHEMA,
            "dataset_id": self.dataset_id,
            "task_id": self.task_id,
            "population": dict(self.population),
            "evidence_scope": dict(self.evidence_scope),
            "membership_kind": self.membership_kind,
            "grid_shape": list(self.grid_shape),
            "candidate_ids": list(self.candidate_ids),
            "row_candidate_ids": list(self.row_candidate_ids),
            "column_candidate_ids": list(self.column_candidate_ids),
            "target_representation": self.target_representation,
        }
        if self.raw_target_label_is_target is not None:
            output["raw_target_label_is_target"] = dict(self.raw_target_label_is_target)
        return output

    @classmethod
    def from_record(cls, value: object) -> CandidateTaskContract:
        if not isinstance(value, Mapping):
            raise ValueError("candidate_task_contract must be a JSON mapping.")
        required = {
            "schema",
            "dataset_id",
            "task_id",
            "population",
            "evidence_scope",
            "membership_kind",
            "grid_shape",
            "candidate_ids",
            "row_candidate_ids",
            "column_candidate_ids",
            "target_representation",
        }
        optional = {"raw_target_label_is_target"}
        unknown = set(value) - required - optional
        missing = required - set(value)
        if missing or unknown:
            raise ValueError(
                "candidate_task_contract fields disagree with the active schema: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        if value.get("schema") != CANDIDATE_TASK_CONTRACT_SCHEMA:
            raise ValueError(
                f"candidate_task_contract.schema must be {CANDIDATE_TASK_CONTRACT_SCHEMA!r}."
            )
        shape = _integer_tuple(value["grid_shape"], "grid_shape")
        if len(shape) != 2:
            raise ValueError("grid_shape must contain exactly two integers.")
        return cls(
            dataset_id=_nonempty_text(value["dataset_id"], "dataset_id"),
            task_id=_nonempty_text(value["task_id"], "task_id"),
            population=_json_mapping(value["population"], "population"),
            evidence_scope=_json_mapping(value["evidence_scope"], "evidence_scope"),
            membership_kind=_nonempty_text(value["membership_kind"], "membership_kind"),
            grid_shape=(shape[0], shape[1]),
            candidate_ids=_integer_tuple(value["candidate_ids"], "candidate_ids"),
            row_candidate_ids=_integer_tuple(value["row_candidate_ids"], "row_candidate_ids"),
            column_candidate_ids=_integer_tuple(
                value["column_candidate_ids"], "column_candidate_ids"
            ),
            target_representation=_nonempty_text(
                value["target_representation"], "target_representation"
            ),
            raw_target_label_is_target=value.get("raw_target_label_is_target"),
        )


@dataclass(frozen=True)
class CandidateMembershipMetadata:
    candidate_ids: np.ndarray
    row_codes: np.ndarray
    column_codes: np.ndarray
    target_rows: np.ndarray
    target_columns: np.ndarray
    selection_ids: np.ndarray
    repetition_indices: np.ndarray
    is_target: np.ndarray


def candidate_task_contract_from_provenance(
    provenance: Mapping[str, Any],
) -> CandidateTaskContract:
    return CandidateTaskContract.from_record(provenance.get("candidate_task_contract"))


def _integer_column(metadata: pd.DataFrame, field: str) -> np.ndarray:
    raw = metadata[field].to_numpy()
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"candidate metadata {field} must contain integers.") from error
    if (
        numeric.ndim != 1
        or not np.isfinite(numeric).all()
        or not np.equal(numeric, np.floor(numeric)).all()
    ):
        raise ValueError(f"candidate metadata {field} must contain finite integers.")
    return numeric.astype(np.int64)


def _raw_label_key(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        numeric = float(value)
        return str(int(numeric)) if numeric.is_integer() else str(numeric)
    return str(value).strip()


def validate_candidate_membership_metadata(
    metadata: pd.DataFrame,
    contract: CandidateTaskContract,
    *,
    labels: np.ndarray | None = None,
) -> CandidateMembershipMetadata:
    """Validate structural and label semantics without acquisition-code assumptions."""

    required = {
        "candidate_id",
        "row_code",
        "col_code",
        "target_row",
        "target_col",
        "selection_id",
        "repetition_index",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"candidate metadata is missing columns {sorted(missing)}.")
    candidate_ids = _integer_column(metadata, "candidate_id")
    row_codes = _integer_column(metadata, "row_code")
    column_codes = _integer_column(metadata, "col_code")
    target_rows = _integer_column(metadata, "target_row")
    target_columns = _integer_column(metadata, "target_col")
    repetitions = _integer_column(metadata, "repetition_index")
    selections = metadata["selection_id"].astype(str).to_numpy()
    if any(not value.strip() for value in selections):
        raise ValueError("candidate metadata selection_id cannot be empty.")
    if np.any(repetitions < 0):
        raise ValueError("candidate metadata repetition_index must be non-negative.")
    if not np.isin(candidate_ids, contract.candidate_ids).all():
        raise ValueError("candidate_id is outside the declared candidate vocabulary.")
    if np.any((target_rows < 0) | (target_rows >= contract.n_rows)) or np.any(
        (target_columns < 0) | (target_columns >= contract.n_columns)
    ):
        raise ValueError("target row/column is outside the declared grid.")
    row_events = candidate_ids < contract.n_rows
    expected_rows = np.where(row_events, candidate_ids, -1)
    expected_columns = np.where(row_events, -1, candidate_ids - contract.n_rows)
    if not np.array_equal(row_codes, expected_rows) or not np.array_equal(
        column_codes, expected_columns
    ):
        raise ValueError("row_code/col_code disagree with canonical candidate_id semantics.")
    derived = (row_events & (row_codes == target_rows)) | (
        ~row_events & (column_codes == target_columns)
    )
    if labels is not None:
        observed_labels = np.asarray(labels)
        if observed_labels.ndim != 1 or len(observed_labels) != len(metadata):
            raise ValueError("candidate labels must be one-dimensional and metadata-aligned.")
        if not np.array_equal(observed_labels, derived.astype(observed_labels.dtype)):
            raise ValueError("candidate labels disagree with candidate-membership semantics.")
    for field in ("raw_is_target", "is_target"):
        if field not in metadata:
            continue
        raw = metadata[field].to_numpy()
        if (
            not all(isinstance(value, (bool, np.bool_, int, np.integer)) for value in raw)
            or not np.isin(raw, (False, True, 0, 1)).all()
        ):
            raise ValueError(f"{field} must contain only boolean target indicators.")
        if not np.array_equal(raw.astype(bool), derived):
            raise ValueError(f"{field} disagrees with candidate-membership semantics.")
    if "raw_target_label" in metadata:
        codebook = contract.raw_target_label_is_target
        if codebook is None:
            raise ValueError(
                "raw_target_label exists but candidate_task_contract has no label codebook."
            )
        try:
            audited = np.asarray(
                [codebook[_raw_label_key(value)] for value in metadata["raw_target_label"]],
                dtype=bool,
            )
        except KeyError as error:
            raise ValueError(
                f"raw_target_label contains undeclared value {error.args[0]!r}."
            ) from error
        if not np.array_equal(audited, derived):
            raise ValueError("raw_target_label disagrees with candidate-membership semantics.")
    return CandidateMembershipMetadata(
        candidate_ids=candidate_ids,
        row_codes=row_codes,
        column_codes=column_codes,
        target_rows=target_rows,
        target_columns=target_columns,
        selection_ids=selections,
        repetition_indices=repetitions,
        is_target=derived,
    )
