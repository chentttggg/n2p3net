"""Typed contracts for candidate-membership decision datasets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

CANDIDATE_TASK_CONTRACT_SCHEMA = "n2p3_candidate_task_contract/1"
MEMBERSHIP_DECISION_SCHEMA = "n2p3_membership_decision/1"


def _membership_texts(values, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of exact string identities")
    values = tuple(values)
    if any(not isinstance(value, str) or not value or value != value.strip() for value in values):
        raise ValueError(f"{name} requires exact nonempty string identities")
    return values


def _membership_integers(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer) or values.dtype == bool:
        raise ValueError(f"{name} must be an aligned integer vector")
    result = np.array(values, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MembershipDecision:
    """One complete requested schedule, with physical events retained when EEG is absent.

    Columns identify final choices; stimulus_indices identify the physical flash
    sets. A row/column flash therefore covers several choice columns. Evidence
    indices map scheduled events to their original epoch rows; -1 means missing.
    This contract contains no target labels or learned candidate preferences.
    """

    candidate_vocabulary: tuple[str, ...]
    membership: np.ndarray
    event_ids: tuple[str, ...]
    stimulus_vocabulary: tuple[str, ...]
    stimulus_indices: np.ndarray
    repetition_indices: np.ndarray
    evidence_indices: np.ndarray
    repetitions: int
    protocol: str = "set_membership"
    timeline_complete: bool = True

    def __post_init__(self):
        for field in ("candidate_vocabulary", "stimulus_vocabulary", "event_ids"):
            values = _membership_texts(getattr(self, field), field)
            minimum = 2 if field == "candidate_vocabulary" else 1
            if len(values) < minimum or len(set(values)) != len(values):
                raise ValueError(f"{field} must contain unique physical identities")
            object.__setattr__(self, field, values)
        if (isinstance(self.repetitions, bool) or not isinstance(self.repetitions, Integral)
                or self.repetitions < 1):
            raise ValueError("repetitions must be a positive integer")
        object.__setattr__(self, "repetitions", int(self.repetitions))
        if not isinstance(self.timeline_complete, (bool, np.bool_)):
            raise ValueError("timeline_complete must be boolean")
        object.__setattr__(self, "timeline_complete", bool(self.timeline_complete))
        if self.protocol not in {"direct", "row_column", "set_membership"}:
            raise ValueError("unsupported membership protocol")
        rows = len(self.event_ids)
        matrix = np.asarray(self.membership)
        if matrix.dtype != bool or matrix.shape != (rows, len(self.candidate_vocabulary)):
            raise ValueError("membership must be a boolean event-by-candidate matrix")
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "membership", matrix)
        for field in ("stimulus_indices", "repetition_indices", "evidence_indices"):
            values = _membership_integers(getattr(self, field), field)
            if len(values) != rows:
                raise ValueError(f"{field} must align with the physical event ledger")
            object.__setattr__(self, field, values)
        if (np.any(self.stimulus_indices < 0)
                or np.any(self.stimulus_indices >= len(self.stimulus_vocabulary))
                or np.any(self.repetition_indices < 0)
                or np.any(self.repetition_indices >= self.repetitions)
                or np.any(self.evidence_indices < -1)):
            raise ValueError("membership indices lie outside their declared axes")
        observed = self.evidence_indices[self.evidence_indices >= 0]
        if len(np.unique(observed)) != len(observed):
            raise ValueError("one physical evidence row cannot stand for multiple scheduled events")
        slots = np.column_stack((self.repetition_indices, self.stimulus_indices))
        if len(np.unique(slots, axis=0)) != rows:
            raise ValueError("duplicate scheduled stimulus within one repetition")
        for stimulus in np.unique(self.stimulus_indices):
            patterns = matrix[self.stimulus_indices == stimulus]
            if not np.all(patterns == patterns[0]):
                raise ValueError("a stimulus identity has inconsistent physical membership")

    @property
    def schedule_complete(self) -> bool:
        return self.timeline_complete and len(self.event_ids) == self.repetitions * len(self.stimulus_vocabulary)

    def prefix(self, repetitions: int) -> MembershipDecision:
        if (isinstance(repetitions, bool) or not isinstance(repetitions, Integral)
                or not 1 <= repetitions <= self.repetitions):
            raise ValueError("prefix repetitions must lie inside the declared schedule")
        selected = self.repetition_indices < repetitions
        return replace(self, membership=self.membership[selected],
            event_ids=tuple(np.asarray(self.event_ids)[selected]),
            stimulus_indices=self.stimulus_indices[selected],
            repetition_indices=self.repetition_indices[selected],
            evidence_indices=self.evidence_indices[selected], repetitions=int(repetitions))

    def validate_epoch_labels(self, labels, target_candidate: int) -> None:
        """Check retained epoch labels against the authorized decision truth.

        ``labels`` uses the same original epoch row axis as evidence_indices.
        Inference never calls this training-only check.
        """
        if (isinstance(target_candidate, bool) or not isinstance(target_candidate, Integral)
                or not 0 <= target_candidate < len(self.candidate_vocabulary)):
            raise ValueError("target candidate must index the frozen choice vocabulary")
        labels = np.asarray(labels)
        observed = self.evidence_indices >= 0
        rows = self.evidence_indices[observed]
        if labels.ndim != 1 or (len(rows) and rows.max() >= len(labels)):
            raise ValueError("epoch labels do not cover the original evidence row axis")
        if not np.array_equal(labels[rows], self.membership[observed, target_candidate]):
            raise ValueError("epoch labels disagree with physical candidate membership")

    def record(self):
        return {"schema": MEMBERSHIP_DECISION_SCHEMA, "protocol": self.protocol,
                "candidate_vocabulary": list(self.candidate_vocabulary),
                "membership": self.membership.tolist(), "event_ids": list(self.event_ids),
                "stimulus_vocabulary": list(self.stimulus_vocabulary),
                "stimulus_indices": self.stimulus_indices.tolist(),
                "repetition_indices": self.repetition_indices.tolist(),
                "evidence_indices": self.evidence_indices.tolist(),
                "repetitions": self.repetitions, "timeline_complete": self.timeline_complete}

    @classmethod
    def from_record(cls, record):
        expected = {"schema", *cls.__dataclass_fields__}
        if not isinstance(record, Mapping) or set(record) != expected or record["schema"] != MEMBERSHIP_DECISION_SCHEMA:
            raise ValueError("membership decision record fields or schema differ")
        return cls(**{key: value for key, value in record.items() if key != "schema"})

    @classmethod
    def direct(cls, candidate_ids, vocabulary, *, event_ids, repetition_indices, repetitions,
               evidence_indices=None, timeline_complete=True):
        vocabulary = _membership_texts(vocabulary, "candidate_vocabulary")
        codes = _membership_texts(candidate_ids, "candidate_ids")
        lookup = {value: index for index, value in enumerate(vocabulary)}
        try:
            indices = np.asarray([lookup[value] for value in codes], dtype=np.int64)
        except KeyError as error:
            raise ValueError("candidate identity outside the frozen vocabulary") from error
        return cls(vocabulary, np.eye(len(vocabulary), dtype=bool)[indices], tuple(event_ids),
            vocabulary, indices, repetition_indices,
            np.arange(len(codes)) if evidence_indices is None else evidence_indices,
            repetitions, "direct", timeline_complete)

    @classmethod
    def row_column(cls, stimulus_ids, *, n_rows, n_columns, event_ids,
                   repetition_indices, repetitions, evidence_indices=None, timeline_complete=True):
        if any(isinstance(size, bool) or not isinstance(size, Integral) or size < 2
               for size in (n_rows, n_columns)):
            raise ValueError("row/column dimensions must be integers of at least two")
        indices = _membership_integers(stimulus_ids, "stimulus_ids")
        if np.any(indices < 0) or np.any(indices >= n_rows + n_columns):
            raise ValueError("stimulus identity outside the row/column vocabulary")
        rows = np.repeat(np.arange(n_rows), n_columns)
        columns = np.tile(np.arange(n_columns), n_rows)
        matrix = ((indices[:, None] < n_rows) & (indices[:, None] == rows)
                  | (indices[:, None] >= n_rows) & (indices[:, None] - n_rows == columns))
        return cls(tuple(f"row:{r}|column:{c}" for r, c in zip(rows, columns, strict=True)),
            matrix, tuple(event_ids), tuple(str(index) for index in range(n_rows + n_columns)),
            indices, repetition_indices, np.arange(len(indices)) if evidence_indices is None else evidence_indices,
            repetitions, "row_column", timeline_complete)

    @classmethod
    def from_timeline(cls, timeline, event_mask, *, repetitions, candidate_vocabulary=None,
                      task_contract=None, repetition_start=0):
        """Adapt an explicitly scoped physical schedule without reading target fields.

        The caller owns the decision/block scope. A later block's repetition
        origin must be declared explicitly rather than inferred from surviving EEG.
        """
        selected = np.asarray(event_mask)
        if selected.dtype != bool or selected.shape != (timeline.n_events,) or not selected.any():
            raise ValueError("membership requires a nonempty explicit scheduled-event scope")
        if (isinstance(repetition_start, bool) or not isinstance(repetition_start, Integral)
                or repetition_start < 0):
            raise ValueError("repetition_start must be a nonnegative integer")
        if len(np.unique(np.asarray(timeline.group_ids)[selected])) != 1 or len(
                np.unique(np.asarray(timeline.subject_ids)[selected])) != 1:
            raise ValueError("one membership decision requires one physical group and participant")
        common = {"event_ids": tuple(np.asarray(timeline.event_ids)[selected]),
                  "repetition_indices": np.asarray(timeline.repetition_indices)[selected] - repetition_start,
                  "evidence_indices": np.asarray(timeline.evidence_indices)[selected],
                  "repetitions": repetitions, "timeline_complete": timeline.complete}
        codes = tuple(np.asarray(timeline.candidate_ids)[selected])
        if task_contract is None:
            if candidate_vocabulary is None:
                raise ValueError("direct membership requires the frozen candidate vocabulary")
            return cls.direct(codes, candidate_vocabulary, **common)
        if not isinstance(task_contract, CandidateTaskContract) or candidate_vocabulary is not None:
            raise ValueError("row/column membership requires one explicit task contract")
        lookup = {str(code): code for code in task_contract.candidate_ids}
        try:
            indices = np.asarray([lookup[code] for code in codes], dtype=np.int64)
        except KeyError as error:
            raise ValueError("scheduled stimulus differs from its row/column task contract") from error
        return cls.row_column(indices, n_rows=task_contract.n_rows,
                              n_columns=task_contract.n_columns, **common)


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
