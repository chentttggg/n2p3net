"""Typed, dimensionless Euclidean-alignment feature artifacts.

An EA transform changes the coordinate system.  The result therefore cannot
be represented as an :class:`EpochDataset`: electrode positions, voltage QC,
and channel masks would acquire a false physical meaning.  This module keeps
the numerical state and its fit authorization next to the transformed feature
array and provides a strict NPZ plus JSON attestation format.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import numpy as np

from data.domain import DOMAIN_ADAPTER_SCHEMA, NATIVE_REFERENCE_SCHEMA, source_domain_ids
from data.epochs import (
    EpochDataset,
    assert_attested_epoch_dataset,
    epoch_row_identity_digest,
    materialize_dataset_identity,
)
from data.euclidean_alignment import (
    EuclideanAlignmentConfig,
    EuclideanAlignmentState,
    fit_euclidean_alignment,
    transform_euclidean_alignment,
)

EA_FEATURE_ARTIFACT_SCHEMA = "n2p3_ea_feature_artifact/2"
EA_FEATURE_RECORD_SCHEMA = "n2p3_ea_feature_attestation/2"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")).hexdigest()


def _array_digest(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                             sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _physical_row_ids(dataset: EpochDataset) -> tuple[str, ...]:
    timeline = dataset.event_timeline.validate(n_epochs=dataset.n_epochs)
    evidence = np.asarray(timeline.evidence_indices, dtype=np.int64)
    event_rows = np.flatnonzero(evidence >= 0)
    if not np.array_equal(np.sort(evidence[event_rows]), np.arange(dataset.n_epochs)):
        raise ValueError("EA authorization requires exactly one physical event per epoch row")
    inverse = np.empty(dataset.n_epochs, dtype=np.int64)
    inverse[evidence[event_rows]] = event_rows
    subjects = np.asarray(dataset.subject_ids).astype(str)
    event_ids = np.asarray(timeline.event_ids).astype(str)[inverse]
    values = tuple(f"{subject}\x1f{event_id}" for subject, event_id in zip(subjects, event_ids, strict=True))
    if len(set(values)) != len(values):
        raise ValueError("EA authorization requires unique physical row identities")
    return values


def physical_row_ids(dataset: EpochDataset) -> tuple[str, ...]:
    """Return stable subject/event identities in the dataset row order."""
    return _physical_row_ids(dataset)


@dataclass(frozen=True)
class EAFitAuthorization:
    """The rows and identities that were legally available to fit EA."""

    parent_cache_sha256: str
    parent_dataset_sha256: str
    parent_row_identity_sha256: str
    fit_scope: str
    authorized_row_ids: tuple[str, ...]
    authorized_participant_keys: tuple[str, ...]
    fit_rows_sha256: str

    def __post_init__(self) -> None:
        for field in ("parent_cache_sha256", "parent_dataset_sha256", "parent_row_identity_sha256", "fit_rows_sha256"):
            _sha(getattr(self, field), field)
        if self.fit_scope not in {"source_rows", "target_calibration", "unlabelled_buffer", "per_domain"}:
            raise ValueError("unsupported EA fit scope")
        rows = tuple(str(value) for value in self.authorized_row_ids)
        people = tuple(str(value) for value in self.authorized_participant_keys)
        if not rows or any(not value or "\x00" in value for value in rows) or len(set(rows)) != len(rows):
            raise ValueError("authorized_row_ids must be unique nonempty physical identities")
        if not people or any(not value or "\x00" in value for value in people) or len(set(people)) != len(people):
            raise ValueError("authorized_participant_keys must be unique nonempty identities")
        object.__setattr__(self, "authorized_row_ids", rows)
        object.__setattr__(self, "authorized_participant_keys", people)

    def record(self) -> dict[str, object]:
        semantic = {"schema": "n2p3_ea_fit_authorization/1", **{
            field.name: (list(getattr(self, field.name)) if isinstance(getattr(self, field.name), tuple)
                         else getattr(self, field.name)) for field in fields(self)
        }, "authorized_row_count": len(self.authorized_row_ids)}
        return {**semantic, "authorization_digest": _digest(semantic)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> EAFitAuthorization:
        if not isinstance(record, Mapping) or record.get("schema") != "n2p3_ea_fit_authorization/1":
            raise ValueError("unsupported EA fit authorization schema")
        expected = {field.name for field in fields(cls)} | {"schema", "authorized_row_count", "authorization_digest"}
        if set(record) != expected:
            raise ValueError("EA fit authorization fields are incomplete")
        semantic = {key: record[key] for key in expected if key != "authorization_digest"}
        if _digest(semantic) != record["authorization_digest"]:
            raise ValueError("EA fit authorization digest mismatch")
        if record["authorized_row_count"] != len(record["authorized_row_ids"]):
            raise ValueError("EA authorization row count mismatch")
        return cls(
            parent_cache_sha256=str(record["parent_cache_sha256"]),
            parent_dataset_sha256=str(record["parent_dataset_sha256"]),
            parent_row_identity_sha256=str(record["parent_row_identity_sha256"]),
            fit_scope=str(record["fit_scope"]),
            authorized_row_ids=tuple(str(value) for value in record["authorized_row_ids"]),
            authorized_participant_keys=tuple(str(value) for value in record["authorized_participant_keys"]),
            fit_rows_sha256=str(record["fit_rows_sha256"]),
        )


def authorization_from_dataset(
    dataset: EpochDataset,
    *,
    fit_scope: str,
    fit_rows: Sequence[int],
) -> EAFitAuthorization:
    """Called by the trusted split/plan owner, never inferred by the EA fitter.

    This records the caller's authorization, not a signature or an independent
    proof that a scientific plan is appropriate. Consumers require the caller's
    expected record independently of the record supplied by an artifact.
    """
    attestation = assert_attested_epoch_dataset(dataset)
    if not bool(np.asarray(dataset.channel_mask, dtype=bool).all()):
        raise ValueError("EA requires every static parent channel to be observed")
    if dataset.trial_channel_mask is not None and not bool(np.asarray(dataset.trial_channel_mask, dtype=bool).all()):
        raise ValueError("EA requires every dynamic parent channel mask entry to be observed")
    row_ids = _physical_row_ids(dataset)
    indices = np.asarray(fit_rows)
    if not np.issubdtype(indices.dtype, np.integer) or indices.dtype.kind == "b":
        raise ValueError("fit_rows must contain integer parent row indices")
    indices = indices.astype(np.int64)
    if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= dataset.n_epochs):
        raise ValueError("fit_rows must be a nonempty vector of valid parent rows")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("fit_rows must not repeat physical rows")
    identity = materialize_dataset_identity(dataset)
    participant_keys = tuple(sorted({identity.record_for(str(dataset.subject_ids[index])).authority_key()
                                      for index in indices}))
    return EAFitAuthorization(
        parent_cache_sha256=attestation["sha256"],
        parent_dataset_sha256=attestation["decoded_dataset_sha256"],
        parent_row_identity_sha256=epoch_row_identity_digest(dataset),
        fit_scope=fit_scope,
        authorized_row_ids=tuple(row_ids[index] for index in indices),
        authorized_participant_keys=participant_keys,
        fit_rows_sha256=_array_digest(indices),
    )


@dataclass(frozen=True)
class EAFitPlan:
    """The caller's fixed scientific authorization and numerical operator recipe.

    A plan is a constraint supplied by a trusted experiment/split owner. Its
    digest identifies bytes/semantics, never certifies who authorized the plan.
    """

    plan_id: str
    authorization: EAFitAuthorization
    reference_mode: str
    config: EuclideanAlignmentConfig

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, str) or not self.plan_id.strip():
            raise ValueError("EA fit plan requires a named experiment/split identity")
        if not isinstance(self.authorization, EAFitAuthorization):
            raise TypeError("EA fit plan requires validated row authorization")
        if self.reference_mode not in {"native_preserve", "common_average"}:
            raise ValueError("unsupported EA fit plan reference mode")
        if not isinstance(self.config, EuclideanAlignmentConfig):
            raise TypeError("EA fit plan requires validated numerical configuration")

    def record(self) -> dict[str, object]:
        return {"schema": "n2p3_ea_fit_plan/1", "plan_id": self.plan_id,
                "authorization": self.authorization.record(), "reference_mode": self.reference_mode,
                "config": asdict(self.config)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> EAFitPlan:
        required = {"schema", "plan_id", "authorization", "reference_mode", "config"}
        if (not isinstance(record, Mapping) or set(record) != required
                or record["schema"] != "n2p3_ea_fit_plan/1"):
            raise ValueError("EA requires a complete current fit plan")
        return cls(plan_id=record["plan_id"],
                   authorization=EAFitAuthorization.from_record(record["authorization"]),
                   reference_mode=record["reference_mode"],
                   config=EuclideanAlignmentConfig(**record["config"]))


@dataclass(frozen=True)
class EAFeatureArtifact:
    """Dimensionless transformed epochs with no physical electrode geometry."""

    name: str
    X: np.ndarray
    y: np.ndarray | None
    subject_ids: np.ndarray
    row_ids: tuple[str, ...]
    component_names: tuple[str, ...]
    component_mask: np.ndarray
    preprocessing: Mapping[str, Any]
    parent_channel_names: tuple[str, ...]
    source_reference: str
    fit_plan: EAFitPlan
    alignment: Mapping[str, Any]
    row_domain_ids: tuple[str, ...]
    _verified_digest: str | None = dataclass_field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._validate_structure(copy_arrays=True)

    def _validate_structure(self, *, copy_arrays: bool = False) -> None:
        if not isinstance(self.fit_plan, EAFitPlan):
            raise TypeError("EA artifact requires a complete EAFitPlan")
        raw_values, raw_labels = np.asarray(self.X), None if self.y is None else np.asarray(self.y)
        if not np.issubdtype(raw_values.dtype, np.floating):
            raise ValueError("EA features must use a floating dtype")
        if raw_labels is not None and not np.issubdtype(raw_labels.dtype, np.integer):
            raise ValueError("EA labels must use an integer dtype when present")
        if np.asarray(self.component_mask).dtype != np.dtype(bool):
            raise ValueError("EA component masks must use boolean dtype")
        array = np.array if copy_arrays else np.asarray
        # Own constructor inputs before freezing them. Revalidation does not copy
        # already-owned arrays or mutate the caller's parent dataset.
        values = array(self.X, dtype=np.float32)
        labels = None if self.y is None else array(self.y, dtype=np.int64)
        subjects = array(self.subject_ids, dtype=str)
        mask = array(self.component_mask, dtype=bool)
        if values.ndim != 3 or min(values.shape) < 1 or not np.isfinite(values).all():
            raise ValueError("EA features must be finite with shape (N,C,T)")
        n_rows, n_components, _ = values.shape
        if (labels is not None and labels.shape != (n_rows,)) or subjects.shape != (n_rows,):
            raise ValueError("EA labels and subject_ids must align with feature rows")
        if mask.shape != (n_rows, n_components) or not bool(mask.all()):
            raise ValueError("EA component masks must be complete after parent mask validation")
        names = tuple(str(name) for name in self.component_names)
        expected_names = tuple(f"ea_component_{index:03d}" for index in range(n_components))
        if names != expected_names:
            raise ValueError("EA component names must be canonical and ordered")
        rows = tuple(str(value) for value in self.row_ids)
        if len(rows) != n_rows or len(set(rows)) != len(rows):
            raise ValueError("EA feature row identities must be unique and align with features")
        if not set(self.authorization.authorized_row_ids) <= set(rows):
            raise ValueError("EA feature rows omit an authorized physical fit row")
        if not isinstance(self.preprocessing, Mapping) or not self.preprocessing:
            raise ValueError("EA artifact requires the parent preprocessing declaration")
        if not self.parent_channel_names or len(self.parent_channel_names) != n_components:
            raise ValueError("EA parent channel order must match the component dimension")
        if not isinstance(self.source_reference, str) or not self.source_reference.strip():
            raise ValueError("EA artifact requires source reference provenance")
        alignment = copy.deepcopy(dict(self.alignment)) if copy_arrays else dict(self.alignment)
        required = {"schema"}
        per_domain = self.authorization.fit_scope == "per_domain"
        required.add("states" if per_domain else "state")
        if set(alignment) != required or alignment["schema"] != "n2p3_euclidean_alignment_fit/3":
            raise ValueError("EA artifact requires the complete current fit contract")
        domains = tuple(self.row_domain_ids)
        if len(domains) != n_rows or any(not isinstance(d, str) or not d for d in domains):
            raise ValueError("EA row domain routing must cover every output row")
        records = alignment["states"] if per_domain else {"__global__": alignment["state"]}
        if set(records) != set(domains):
            raise ValueError("EA states must cover exactly all output domains")
        for domain, record in records.items():
            state = EuclideanAlignmentState.from_record(record)
            if (state.fit_scope != self.authorization.fit_scope
                    or state.domain_id != (domain if per_domain else None)
                    or state.inverse_sqrt_reference.shape != (n_components, n_components)):
                raise ValueError("EA state scope/domain/dimension disagrees with output")
        object.__setattr__(self, "X", np.ascontiguousarray(values))
        object.__setattr__(self, "y", None if labels is None else np.ascontiguousarray(labels))
        object.__setattr__(self, "subject_ids", np.ascontiguousarray(subjects))
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "component_names", names)
        object.__setattr__(self, "component_mask", np.ascontiguousarray(mask))
        object.__setattr__(self, "parent_channel_names", tuple(str(value) for value in self.parent_channel_names))
        object.__setattr__(self, "source_reference", self.source_reference.strip())
        object.__setattr__(self, "alignment", alignment)
        if copy_arrays:
            object.__setattr__(self, "preprocessing", copy.deepcopy(dict(self.preprocessing)))
        object.__setattr__(self, "row_domain_ids", domains)
        for value in (self.X, self.y, self.subject_ids, self.component_mask):
            if value is not None:
                value.setflags(write=False)

    @property
    def authorization(self) -> EAFitAuthorization:
        """The authorization is owned solely by the fixed fit plan."""
        return self.fit_plan.authorization

    @property
    def n_epochs(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.X.shape[1])

    @property
    def coordinate_system(self) -> str:
        return "dimensionless_euclidean_components"

    def semantic_record(self) -> dict[str, object]:
        return {
            "schema": EA_FEATURE_ARTIFACT_SCHEMA,
            "name": self.name,
            "X_sha256": _array_digest(self.X),
            "y_sha256": None if self.y is None else _array_digest(self.y),
            "subject_ids_sha256": _array_digest(self.subject_ids),
            "row_ids": list(self.row_ids),
            "component_names": list(self.component_names),
            "component_mask_sha256": _array_digest(self.component_mask),
            "preprocessing": dict(self.preprocessing),
            "parent_channel_names": list(self.parent_channel_names),
            "source_reference": self.source_reference,
            "fit_plan": self.fit_plan.record(),
            "alignment": self.alignment,
            "row_domain_ids": list(self.row_domain_ids),
            "coordinate_system": self.coordinate_system,
            "output_unit": "dimensionless",
        }

    def record(self, *, cache_sha256: str | None = None, byte_size: int | None = None) -> dict[str, object]:
        semantic = self.semantic_record()
        record = {**semantic, "artifact_digest": _digest(semantic)}
        if cache_sha256 is not None:
            record["cache_attestation"] = {
                "schema": EA_FEATURE_RECORD_SCHEMA,
                "sha256": _sha(cache_sha256, "cache_sha256"),
                "byte_size": int(byte_size),
            }
        return record


    def transform(self, dataset: EpochDataset) -> np.ndarray:
        """Apply the validated frozen state to attested inputs without fitting."""
        if self._verified_digest is None or self._verified_digest != _digest(self.semantic_record()):
            raise ValueError("EA prediction requires an unchanged, parent-verified artifact")
        assert_attested_epoch_dataset(dataset)
        _assert_parent_geometry(dataset, self)
        return _transform(dataset, self.fit_plan, self.alignment)


def _record_path(path: Path) -> Path:
    return path.with_suffix(".record.json")


def assert_reference_mode(dataset: EpochDataset, declared: str) -> None:
    """EA preserves and verifies the already applied physical reference."""
    adapter = dataset.provenance.get("domain_adapter")
    reference = str(dataset.provenance.get("source_reference", "")).strip()
    if declared == "common_average":
        if (not isinstance(adapter, Mapping) or adapter.get("schema") != DOMAIN_ADAPTER_SCHEMA
                or not reference.startswith("common_average[")):
            raise ValueError("common_average requires a provenance-verified common average cache")
    elif declared == "native_preserve":
        if adapter is not None:
            if (not isinstance(adapter, Mapping) or adapter.get("schema") != NATIVE_REFERENCE_SCHEMA
                    or not reference.startswith("native:")):
                raise ValueError("native_preserve disagrees with the parent's reference adapter")
        elif not reference or reference == "unspecified" or reference.startswith("common_average["):
            raise ValueError("native_preserve requires an explicit non-CAR source reference")
    else:
        raise ValueError("unsupported EA reference_mode")


def _assert_observed(dataset: EpochDataset) -> None:
    if not np.asarray(dataset.channel_mask).all() or (
        dataset.trial_channel_mask is not None and not np.asarray(dataset.trial_channel_mask).all()
    ):
        raise ValueError("EA requires complete observed static and dynamic channel masks")


def _authorized_indices(dataset: EpochDataset, expected: EAFitAuthorization) -> np.ndarray:
    if not isinstance(expected, EAFitAuthorization):
        raise TypeError("caller must provide an expected EAFitAuthorization from its plan")
    attestation = assert_attested_epoch_dataset(dataset)
    if expected.parent_cache_sha256 != attestation["sha256"]:
        raise ValueError("EA artifact parent cache attestation differs")
    if expected.parent_dataset_sha256 != attestation["decoded_dataset_sha256"]:
        raise ValueError("EA artifact parent decoded dataset differs")
    row_ids = physical_row_ids(dataset)
    index = {row: i for i, row in enumerate(row_ids)}
    if not set(expected.authorized_row_ids) <= set(index):
        raise ValueError("EA authorized physical fit rows are absent from actual parent")
    rows = np.asarray([index[row] for row in expected.authorized_row_ids], dtype=np.int64)
    observed = authorization_from_dataset(dataset, fit_scope=expected.fit_scope, fit_rows=rows)
    if observed.record() != expected.record():
        raise ValueError("EA expected authorization disagrees with actual parent fit rows/participants")
    return rows


def _domains(dataset: EpochDataset, scope: str) -> np.ndarray:
    if scope == "per_domain":
        if dataset.provenance.get("source_domain_axis") is None:
            raise ValueError("per_domain EA requires a source_domain_axis provenance")
        return source_domain_ids(dataset).astype(str)
    return np.full(dataset.n_epochs, "__global__")


def _fit_records(dataset: EpochDataset, rows: np.ndarray, scope: str,
                 config: EuclideanAlignmentConfig) -> dict[str, dict]:
    domains = _domains(dataset, scope)
    if set(domains[rows]) != set(domains):
        raise ValueError("EA fit authorization must cover every output domain; missing domain")
    return {
        domain: fit_euclidean_alignment(
            dataset.X[rows[domains[rows] == domain]], fit_scope=scope,
            domain_id=domain if scope == "per_domain" else None, config=config,
        ).record()
        for domain in sorted(set(domains))
    }


def _transform(dataset: EpochDataset, plan: EAFitPlan, alignment: Mapping[str, Any]) -> np.ndarray:
    _assert_observed(dataset)
    domains = _domains(dataset, plan.authorization.fit_scope)
    records = alignment["states"] if plan.authorization.fit_scope == "per_domain" else {
        "__global__": alignment["state"]}
    if not set(domains) <= set(records):
        raise ValueError("EA frozen prediction state lacks an input domain")
    output = np.full(dataset.X.shape, np.nan, dtype=np.float32)
    for domain in sorted(set(domains)):
        mask = domains == domain
        output[mask] = transform_euclidean_alignment(
            dataset.X[mask], EuclideanAlignmentState.from_record(records[domain]))
    if not np.isfinite(output).all():
        raise ValueError("EA transform did not assign finite features to every output row")
    return output


def _assert_parent_geometry(dataset: EpochDataset, artifact: EAFeatureArtifact) -> None:
    _assert_observed(dataset)
    if (tuple(dataset.channel_names) != artifact.parent_channel_names
            or dataset.preprocessing.record() != artifact.preprocessing
            or dataset.provenance.get("source_reference") != artifact.source_reference):
        raise ValueError("EA parent channel/preprocessing/reference geometry differs")
    assert_reference_mode(dataset, artifact.fit_plan.reference_mode)


def fit_ea_feature_artifact(
    parent: EpochDataset, *, expected_plan: EAFitPlan, name: str | None = None,
) -> EAFeatureArtifact:
    """Resolve plan rows, fit states, route domains and transform in one module."""
    if not isinstance(expected_plan, EAFitPlan):
        raise TypeError("EA fitting requires the caller's fixed EAFitPlan")
    auth = expected_plan.authorization
    rows = _authorized_indices(parent, auth)
    assert_reference_mode(parent, expected_plan.reference_mode)
    records = _fit_records(parent, rows, auth.fit_scope, expected_plan.config)
    alignment = {
        "schema": "n2p3_euclidean_alignment_fit/3",
        **({"states": records} if auth.fit_scope == "per_domain"
           else {"state": records["__global__"]}),
    }
    artifact = EAFeatureArtifact(
        name=name or f"{parent.name}-ea-{auth.fit_scope}",
        X=_transform(parent, expected_plan, alignment), y=parent.y, subject_ids=parent.subject_ids,
        row_ids=physical_row_ids(parent),
        component_names=tuple(f"ea_component_{i:03d}" for i in range(parent.n_channels)),
        component_mask=np.ones((parent.n_epochs, parent.n_channels), dtype=bool),
        preprocessing=parent.preprocessing.record(), parent_channel_names=parent.channel_names,
        source_reference=parent.provenance["source_reference"], fit_plan=expected_plan, alignment=alignment,
        row_domain_ids=tuple(_domains(parent, auth.fit_scope)),
    )
    object.__setattr__(artifact, "_verified_digest", _digest(artifact.semantic_record()))
    return artifact


def assert_parent_replay(
    artifact: EAFeatureArtifact, *, parent: EpochDataset,
    expected_plan: EAFitPlan,
) -> None:
    """Replay actual authorized input through fit and transform before admission."""
    artifact._validate_structure()
    if not isinstance(expected_plan, EAFitPlan) or artifact.fit_plan.record() != expected_plan.record():
        raise ValueError("EA artifact fit plan differs from caller's expected plan")
    rows = _authorized_indices(parent, expected_plan.authorization)
    _assert_parent_geometry(parent, artifact)
    if (artifact.row_ids != physical_row_ids(parent)
            or not np.array_equal(artifact.y, parent.y)
            or not np.array_equal(artifact.subject_ids, parent.subject_ids)
            or artifact.row_domain_ids != tuple(_domains(parent, expected_plan.authorization.fit_scope))):
        raise ValueError("EA output rows/labels/subjects/domain routing differ from actual parent")
    expected_records = _fit_records(parent, rows, expected_plan.authorization.fit_scope,
                                   expected_plan.config)
    records = artifact.alignment.get("states", {"__global__": artifact.alignment.get("state")})
    matrix_fields = {"mean_trial_covariance", "reference_covariance", "inverse_sqrt_reference"}
    for domain, expected in expected_records.items():
        record = records[domain]
        for key in expected:
            if key == "state_digest":
                continue
            if key in matrix_fields:
                value = np.asarray(expected[key])
                if not np.allclose(record[key], value, rtol=1e-9,
                                   atol=max(np.abs(value).max(), np.finfo(float).tiny) * 1e-11):
                    raise ValueError(f"EA fit replay disagrees with actual authorized input: {key}")
            elif key == "covariance_condition_number":
                if not np.isclose(record[key], expected[key], rtol=1e-9, atol=0):
                    raise ValueError("EA fit condition number replay differs")
            elif record[key] != expected[key]:
                raise ValueError(f"EA fit replay disagrees with actual authorized input: {key}")
    transformed = _transform(parent, expected_plan, artifact.alignment)
    if not np.array_equal(artifact.X, transformed):
        raise ValueError("EA feature output differs from the actual parent/state transform replay")
    object.__setattr__(artifact, "_verified_digest", _digest(artifact.semantic_record()))


def save_ea_feature_artifact(
    path: str | Path, artifact: EAFeatureArtifact, *, parent: EpochDataset,
    expected_plan: EAFitPlan, compressed: bool = True,
) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("EA feature artifacts must end in .npz")
    assert_parent_replay(artifact, parent=parent, expected_plan=expected_plan)
    payload = {
        "schema": np.asarray(EA_FEATURE_ARTIFACT_SCHEMA), "name": np.asarray(artifact.name),
        "X": artifact.X, "y": np.empty(0, dtype=np.int64) if artifact.y is None else artifact.y,
        "has_labels": np.asarray(artifact.y is not None), "subject_ids": artifact.subject_ids,
        "row_ids": np.asarray(artifact.row_ids, dtype=str),
        "row_domain_ids": np.asarray(artifact.row_domain_ids, dtype=str),
        "component_names": np.asarray(artifact.component_names, dtype=str),
        "component_mask": artifact.component_mask,
        "preprocessing_json": np.asarray(json.dumps(dict(artifact.preprocessing), sort_keys=True)),
        "parent_channel_names": np.asarray(artifact.parent_channel_names, dtype=str),
        "source_reference": np.asarray(artifact.source_reference),
        "fit_plan_json": np.asarray(json.dumps(artifact.fit_plan.record(), sort_keys=True)),
        "alignment_json": np.asarray(json.dumps(artifact.alignment, sort_keys=True)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    (np.savez_compressed if compressed else np.savez)(temporary, **payload)
    temporary.replace(path)
    file_sha, size = _file_digest(path)
    _record_path(path).write_text(json.dumps(artifact.record(cache_sha256=file_sha, byte_size=size),
                                          ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_ea_feature_artifact(
    path: str | Path, *, parent: EpochDataset, expected_plan: EAFitPlan,
) -> EAFeatureArtifact:
    """Load the exact verified bytes and replay against independently supplied plan/parent."""
    path = Path(path)
    record = json.loads(_record_path(path).read_text(encoding="utf-8"))
    attestation = record.get("cache_attestation")
    if not isinstance(attestation, Mapping) or attestation.get("schema") != EA_FEATURE_RECORD_SCHEMA:
        raise ValueError("EA artifact attestation is missing or unsupported")
    # Keep one immutable verified snapshot, spilling large artifacts to disk.
    # Hashing then reopening the source would allow a different archive at decode.
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024**2, mode="w+b") as snapshot:
        digest = hashlib.sha256()
        observed_size = 0
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024**2), b""):
                snapshot.write(block)
                digest.update(block)
                observed_size += len(block)
        if (digest.hexdigest() != attestation.get("sha256")
                or observed_size != attestation.get("byte_size")):
            raise ValueError("EA artifact bytes disagree with its attestation")
        snapshot.seek(0)
        with np.load(snapshot, allow_pickle=False) as archive:
            required = {"schema", "name", "X", "y", "has_labels", "subject_ids", "row_ids", "row_domain_ids",
                        "component_names", "component_mask", "preprocessing_json", "parent_channel_names",
                        "source_reference", "fit_plan_json", "alignment_json"}
            if set(archive.files) != required:
                raise ValueError("EA artifact fields are incomplete or unsupported")
            if str(np.asarray(archive["schema"]).item()) != EA_FEATURE_ARTIFACT_SCHEMA:
                raise ValueError("unsupported EA feature artifact schema; replay old artifacts as historical evidence")
            artifact = EAFeatureArtifact(
                name=str(archive["name"].item()), X=archive["X"],
                y=archive["y"] if bool(archive["has_labels"].item()) else None,
                subject_ids=archive["subject_ids"], row_ids=tuple(archive["row_ids"].astype(str)),
                row_domain_ids=tuple(archive["row_domain_ids"].astype(str)),
                component_names=tuple(archive["component_names"].astype(str)),
                component_mask=archive["component_mask"],
                preprocessing=json.loads(str(archive["preprocessing_json"].item())),
                parent_channel_names=tuple(archive["parent_channel_names"].astype(str)),
                source_reference=str(archive["source_reference"].item()),
                fit_plan=EAFitPlan.from_record(json.loads(str(archive["fit_plan_json"].item()))),
                alignment=json.loads(str(archive["alignment_json"].item())),
            )
    if _digest(artifact.semantic_record()) != record.get("artifact_digest"):
        raise ValueError("EA artifact semantic digest mismatch")
    assert_parent_replay(artifact, parent=parent, expected_plan=expected_plan)
    return artifact


__all__ = [
    "EA_FEATURE_ARTIFACT_SCHEMA", "EA_FEATURE_RECORD_SCHEMA", "EAFitAuthorization", "EAFitPlan", "EAFeatureArtifact",
    "authorization_from_dataset", "physical_row_ids", "fit_ea_feature_artifact", "save_ea_feature_artifact",
    "load_ea_feature_artifact", "assert_parent_replay", "assert_reference_mode",
]
