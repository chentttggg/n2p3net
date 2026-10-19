"""Source-only temporal subspaces for heterogeneous observed EEG montages.

Source ERP contrasts contribute equal normalized temporal Gram matrices per
participant. Only their temporal eigenspace transfers: target spatial filters
and discrimination are fitted on the target's observed electrodes. A fixed
orthonormal cosine basis is the source-free control. Both are frozen before
target labels are consumed and carry distinct roles and numeric digests.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import numpy as np

from data.epochs import (
    PreprocessingSpec,
    loaded_epoch_cache_attestation,
    materialize_dataset_identity,
)
from data.exclusions import assert_origin_keys_allowed
from data.identity import (
    DatasetIdentityTable,
    ParticipantIdentityRecord,
    source_dataset_ids_from_identity,
)
from research.contracts import semantic_sha256

TEMPORAL_BASIS_SCHEMA = "n2p3_temporal_basis/1"


def resolve_temporal_basis(dataset, *, kind, rank, source_basis=None):
    """Resolve predeclared target features without fitting to target labels."""
    if kind == "fixed_cosine":
        if source_basis is not None:
            raise ValueError("fixed temporal control cannot consume a source basis")
        return TemporalBasis.fixed_cosine(dataset.preprocessing, rank=rank)
    if kind != "source_supervised" or not isinstance(source_basis, TemporalBasis) or not source_basis.source_fitted:
        raise ValueError("source temporal route requires its separate source-fitted basis artifact")
    source_basis = TemporalBasis.from_record(source_basis.record())
    if source_basis.matrix.shape[1] != rank:
        raise ValueError("source temporal basis rank differs from the predeclared recipe")
    source_basis.validate_target(dataset)
    return source_basis


def _numeric_sha(values):
    value = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((value.dtype.str, value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _rank(rank, times):
    if type(rank) is not int or not 1 <= rank <= times:
        raise ValueError("temporal rank must be a positive integer within the input time dimension")


def _cosine_basis(n_times, rank):
    times = np.arange(n_times)[:, None] + .5
    basis = np.cos(np.pi * times * np.arange(rank)[None, :] / n_times)
    basis[:, 0] /= np.sqrt(n_times)
    basis[:, 1:] *= np.sqrt(2 / n_times)
    return basis


@dataclass(frozen=True)
class TemporalBasis:
    """Read-only basis with source fit receipts or an analytic provenance role."""

    _record: dict

    @property
    def matrix(self):
        return np.asarray(self._record["basis"], dtype=np.float64)

    @property
    def source_fitted(self):
        return self._record["fit_role"] == "source_supervised"

    def record(self):
        return copy.deepcopy(self._record)

    @classmethod
    def from_record(cls, record):
        record = copy.deepcopy(record)
        if record.get("schema") != TEMPORAL_BASIS_SCHEMA or record.get("fit_role") not in {"source_supervised", "fixed_analytic"}:
            raise ValueError("unsupported temporal basis artifact schema or role")
        if record.get("digest") != semantic_sha256({k: v for k, v in record.items() if k != "digest"}):
            raise ValueError("temporal basis digest mismatch")
        spec = PreprocessingSpec(**record["preprocessing"])
        _rank(record["rank"], spec.n_times)
        basis = np.asarray(record["basis"], dtype=np.float64)
        if basis.shape != (spec.n_times, record["rank"]) or not np.isfinite(basis).all() or not np.allclose(
            basis.T @ basis, np.eye(record["rank"]), rtol=1e-10, atol=1e-10
        ):
            raise ValueError("temporal basis must have finite orthonormal columns")
        receipts = record.get("fit_receipts")
        if not isinstance(receipts, list) or (record["fit_role"] == "source_supervised" and not receipts):
            raise ValueError("source temporal basis requires physical fit receipts")
        if record["fit_role"] == "fixed_analytic" and (receipts or record.get("source_studies")):
            raise ValueError("fixed analytic basis cannot claim source training")
        if record["fit_role"] == "fixed_analytic":
            expected = _cosine_basis(spec.n_times, record["rank"])
            if not np.array_equal(basis, expected):
                raise ValueError("fixed analytic basis must equal the declared cosine formula")
        if record["fit_role"] == "source_supervised":
            participants = []
            studies = set()
            for item in receipts:
                if (not item.get("participant_key") or not item.get("origin_keys")
                    or not item.get("event_ids") or len(set(item["event_ids"])) != len(item["event_ids"])
                    or any(type(item.get(key)) is not str or len(item[key]) != 64
                           for key in ("values_sha256", "labels_sha256", "observed_sha256"))):
                    raise ValueError("source temporal fit receipt is incomplete")
                identity = ParticipantIdentityRecord.from_payload(item["identity"])
                assert_origin_keys_allowed(identity.origin_subject_keys)
                if (identity.authority_key() != item["participant_key"] or
                    list(identity.origin_subject_keys) != item["origin_keys"] or
                    list(identity.global_person_keys) != item["global_keys"]):
                    raise ValueError("temporal receipt participant identity differs")
                studies.update(source_dataset_ids_from_identity(DatasetIdentityTable((identity,))))
                participants.append(item["participant_key"])
            if len(set(participants)) != len(participants):
                raise ValueError("source temporal fit cannot count a participant twice")
            if sorted(studies) != record.get("source_studies"):
                raise ValueError("temporal source study ledger differs from participant origins")
        return cls(record)

    @classmethod
    def _build(cls, *, spec, basis, role, receipts, studies):
        record = {"schema": TEMPORAL_BASIS_SCHEMA, "fit_role": role,
            "preprocessing": spec.record(), "rank": basis.shape[1], "basis": basis.tolist(),
            "fit_receipts": receipts, "source_studies": sorted(studies),
            "risk": "equal_participant_normalized_ERP_contrast" if role == "source_supervised" else "none"}
        return cls.from_record({**record, "digest": semantic_sha256(record)})

    @classmethod
    def fixed_cosine(cls, spec: PreprocessingSpec, *, rank: int):
        _rank(rank, spec.n_times)
        basis = _cosine_basis(spec.n_times, rank)
        return cls._build(spec=spec, basis=basis, role="fixed_analytic", receipts=[], studies=[])

    @classmethod
    def fit_source(cls, scopes, *, rank: int):
        """Fit explicit (dataset, authorized integer rows) scopes, source only.

        Each participant must be supplied in one scope, so recording count
        cannot silently multiply its statistical weight. Channel intersection
        is local to that participant and requires actual observations.
        """
        receipts, matrices, studies, participants = [], [], set(), set()
        spec = None
        for dataset, requested in scopes:
            dataset.validate(require_labels=True)
            if spec is None:
                spec = dataset.preprocessing
                _rank(rank, spec.n_times)
            if dataset.preprocessing != spec:
                raise ValueError("source temporal preprocessing contracts must match")
            rows = np.asarray(requested)
            if (rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows)
                or np.any(rows < 0) or np.any(rows >= dataset.n_epochs) or len(set(rows.tolist())) != len(rows)):
                raise ValueError("temporal fit requires explicit unique in-range integer rows")
            table = materialize_dataset_identity(dataset)
            observed = np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2]).copy()
            if dataset.trial_channel_mask is not None:
                observed &= dataset.trial_channel_mask
            timeline = dataset.event_timeline
            events = np.flatnonzero(timeline.evidence_indices >= 0)
            if not np.array_equal(np.sort(timeline.evidence_indices[events]), np.arange(dataset.n_epochs)):
                raise ValueError("source temporal fit requires one physical event per input row")
            inverse = np.empty(dataset.n_epochs, dtype=int)
            inverse[timeline.evidence_indices[events]] = events
            for subject in np.unique(dataset.subject_ids[rows]):
                selected = rows[dataset.subject_ids[rows] == subject]
                identity = table.record_for(subject)
                key = identity.authority_key()
                if key in participants:
                    raise ValueError("source temporal fit cannot count a participant twice")
                participants.add(key)
                labels = dataset.y[selected]
                if set(np.unique(labels)) != {0, 1} or min(np.bincount(labels.astype(int))) < 2:
                    raise ValueError("each source participant requires two observations per binary class")
                channels = np.flatnonzero(observed[selected].all(axis=0))
                if not len(channels):
                    raise ValueError("source temporal fitting requires observed channels within each participant")
                X = np.asarray(dataset.X[selected][:, channels], dtype=np.float64)
                if not np.isfinite(X).all():
                    raise ValueError("source temporal observations must be finite")
                contrast = X[labels == 1].mean(axis=0) - X[labels == 0].mean(axis=0)
                gram = contrast.T @ contrast
                energy = np.trace(gram)
                if not np.isfinite(energy) or energy <= 0:
                    raise ValueError("source ERP contrast has no identifiable temporal energy")
                matrices.append(gram / energy)
                receipts.append({"participant_key": key, "origin_keys": list(identity.origin_subject_keys),
                    "global_keys": list(identity.global_person_keys), "rows": selected.tolist(),
                    "event_ids": timeline.event_ids[inverse[selected]].tolist(),
                    "observed_channels": [dataset.channel_names[c] for c in channels],
                    "values_sha256": _numeric_sha(X), "labels_sha256": _numeric_sha(labels),
                    "observed_sha256": _numeric_sha(observed[selected]),
                    "parent_cache_attestation": (None if dataset.verified_cache_attestation is None else
                                                  loaded_epoch_cache_attestation(dataset)),
                    "identity": identity.payload()})
                studies.update(source_dataset_ids_from_identity(table.subset([str(subject)])))
        if spec is None or not matrices:
            raise ValueError("source temporal fitting requires nonempty authorized source scopes")
        values, vectors = np.linalg.eigh(np.mean(matrices, axis=0))
        if values[-rank] <= values[-1] * 1e-10:
            raise ValueError("requested temporal rank exceeds identifiable source ERP subspace")
        basis = vectors[:, -rank:][:, ::-1]
        for column in range(rank):
            if basis[np.argmax(np.abs(basis[:, column])), column] < 0:
                basis[:, column] *= -1
        return cls._build(spec=spec, basis=basis, role="source_supervised", receipts=receipts, studies=studies)

    def verify_source(self, scopes):
        """Replay source fitting to bind receipts, actual samples and numeric basis."""
        if not self.source_fitted:
            raise ValueError("fixed temporal control has no source fitting to verify")
        repeated = type(self).fit_source(scopes, rank=self.matrix.shape[1])
        if repeated.record() != self.record():
            raise ValueError("temporal source replay differs in actual fit observations or numeric state")

    def validate_target(self, dataset):
        if dataset.preprocessing != PreprocessingSpec(**self._record["preprocessing"]):
            raise ValueError("temporal basis target preprocessing contract differs")
        table = materialize_dataset_identity(dataset)
        source_keys = {key for row in self._record["fit_receipts"] for key in
                       [row["participant_key"], *row["origin_keys"], *row["global_keys"]]}
        target_keys = {key for row in table.records for key in
                       [row.authority_key(), *row.origin_subject_keys, *row.global_person_keys]}
        if source_keys & target_keys:
            raise ValueError("source temporal basis contains target participant")
        if set(source_dataset_ids_from_identity(table)) & set(self._record["source_studies"]):
            raise ValueError("source temporal basis must exclude the complete target study")

    def project(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3 or X.shape[2] != self.matrix.shape[0] or not np.isfinite(X).all():
            raise ValueError("temporal projection requires finite aligned (N,C,T) input")
        return X @ self.matrix
