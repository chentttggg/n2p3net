"""Typed original candidate recordings consumed by the shared prefix issuer.

Dataset decoders own byte admission, physical clocks and private truth. This
interface does not authorize data; the registered decoder's original factory
supplies it to CalibrationPrefixCompiler, which grants only the requested C.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from data.candidate_task import MembershipDecision
from research.contracts import semantic_sha256


@dataclass(frozen=True)
class OriginalCandidateSelection:
    selection_id: str
    event_indices: tuple[int, ...]
    decision: MembershipDecision
    parent_target_candidate: str | None

    def public_record(self):
        return {"selection_id": self.selection_id, "event_indices": list(self.event_indices),
                "decision": self.decision.record()}


@dataclass(frozen=True)
class OriginalCandidateRecording:
    record_id: str
    source_sha256: str
    source_relative_path: str
    participant_key: str
    source_subject_id: str
    session_id: str
    run_id: str
    sample_rate_hz: float
    channel_names: tuple[str, ...]
    reference: str
    native_sample_count: int | None
    native_onsets: np.ndarray
    authorized_binary_labels: np.ndarray
    selections: tuple[OriginalCandidateSelection, ...]
    source_receipt: dict
    read_native: Callable[[int, int], np.ndarray]
    unavailable_public_onsets: np.ndarray | None = None

    def __post_init__(self):
        from transfer.noise_model import require_sha256

        require_sha256(self.source_sha256, "original candidate acquisition")
        if self.native_sample_count is not None and (type(self.native_sample_count) is not int or self.native_sample_count <= 0):
            raise ValueError("original acquisition sample count requires a positive Python integer")
        onsets, labels = np.asarray(self.native_onsets), np.asarray(self.authorized_binary_labels)
        if (onsets.ndim != 1 or onsets.dtype.kind not in "iu" or labels.shape != onsets.shape
                or labels.dtype != bool or not len(onsets)
                or any(s.parent_target_candidate is not None and s.parent_target_candidate not in s.decision.candidate_vocabulary
                       for s in self.selections)
                or sorted(i for s in self.selections for i in s.event_indices) != list(range(len(onsets)))):
            raise ValueError("original recording schedule/clock/binary labels do not share one event axis")
        for selection in self.selections:
            indices = np.asarray(selection.event_indices, dtype=np.int64)
            q = (None if selection.parent_target_candidate is None else
                 selection.decision.candidate_vocabulary.index(selection.parent_target_candidate))
            if (len(indices) != len(selection.decision.event_ids)
                    or q is not None and not np.array_equal(labels[indices], selection.decision.membership[:, q])):
                raise ValueError("original recording binary labels differ from parent truth/set membership")
        for key, value in (("native_onsets", onsets), ("authorized_binary_labels", labels)):
            copy = np.ascontiguousarray(value)
            object.__setattr__(self, key, np.frombuffer(copy.tobytes(), dtype=copy.dtype).reshape(copy.shape))
        from transfer.noise_model import freeze_metadata

        object.__setattr__(self, "source_receipt", freeze_metadata(self.source_receipt))
        if self.unavailable_public_onsets is not None:
            public = np.asarray(self.unavailable_public_onsets, dtype=np.int64)
            if public.shape != onsets.shape or np.any(public < 0) or np.any(np.diff(public) <= 0):
                raise ValueError("unavailable physical schedule clocks must retain actual ordered original markers")
            object.__setattr__(self, "unavailable_public_onsets", np.frombuffer(public.tobytes(), dtype=np.int64))

    def public_record(self):
        return {"record_id": self.record_id, "source_sha256": self.source_sha256,
            "source_relative_path": self.source_relative_path, "participant_key": self.participant_key,
            "source_subject_id": self.source_subject_id, "session_id": self.session_id, "run_id": self.run_id,
            "sample_rate_hz": self.sample_rate_hz, "channel_names": list(self.channel_names),
            "reference": self.reference, "native_sample_count": self.native_sample_count,
            "native_onsets": self.native_onsets.tolist(),
            "unavailable_public_onsets": None if self.unavailable_public_onsets is None else self.unavailable_public_onsets.tolist(),
            "selections": [s.public_record() for s in self.selections],
            "source_receipt": json.loads(json.dumps(self.source_receipt, allow_nan=False))}


@dataclass(frozen=True)
class CandidateDatasetAdapter:
    dataset_id: str
    participant_key: str
    source_subject_id: str
    source_format: str
    recordings: tuple[OriginalCandidateRecording, ...]
    source_artifacts: tuple[dict, ...]
    candidate_task_contract: dict
    endpoint: str
    original_factory_source_sha256: str
    requested_future_record_ids: tuple[str, ...] | None = None

    def __post_init__(self):
        from transfer.noise_model import freeze_metadata

        if (not self.recordings or len({r.record_id for r in self.recordings}) != len(self.recordings)
                or any(r.participant_key != self.participant_key or r.source_subject_id != self.source_subject_id
                       for r in self.recordings)):
            raise ValueError("candidate dataset adapter requires its original unique participant recording roster")
        object.__setattr__(self, "recordings", tuple(self.recordings))
        object.__setattr__(self, "source_artifacts", freeze_metadata(self.source_artifacts))
        object.__setattr__(self, "candidate_task_contract", freeze_metadata(self.candidate_task_contract))
        if self.requested_future_record_ids is not None:
            names = tuple(self.requested_future_record_ids)
            if not names or len(set(names)) != len(names) or not set(names) <= {r.record_id for r in self.recordings}:
                raise ValueError("future record roster must explicitly retain unique original acquisition identities")
            object.__setattr__(self, "requested_future_record_ids", names)

    def record(self):
        return {"schema": "n2p3_original_candidate_dataset_adapter/1", "dataset_id": self.dataset_id,
            "participant_key": self.participant_key, "source_subject_id": self.source_subject_id,
            "source_format": self.source_format, "recordings": [r.public_record() for r in self.recordings],
            "source_artifacts": json.loads(json.dumps(self.source_artifacts, allow_nan=False)),
            "candidate_task_contract": json.loads(json.dumps(self.candidate_task_contract, allow_nan=False)),
            "endpoint": self.endpoint, "original_factory_source_sha256": self.original_factory_source_sha256,
            "requested_future_record_ids": None if self.requested_future_record_ids is None else list(self.requested_future_record_ids)}

    @property
    def digest(self):
        return semantic_sha256(self.record())


def resolve_candidate_dataset_adapter(declaration):
    """One registered original-data decoder dispatch, reused by replay/parent."""
    if declaration["dataset_id"] == "Huebner2017" and declaration["source_format"] == "original_vmrk_plus_NEMAR_BDF":
        from data.huebner2017_candidate import huebner_dataset_adapter

        return huebner_dataset_adapter(declaration["record_sources"], evaluation_session_ids=declaration.get("evaluation_session_ids"))
    if declaration["dataset_id"] == "BI2014a" and declaration["source_format"] in {
            "original_mat", "original_csv", "original_mat_zip", "original_csv_zip"}:
        from data.bi2014a_recording import bi2014a_dataset_adapter

        return bi2014a_dataset_adapter(declaration["record_sources"], source_format=declaration["source_format"])
    raise ValueError("candidate source declaration has no registered original decoder/format authority")
