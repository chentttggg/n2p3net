"""Reuse an original C-noise fit only after two actual live C input relations.

Scope/import identities stay distinct. The original strict parent replay is
performed first; the new side then proves exact numerical inputs and full
parameter-law equality. No fitted provenance or raw registry is rewritten.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from data.brainsync_raw_authority import require_calibration_labels, require_raw_capability
from research.conditional_response_graph import string_rows
from research.frozen_metadata import freeze_metadata
from research.target_noise_operand import (
    SEGMENT_LAW,
    geometry_from_capability,
    replay_c_residual_inputs,
)
from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_REUSES = WeakKeyDictionary()


def exact_parameter_law_relation(old_graph, actual_graph):
    """Compare actual named generators; raw/producer metadata are separate."""
    arrays = ("source_mean", "source_factor")
    for name in arrays:
        before, after = getattr(old_graph, name), getattr(actual_graph, name)
        if before.dtype != after.dtype or not np.array_equal(before, after):
            raise ValueError("the new target has another original source parameter law")
    exact_fields = ("source_map", "reference_chart_mean", "conditional_gain", "conditional_covariance")
    if (any(getattr(old_graph, name) != getattr(actual_graph, name) for name in exact_fields)
            or old_graph.coefficient_count != actual_graph.coefficient_count
            or old_graph.chart.exact_matrix != actual_graph.chart.exact_matrix
            or old_graph.chart.exact_inverse != actual_graph.chart.exact_inverse):
        raise ValueError("the new target changed exact common/conditional parameter operators")
    for name in ("function_dictionary_sha256", "function_source_clock_hz", "target_observation_geometry",
                 "target_participant_key", "source_codec_sha256"):
        if metadata_digest(old_graph.binding[name]) != metadata_digest(actual_graph.binding[name]):
            raise ValueError("the new target changed the source function or target physical parameter law")
    atoms = {name: array_digest(getattr(actual_graph, name)) for name in arrays}
    atoms.update({name: metadata_digest([str(value) for value in getattr(actual_graph, name)])
        if name == "reference_chart_mean" else metadata_digest(string_rows(getattr(actual_graph, name)))
        for name in exact_fields})
    return {"old_graph_identity": old_graph.digest(), "actual_graph_identity": actual_graph.digest(),
        "same_exact_parameter_law": True, "generator_atoms": atoms,
        "same_native_chart": metadata_digest(actual_graph.chart.record()),
        "source_and_conditional_latents_shared_by_one_actual_engine": True,
        "observation_permission_and_consumer_metadata_not_used_as_parameter_law": True}


class _Capture:
    def __init__(self, callback):
        self.callback, self.outputs = callback, {}

    def __call__(self, cap, labels):
        result = self.callback(cap, labels)
        self.outputs[cap.identity_sha256] = result
        return result


def _mean_pair(result):
    values, error, binding = result
    values, error = np.asarray(values), np.asarray(error)
    if values.ndim == 3:
        values, error = values[:, :, 0], error[:, :, 0]
    return values, error, binding


def _require_callback_graph(callback, graph, *, side):
    """Bind the law being compared to both actual owners of its C mean."""
    expected = graph.digest()
    if (callback.graph.digest() != expected
            or callback.compiler.graph.digest() != expected):
        raise ValueError(f"{side} fixed C mean is not compiled from its declared actual source graph")
    return expected


def _require_mean_graph_binding(binding, expected, *, side):
    if binding["source_response_model_identity"] != expected:
        raise ValueError(f"{side} actual C mean output belongs to another source graph")


@dataclass(frozen=True, eq=False)
class ReusedCNoise:
    original_operand: object
    programme_admission: object
    record: dict
    identity: str

    @property
    def noise(self):
        return self.original_operand.noise

    @property
    def eta(self):
        return self.original_operand.eta

    def carrier(self):
        return self.original_operand.carrier()

    def segment_binding(self, cap, *, noise_segment_id):
        from research.brainsync_programme_admission import require_actual_programme

        programme = require_actual_programme(self.programme_admission)
        registered = _REUSES.get(self)
        cap = require_raw_capability(cap)
        if registered is None or registered[0] is not self.record or cap not in (
                programme.calibration_caps + programme.held_target.caps):
            raise ValueError("reused C noise requires its actual original C/held programme relation")
        if geometry_from_capability(cap) != self.noise.geometry:
            raise ValueError("reused C-noise numerical coordinates differ from actual observations")
        return {"law": SEGMENT_LAW, "noise_segment_id": noise_segment_id,
                "acquisition_segment_id": cap.public_binding["segment_id"], "owner": self.identity}


def admit_reused_c_noise(original_operand, old_records, new_records, *, old_mean_callback,
                         new_mean_callback, old_graph, actual_graph, programme_admission):
    from research.brainsync_programme_admission import require_actual_programme

    programme = require_actual_programme(programme_admission)
    old_records, new_records = tuple(old_records), tuple(new_records)
    old_graph_identity = _require_callback_graph(old_mean_callback, old_graph, side="original")
    actual_graph_identity = _require_callback_graph(new_mean_callback, actual_graph, side="new")
    captured = _Capture(old_mean_callback)
    original_parent = replay_c_residual_inputs(original_operand, old_records, mean_callback=captured)
    law = exact_parameter_law_relation(old_graph, actual_graph)
    old_caps = tuple(cap for record in old_records for cap in record.raw_capabilities)
    new_caps = tuple(cap for record in new_records for cap in record.raw_capabilities)
    if (new_caps != programme.calibration_caps or len(old_caps) != len(new_caps)
            or tuple(record.labels for record in new_records) != (programme.calibration_labels,)):
        raise ValueError("C-noise reuse must cover the actual complete new programme C input")
    old_labels = tuple(require_calibration_labels(record.labels) for record in old_records)
    new_labels = tuple(require_calibration_labels(record.labels) for record in new_records)
    if len(old_labels) != len(new_labels) or any(a.labels != b.labels for a, b in zip(old_labels, new_labels, strict=True)):
        raise ValueError("new C-noise input changed actual original authorized binary event bits")
    rows = []
    old_noise_records = original_operand.noise.provenance["records"]
    if len(old_noise_records) != len(old_caps):
        raise ValueError("the original fitted-noise record roster differs from its actual parent")
    for old_cap, new_cap, fitted in zip(old_caps, new_caps, old_noise_records, strict=True):
        require_raw_capability(old_cap)
        require_raw_capability(new_cap)
        before, after = old_cap.public_binding, new_cap.public_binding
        keys = ("participant_id", "record_id", "segment_id", "sample_start", "prefix_stop", "full_stop",
                "available_stop", "public_protocol", "public_physical_history", "operator", "reference", "signal_unit")
        if (any(metadata_digest(before[key]) != metadata_digest(after[key]) for key in keys)
                or old_cap.events != new_cap.events or old_cap.candidate_vocabulary != new_cap.candidate_vocabulary
                or geometry_from_capability(old_cap) != geometry_from_capability(new_cap)):
            raise ValueError("new C programme changed the actual observation/event/clock/geometry inputs")
        for name in ("eeg", "eog", "membership"):
            a, b = getattr(old_cap, name), getattr(new_cap, name)
            if a.dtype != b.dtype or not np.array_equal(a, b):
                raise ValueError("new C programme changed original numeric raw or public membership")
        old_mean, old_error, old_binding = _mean_pair(captured.outputs[old_cap.identity_sha256])
        record = next(record for record in new_records if new_cap in record.raw_capabilities)
        new_mean, new_error, new_binding = _mean_pair(new_mean_callback(new_cap, record.labels))
        _require_mean_graph_binding(old_binding, old_graph_identity, side="original")
        _require_mean_graph_binding(new_binding, actual_graph_identity, side="new")
        if (old_mean.dtype != new_mean.dtype or old_error.dtype != new_error.dtype
                or array_digest(old_mean, old_error) != array_digest(new_mean, new_error)):
            raise ValueError("new fixed C mean programme changed its actual numerical residual input")
        # This is the original fitter's exact mean-image/I7/zero-nuisance chart.
        old_residual = old_cap.eeg - old_mean @ np.eye(7) - np.zeros_like(old_mean)
        new_residual = new_cap.eeg - new_mean @ np.eye(7) - np.zeros_like(new_mean)
        if (not np.array_equal(old_residual, new_residual)
                or array_digest(old_residual) != fitted["residual_sha256"]
                or array_digest(new_residual) != fitted["residual_sha256"]
                or fitted["record_id"] != before["record_segment_id"]
                or tuple(fitted["read_support"]) != (before["sample_start"], before["full_stop"])):
            raise ValueError("the original C-noise fitter did not consume these exact residual bytes/support")
        rows.append({"record_segment_id": before["record_segment_id"],
            "old_capability": old_cap.identity_sha256, "new_capability": new_cap.identity_sha256,
            "old_scope": before["independent_recording_scope_sha256"],
            "new_scope": after["independent_recording_scope_sha256"],
            "raw_arrays_sha256": array_digest(new_cap.eeg, new_cap.eog, new_cap.membership),
            "mean_error_sha256": array_digest(new_mean, new_error), "residual_sha256": array_digest(new_residual),
            "old_mean_binding": metadata_digest(old_binding), "new_mean_binding": metadata_digest(new_binding),
            "original_fitted_residual_ledger": fitted,
            "actual_numeric_inputs_equal": True})
    record = freeze_metadata({"schema": "actual_C_noise_numeric_input_and_permission_transition/1",
        "old_parameter_operand_identity": original_operand.identity,
        "original_complete_noise_record_sha256": metadata_digest(original_operand.noise.record()),
        "actual_programme_admission": programme.identity, "original_strict_parent_replay": original_parent,
        "same_parameter_law": law, "actual_C_records": rows,
        "old_label_capability_identities": [labels.identity_sha256 for labels in old_labels],
        "new_label_capability_identities": [labels.identity_sha256 for labels in new_labels],
        "eta": original_operand.eta, "noise_parameter_refits": 0, "response_C_updates_here": 0,
        "original_noise_provenance_rewritten": False, "producer": IMPLEMENTATION_SHA256})
    result = ReusedCNoise(original_operand, programme, record, metadata_digest(record))
    _REUSES[result] = (record, old_records, new_records, old_mean_callback, new_mean_callback)
    return result


def require_reused_c_noise(operand):
    from research.brainsync_programme_admission import require_actual_programme

    registered = _REUSES.get(operand) if type(operand) is ReusedCNoise else None
    if registered is None or registered[0] is not operand.record:
        raise ValueError("the joint learner requires its original live C-noise input reuse relation")
    require_actual_programme(operand.programme_admission)
    operand.noise.assert_integrity()
    return operand
