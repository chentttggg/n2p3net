"""Actual binary-labelled C exposure design in the shared response-parameter law.

This is a C model admission, consuming canonical raw/label capabilities. It
does not create raw permissions or alter the held unknown-q admission owner.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from data.brainsync_raw_authority import (
    compile_physical_clock,
    require_calibration_labels,
    require_raw_capability,
)
from research.frozen_metadata import freeze_metadata
from research.physical_affine_frontier import PhysicalAffineSegment
from research.physical_operator_binding import require_brainsync_clock, require_brainsync_plans
from research.physical_response_design import PhysicalResponsePlan
from research.target_noise_operand import geometry_from_capability
from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_C_SEGMENTS = WeakKeyDictionary()


class FixedLabelPhysicalResponse:
    """The original C bits directly select target increments; only shape is latent."""

    def __init__(self, cap, labels, source, dictionary, *, block_id, shape_variable,
                 shapes, clock_program, maximum_image_cache_bytes=64 * 1024**2):
        cap, labels = require_raw_capability(cap), require_calibration_labels(labels)
        b = cap.public_binding
        if (b["role"] != "calibration" or labels.public_binding["record_id"] != b["record_id"]
                or labels.public_binding["independent_recording_scope_sha256"] != b["independent_recording_scope_sha256"]):
            raise ValueError("fixed C response requires original same-record raw and binary label authority")
        if (dictionary.digest() != source.graph.binding["function_dictionary_sha256"]
                or dictionary.sample_rate_hz != source.graph.binding["function_source_clock_hz"]):
            raise ValueError("fixed C response dictionary/clock differs from its actual source parameter law")
        if metadata_digest(source.graph.binding["target_observation_geometry"]) != metadata_digest(
                geometry_from_capability(cap).record()):
            raise ValueError("fixed C response source law differs from its original native observation geometry")
        self.cap, self.labels, self.source = cap, labels, source
        self.plan = PhysicalResponsePlan(compile_physical_clock(cap), dictionary, clock_program=clock_program,
            exposure_response_law="onset_impulse_with_actual_duration_provenance/1",
            candidate_vocabulary=cap.candidate_vocabulary)
        self.block_id, self.shape_variable, self.shapes = block_id, shape_variable, tuple(shapes)
        self.scope = (shape_variable,)
        self.rank, self.channels, self.source_chart = source.rank, source.channels, source.source_chart
        self.physical_capability_identity = cap.identity_sha256
        self.physical_target_lock = b["target_lock_group"]
        self.physical_candidate_variable = None
        self.physical_clock_program = clock_program
        self.physical_shape_block_id, self.physical_shape_variable = block_id, shape_variable
        self.physical_shape_parameters = tuple(tuple(vars(state).items()) for state in self.shapes)
        events = tuple(e for e in self.plan.exposures if e["block_id"] == block_id)
        self.physical_exposure_ids = tuple(e["event_id"] for e in events)
        self.physical_unknown_identity_bindings = {}
        by_id = {label.event_id: label for label in labels.labels}
        active = {h["event_id"] for h in self.plan.support(block_id, self.shapes,
            native_start=b["sample_start"], native_stop=b["full_stop"])["hulls"]}
        self.indicators, self.event_label_rows = {}, []
        for event in events:
            label = by_id.get(event["event_id"])
            if label is None:
                if event["event_id"] in active:
                    raise ValueError("active C exposure requires its original binary bit or an explicit scoped C-state law")
                self.indicators[event["event_id"]] = 0
                self.event_label_rows.append({"event_id": event["event_id"],
                    "role": "all_positive_shape_supports_outside_authorized_C_rows"})
            else:
                if (label.sample, label.block_id, label.trial_index) != (
                        event["sample"], event["block_id"], event["trial_index"]):
                    raise ValueError("C bit and actual physical exposure have different original identities")
                self.indicators[event["event_id"]] = label.binary_label
                self.event_label_rows.append({"event_id": event["event_id"], "sample": label.sample,
                    "binary_label": label.binary_label, "role": "original_authorized_C_bit"})
        self.indicators = freeze_metadata(self.indicators)
        self.event_label_rows = freeze_metadata(self.event_label_rows)
        self.identity = metadata_digest({"schema": "fixed_binary_C_physical_response/1",
            "raw": cap.identity_sha256, "labels": labels.identity_sha256, "plan": self.plan.identity,
            "block": block_id, "shape_variable": shape_variable, "shapes": self.physical_shape_parameters,
            "actual_binary_design": self.event_label_rows, "parameter_source_chart": source.source_chart,
            "exact_template_header": source.header, "producer": IMPLEMENTATION_SHA256})
        self.maximum_image_cache_bytes = maximum_image_cache_bytes
        self._designs, self._images = OrderedDict(), OrderedDict()
        self._image_bytes = 0

    def support(self, *, native_start, native_stop):
        return self.plan.support(self.block_id, self.shapes, native_start=native_start, native_stop=native_stop)

    def _design(self, selected_states, start, stop):
        h = selected_states[self.shape_variable].shape
        key = start, stop, h
        found = self._designs.pop(key, None)
        if found is not None:
            self._designs[key] = found
            return found
        state = self.shapes[h]
        rows = tuple(range(start, stop))
        k = self.plan.dictionary.matrix.shape[1]
        design, error, absolute = (np.zeros((len(rows), 2 * k)) for _ in range(3))
        ledger = []
        events = [e for e in self.plan.exposures if e["block_id"] == self.block_id]
        for event in events:
            basis, binding = self.plan.exposure_basis(rows, event["event_id"], state)
            bit = self.indicators[event["event_id"]]
            common = state.common_amplitude * basis.matrix
            target = bit * state.target_amplitude * basis.matrix
            design[:, :k] += common
            design[:, k:] += target
            absolute[:, :k] += abs(common)
            absolute[:, k:] += abs(target)
            error[:, :k] += abs(state.common_amplitude) * basis.error + np.finfo(float).eps * abs(common)
            error[:, k:] += bit * abs(state.target_amplitude) * basis.error + np.finfo(float).eps * abs(target)
            ledger.append({**binding, "fixed_C_target_indicator": bit})
        gamma = len(events) * np.finfo(float).eps / (1 - len(events) * np.finfo(float).eps)
        error += gamma * absolute
        binding = {"C_response_identity": self.identity, "shape_index": h,
            "original_C_event_bits": self.event_label_rows, "physical_exposure_ledger": ledger,
            "numeric_design_identity": array_digest(design, error), "held_q_used": False}
        found = design, error, binding
        self._designs[key] = found
        # A short bounded physical row cache, independent of roster size.
        while len(self._designs) > 8:
            self._designs.popitem(last=False)
        return found

    def _image(self, key, factory):
        found = self._images.pop(key, None)
        if found is None:
            found = factory()
            size = found.values.nbytes + found.absolute_error.nbytes
            while self._images and self._image_bytes + size > self.maximum_image_cache_bytes:
                _, old = self._images.popitem(last=False)
                self._image_bytes -= old.values.nbytes + old.absolute_error.nbytes
            if size <= self.maximum_image_cache_bytes:
                self._images[key] = found
                self._image_bytes += size
        else:
            self._images[key] = found
        return found

    def affine(self, selected_states, *, native_start, native_stop):
        design, error, binding = self._design(selected_states, native_start, native_stop)
        key = "absolute", binding["numeric_design_identity"]
        image = self._image(key, lambda: self.source.compile_temporal(design, temporal_error=error,
            query_binding={"fixed_binary_C_design": key[1], "C_response": self.identity}))
        return image.values, image.absolute_error, {"actual_C_design": binding, "exact_source_image": image.binding}

    def affine_difference(self, selected_states, reference_states, *, native_start, native_stop):
        actual, actual_error, actual_binding = self._design(selected_states, native_start, native_stop)
        reference, reference_error, reference_binding = self._design(reference_states, native_start, native_stop)
        a, b = actual_binding["numeric_design_identity"], reference_binding["numeric_design_identity"]
        same_physical_query = actual_binding["shape_index"] == reference_binding["shape_index"]
        error = np.zeros_like(actual_error) if same_physical_query else actual_error + reference_error
        error = np.where(error == 0, 0., np.nextafter(error, np.inf))
        key = "difference", a, b, array_digest(error)
        image = self._image(key, lambda: self.source.compile_temporal_difference(actual, reference,
            temporal_error=error, query_binding={"actual_C_design": a, "reference_C_design": b}))
        return image.values, image.absolute_error, {"actual_C_design": actual_binding,
            "reference_C_design": reference_binding, "exact_source_image": image.binding}


def calibration_segment_from_capabilities(cap, labels, carrier, responses, rosters, drift_group, *,
        noise_segment_declaration, clock_program, current_variable, q_values, maximum_intermediate_entries):
    """One live C-model join over existing canonical raw and label registries."""
    cap, labels = require_raw_capability(cap), require_calibration_labels(labels)
    b, geometry = cap.public_binding, carrier.noise.geometry
    if (b["role"] != "calibration" or labels.public_binding["record_id"] != b["record_id"]
            or labels.public_binding["independent_recording_scope_sha256"] != b["independent_recording_scope_sha256"]
            or drift_group.clock_domain != b["record_id"]):
        raise ValueError("C model compilation needs actual same-record calibration raw/labels and drift clock")
    if (geometry.channels != tuple(b["eeg_channels"]) or geometry.sfreq != b["sample_rate_hz"]
            or geometry.unit != b["signal_unit"] or geometry.reference != b["reference"]
            or geometry.operator_identity != b["operator"]):
        raise ValueError("actual complete C-noise geometry differs from original native C observations")
    responses = tuple(responses)
    represented = set()
    blocks = set()
    for response in responses:
        if (response.cap is not cap or response.labels is not labels
                or response.physical_clock_program != clock_program
                or response.block_id in blocks or current_variable in response.scope):
            raise ValueError("fixed C callback must bind original bits/clock and its own independent shape scope")
        roster = next(r for r in rosters if r.selection_id == response.shape_variable)
        if (len(roster.states) != len(response.shapes)
                or {state.shape for state in roster.states} != set(range(len(response.shapes)))):
            raise ValueError("C response must retain all original positive shape states")
        blocks.add(response.block_id)
        represented.update(response.physical_exposure_ids)
    if represented != {e["event_id"] for e in b["public_physical_history"]["exposures"]}:
        raise ValueError("C model must consider every original available completed/interrupted exposure")
    if noise_segment_declaration["acquisition_segment_id"] != b["segment_id"]:
        raise ValueError("C-noise segment declaration differs from the original acquisition segment")
    operators = require_brainsync_plans((response.plan for response in responses), cap, clock_program=clock_program)
    model = PhysicalAffineSegment(cap.eeg, carrier, compile_physical_clock(cap), responses, rosters, drift_group,
        record_id=b["record_segment_id"], origin_sample=b["sample_start"], clock_program=clock_program,
        current_variable=current_variable, q_values=q_values,
        maximum_intermediate_entries=maximum_intermediate_entries, original_input_identity=cap.identity_sha256,
        noise_segment_declaration=noise_segment_declaration)
    _C_SEGMENTS[model] = (model.identity, cap, labels, tuple(r.identity for r in responses), operators)
    return model


def require_calibration_physical_segment(model, cap, labels):
    cap, labels = require_raw_capability(cap), require_calibration_labels(labels)
    original = _C_SEGMENTS.get(model)
    if (original is None or original[:3] != (model.identity, cap, labels)
            or original[3] != tuple(r.identity for r in model.responses)):
        raise ValueError("C response update requires its original live raw/label/model compilation relation")
    require_brainsync_clock(model.clock, cap)
    operators = require_brainsync_plans((response.plan for response in model.responses), cap,
        clock_program=model.clock_program)
    if metadata_digest(operators) != metadata_digest(original[4]):
        raise ValueError("the original C raw-bound physical operator changed after parent admission")
    return model
