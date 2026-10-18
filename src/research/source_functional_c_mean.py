"""Actual physical source-functionals -> fixed C plugin mean, without hidden q.

Completed exposures use their original granted binary C bits. A fixed response
state and any extra joint exposure mean law are explicit caller programme inputs.
Only the original source-functional mean compiler evaluates the response law.
"""
from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np

from data.brainsync_raw_authority import (
    compile_physical_clock,
    require_calibration_labels,
    require_raw_capability,
)
from research.physical_response_design import PhysicalResponsePlan
from research.target_noise_operand import geometry_from_capability
from transfer.native_response_dictionary import NativeResponseDictionary
from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _q(value):
    return value if isinstance(value, Fraction) else Fraction.from_float(float(value))


def _upper(value):
    rounded = float(value)
    return float(np.nextafter(rounded, np.inf)) if Fraction.from_float(rounded) < value else rounded


class SourceFunctionalCMean:
    """One declared fixed-shape source-prior mean for the C residual programme.

    `mean_compiler` is the real source-functional owner's mean-only compiler;
    it shares this graph and returns values/error/binding with a final width1.
    The callback never receives an issuer, manifest or session target.
    """

    def __init__(self, graph, dictionary, mean_compiler, *, states_by_block,
                 clock_program, extra_exposure_mean_law=None):
        if not isinstance(dictionary, NativeResponseDictionary):
            raise TypeError("C mean requires the original source-native function dictionary owner")
        if dictionary.sample_rate_hz != graph.binding["function_source_clock_hz"]:
            raise ValueError("C mean function clock differs from the original source response graph")
        if dictionary.digest() != graph.binding["function_dictionary_sha256"]:
            raise ValueError("C mean function dictionary differs from the original source response graph")
        if (graph.digest() != mean_compiler.graph.digest()
                or graph.coefficient_count != 2 * dictionary.matrix.shape[1]
                or graph.chart.native_dimension != 7):
            raise ValueError("C mean requires the actual complete source graph/function coefficient chart")
        self.graph, self.dictionary, self.compiler = graph, dictionary, mean_compiler
        self.states = dict(states_by_block)
        self.clock_program, self.extra_law = clock_program, extra_exposure_mean_law
        self.identity = metadata_digest({"source_graph": graph.digest(), "dictionary": dictionary.digest(),
            "states_by_block": {str(block): vars(state) for block, state in self.states.items()},
            "clock_program": clock_program, "extra_exposure_mean_law": self._extra_record(),
            "mean_role": "source_prior_fixed_before_target_C_response_update", "compiler": IMPLEMENTATION_SHA256})
        # Unit coefficient functionals obtain an outward bound for physical
        # dictionary interpolation error without materializing a rounded beta.
        p, c, d = graph.coefficient_count, graph.chart.native_dimension, graph.dimension
        query = np.empty((p, c, d), dtype=object)
        query.fill(Fraction())
        for coefficient in range(p):
            for channel in range(c):
                query[coefficient, channel, coefficient * c:(coefficient + 1) * c] = graph.chart.exact_inverse[channel]
        values, errors, _ = self.compiler.compile_mean(query, query_binding={"role": "C_mean_error_unit_functionals"})
        self.coefficient_abs_bounds = tuple(tuple(_q(abs(values[i, j, 0])) + _q(errors[i, j, 0])
                                                   for j in range(c)) for i in range(p))

    def _extra_record(self):
        return None if self.extra_law is None else self.extra_law.record()

    def __call__(self, capability, label_capability):
        cap = require_raw_capability(capability)
        labels = require_calibration_labels(label_capability)
        b = cap.public_binding
        if (b["role"] != "calibration" or labels.public_binding["record_id"] != b["record_id"]
                or labels.public_binding["independent_recording_scope_sha256"] != b["independent_recording_scope_sha256"]):
            raise ValueError("source C mean requires the actual same-record calibration label authority")
        if metadata_digest(self.graph.binding["target_observation_geometry"]) != metadata_digest(
                geometry_from_capability(cap).record()):
            raise ValueError("source graph C mean and actual native observation operator/geometry differ")
        plan = PhysicalResponsePlan(compile_physical_clock(cap), self.dictionary,
            clock_program=self.clock_program, exposure_response_law="onset_impulse_with_actual_duration_provenance/1",
            candidate_vocabulary=cap.candidate_vocabulary)
        start, stop = b["sample_start"], b["full_stop"]
        native_rows = tuple(range(start, stop))
        events = [event for event in plan.exposures if event["sample"] < stop]
        by_id = {label.event_id: label for label in labels.labels}
        p, c, k = self.graph.coefficient_count, self.graph.chart.native_dimension, self.dictionary.matrix.shape[1]
        design = [[Fraction() for _ in range(p)] for _ in native_rows]
        design_error = [[Fraction() for _ in range(p)] for _ in native_rows]
        ledgers, used, unknown_proofs = [], set(), []
        support_by_block = {block: plan.support(block, (self.states[block],), native_start=start, native_stop=stop)
                            for block in {event["block_id"] for event in events}}
        active = {hull["event_id"] for support in support_by_block.values() for hull in support["hulls"]}
        for event in events:
            event_id = event["event_id"]
            state = self.states[event["block_id"]]
            if event["exposure_role"] == "completed_protocol_stimulus":
                label = by_id.get(event_id)
                if label is None or (label.sample, label.block_id, label.trial_index) != (
                        event["sample"], event["block_id"], event["trial_index"]):
                    raise ValueError("completed C exposure needs its original authorized binary event label")
                target_indicator = Fraction(label.binary_label)
                used.add(event_id)
            elif event_id not in active:
                target_indicator = Fraction()
                unknown_proofs.append({"event_id": event_id, "role": "zero_response_on_granted_rows_no_stimulus_prior",
                    "support_identity": metadata_digest(support_by_block[event["block_id"]])})
            elif self.extra_law is not None:
                target_indicator = _q(self.extra_law.mean_target_indicator(event, cap, labels))
                if not 0 <= target_indicator <= 1:
                    raise ValueError("explicit joint exposure mean must give a target indicator expectation in [0,1]")
            else:
                raise ValueError("active interrupted C exposure needs an explicit joint exposure mean law")
            basis, ledger = plan.exposure_basis(native_rows, event_id, state)
            common, target = _q(state.common_amplitude), _q(state.target_amplitude) * target_indicator
            for row in range(len(native_rows)):
                for column in range(k):
                    value, error = _q(basis.matrix[row, column]), _q(basis.error[row, column])
                    design[row][column] += common * value
                    design[row][k + column] += target * value
                    design_error[row][column] += abs(common) * error
                    design_error[row][k + column] += abs(target) * error
            ledgers.append({**ledger, "C_target_indicator_exact": str(target_indicator),
                            "indicator_origin": "original_C_bit" if event_id in used else "explicit_mean_law_or_zero_support"})
        query = np.empty((len(native_rows), c, self.graph.dimension), dtype=object)
        query.fill(Fraction())
        for row in range(len(native_rows)):
            for channel in range(c):
                for coefficient, value in enumerate(design[row]):
                    if value:
                        for chart_column, weight in enumerate(self.graph.chart.exact_inverse[channel]):
                            query[row, channel, coefficient * c + chart_column] = value * weight
        values, errors, compiled = self.compiler.compile_mean(query, query_binding={
            "physical_plan": plan.identity, "C_noise_mean_programme": self.identity,
            "actual_physical_exposures": ledgers, "authorized_C_label_capability": labels.identity_sha256})
        self.last_cache_statistics = compiled.get("cache")
        compiled = {key: value for key, value in compiled.items() if key != "cache"}
        mean, bound = values[:, :, 0], errors[:, :, 0].copy()
        for row in range(len(native_rows)):
            for channel in range(c):
                original = sum((design_error[row][j] * self.coefficient_abs_bounds[j][channel]
                                for j in range(p)), Fraction())
                bound[row, channel] = _upper(original + _q(bound[row, channel]))
        return mean, bound, {"source_response_model_identity": self.graph.digest(),
            "mean_role": "original_source_functional_prior_mean", "target_response_C_updates_before_noise_fit": 0,
            "used_label_event_ids": sorted(used), "considered_exposure_ids": [event["event_id"] for event in events],
            "unknown_exposure_law": None if self.extra_law is None and not unknown_proofs else {
                "joint_mean_law": self._extra_record(), "zero_support_proofs": unknown_proofs},
            "physical_plan_identity": plan.identity, "C_fixed_mean_programme": self.identity,
            "source_functional_mean_compiler": compiled, "mean_image_sha256": array_digest(mean, bound),
            "actual_physical_exposure_ledger": ledgers, "clock_program": self.clock_program,
            "no_hidden_session_target_argument": True, "source_response_F_rescaled": False}
