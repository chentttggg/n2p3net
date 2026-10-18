"""Actual numerical operator inputs and the original BrainSync raw-parent join.

Numeric clocks remain free inputs. Only the live parent requires the original
clock implementation and independently reconstructs its actual cap operands.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from research.frozen_metadata import freeze_metadata
from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def clock_operator_inputs(clock):
    return {"clock_owner": [type(clock).__module__, type(clock).__qualname__],
        "capability_identity": clock.capability_identity_sha256,
        "actual_public_binding": clock.public_binding}


def plan_operator_inputs(plan):
    """Read the operands the implementation actually consumes, excluding caches."""
    membership = plan.membership
    return freeze_metadata({"schema": "physical_response_actual_operator_inputs/1",
        "clock": clock_operator_inputs(plan.chart), "plan_binding": plan.binding,
        "history": plan.history, "exposures": plan.exposures, "vocabulary": plan.vocabulary,
        "dictionary": plan.dictionary.digest(), "clock_program": plan.clock_program,
        "exposure_response_law": plan.exposure_response_law,
        "membership_owner": [type(membership).__module__, type(membership).__qualname__],
        "membership_vocabulary": membership.vocabulary,
        "membership_positions": list(membership._positions.items()),
        "membership_rows": membership._rows, "unknown_options": membership._options,
        "membership_record": membership.record, "membership_identity": membership.identity,
        "operator_input_compiler": IMPLEMENTATION_SHA256})


def _require_clock_matches(clock, expected):
    from data.brainsync_raw_authority import PhysicalClockChart

    if (type(clock) is not PhysicalClockChart
            or metadata_digest(clock_operator_inputs(clock)) != metadata_digest(clock_operator_inputs(expected))):
        raise ValueError("the actual physical clock/operator differs from its original registered BrainSync raw capability")
    return expected


def require_brainsync_clock(clock, cap):
    from data.brainsync_raw_authority import compile_physical_clock

    # The original compiler validates the live raw capability exactly once.
    return _require_clock_matches(clock, compile_physical_clock(cap))


def _require_plan_matches(plan, cap, clock, *, clock_program):
    from research.exposure_membership import ExposureCandidateMembership
    from research.physical_response_design import PhysicalResponsePlan

    if type(plan) is not PhysicalResponsePlan or type(plan.membership) is not ExposureCandidateMembership:
        raise ValueError("the original raw parent requires the actual canonical physical plan and membership owner")
    _require_clock_matches(plan.chart, clock)
    if clock_program not in {"timeline_wall", "delivery_anchor"} or plan.clock_program != clock_program:
        raise ValueError("the original BrainSync parent requires its actual supported recording clock programme")
    expected = PhysicalResponsePlan(clock, plan.dictionary, clock_program=clock_program,
        exposure_response_law=plan.exposure_response_law, candidate_vocabulary=cap.candidate_vocabulary)
    actual_inputs, expected_inputs = plan_operator_inputs(plan), plan_operator_inputs(expected)
    if metadata_digest(actual_inputs) != metadata_digest(expected_inputs) or plan.identity != expected.identity:
        raise ValueError("the actual physical plan/history/exposures/membership differ from the original raw operator")
    return {"actual_plan_identity": plan.identity, "actual_operator_inputs": metadata_digest(actual_inputs),
        "raw_capability_identity": cap.identity_sha256, "clock_program": clock_program,
        "original_full_public_history_reconstructed": True, "join_producer": IMPLEMENTATION_SHA256}


def require_brainsync_plans(plans, cap, *, clock_program):
    from data.brainsync_raw_authority import compile_physical_clock

    clock = compile_physical_clock(cap)
    return tuple(_require_plan_matches(plan, cap, clock, clock_program=clock_program) for plan in plans)


def require_brainsync_plan(plan, cap, *, clock_program):
    return require_brainsync_plans((plan,), cap, clock_program=clock_program)[0]


_CALLBACK_FIELDS = ("scope", "rank", "channels", "source_chart", "physical_capability_identity",
    "physical_target_lock", "physical_candidate_variable", "physical_clock_program", "physical_shape_block_id",
    "physical_shape_variable", "physical_shape_parameters", "physical_exposure_ids", "physical_unknown_identity_bindings")


def _require_callback_matches(callback, cap, clock, *, clock_program):
    from research.physical_affine_frontier import PhysicalResponseAffine
    from research.physical_source_templates import PhysicalTemplateResponse
    from research.scoped_physical_response import ScopedPhysicalResponse

    if type(callback) not in (PhysicalResponseAffine, PhysicalTemplateResponse):
        raise ValueError("the original raw parent requires its actual direct or exact-template physical callback owner")
    scoped = callback.response
    if type(scoped) is not ScopedPhysicalResponse:
        raise ValueError("the actual raw-bound callback requires its original declared scoped physical response")
    joined = _require_plan_matches(scoped.plan, cap, clock, clock_program=clock_program)
    # Re-evaluate the existing scope/assignment contract from current inputs;
    # it adds no state or prior and accepts the caller's legitimate joint law.
    expected_scope = ScopedPhysicalResponse(scoped.plan, scoped.block_id, scoped.candidate_variable,
        scoped.shape_variable, scoped.shapes, scoped.exposure_assignments)
    if type(callback) is PhysicalResponseAffine:
        expected = PhysicalResponseAffine(expected_scope, callback.mean, callback.factor)
    else:
        expected = PhysicalTemplateResponse(expected_scope, callback.source)
        if tuple(callback.source.channel_ids) != tuple(cap.public_binding["eeg_channels"]):
            raise ValueError("template response observation coordinates differ from the original raw EEG columns")
    actual_fields = {name: getattr(callback, name) for name in _CALLBACK_FIELDS}
    expected_fields = {name: getattr(expected, name) for name in _CALLBACK_FIELDS}
    if (callback.identity != expected.identity or metadata_digest(actual_fields) != metadata_digest(expected_fields)
            or callback.channels != len(cap.public_binding["eeg_channels"])):
        raise ValueError("the actual callback operator parameters differ from its original physical descriptor")
    return {**joined, "callback_identity": callback.identity,
        "callback_owner": [type(callback).__module__, type(callback).__qualname__],
        "actual_callback_fields": metadata_digest(actual_fields)}


def require_brainsync_callback(callback, cap, *, clock_program):
    from data.brainsync_raw_authority import compile_physical_clock

    return _require_callback_matches(callback, cap, compile_physical_clock(cap), clock_program=clock_program)


def brainsync_segment_operator(segment, cap):
    """Rejoin current operands before publication or admitted-parent reuse."""
    from data.brainsync_raw_authority import compile_physical_clock

    clock = compile_physical_clock(cap)
    b, geometry = cap.public_binding, segment.carrier.noise.geometry
    _require_clock_matches(segment.clock, clock)
    if (segment.clock_program not in {"timeline_wall", "delivery_anchor"}
            or segment.group.clock_domain != b["record_id"]
            or segment.group.clock_program != segment.clock_program
            or segment.record_id != b["record_segment_id"] or segment.origin != b["sample_start"]
            or segment.original_input_identity != cap.identity_sha256
            or segment.q_values != cap.candidate_vocabulary
            or segment.variable_ids[segment.current] != b["target_lock_group"]
            or array_digest(segment.y) != array_digest(cap.eeg)
            or geometry.channels != tuple(b["eeg_channels"]) or geometry.unit != b["signal_unit"]
            or geometry.reference != b["reference"] or geometry.sfreq != b["sample_rate_hz"]
            or geometry.operator_identity != b["operator"]
            or segment.noise_segment_declaration["acquisition_segment_id"] != b["segment_id"]):
        raise ValueError("the admitted segment no longer uses its original raw coordinates, clock, support and operator")
    callbacks = tuple(_require_callback_matches(callback, cap, clock, clock_program=segment.clock_program)
        for callback in segment.responses)
    return freeze_metadata({"schema": "actual_BrainSync_segment_operator_join/1",
        "raw": cap.identity_sha256, "segment_identity": segment.identity,
        "clock": clock_operator_inputs(segment.clock), "callbacks": callbacks,
        "drift_group": segment.group.identity, "carrier": segment.carrier.identity,
        "noise_segment": segment.noise_segment_declaration, "producer": IMPLEMENTATION_SHA256})
