"""Live source/C/noise joins for one response-C and a complete held decision."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from research.brainsync_programme_admission import require_programme_transport
from research.calibration_physical_response import require_calibration_physical_segment
from research.frozen_metadata import plain_metadata
from research.grouped_physical_prediction import publish_grouped_physical
from research.noise_reuse_relation import require_reused_c_noise
from transfer.noise_model import metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def require_complete_learner_inputs(admission, noise, engine, target, original_segments, held_caps):
    admission = require_programme_transport(admission)
    noise = require_reused_c_noise(noise)
    programme = admission.programme
    if noise.programme_admission is not programme or engine.graph.digest() != admission.graph.digest():
        raise ValueError("the full learner must share its actual admitted source/C/held and C-noise programme")
    held_caps, segments = tuple(held_caps), tuple(original_segments)
    caps = programme.calibration_caps + programme.held_target.caps
    if (held_caps != programme.held_target.caps or len(segments) != len(caps)
            or len(target.records) != len(caps)):
        raise ValueError("full learner publication requires every actual C and independent held acquisition segment")
    c_groups, held_groups = set(), set()
    for i, (cap, original, (prepared, p, f)) in enumerate(zip(caps, segments, target.records, strict=True)):
        b = cap.public_binding
        if (original.original_input_identity != cap.identity_sha256 or prepared.base_identity != original.identity
                or not np.array_equal(original.y, cap.eeg) or original.origin != b["sample_start"]
                or f != b["full_stop"] - b["sample_start"]
                or original.carrier.noise is not noise.noise or original.carrier.eta != noise.eta
                or original.noise_segment_declaration != noise.segment_binding(cap,
                    noise_segment_id=original.noise_segment_declaration["noise_segment_id"])):
            raise ValueError("the response-C/held numerical model changed its actual raw/noise/endpoint operands")
        if any(callback.source is not engine for callback in original.responses):
            raise ValueError("C and held callbacks must use the same actual shared Gaussian engine object")
        if i < len(programme.calibration_caps):
            require_calibration_physical_segment(original, cap, programme.calibration_labels)
            if p != f or any(target.rosters[target.current].selection_id in callback.scope
                    for callback in original.responses):
                raise ValueError("the same full labelled C likelihood must enter P/F once without a held-q assignment")
            c_groups.add(original.group.group_id)
        else:
            if p != b["prefix_stop"] - b["sample_start"]:
                raise ValueError("held prefix must retain its actual predeclared raw request")
            held_groups.add(original.group.group_id)
    if c_groups & held_groups:
        raise ValueError("independent C and held recordings must retain their distinct original physical drift measures")
    return {"schema": "actual_source_C_noise_one_response_C_held_full_learner/1",
        "actual_source_transport": plain_metadata(admission.record), "actual_programme_identity": programme.identity,
        "C_noise_reuse_identity": noise.identity, "C_noise_reuse_relation": plain_metadata(noise.record),
        "actual_C_raw_identities": [cap.identity_sha256 for cap in programme.calibration_caps],
        "actual_C_labels_identity": programme.calibration_labels.identity_sha256,
        "actual_held_raw_identities": [cap.identity_sha256 for cap in held_caps],
        "C_full_at_both_endpoints": True, "response_C_update_count": 1,
        "original_source_prior_count": 1, "C_and_held_share_actual_engine": True,
        "source_chart": engine.source_chart, "rank": engine.rank,
        "C_held_independent_drift_groups": [sorted(c_groups), sorted(held_groups)],
        "source_fits": 0, "noise_parameter_refits": 0, "wrapper_producer": IMPLEMENTATION_SHA256}


def publish_complete_learner(verified_inference, admission, noise, engine, target, original_segments,
        held_caps, *, scoring_law, expert_identity, quality_policy):
    relation = require_complete_learner_inputs(admission, noise, engine, target, original_segments, held_caps)
    prediction = publish_grouped_physical(verified_inference, target, original_segments, held_caps,
        scoring_law=scoring_law, expert_identity=expert_identity, quality_policy=quality_policy)
    scope = {**prediction.decision_scope, "full_learner_relation": relation,
        "full_learner_relation_sha256": metadata_digest(relation)}
    return replace(prediction, decision_scope=scope)
