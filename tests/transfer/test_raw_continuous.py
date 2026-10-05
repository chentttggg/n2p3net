from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace

import mne
import numpy as np
import pytest
from scipy.io import savemat

from data.bnci2014_008_candidate import read_bnci2014_008_candidate_record
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from data.preprocess import design_continuous_sos
from experiments.run_bnci_continuous_study import _cell, run_study
from tests.support.bnci import bnci_mat_fields
from transfer.continuous_decision import ContinuousDecisionModel, smooth_erp_basis
from transfer.raw_continuous import (
    CANDIDATES,
    SFREQ,
    ContinuousInputPlan,
    _calibration_view,
    _membership,
    _prepare_record,
    prepare_bnci_continuous,
)


@pytest.fixture
def record_bytes():
    fields = bnci_mat_fields()
    fields["X"] = np.random.default_rng(411).normal(0, 4, fields["X"].shape).astype(np.float32)
    buffer = io.BytesIO()
    savemat(buffer, {"data": fields}, do_compression=True, long_field_names=True)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    record = read_bnci2014_008_candidate_record(io.BytesIO(payload), source_relative_path="A01.mat", source_sha256=digest)
    return payload, digest, record


def test_attested_raw_prefixes_keep_all_labels_and_fixed_test(record_bytes, tmp_path):
    payload, digest, record = record_bytes
    path = tmp_path / "A01.mat"
    path.write_bytes(payload)
    views = prepare_bnci_continuous(path, source_sha256=digest, budgets=(200, 800))
    assert set(views.calibration) == {200, 800}
    assert len(views.decisions) == 29
    assert sum(d.samples is None for d in views.decisions) == 1
    assert not hasattr(views.decisions[1], "target_mask")
    assert views.ledger["different_decisions_use_disjoint_sample_positions"]
    assert views.ledger["minimum_gap_seconds"] > 1.2
    for budget in (200, 800):
        entry = views.ledger["calibration"][str(budget)]
        assert entry["retained_label_count"] == budget
        assert entry["stop_sample"] == int(record.flash_sample[budget])
        assert len(views.calibration[budget].target_mask) == budget
        assert entry["right_censored_event_indices"]
        assert entry["stop_sample"] <= views.ledger["test"]["start_sample"]
    with pytest.raises(ValueError, match="frozen source"):
        prepare_bnci_continuous(path, source_sha256="0" * 64, budgets=(200, 800))


def test_unknown_next_labels_and_samples_cannot_change_calibration_fit(record_bytes):
    _, _, record = record_bytes
    plan = ContinuousInputPlan(n_basis=2)
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    sos = design_continuous_sos(SFREQ, l_freq=spec.l_freq, h_freq=spec.h_freq,
        order=spec.filter_order, notch_freqs_hz=spec.notch_freqs_hz, notch_quality_factor=spec.notch_quality_factor)
    pad = int(mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)["padlen"])
    stop = int(record.flash_sample[199]) + round(1.2 * SFREQ) + 1
    original, receipt = _calibration_view(record, 200, stop, plan, sos, pad)
    values = record.eeg_uv.copy()
    values[int(record.flash_sample[200]):] += 100000
    labels = record.raw_is_target.copy()
    labels[200:] = ~labels[200:]
    changed, changed_receipt = _calibration_view(replace(record, eeg_uv=values, raw_is_target=labels),
                                               200, stop, plan, sos, pad)
    np.testing.assert_array_equal(original.samples, changed.samples)
    np.testing.assert_array_equal(original.target_mask, changed.target_mask)
    assert receipt == changed_receipt
    settings = dict(sfreq=SFREQ, channels=tuple(plan.record()["channels"]), candidate_ids=CANDIDATES,
                    operator_identity="counterexample", basis=smooth_erp_basis(SFREQ, 1.2, 2))
    left = ContinuousDecisionModel.fit([original], **settings)
    right = ContinuousDecisionModel.fit([changed], **settings)
    np.testing.assert_array_equal(left.kernel_mean, right.kernel_mean)
    np.testing.assert_array_equal(left.innovation_covariance, right.innovation_covariance)


def test_membership_uses_flash_codes_and_checks_real_interselection_gaps(record_bytes):
    _, _, record = record_bytes
    member = _membership(np.arange(12))
    assert member.shape == (12, 36)
    np.testing.assert_array_equal(member.sum(axis=0), np.full(36, 2))
    np.testing.assert_array_equal(member.sum(axis=1), np.full(12, 6))
    # Change a real schedule gap to less than the model support. The scope owner
    # refuses that decision; it cannot infer neighbor target truth as context.
    flashes = record.flash_sample.copy()
    left = np.flatnonzero(record.selection_index == 6)[-1]
    right = np.flatnonzero(record.selection_index == 7)[0]
    shift = int(flashes[right] - flashes[left] - 64)
    flashes[right:] -= shift
    views = _prepare_record(replace(record, flash_sample=flashes), (200, 800), ContinuousInputPlan())
    assert any(d.failure_reason == "previous_unknown_decision_overlaps_finite_response_support" for d in views.decisions)


def test_actual_scope_fit_freeze_reload_and_requested_decision_ledger(record_bytes, tmp_path):
    _, _, record = record_bytes
    plan = ContinuousInputPlan(n_basis=2)
    views = _prepare_record(record, (200, 800), plan)
    # Exercise one partial and one complete physical decision, both remain in
    # the denominator. The separately declared real run covers all 8 people.
    views = replace(views, decisions=views.decisions[:2])
    result = _cell(views, 200, plan, tmp_path / "K200", "a" * 64, evidence_scope="software_correctness")
    counts = result["decision_accounting"]["by_evidence_level"]["10"]
    assert result["status"] == "completed", result
    assert counts["requested"] == counts["accounted"] == 2
    assert counts["incomplete"] == 1
    assert counts["valid"] == 1
    assert result["nll_scored_decisions"] == result["nll_missing_decisions"] == 1
    assert result["retained_label_count"] == 200
    assert result["evidence_scope"] == "software_correctness"
    assert result["mean_actual_offline_latency_s"] > result["mean_finite_response_latency_s"]


def test_run_failure_is_closed_and_all_requests_remain_visible(record_bytes, tmp_path):
    payload, _, _ = record_bytes
    source = tmp_path / "A01.mat"
    source.write_bytes(payload)
    plan = {"schema": "n2p3_bnci_continuous_study/1",
            "records": [{"subject": "A01", "path": str(source), "sha256": "0" * 64}],
            "budgets": [200, 800], "evidence_scope": "software_correctness", "cpu_threads": 1,
            "continuous_input": {}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="frozen source"):
        run_study(path, output)
    closed = json.loads((output / "result.json").read_text())
    assert closed["status"] == "failed"
    assert [row["K"] for row in closed["requested_cell_outcomes"]] == [200, 800]
    assert all(row["status"] == "not_executed_due_to_run_failure" for row in closed["requested_cell_outcomes"])
    original = (output / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_study(path, output)
    assert (output / "result.json").read_bytes() == original
