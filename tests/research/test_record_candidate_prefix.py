"""Record-clock/label permission counterexamples for the shared input owner."""
from dataclasses import replace

import numpy as np

from data.candidate_input import (
    CandidateDatasetAdapter,
    OriginalCandidateRecording,
    OriginalCandidateSelection,
)
from data.candidate_task import MembershipDecision
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from data.identity import origin_subject_key
from transfer.ensemble_views import RecordCalibrationPrefixCompiler, SelectionEpochProfile


def _adapter():
    calls, records = [], []
    participant = origin_subject_key("Huebner2017", "synthetic")
    for run in range(2):
        identity = f"actual-software-record-{run}"
        onsets = np.array([1000, 1250, 5000, 5250], dtype=np.int64)
        labels = np.array([True, False, False, True])
        selections = []
        for index in range(2):
            decision = MembershipDecision.direct(("q0", "q1"), ("q0", "q1"),
                event_ids=(f"{identity}:{2*index}", f"{identity}:{2*index+1}"),
                repetition_indices=np.zeros(2, dtype=np.int64), repetitions=1)
            selections.append(OriginalCandidateSelection(f"{identity}:selection:{index}",
                (2*index, 2*index+1), decision, f"q{index}"))
        def read(start, stop, record_key=identity):
            calls.append((record_key, start, stop))
            return np.zeros((stop-start, 2), dtype=float)
        records.append(OriginalCandidateRecording(identity, str(run+1)*64, identity+".raw", participant,
            "synthetic", "0", str(run), 1000., ("Fz", "Cz"), "nose", 7000, onsets, labels,
            tuple(selections), {"software_fixture": True}, read))
    adapter = CandidateDatasetAdapter("Huebner2017", participant, "synthetic", "explicit_software_fixture",
        tuple(records), (), {}, "actual_fixture_full_selection", "a"*64)
    return adapter, calls


def test_global_label_budget_retains_record_clock_reset_and_no_outside_c_reads():
    adapter, calls = _adapter()
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    compiler = RecordCalibrationPrefixCompiler(adapter, budgets=(0, 3, 5), profiles=(),
        authorization_preprocessing=spec, response_seconds=.2, pre_roll_seconds=.2)
    prefix = compiler.compile(5)
    assert len(prefix.calibration_records) == 2
    assert [r.onsets[0] for r in prefix.calibration_records] == [1000, 1000]
    assert sum(len(r.onsets) for r in prefix.calibration_records) == 5
    second = prefix.target_contract["record_scopes"][adapter.recordings[1].record_id]
    assert second["global_label_indices"] == (4, 5)
    assert second["read_sample_support"][1] <= 1250
    assert all(stop <= 1250 for identity, _, stop in calls if identity == adapter.recordings[1].record_id)


def test_requested_large_k_alias_and_common_future_preserve_every_physical_selection():
    adapter, _ = _adapter()
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    compiler = RecordCalibrationPrefixCompiler(adapter, budgets=(0, 3, 2000), profiles=(),
        authorization_preprocessing=spec, response_seconds=.2, pre_roll_seconds=.2)
    views = compiler.independent_views()
    assert views.ledger["requested_to_actual_K"]["2000"] == 6
    assert 2000 not in views.calibration and 6 in views.calibration
    assert len(views.decisions) == 1 and views.decisions[0].decision_id.endswith("selection:1")
    assert views.decisions[0].decision.schedule_complete


def test_forward_group_join_uses_original_record_and_keeps_future_labels_hidden():
    adapter, _ = _adapter()
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    compiler = RecordCalibrationPrefixCompiler(adapter, budgets=(0, 6), profiles=(),
        authorization_preprocessing=spec, response_seconds=.2, pre_roll_seconds=.2)
    prefix = compiler.compile(6)
    held = prefix.complete_groups[-1]
    earlier = compiler.before_group(prefix, held)
    assert earlier.budget == 4
    assert len(earlier.calibration_records) == 1
    assert all(r.record_id != adapter.recordings[1].record_id for r in earlier.calibration_records)
    for view in compiler.independent_views().decisions:
        assert not hasattr(view, "authorized_binary_labels") and not hasattr(view, "parent_target_candidate")


def test_declared_missing_slots_require_measured_reference_and_keep_explicit_mask():
    adapter, _ = _adapter()
    spec = replace(preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT), baseline_mode="none")
    profile = SelectionEpochProfile.from_preprocessing(spec, source_sample_rate_hz=1000.,
        channel_names=("Fz", "Pz"), reference="nose", reference_channel="Cz", allow_missing_channels=True)
    compiler = RecordCalibrationPrefixCompiler(adapter, budgets=(0, 2), profiles=(profile,),
        authorization_preprocessing=spec, response_seconds=.2, pre_roll_seconds=.2)
    metadata = compiler.compile(0).trial_target(profile)
    assert metadata.channel_names == ("FZ", "PZ")
    assert metadata.provenance["selection_epoch_capability"]["profile"]["allow_missing_channels"] is True
