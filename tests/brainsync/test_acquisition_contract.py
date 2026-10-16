import json
import subprocess
import textwrap

import mne
import numpy as np

from data import contract
from tests import PROJECT_ROOT
from tests.support.io import read_json


def test_exported_epoch_support_is_derived_from_canonical_model_contract():
    record = contract.epoch_support_record()
    source = contract.SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
    assert record["tmin_ms"] == source.tmin_ms
    assert record["tmax_ms"] == source.tmax_ms
    assert record["model_sample_rate_hz"] == source.sample_rate_hz
    assert record["n_times"] == source.n_times
    path = PROJECT_ROOT / "configs/brainsync_epoch_support.json"
    assert read_json(path) == record


def test_left_earlobe_reference_export_matches_backend_contract():
    record = contract.brainsync_reference_record()
    assert record == {
        "schema": "brainsync-acquisition-reference/1",
        "label": "left earlobe",
        "side": "left",
        "site": "earlobe",
        "electrode": "ear_clip",
    }
    root = PROJECT_ROOT
    assert read_json(root / "configs/brainsync_acquisition_reference.json") == record


def test_frontend_reference_matches_backend_contract(frontend_runtime):
    frontend, _ = frontend_runtime
    path = frontend / "models/acquisition_reference.json"
    assert read_json(path) == contract.brainsync_reference_record()


def test_cz_reference_removes_one_dimension_and_transmits_cz_noise():
    matrix = np.delete(np.eye(7) - np.eye(7)[1], 1, axis=0)
    assert np.linalg.matrix_rank(matrix) == 6
    np.testing.assert_array_equal(matrix @ np.ones(7), np.zeros(6))
    source = np.arange(21, dtype=float).reshape(7, 3)
    common = np.array([2.0, 5.0, -3.0])
    np.testing.assert_array_equal(matrix @ (source + common), matrix @ source)
    cz_noise = np.zeros_like(source)
    cz_noise[1] = common
    np.testing.assert_array_equal(matrix @ cz_noise, np.broadcast_to(-common, (6, 3)))
    difference = matrix @ source
    reconstructed = np.insert(difference + source[1], 1, source[1], axis=0)
    np.testing.assert_array_equal(reconstructed, source)


def test_actual_frontend_ten_block_schedule_has_450_complete_backend_epochs(frontend_runtime):
    from data.preprocess import preprocess

    frontend, executable = frontend_runtime
    code = textwrap.dedent("""
        import asyncio, json
        from types import SimpleNamespace
        from unittest.mock import patch
        from gtn_workstation.app import configure_runtime_paths
        configure_runtime_paths()
        from PySide6.QtCore import QCoreApplication
        from gtn_workstation.runtime import ExperimentController
        from gtn_workstation.models import ExperimentConfig
        app = QCoreApplication([])
        clock = SimpleNamespace(now=10.0)
        events = []
        async def synthetic_sample_support(_stage, _request):
            # This fixture tests scheduling and backend epoch geometry, not acquisition readiness.
            return None
        controller = ExperimentController(SimpleNamespace(event=lambda *_: None),
                                          recording_support=synthetic_sample_support)
        controller._recording_mode = True
        controller.stimulus_event.connect(events.append)
        async def advance(deadline):
            clock.now = max(clock.now, deadline)
            return False
        controller._wait_or_pause = advance
        with patch('gtn_workstation.runtime.time.monotonic', side_effect=lambda: clock.now):
            asyncio.run(controller._run(ExperimentConfig(blocks=10, digits_per_block=45)))
        print(json.dumps(dict(events=events, end=clock.now)))
    """)
    payload = json.loads(
        subprocess.check_output([str(executable), "-c", code], cwd=frontend, text=True)
    )
    onsets = [row for row in payload["events"] if row["kind"] == "onset"]
    rest_starts = [
        row["software_rest_start_monotonic"] - 9
        for row in payload["events"]
        if row["kind"] == "rest_start"
    ]
    rest_ends = [
        row["software_rest_end_monotonic"] - 9
        for row in payload["events"]
        if row["kind"] == "rest_end"
    ]
    assert len(onsets) == 450 and len(rest_starts) == len(rest_ends) == 9
    samples = np.array([round((row["software_onset_monotonic"] - 9) * 250) for row in onsets])
    assert payload["end"] - onsets[-1]["software_onset_monotonic"] >= 1.2
    signal = np.random.default_rng(81).normal(0, 2e-6, round((payload["end"] - 9 + 1) * 250))
    raw = mne.io.RawArray(signal[None, :], mne.create_info(["Cz"], 250, "eeg"), verbose=False)
    raw.set_annotations(
        mne.Annotations(rest_starts, np.array(rest_ends) - rest_starts, ["BAD_brainsync_rest"] * 9)
    )
    result = preprocess(
        raw,
        np.column_stack([samples, np.zeros(450, dtype=int), np.ones(450, dtype=int)]),
        sfreq=128,
        l_freq=0.1,
        h_freq=30,
        tmin=-0.2,
        tmax=1.2,
        n_times=179,
        filter_phase="forward",
        causal_iir_initial_state=contract.CAUSAL_IIR_INITIAL_STATE,
    )
    assert result.n_epochs == 450
    np.testing.assert_array_equal(np.bincount(result.event_indices // 45), np.full(10, 45))
