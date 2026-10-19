"""The actual report owner keeps a legal frozen-QC rejection explanatory."""
from __future__ import annotations

import json
from types import SimpleNamespace

from data.brainsync import BrainSyncNoUsableEvidenceError
from data.brainsync_contract import validate_brainsync_bids_recording
from experiments import report_brainsync_quality as report
from tests.support.brainsync import write_bids_session


def test_configured_model_full_qc_rejection_keeps_original_report_status_and_denominator(tmp_path, monkeypatch):
    root = write_bids_session(tmp_path/'session', session_id='full-qc-rejection', target=2,
        started_utc='2026-10-19T00:00:00Z', blocks=(tuple(range(1, 10))*5,))
    artifact = tmp_path/'frozen_model.pt'
    artifact.write_bytes(b'existing frozen numerical model intercepted only at its loader')
    actual_ledger = {'role': 'synthetic_frozen_QC_rejection', 'scheduled_trials': 45}
    def reject_after_original_admission(path):
        validated = validate_brainsync_bids_recording(path)
        try:
            failure = BrainSyncNoUsableEvidenceError(validated, actual_ledger)
        finally:
            validated.bids.close()
        raise failure
    predictor = SimpleNamespace(epoch_quality=None, load_inference_session=reject_after_original_admission)
    monkeypatch.setattr(report, 'load_calibrated_predictor', lambda *args, **kwargs: predictor)
    output = tmp_path/'report'
    result = report.main(['--session-dir', str(root), '--output-dir', str(output), '--model-artifact', str(artifact)])
    saved = json.loads((output/'report.json').read_bytes())
    assert saved == json.loads(json.dumps(result))
    assert saved['status'] == 'not_evaluable' and 'error' not in saved
    assert saved['continuous_quality'] == actual_ledger
    assert saved['counts'] == {'scheduled_trials': 45, 'preprocessed_epochs': 0,
        'preprocessing_unavailable': 45, 'qc_retained_epochs': 0}
    assert saved['target_labels_used'] is False and saved['fit_performed'] is False
    assert saved['classifier_evaluated'] is False and (output/'report.html').is_file()
