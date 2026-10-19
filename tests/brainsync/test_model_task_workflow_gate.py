"""One actual CLI control for a readable numeric-only personal artifact."""
from __future__ import annotations

import json
from types import MethodType, SimpleNamespace

import pytest

from experiments import brainsync_model_task as task
from transfer.calibrated import CalibratedPredictor


def test_inspect_checks_actual_workflow_gate_before_any_workflow_metadata_dereference(tmp_path, monkeypatch):
    artifact = tmp_path/'numeric_only.pt'
    artifact.write_bytes(b'existing readable numeric artifact intercepted only at the numerical loader')
    numeric = SimpleNamespace(metadata={})
    numeric.validate_personal_workflow_metadata = MethodType(CalibratedPredictor.validate_personal_workflow_metadata, numeric)
    seen = []
    def load(stream, *, device):
        assert stream.read() == artifact.read_bytes()
        seen.append(device)
        return numeric
    monkeypatch.setattr(task, 'load_calibrated_predictor', load)
    output = tmp_path/'inspect'
    with pytest.raises(SystemExit) as failure:
        task.main(['--task', 'inspect', '--model-artifact', str(artifact), '--output-dir', str(output)])
    record = json.loads((output/'summary.json').read_bytes())
    assert failure.value.code == 1 and seen == ['cpu']
    assert record['status'] == 'failed' and record['error']['type'] == 'ValueError'
    assert 'personal workflow metadata is missing' in record['error']['message']
    assert 'subject_id' in record['error']['message'] and 'input_request' in record['error']['message']
