"""Numeric persistence stays general; the personal application owns workflow admission."""
from __future__ import annotations

import json

import pytest
import torch

from baselines.features import downsampled_time_points
from experiments.brainsync_model_task import main as model_task
from research.contracts import semantic_sha256
from tests.support.datasets import binary_dataset
from transfer.calibrated import (
    CalibratedPredictor,
    _input_contract,
    _state_digest,
    load_calibrated_predictor,
    save_calibrated_predictor,
)


def _numeric_predictor():
    dataset = binary_dataset(count=4)
    features = dataset.n_channels * downsampled_time_points(
        dataset.n_times, dataset.preprocessing.sfreq, target_hz=20.)
    metadata = {'model_kind': 'target_only_shrinkage_lda',
                'input_contract': _input_contract(dataset),
                'participant_key': dataset.identity_table.authority_keys()[0]}
    state = {'coefficient': torch.ones(features), 'feature_mean': torch.zeros(features),
             'feature_scale': torch.ones(features), 'intercept': torch.zeros(1)}
    return CalibratedPredictor(metadata, state)


def test_minimal_numeric_codec_roundtrip_preserves_absent_workflow_metadata(tmp_path):
    predictor = _numeric_predictor()
    restored = load_calibrated_predictor(save_calibrated_predictor(tmp_path / 'numeric.pt', predictor))
    assert restored.metadata == predictor.metadata
    assert 'selection' not in restored.metadata
    for name, value in predictor.state.items():
        assert torch.equal(value, restored.state[name])


@pytest.mark.parametrize('schema', ['n2p3_calibrated_predictor/1', 'n2p3_calibrated_predictor/3',
                                   'n2p3_calibrated_predictor/4', 'n2p3_calibrated_predictor/5'])
def test_historical_and_current_numeric_schema_load_without_personal_selection(tmp_path, schema):
    predictor = _numeric_predictor()
    path = tmp_path / 'numeric.pt'
    torch.save({'schema': schema, 'metadata': predictor.metadata,
                'metadata_digest': semantic_sha256(predictor.metadata), 'state': predictor.state,
                'state_digest': _state_digest(predictor.state),
                'selection_contract': 'n2p3_fixed_or_historical_calibration/1'}, path)
    assert load_calibrated_predictor(path).metadata == predictor.metadata


def test_actual_personal_cli_rejects_numeric_only_artifact_at_workflow_gate(tmp_path):
    path = save_calibrated_predictor(tmp_path / 'numeric.pt', _numeric_predictor())
    output = tmp_path / 'inspect'
    with pytest.raises(SystemExit) as failed:
        model_task(['--task', 'inspect', '--model-artifact', str(path), '--output-dir', str(output)])
    assert failed.value.code == 1
    record = json.loads((output / 'summary.json').read_bytes())
    assert record['status'] == 'failed' and record['error']['type'] == 'ValueError'
    assert 'personal workflow metadata is missing' in record['error']['message']
    assert 'selection' in record['error']['message']
