"""The named workstation entry binds a source and rejects processing drift."""
import hashlib
import json
from pathlib import Path

import pytest

from transfer.workstation_profiles import load_old_basic_profile, validate_executed_profile_source


@pytest.fixture
def profile(tmp_path):
    root = Path(__file__).resolve().parents[2]
    record = json.loads((root / 'configs/brainsync_old_basic_nn_source.json').read_text())
    source = tmp_path / 'source.pt'
    source.write_bytes(b'fixed source identity')
    record['source']['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    record['source']['path'] = source.name
    record['preprocessing'] = str(root / 'configs/brainsync_bidirectional_preprocessing.json')
    record['training_budget'] = str(root / 'configs/unified_training_budget.json')
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(record))
    return path, source


def test_profile_preserves_actual_full_calibration_program_and_pins_source(profile):
    path, source = profile
    result = load_old_basic_profile(path, source)
    validate_executed_profile_source(result['record'], hashlib.sha256(source.read_bytes()).hexdigest())
    recipe = result['recipe']
    assert recipe.initialization == 'source' and recipe.epochs == 90
    assert recipe.batch_size == 512 and recipe.lr == 0.001
    assert recipe.bn_buffers == 'calibration' and recipe.dropout == 'off'
    assert recipe.input_statistics == 'target_prefix' and recipe.candidate_loss_weight == 0
    assert result['source_checkpoint'] == source.resolve()
    source.write_bytes(b'another otherwise readable checkpoint')
    with pytest.raises(ValueError, match='SHA-256'):
        load_old_basic_profile(path, source)
    with pytest.raises(ValueError, match='executed source SHA-256'):
        validate_executed_profile_source(result['record'], hashlib.sha256(source.read_bytes()).hexdigest())


@pytest.mark.parametrize('overrides', [
    ['--ocular-correction'], ['--epoch-qc'], ['--reference-mode', 'native'],
    ['--selector-mode', 'unified'], ['--seed', '3'], ['--preprocessing-config', 'other.json'],
])
def test_cli_profile_rejects_processing_and_recipe_overrides(tmp_path, overrides):
    from experiments.brainsync_model_task import main
    with pytest.raises(SystemExit) as error:
        main(['--task', 'calibrate', '--calibration-profile', 'old_basic_nn_source',
              '--model-artifact', str(tmp_path / 'model.pt'),
              '--output-dir', str(tmp_path / 'run'), *overrides])
    assert error.value.code == 2
    assert not (tmp_path / 'model.pt').exists()


def test_wrong_source_is_reported_before_any_fitting(tmp_path):
    from experiments.brainsync_model_task import main
    source = tmp_path / 'wrong.pt'
    source.write_bytes(b'wrong weights')
    with pytest.raises(SystemExit) as error:
        main(['--task', 'calibrate', '--calibration-profile', 'old_basic_nn_source',
              '--source-checkpoint', str(source), '--model-artifact', str(tmp_path / 'model.pt'),
              '--output-dir', str(tmp_path / 'run')])
    assert error.value.code == 1
    result = json.loads((tmp_path / 'run/summary.json').read_bytes())
    assert result['status'] == 'failed' and 'SHA-256' in result['error']['message']
    assert not (tmp_path / 'model.pt').exists()
