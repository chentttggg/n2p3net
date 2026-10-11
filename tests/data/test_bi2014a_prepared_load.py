import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from data.bi2014a_candidate import (
    build_bi2014a_prepared_participant,
    load_bi2014a_prepared_participant,
)
from data.epochs import save_epoch_dataset
from tests.data.test_bi2014a_scoped_preparation import spec, subject


@pytest.fixture(scope='module')
def sealed_episode_pair(tmp_path_factory):
    root = tmp_path_factory.mktemp('bi-episode-mapping')
    directory, _ = subject(root, tail=2)
    prepared = build_bi2014a_prepared_participant(directory, preprocessing=spec())
    output = root / 'prepared'
    output.mkdir()
    ledger = deepcopy(prepared.ledger)
    for role in ('calibration', 'future'):
        path = save_epoch_dataset(output / f'{role}.npz', getattr(prepared, role), compressed=False)
        ledger[f'{role}_cache'] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / 'preparation.json').write_text(json.dumps(ledger))
    return output, ledger


@pytest.mark.parametrize('changed', ['other_episode_indices', 'episode_id', 'raw_depth', 'censor', 'horizon'])
def test_readonly_loader_binds_every_descriptor_to_actual_future_timeline(sealed_episode_pair, changed):
    output, original = sealed_episode_pair
    ledger = deepcopy(original)
    first = ledger['future_decisions'][0]
    if changed == 'other_episode_indices':
        first['membership_decision']['evidence_indices'] = ledger['future_decisions'][1]['membership_decision']['evidence_indices']
    elif changed == 'episode_id':
        first['episode_id'] = ledger['future_decisions'][1]['episode_id']
    elif changed == 'raw_depth':
        first['complete_raw_repetitions'] += 1
    elif changed == 'censor':
        first['terminal_censor'] = not first['terminal_censor']
    else:
        first['horizons'][0]['decision']['evidence_indices'] = ledger['future_decisions'][1]['horizons'][0]['decision']['evidence_indices']
    path = output / 'preparation.json'
    try:
        path.write_text(json.dumps(ledger))
        # The immutable numeric caches still match. The decoder must also bind
        # the external episode description to its actual scheduled rows.
        for role in ('calibration', 'future'):
            assert hashlib.sha256((output / f'{role}.npz').read_bytes()).hexdigest() == original[f'{role}_cache']['sha256']
        with pytest.raises(ValueError, match='physical decision ledger'):
            load_bi2014a_prepared_participant(output)
    finally:
        path.write_text(json.dumps(original))


def test_readonly_loader_preserves_terminal_partial_and_original_horizon_manifest(sealed_episode_pair):
    output, original = sealed_episode_pair
    loaded = load_bi2014a_prepared_participant(output)
    assert not loaded.future_decisions[-1].schedule_complete
    assert loaded.ledger['future_decisions'] == original['future_decisions']
    legacy = deepcopy(original)
    legacy.pop('maximum_horizon', None)
    path = output / 'preparation.json'
    try:
        path.write_text(json.dumps(legacy))
        assert [value.record() for value in load_bi2014a_prepared_participant(output).future_decisions] == [
            value.record() for value in loaded.future_decisions]
    finally:
        path.write_text(json.dumps(original))


def test_descriptor_cannot_infer_a_complete_raw_horizon_from_unknown_schedule(sealed_episode_pair):
    from data.bi2014a_candidate import _prepared_future_decisions

    output, _ = sealed_episode_pair
    future = load_bi2014a_prepared_participant(output).future
    unknown = replace(future, event_timeline=replace(future.event_timeline, complete=False))
    with pytest.raises(ValueError, match='known original schedule'):
        _prepared_future_decisions(unknown, 8)


def test_prepared_load_binds_original_pair_and_hidden_future(tmp_path):
    directory, _ = subject(tmp_path)
    prepared = build_bi2014a_prepared_participant(directory, preprocessing=spec())
    output = tmp_path / 'prepared'
    output.mkdir()
    ledger = dict(prepared.ledger)
    for role in ['calibration', 'future']:
        path = save_epoch_dataset(output / f'{role}.npz', getattr(prepared, role), compressed=False)
        ledger[f'{role}_cache'] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / 'preparation.json').write_text(json.dumps(ledger))
    loaded = load_bi2014a_prepared_participant(output)
    np.testing.assert_array_equal(loaded.calibration.X, prepared.calibration.X)
    assert loaded.future.y is None
    assert [value.record() for value in loaded.future_decisions] == [value.record() for value in prepared.future_decisions]
    (output / 'future.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='sealed preparation'):
        load_bi2014a_prepared_participant(output)
