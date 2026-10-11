import hashlib
import json

import numpy as np
import pytest

from data.bi2014a_candidate import (
    build_bi2014a_prepared_participant,
    load_bi2014a_prepared_participant,
)
from data.epochs import save_epoch_dataset
from tests.data.test_bi2014a_scoped_preparation import spec, subject


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
