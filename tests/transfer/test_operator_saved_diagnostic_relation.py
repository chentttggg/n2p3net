"""Re-signed derived-metadata corruption and an absent-field legacy positive."""
import json

import numpy as np
import pytest

from transfer.noise_model import JointNoiseModel, metadata_digest
from transfer.operator_lda import OperatorAwareLDA


def test_saved_regularization_diagnostic_is_checked_and_legacy_absence_remains_readable(tmp_path):
    raw = np.array([[-1.], [1.], [-1.], [1.]])
    noise = JointNoiseModel.fit(raw, np.empty((4, 0)), fit_scope='source_training',
        scope_id='source', draw_ids=['0', '1', '2', '3'], shrinkage=0.)
    features, labels = raw+np.array([[0.], [0.], [1.], [1.]]), np.array([0, 0, 1, 1])
    model = OperatorAwareLDA.fit(features, labels, noise=noise, raw_operator=np.eye(1),
        state_operator=np.empty((1, 0)), calibration_ids=['C0', 'C1', 'C2', 'C3'],
        scope_id='C', input_identity='I', ridge_relative=.1)
    path = tmp_path/'original.npz'
    model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != 'metadata'}
        original = json.loads(str(archive['metadata']))
    changed = json.loads(json.dumps(original))
    changed['regularization_diagnostics']['retained_directions'] = 0
    changed['regularization_diagnostics']['ridge_variance_on_retained_support'] = 0.
    changed.pop('artifact_sha256')
    changed['artifact_sha256'] = metadata_digest(changed)
    corrupt = tmp_path/'corrupt.npz'
    np.savez_compressed(corrupt, **arrays, metadata=np.asarray(json.dumps(changed, sort_keys=True)))
    with pytest.raises(ValueError, match='regularization diagnostic'):
        OperatorAwareLDA.load(corrupt)
    # Old schema records without this newly introduced derived field retain
    # their actual numeric payload and original producer provenance.
    legacy = json.loads(json.dumps(original))
    legacy.pop('regularization_diagnostics')
    legacy.pop('artifact_sha256')
    legacy['artifact_sha256'] = metadata_digest(legacy)
    legacy_path = tmp_path/'legacy-missing-field.npz'
    np.savez_compressed(legacy_path, **arrays, metadata=np.asarray(json.dumps(legacy, sort_keys=True)))
    loaded = OperatorAwareLDA.load(legacy_path)
    assert loaded.provenance == model.provenance
    assert loaded.regularization_diagnostics == model.regularization_diagnostics
    np.testing.assert_array_equal(loaded.predict_logit(features, input_identity='I'),
        model.predict_logit(features, input_identity='I'))
