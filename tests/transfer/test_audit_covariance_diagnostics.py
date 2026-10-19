"""Independent sample-rank and score-variance controls, no EEG or model training."""
import json

import numpy as np
import pytest

from transfer.noise_model import JointNoiseModel
from transfer.operator_lda import ConditioningObservationSupport, OperatorAwareLDA


def test_three_rows_do_not_identify_every_shrinkage_covariance_direction(tmp_path):
    # Any centered three-row matrix has rank <= 2. A positive shrinkage
    # covariance may be full rank, which is a different statement.
    raw = np.array([[1., 0., 2., 0.], [0., 1., 0., 0.], [-1., -1., -2., 0.]])
    noise = JointNoiseModel.fit(raw, np.empty((3, 0)), fit_scope="source_training",
        scope_id="three-rows", draw_ids=['a', 'b', 'c'], shrinkage=1.)
    diagnostic = noise.estimation_diagnostics()
    assert diagnostic['draws_per_joint_dimension'] == .75
    assert diagnostic['empirical_resolution']['active_dimensions'] == 3
    assert diagnostic['empirical_resolution']['centered_rank_upper_bound'] == 2
    assert diagnostic['empirical_null_directions_at_least'] == 1
    assert diagnostic['shrinkage_fills_empirical_null_directions'] is True
    assert np.linalg.matrix_rank(noise.covariance) == 3
    assert noise.covariance[-1, -1] == 0  # no fabricated variance for the constant coordinate
    assert diagnostic['statistical_accuracy_certified'] is False
    path = tmp_path/'three.npz'
    noise.save(path)
    loaded = JointNoiseModel.load(path)
    assert loaded.estimation_diagnostics() == diagnostic
    assert loaded.provenance == noise.provenance
    json.dumps(diagnostic, allow_nan=False)


def test_old_numeric_provenance_does_not_acquire_empirical_evidence():
    raw = np.array([[1., 2.], [-1., -2.], [0., 0.]])
    current = JointNoiseModel.fit(raw, np.empty((3, 0)), fit_scope="source_training",
        scope_id="old-producer", draw_ids=['a', 'b', 'c'], shrinkage=.5)
    provenance = json.loads(json.dumps(current.provenance))
    provenance.pop('empirical_resolution')
    provenance['implementation_sha256'] = '1'*64
    old = JointNoiseModel(current.mean, current.covariance, 2, provenance)
    assert old.provenance['implementation_sha256'] == '1'*64
    diagnostic = old.estimation_diagnostics()
    assert diagnostic['empirical_resolution'] is None
    assert diagnostic['shrinkage_fills_empirical_null_directions'] is None


def test_ridge_regularizes_inside_the_same_covariance_support_and_score_ratio():
    # Orthogonal zero-mean rows yield a diagonal empirical covariance with
    # eigenvalues 1 and 1e-9. Both remain represented even when ridge strongly
    # attenuates precision in the weak direction; support is not ridge-sized.
    raw = np.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
    raw[:, 1] *= np.sqrt(1e-9)
    noise = JointNoiseModel.fit(raw, np.empty((4, 0)), fit_scope="source_training",
        scope_id="known-diagonal", draw_ids=['a', 'b', 'c', 'd'], shrinkage=0.)
    labels = np.array([0, 0, 1, 1])
    X = raw+labels[:, None]*np.array([.1, .001])
    kwargs = dict(noise=noise, raw_operator=np.eye(2), state_operator=np.empty((2, 0)),
        calibration_ids=['c0', 'c1', 'c2', 'c3'], scope_id='C', input_identity='I')
    unregularized = OperatorAwareLDA.fit(X, labels, ridge_relative=0., **kwargs)
    model = OperatorAwareLDA.fit(X, labels, ridge_relative=1e-3, **kwargs)
    np.testing.assert_array_equal(model.identifiable_projector, unregularized.identifiable_projector)
    diagnostic = model.regularization_diagnostics
    assert diagnostic['retained_directions'] == 2
    eigen = np.array([1e-9, 1.])
    ridge = 1e-3*eigen.mean()
    np.testing.assert_allclose(diagnostic['directional_precision_retention'], eigen/(eigen+ridge))
    report = model.calibration_diagnostic(X, labels, calibration_ids=kwargs['calibration_ids'])
    residual = X-np.stack([X[labels == k].mean(axis=0) for k in (0, 1)])[labels]
    empirical = np.sum((residual@model.weights)**2)/(len(X)-2)
    predicted = float(model.weights@noise.covariance@model.weights)
    assert report['empirical_to_noise_score_variance'] == pytest.approx(empirical/predicted)
    assert report['inference_role'].endswith('no_acceptance_gate')
    changed_operator = OperatorAwareLDA.fit(X, labels, ridge_relative=1e-3,
        **{**kwargs, 'raw_operator': 2*np.eye(2)})
    changed_report = changed_operator.calibration_diagnostic(X, labels,
        calibration_ids=kwargs['calibration_ids'])
    assert changed_report['empirical_to_noise_score_variance'] == pytest.approx(
        report['empirical_to_noise_score_variance']/4)
    with pytest.raises(ValueError, match='exact original'):
        model.calibration_diagnostic(X*2, labels, calibration_ids=kwargs['calibration_ids'])


def test_identical_class_means_have_no_undefined_ratio_in_json():
    raw = np.array([[-1.], [1.], [-1.], [1.]])
    noise = JointNoiseModel.fit(raw, np.empty((4, 0)), fit_scope='source_training',
        scope_id='source', draw_ids=['0', '1', '2', '3'], shrinkage=0.)
    labels = np.array([0, 0, 1, 1])
    model = OperatorAwareLDA.fit(raw, labels, noise=noise, raw_operator=np.eye(1),
        state_operator=np.empty((1, 0)), calibration_ids=['0', '1', '2', '3'],
        scope_id='C', input_identity='I')
    report = model.calibration_diagnostic(raw, labels, calibration_ids=['0', '1', '2', '3'])
    assert report['empirical_to_noise_score_variance'] is None
    json.dumps(report, allow_nan=False)


def test_conditional_score_diagnostic_uses_original_raw_and_unridged_schur(tmp_path):
    raw = np.array([[-1.], [-1.], [1.], [1.], [-1.], [-1.], [1.], [1.]])
    independent = np.array([[-1.], [1.], [-1.], [1.], [1.], [-1.], [1.], [-1.]])
    state = .4*raw+.2*independent
    noise = JointNoiseModel.fit(raw, state, fit_scope='source_training', scope_id='S',
        draw_ids=list(map(str, range(8))), shrinkage=0.)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    features = state+.3*labels[:, None]
    ids = [f'C{i}' for i in range(8)]
    model = OperatorAwareLDA.fit(features, labels, noise=noise, raw_operator=np.zeros((1, 1)),
        state_operator=np.eye(1), conditioning_features=raw,
        observation_support=ConditioningObservationSupport((0, 1), (1, 2), 2),
        calibration_ids=ids, scope_id='C', input_identity='I', ridge_relative=.5)
    diagnostic = model.calibration_diagnostic(features, labels, calibration_ids=ids, observed_raw=raw)
    residual = features-model.class_means[labels]-(raw-model.conditioning_class_means[labels])@model.conditioning_gain.T
    score_variance = np.sum((residual@model.weights)**2)/6
    schur = noise.covariance[1, 1]-noise.covariance[0, 1]**2/noise.covariance[0, 0]
    assert diagnostic['empirical_score_variance'] == pytest.approx(score_variance)
    assert diagnostic['predicted_noise_score_variance'] == pytest.approx(model.weights[0]**2*schur)
    assert diagnostic['regularized_score_variance'] > diagnostic['predicted_noise_score_variance']
    path = tmp_path/'conditional.npz'
    model.save(path)
    restored = OperatorAwareLDA.load(path)
    assert restored.calibration_diagnostic(features, labels, calibration_ids=ids, observed_raw=raw) == diagnostic
    with pytest.raises(ValueError, match='exact original'):
        restored.calibration_diagnostic(features, labels, calibration_ids=ids, observed_raw=raw+.01)
