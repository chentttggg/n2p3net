from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import block_diag

from tests.transfer.test_continuous_fixed_prefix import (
    analytic_model,
    dense_noise,
    diffuse_density,
    score,
)
from transfer.continuous_decision import (
    _conditional_prediction,
    _ConditionalRecordWhitening,
    _event_design,
)


@pytest.mark.parametrize('order', [0, 8, 32])
@pytest.mark.parametrize('early_events', [False, True])
def test_common_whitening_preserves_all_candidate_prefix_scatter_and_logdet_bitwise(order, early_events):
    model, samples, onsets, membership = analytic_model(order)
    if not early_events:
        onsets = np.array([33, 44, 58, 72])
    designs = [_event_design(onsets, target, len(samples), model.basis) for target in membership.T]
    prepared = _ConditionalRecordWhitening(samples, designs[0][:, :model.basis.shape[1]],
                                           model.noise, model.conditioning_samples)
    for design in designs:
        original = _conditional_prediction([samples], [design], model.noise, model.conditioning_samples,
            model.kernel_mean, model.kernel_row_covariance)
        actual = prepared.predict(design, model.kernel_mean, model.kernel_row_covariance)
        np.testing.assert_array_equal(actual[0], original[0])
        assert actual[1:3] == original[1:3]


def test_score_call_owns_whitening_and_never_reuses_another_view_or_noise(monkeypatch):
    from transfer.temporal_noise import TemporalNoise

    model, samples, onsets, membership = analytic_model(8)
    calls = []
    original = TemporalNoise.whiten

    def record(self, value):
        calls.append((id(self), value.shape))
        return original(self, value)

    monkeypatch.setattr(TemporalNoise, 'whiten', record)
    first = score(model, samples, onsets, membership)
    assert len(calls) == len(model.candidate_ids) + 1
    assert calls[0][1][1] == samples.shape[1] + model.basis.shape[1] + 2
    assert all(value[1][1] == model.basis.shape[1] for value in calls[1:])
    changed = samples.copy()
    changed[0, 0] += .1
    second = score(model, changed, onsets, membership)
    assert len(calls) == 2 * (len(model.candidate_ids) + 1)
    assert not np.array_equal(first.log_evidence, second.log_evidence)
    another, _, _, _ = analytic_model(32)
    score(another, samples, onsets, membership)
    assert calls[-1][0] != calls[0][0]


def test_full_record_startup_is_kept_and_candidate_prefix_is_not_assumed_common():
    model, samples, onsets, membership = analytic_model(8)
    first = _event_design(onsets, membership[:, 0], len(samples), model.basis)
    other = _event_design(onsets, membership[:, 2], len(samples), model.basis)
    common = _ConditionalRecordWhitening(samples, first[:, :model.basis.shape[1]], model.noise, 32)
    before = common.predict(first, model.kernel_mean, model.kernel_row_covariance)
    after = common.predict(other, model.kernel_mean, model.kernel_row_covariance)
    assert not np.array_equal(before[0], after[0])
    actual = model.noise.whiten(first)[32:]
    restarted = model.noise.whiten(first[32:])
    assert not np.array_equal(actual, restarted)


@pytest.mark.parametrize('change', ['shape', 'scope', 'membership', 'layout'])
def test_public_score_keeps_current_record_and_operator_gates(change):
    model, samples, onsets, membership = analytic_model(8)
    settings = dict(operator_identity=model.operator_identity, sfreq=model.sfreq, channels=model.channels,
                    candidate_ids=model.candidate_ids, record_id='actual-held')
    if change == 'shape':
        samples = samples[:, :1]
    elif change == 'scope':
        settings['record_id'] = ''
    elif change == 'membership':
        membership = membership[:, :2]
    else:
        settings['operator_identity'] = 'unknown-projection'
    with pytest.raises(ValueError):
        model.score(samples, onsets, membership, **settings)


def test_multirecord_joint_posterior_keeps_one_kernel_and_independent_drifts():
    model, samples, onsets, membership = analytic_model(8)
    design = _event_design(onsets, membership[:, 0], len(samples), model.basis)
    second = samples * 1.2
    first = _conditional_prediction([samples, second], [design, design], model.noise, 32,
                                    model.kernel_mean, model.kernel_row_covariance, posterior=True)
    drift = np.column_stack((np.ones(len(samples)), (2*np.arange(len(samples))-31)/31))
    time = dense_noise(model.noise.coefficients, len(samples))
    full = diffuse_density(np.vstack((samples,second)), np.vstack((design,design)), block_diag(drift,drift),
        block_diag(time,time), model.kernel_mean, model.kernel_row_covariance, model.innovation_covariance)
    prefix = diffuse_density(np.vstack((samples[:32],second[:32])), np.vstack((design[:32],design[:32])),
        block_diag(drift[:32],drift[:32]), block_diag(time[:32,:32],time[:32,:32]),
        model.kernel_mean, model.kernel_row_covariance, model.innovation_covariance)
    scatter, logdet, n_tail, _, _ = first
    value = -.5 * (np.trace(np.linalg.solve(model.innovation_covariance,scatter)) + 2*logdet
                   + n_tail*(np.linalg.slogdet(model.innovation_covariance)[1]+2*np.log(2*np.pi)))
    np.testing.assert_allclose(value, full-prefix, rtol=1e-10, atol=3e-9)
    assert first[2] == 2 * (len(samples) - 32)
    with pytest.raises(ValueError):
        _conditional_prediction([samples, second], [design], model.noise, 32,
                                model.kernel_mean, model.kernel_row_covariance)


def test_analytic_metadata_and_numeric_precision_are_unchanged_by_scoring():
    model, samples, onsets, membership = analytic_model(8)
    before = model._metadata()
    score(model, samples, onsets, membership)
    assert model._metadata() == before
    with pytest.raises(ValueError):
        replace(model, kernel_row_covariance=np.zeros_like(model.kernel_row_covariance))
