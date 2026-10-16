"""Cached reexport retains the source law while admitting a new excluded target."""

from dataclasses import replace

import numpy as np
import pytest

from research.channel_noise_mechanism import FittedChannelNoise, FrozenERPResponse
from research.joint_flat_authority import (
    build_source_parameters,
    source_parameters_from_saved_model,
)
from tests.research.test_joint_flat_response import compiler_fixture
from transfer.continuous_decision import ContinuousDecisionModel
from transfer.joint_flat_programme import FrozenJointFlatSource
from transfer.noise_model import metadata_digest


def test_cached_source_reexport_preserves_original_law_without_factory_or_rng(tmp_path, monkeypatch):
    compiler = compiler_fixture('A01.mat', 840)
    target = compiler_fixture('A02.mat', 841)
    cohort = ((compiler, compiler.compile(120)),)
    path = tmp_path/'source.npz'
    parameters, _ = build_source_parameters(cohort, target_participant_key='first-target',
        degrees_of_freedom=4., relative_noise_shape=1., basis=np.ones((8, 1)), ridge=1.,
        noise_order=0, conditioning_samples=4, source_model_path=path)
    frozen = FrozenJointFlatSource.from_parameters(parameters, original_model_content=path.read_bytes())

    def forbidden(*_args, **_kwargs):
        raise AssertionError('cached source reexport called a factory, fit or RNG')

    monkeypatch.setattr(FittedChannelNoise, 'from_temporal_noise', forbidden)
    monkeypatch.setattr(FittedChannelNoise, 'fit', forbidden)
    monkeypatch.setattr(FrozenERPResponse, 'from_continuous_model', forbidden)
    monkeypatch.setattr(ContinuousDecisionModel, 'fit', forbidden)
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    arguments = dict(target_participant_key='second-target', degrees_of_freedom=4.,
        relative_noise_shape=1., source_model_path=path, frozen_source=frozen)
    rebound, model = source_parameters_from_saved_model(cohort, **arguments)
    assert rebound.prior is frozen.parameters.prior and rebound.noise is frozen.parameters.noise
    assert rebound.hyperparameters is frozen.parameters.hyperparameters
    np.testing.assert_array_equal(rebound.basis, frozen.parameters.basis)
    assert model.digest() == frozen.parameters.binding['source_model_digest']
    assert metadata_digest({**rebound.binding, 'excluded_target_participant': 'first-target'}) == parameters.digest
    assert rebound.binding['excluded_target_participant'] == 'second-target'
    with pytest.raises(ValueError, match='excluded target'):
        source_parameters_from_saved_model(cohort, **{**arguments,
            'target_participant_key': cohort[0][1].target_contract['participant_key']})
    with pytest.raises(ValueError, match='complete frozen parameter law'):
        source_parameters_from_saved_model(cohort, **{**arguments, 'relative_noise_shape': 2.})
    from transfer import ensemble_fit
    from transfer.ensemble import EnsembleProgram
    from transfer.joint_flat_programme import JointFlatProgram, JointFlatSamplingProgram

    sampler = JointFlatSamplingProgram(4., 8, 8, (2, 3), 1, 2, 1., 1.2, 0., 0., .1)
    spec = ensemble_fit.CalibrationExpertProgram('joint', 'joint_flat',
        joint_flat=JointFlatProgram(sampler, 2, 'whole_observed_current_q'))
    prefix, cache = target.compile(0), {}
    first = ensemble_fit.fit_calibration_prefix(prefix, (spec,), EnsembleProgram.equal_weights('original', ('joint',)),
        source_checkpoints={'joint': frozen}, prefix_compiler=target, _prefix_fit_cache=cache)
    monkeypatch.setattr(ensemble_fit, '_fit_expert', forbidden)
    second = ensemble_fit.fit_calibration_prefix(prefix, (spec,), EnsembleProgram.equal_weights('consumer', ('joint',)),
        source_checkpoints={'joint': frozen}, prefix_compiler=target, _prefix_fit_cache=cache)
    assert first._experts[0] is second._experts[0]
    assert first._experts[0].predictor.provenance == second._experts[0].predictor.provenance
    # A matching dictionary key cannot admit a changed native target producer.
    original_expert = first._experts[0]
    changed = replace(original_expert.predictor, provenance={**original_expert.predictor.provenance,
        'requested_labels': 1})
    cache[original_expert.fit_digest] = replace(original_expert, predictor=changed, model_digest=changed.digest())
    with pytest.raises(ValueError, match='joint actual C posterior'):
        ensemble_fit.fit_calibration_prefix(prefix, (spec,), EnsembleProgram.equal_weights('poisoned', ('joint',)),
            source_checkpoints={'joint': frozen}, prefix_compiler=target, _prefix_fit_cache=cache)
    changed = replace(original_expert.predictor, provenance={**original_expert.predictor.provenance,
        'ensemble_fit_binding': {**original_expert.predictor.provenance['ensemble_fit_binding'],
            'OOF_probability_sha256': '1'*64}})
    cache[original_expert.fit_digest] = replace(original_expert, predictor=changed, model_digest=changed.digest())
    with pytest.raises(ValueError, match='uncalibrated base model digest'):
        ensemble_fit.fit_calibration_prefix(prefix, (spec,), EnsembleProgram.equal_weights('OOF-poisoned', ('joint',)),
            source_checkpoints={'joint': frozen}, prefix_compiler=target, _prefix_fit_cache=cache)
    # A valid NPZ with a changed byte envelope still decodes to the same model.
    path.write_bytes(path.read_bytes()+b'changed-original-envelope')
    with pytest.raises(ValueError, match='original NPZ'):
        source_parameters_from_saved_model(cohort, **arguments)
