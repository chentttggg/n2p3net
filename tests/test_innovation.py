from __future__ import annotations

import pytest
import torch

from models.innovation import (
    CausalInnovationDecoder,
    CausalInnovationOutput,
    CausalObservationEncoder,
)
from models.n2p3net import N2P3Net
from train import prequential as prequential_module
from train.contracts import GenerativeProfile
from train.prequential import (
    _gaussian_nll_per_time,
    causal_adaptive_ar1_prediction,
    causal_ar_prediction,
    estimate_generative_profile,
    nested_prequential_training_loss,
    prequential_log_likelihood_ratio,
    prequential_nll,
    prequential_score_per_trial,
)
from train.prequential_audit import _subject_means, audit_prequential_model


def test_default_innovation_tcn_has_registered_receptive_field() -> None:
    encoder = CausalObservationEncoder(3, d_model=8)
    assert encoder.receptive_field == 249


def test_decoder_excludes_current_feature_from_every_moment() -> None:
    torch.manual_seed(3)
    decoder = CausalInnovationDecoder(7, 3, covariance_rank=2).eval()
    with torch.no_grad():
        decoder.mean_projection.weight.normal_(0.0, 0.1)
        decoder.diagonal_projection.weight.normal_(0.0, 0.1)
        decoder.factor_projection.weight.normal_(0.0, 0.1)
    features = torch.randn(2, 2, 40, 7)
    changed = features.clone()
    changed[:, :, 19] += 100.0

    with torch.no_grad():
        original = decoder(features)
        intervened = decoder(changed)

    for first, second in (
        (original.history_correction, intervened.history_correction),
        (original.log_variance_scale, intervened.log_variance_scale),
        (original.factor_scale, intervened.factor_scale),
    ):
        time_dim = 3 if first.dim() == 4 and first.shape[2] == 3 else 2
        assert torch.equal(first.select(time_dim, 19), second.select(time_dim, 19))
        assert not torch.equal(first.select(time_dim, 20), second.select(time_dim, 20))


def _small_model() -> N2P3Net:
    model = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        n_time=64,
        tmin_ms=-200.0,
        tmax_ms=800.0,
        sfreq=64.0,
        baseline_mode="none",
        d_model=16,
        temporal_kernels=(13,),
        filters_per_scale=2,
        encoder_depth=1,
        use_innovation_likelihood=True,
        innovation_d_model=8,
        innovation_dilations=(1, 2),
    ).eval()
    with torch.no_grad():
        model.innovation_decoder.mean_projection.weight.normal_(0.0, 0.1)
        model.innovation_decoder.diagonal_projection.weight.normal_(0.0, 0.1)
        model.innovation_decoder.factor_projection.weight.normal_(0.0, 0.1)
    return model


def test_full_likelihood_graph_is_strict_past_at_intervention_time() -> None:
    torch.manual_seed(5)
    model = _small_model()
    observation = torch.randn(2, 3, 64)
    changed = observation.clone()
    changed[:, :, 31] += 50.0
    class_means = torch.randn(2, 3, 64)

    with torch.no_grad():
        original = model(observation, likelihood_class_means=class_means).likelihood
        intervened = model(changed, likelihood_class_means=class_means).likelihood
    assert original is not None and intervened is not None

    original_moments = original.causal_innovation
    changed_moments = intervened.causal_innovation
    assert torch.equal(
        original_moments.history_correction[:, :, :, 31],
        changed_moments.history_correction[:, :, :, 31],
    )
    assert torch.equal(
        original_moments.log_variance_scale[:, :, 31],
        changed_moments.log_variance_scale[:, :, 31],
    )
    assert torch.equal(
        original_moments.factor_scale[:, :, 31],
        changed_moments.factor_scale[:, :, 31],
    )
    assert not torch.equal(
        original_moments.history_correction[:, :, :, 32],
        changed_moments.history_correction[:, :, :, 32],
    )


def test_hypotheses_use_separate_strict_past_histories() -> None:
    torch.manual_seed(7)
    model = _small_model()
    observation = torch.randn(2, 3, 64)
    class_means = torch.zeros(2, 3, 64)
    class_means[1, :, :40] = 1.5
    with torch.no_grad():
        decomposition = model(
            observation,
            likelihood_class_means=class_means,
        ).likelihood
    assert decomposition is not None
    correction = decomposition.causal_innovation.history_correction
    assert not torch.equal(correction[:, 0, :, 20:], correction[:, 1, :, 20:])


def test_woodbury_nll_matches_dense_gaussian() -> None:
    torch.manual_seed(11)
    residual = torch.randn(4, 9, 5, dtype=torch.float64)
    diagonal = torch.rand(4, 9, 5, dtype=torch.float64) + 0.5
    factor = torch.randn(4, 9, 5, 2, dtype=torch.float64) * 0.2
    actual = _gaussian_nll_per_time(residual, diagonal, factor)

    covariance = torch.diag_embed(diagonal) + factor @ factor.transpose(-1, -2)
    chol = torch.linalg.cholesky(covariance)
    solved = torch.cholesky_solve(residual[..., None], chol).squeeze(-1)
    expected = 0.5 * (
        (residual * solved).sum(dim=-1)
        + 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
        + residual.shape[-1] * torch.log(torch.tensor(2.0 * torch.pi, dtype=torch.float64))
    )
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("rank", [1, 2])
def test_woodbury_fast_paths_match_dense_gradients(rank: int) -> None:
    torch.manual_seed(37 + rank)
    residual = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
    diagonal = (torch.rand(2, 5, 4, dtype=torch.float64) + 0.5).requires_grad_()
    factor = (torch.randn(2, 5, 4, rank, dtype=torch.float64) * 0.2).requires_grad_()
    actual = _gaussian_nll_per_time(residual, diagonal, factor).sum()
    actual_gradients = torch.autograd.grad(actual, (residual, diagonal, factor))

    residual_ref = residual.detach().requires_grad_()
    diagonal_ref = diagonal.detach().requires_grad_()
    factor_ref = factor.detach().requires_grad_()
    covariance = torch.diag_embed(diagonal_ref) + factor_ref @ factor_ref.transpose(-1, -2)
    expected = torch.distributions.MultivariateNormal(
        torch.zeros_like(residual_ref), covariance_matrix=covariance
    ).log_prob(residual_ref).neg().sum()
    expected_gradients = torch.autograd.grad(expected, (residual_ref, diagonal_ref, factor_ref))
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.allclose(actual_gradient, expected_gradient, atol=1e-9, rtol=1e-9)


def test_convolutional_var_matches_strict_past_lag_loop_and_gradients() -> None:
    torch.manual_seed(47)
    signal = torch.randn(3, 4, 19, dtype=torch.float64, requires_grad=True)
    coefficients = torch.randn(6, 4, 4, dtype=torch.float64, requires_grad=True)
    actual = causal_ar_prediction(signal, coefficients)
    expected = torch.zeros_like(signal)
    for lag, matrix in enumerate(coefficients, start=1):
        if lag >= signal.shape[-1]:
            break
        expected[:, :, lag:] = expected[:, :, lag:] + torch.einsum(
            "cd,bdt->bct", matrix, signal[:, :, :-lag]
        )
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

    actual_gradients = torch.autograd.grad(actual.square().sum(), (signal, coefficients))
    expected_gradients = torch.autograd.grad(expected.square().sum(), (signal, coefficients))
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.allclose(actual_gradient, expected_gradient, atol=1e-10, rtol=1e-10)


def test_single_label_likelihood_matches_selected_full_hypothesis() -> None:
    torch.manual_seed(53)
    model = _small_model()
    observation = torch.randn(5, 3, 64)
    class_means = torch.randn(2, 3, 64)
    labels = torch.tensor([0, 1, 1, 0, 1])
    with torch.no_grad():
        full = model(observation, likelihood_class_means=class_means).likelihood
        single = model(
            observation,
            likelihood_class_means=class_means,
            likelihood_labels=labels,
        ).likelihood
    assert full is not None and single is not None
    rows = torch.arange(labels.numel())
    assert single.causal_innovation.history_correction.shape[1] == 1
    assert torch.equal(single.causal_innovation.hypothesis_labels, labels)
    assert torch.allclose(
        single.causal_innovation.history_correction[:, 0],
        full.causal_innovation.history_correction[rows, labels],
        atol=1e-7,
    )
    assert torch.allclose(
        single.causal_innovation.log_variance_scale[:, 0],
        full.causal_innovation.log_variance_scale[rows, labels],
        atol=1e-7,
    )
    assert torch.allclose(
        single.causal_innovation.factor_scale[:, 0],
        full.causal_innovation.factor_scale[rows, labels],
        atol=1e-7,
    )
    training = torch.randn(24, 3, 64)
    training_labels = torch.tensor([0, 1] * 12, dtype=torch.float32)
    profile = estimate_generative_profile(
        training,
        training_labels,
        sfreq=64.0,
        tmin_ms=-200.0,
        score_interval_ms=(0.0, 700.0),
        ar_order=2,
    )
    full_loss = nested_prequential_training_loss(
        observation,
        full.causal_innovation,
        profile,
        labels,
        covariance_weight=0.4,
    )
    single_loss = nested_prequential_training_loss(
        observation,
        single.causal_innovation,
        profile,
        labels,
        covariance_weight=0.4,
    )
    assert torch.allclose(single_loss, full_loss, atol=1e-7)


def test_adaptive_ar1_never_uses_current_innovation() -> None:
    torch.manual_seed(13)
    innovation = torch.randn(2, 3, 40)
    changed = innovation.clone()
    changed[:, :, 24] += 100.0
    original = causal_adaptive_ar1_prediction(innovation, min_history=3)
    intervened = causal_adaptive_ar1_prediction(changed, min_history=3)
    assert torch.equal(original[:, :, 24], intervened[:, :, 24])
    assert not torch.equal(original[:, :, 25], intervened[:, :, 25])


def test_profile_rejects_prestimulus_scoring_after_full_baseline_transform() -> None:
    torch.manual_seed(17)
    observations = torch.randn(12, 3, 64)
    labels = torch.tensor([0, 1] * 6, dtype=torch.float32)
    with pytest.raises(ValueError, match="pre-stimulus"):
        estimate_generative_profile(
            observations,
            labels,
            sfreq=64.0,
            tmin_ms=-200.0,
            score_interval_ms=(-100.0, 400.0),
            ar_order=2,
        )


def test_prequential_llr_can_retain_class_conditional_power_difference() -> None:
    torch.manual_seed(19)
    train = torch.cat((torch.randn(20, 2, 40), 2.5 * torch.randn(20, 2, 40)))
    labels = torch.cat((torch.zeros(20), torch.ones(20)))
    profile = estimate_generative_profile(
        train,
        labels,
        sfreq=40.0,
        tmin_ms=0.0,
        score_interval_ms=(0.0, 1000.0),
        ar_order=2,
    )
    observation = 2.5 * torch.randn(6, 2, 40)
    innovation = CausalInnovationOutput(
        history_correction=torch.zeros(6, 2, 2, 40),
        log_variance_scale=torch.zeros(6, 2, 40, 2),
        factor_scale=torch.zeros(6, 2, 40, 2, 2),
    )
    llr = prequential_log_likelihood_ratio(
        observation,
        innovation,
        profile,
        variant="m0",
    )
    assert llr.mean() > 0.0


def test_all_missing_observation_has_zero_llr_and_no_trainable_score() -> None:
    torch.manual_seed(23)
    training = torch.randn(24, 3, 32)
    labels = torch.tensor([0, 1] * 12, dtype=torch.float32)
    profile = estimate_generative_profile(
        training,
        labels,
        sfreq=32.0,
        tmin_ms=0.0,
        score_interval_ms=(0.0, 1000.0),
        ar_order=2,
    )
    observation = torch.full((4, 3, 32), 1e6)
    missing = torch.zeros(4, 3, dtype=torch.bool)
    innovation = CausalInnovationOutput(
        history_correction=torch.randn(4, 2, 3, 32),
        log_variance_scale=torch.randn(4, 2, 32, 3),
        factor_scale=torch.randn(4, 2, 32, 3, 2),
    )

    score = prequential_score_per_trial(
        observation,
        innovation,
        profile,
        hypothesis=0,
        variant="m0",
        observation_mask=missing,
    )
    llr = prequential_log_likelihood_ratio(
        observation,
        innovation,
        profile,
        variant="m0",
        observation_mask=missing,
    )

    assert torch.equal(score.nll_sum, torch.zeros(4))
    assert torch.equal(score.observed_scalar_count, torch.zeros(4))
    assert not score.valid.any()
    assert torch.isnan(score.nll_per_observed_scalar).all()
    assert torch.equal(llr, torch.zeros(4))


def test_end_to_end_all_missing_model_input_cannot_recreate_old_large_llr() -> None:
    torch.manual_seed(27)
    model = _small_model()
    training = torch.randn(24, 3, 64)
    labels = torch.tensor([0, 1] * 12, dtype=torch.float32)
    profile = estimate_generative_profile(
        training,
        labels,
        sfreq=64.0,
        tmin_ms=-200.0,
        score_interval_ms=(0.0, 700.0),
        ar_order=2,
    )
    missing = torch.zeros(2, 3, dtype=torch.bool)
    with torch.no_grad():
        likelihood = model(
            torch.full((2, 3, 64), 1e6),
            channel_mask=missing,
            likelihood_class_means=profile.class_means,
        ).likelihood
    assert likelihood is not None
    llr = prequential_log_likelihood_ratio(
        likelihood.likelihood_observation,
        likelihood.causal_innovation,
        profile,
        variant="m3_low_rank_dynamic",
        observation_mask=likelihood.observation_mask,
    )
    assert torch.equal(llr, torch.zeros(2))


def test_nonfinite_observed_sample_is_rejected_but_explicit_missing_is_allowed() -> None:
    model = _small_model()
    observation = torch.randn(2, 3, 64)
    observation[:, 1, 10] = torch.nan
    with pytest.raises(ValueError, match=r"Observed Stage-0.*must be finite"):
        model(observation)

    mask = torch.tensor([True, False, True])
    output = model(observation, channel_mask=mask)
    assert output.likelihood is not None
    assert torch.isfinite(output.likelihood.likelihood_observation).all()


def test_missing_channel_cannot_change_observed_density_or_hypothesis_history() -> None:
    torch.manual_seed(31)
    model = _small_model()
    observation = torch.randn(2, 3, 64)
    changed = observation.clone()
    changed[:, 1] = 1e6
    class_means = torch.randn(2, 3, 64)
    channel_mask = torch.tensor([[True, False, True], [True, False, True]])

    with torch.no_grad():
        original = model(
            observation,
            channel_mask=channel_mask,
            likelihood_class_means=class_means,
        ).likelihood
        intervened = model(
            changed,
            channel_mask=channel_mask,
            likelihood_class_means=class_means,
        ).likelihood

    assert original is not None and intervened is not None
    assert torch.equal(original.observation_mask, channel_mask)
    assert torch.equal(intervened.observation_mask, channel_mask)
    assert torch.equal(
        original.causal_innovation.history_correction,
        intervened.causal_innovation.history_correction,
    )
    assert torch.equal(
        original.causal_innovation.log_variance_scale,
        intervened.causal_innovation.log_variance_scale,
    )
    assert torch.equal(
        original.causal_innovation.factor_scale,
        intervened.causal_innovation.factor_scale,
    )


def test_masked_low_rank_score_matches_explicit_gaussian_marginal() -> None:
    torch.manual_seed(37)
    batch, channels, n_time = 3, 4, 12
    observation = torch.randn(batch, channels, n_time)
    labels = torch.tensor([0, 1] * 12, dtype=torch.float32)
    profile = estimate_generative_profile(
        torch.randn(24, channels, n_time),
        labels,
        sfreq=12.0,
        tmin_ms=0.0,
        score_interval_ms=(0.0, 1000.0),
        ar_order=2,
    )
    mask = torch.tensor(
        [[True, False, True, False], [True, True, False, False], [False, True, True, True]]
    )
    innovation = CausalInnovationOutput(
        history_correction=torch.zeros(batch, 2, channels, n_time),
        log_variance_scale=torch.zeros(batch, 2, n_time, channels),
        factor_scale=torch.randn(batch, 2, n_time, channels, 2) * 0.1,
    )
    score = prequential_score_per_trial(
        observation,
        innovation,
        profile,
        hypothesis=1,
        variant="m3_low_rank_dynamic",
        observation_mask=mask,
    )

    class_mean = profile.class_means[1].to(observation.dtype)
    centered = (observation - class_mean) * mask[:, :, None]
    ar = causal_ar_prediction(centered, profile.ar_coefficients.to(observation.dtype))
    ar = ar * mask[:, :, None]
    ar_error = (centered - ar) * mask[:, :, None]
    mean = class_mean + ar + causal_adaptive_ar1_prediction(ar_error) * mask[:, :, None]
    residual = (observation - mean).transpose(1, 2)
    diagonal0 = profile.ar_low_rank_diagonal[1].to(observation.dtype)
    factor0 = profile.ar_low_rank_factor[1].to(observation.dtype)
    expected = []
    for row in range(batch):
        selected = mask[row]
        diagonal = diagonal0[selected][None].expand(n_time, -1)
        factor = factor0[selected][None].expand(n_time, -1, -1)
        factor = (
            factor
            + innovation.factor_scale[row, 1, :, selected].to(observation.dtype)
            * (diagonal.sqrt()[..., None])
        )
        covariance = torch.diag_embed(diagonal) + factor @ factor.transpose(-1, -2)
        chol = torch.linalg.cholesky(covariance)
        value = residual[row, :, selected]
        solved = torch.cholesky_solve(value[..., None], chol).squeeze(-1)
        per_time = 0.5 * (
            (value * solved).sum(-1)
            + 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
            + int(selected.sum()) * torch.log(torch.tensor(2.0 * torch.pi))
        )
        expected.append(per_time.sum())
    assert torch.allclose(score.nll_sum, torch.stack(expected), atol=1e-5, rtol=1e-5)


def test_llr_is_sum_of_per_time_evidence_not_a_time_average() -> None:
    torch.manual_seed(41)
    train = torch.cat((torch.randn(16, 2, 20), torch.randn(16, 2, 20) + 0.5))
    labels = torch.cat((torch.zeros(16), torch.ones(16)))
    profile = estimate_generative_profile(
        train,
        labels,
        sfreq=20.0,
        tmin_ms=0.0,
        score_interval_ms=(0.0, 1000.0),
        ar_order=2,
    )
    observation = torch.randn(4, 2, 20)
    zero = CausalInnovationOutput(
        history_correction=torch.zeros(4, 2, 2, 20),
        log_variance_scale=torch.zeros(4, 2, 20, 2),
        factor_scale=torch.zeros(4, 2, 20, 2, 2),
    )
    negative = prequential_score_per_trial(observation, zero, profile, hypothesis=0, variant="m0")
    positive = prequential_score_per_trial(observation, zero, profile, hypothesis=1, variant="m0")
    llr = prequential_log_likelihood_ratio(observation, zero, profile, variant="m0")

    assert torch.allclose(llr, (negative.nll_per_time - positive.nll_per_time).sum(dim=1))
    assert torch.allclose(llr, negative.nll_sum - positive.nll_sum)
    assert torch.allclose(
        negative.nll_per_observed_scalar,
        negative.nll_sum / negative.observed_scalar_count,
    )


def test_nested_training_loss_reuses_neural_mean_computation(monkeypatch) -> None:
    torch.manual_seed(43)
    channels, n_time = 3, 20
    training = torch.randn(24, channels, n_time)
    training_labels = torch.tensor([0, 1] * 12, dtype=torch.float32)
    profile = estimate_generative_profile(
        training,
        training_labels,
        sfreq=20.0,
        tmin_ms=0.0,
        score_interval_ms=(0.0, 1000.0),
        ar_order=2,
    )
    observation = torch.randn(6, channels, n_time)
    labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float32)
    innovation = CausalInnovationOutput(
        history_correction=torch.randn(6, 2, channels, n_time) * 0.01,
        log_variance_scale=torch.randn(6, 2, n_time, channels) * 0.01,
        factor_scale=torch.randn(6, 2, n_time, channels, 2) * 0.01,
    )
    weight = 0.4
    expected_m1 = prequential_nll(
        observation, innovation, profile, labels, variant="m1"
    )
    expected_m3 = prequential_nll(
        observation,
        innovation,
        profile,
        labels,
        variant="m3_low_rank_dynamic",
    )

    calls = 0
    original = prequential_module.causal_ar_prediction

    def counted_ar_prediction(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(prequential_module, "causal_ar_prediction", counted_ar_prediction)
    actual = nested_prequential_training_loss(
        observation,
        innovation,
        profile,
        labels,
        covariance_weight=weight,
    )

    assert torch.allclose(actual, (expected_m1 + weight * expected_m3) / (1.0 + weight))
    assert calls == 1


def test_audit_aggregation_weights_subjects_not_trial_counts() -> None:
    values = torch.tensor([1.0, 1.0, 1.0, 1.0, 9.0])
    subjects = torch.tensor([0, 0, 0, 0, 1])
    means = _subject_means(values, subjects)
    assert torch.equal(means, torch.tensor([1.0, 9.0]))
    assert means.mean().item() == 5.0


def test_zero_innovation_outputs_cannot_pass_as_incremental_families() -> None:
    torch.manual_seed(29)
    training = torch.randn(48, 2, 32)
    training_labels = torch.tensor([0, 1] * 24, dtype=torch.float32)
    profile = estimate_generative_profile(
        training,
        training_labels,
        sfreq=32.0,
        tmin_ms=0.0,
        score_interval_ms=(0.0, 1000.0),
        ar_order=2,
    )
    audit = torch.randn(32, 2, 32)
    labels = torch.tensor([0, 1] * 16, dtype=torch.float32)
    subjects = torch.arange(4).repeat_interleave(8)
    zero = CausalInnovationOutput(
        history_correction=torch.zeros(32, 2, 2, 32),
        log_variance_scale=torch.zeros(32, 2, 32, 2),
        factor_scale=torch.zeros(32, 2, 32, 2, 2),
    )

    report = audit_prequential_model(
        audit,
        zero,
        profile,
        labels,
        subjects,
    )

    zero_children = {"m1": "linear_ar", "m2_diag": "m1", "m3_low_rank_dynamic": "m2_low_rank"}
    for child, parent in zero_children.items():
        assert report.nll_by_variant[child] == pytest.approx(
            report.nll_by_variant[parent], abs=1e-7
        )
        assert not report.checks_by_variant[child]["incremental_predictive_skill_vs_parent"]
        assert child not in report.eligible_variants


def test_dynamic_low_rank_candidate_must_beat_both_direct_parents(monkeypatch) -> None:
    from train import prequential_audit as module

    values = {
        "m0": torch.full((8,), 10.0),
        "linear_ar": torch.full((8,), 8.0),
        "m1": torch.full((8,), 6.0),
        "m2_diag": torch.full((8,), 4.0),
        "m2_low_rank": torch.full((8,), 5.0),
        # Beats static low-rank but loses to dynamic diagonal.
        "m3_low_rank_dynamic": torch.full((8,), 4.5),
    }

    monkeypatch.setattr(
        module,
        "prequential_nll_per_trial",
        lambda *args, variant, **kwargs: values[variant],
    )
    monkeypatch.setattr(
        module,
        "_standardized_residual_diagnostics",
        lambda *args, **kwargs: module._ResidualDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0),
    )
    profile = GenerativeProfile(
        class_means=torch.zeros(2, 1, 4),
        class_channel_variances=torch.ones(2, 1),
        ar_coefficients=torch.zeros(1, 1, 1),
        ar_channel_variances=torch.ones(2, 1),
        target_rate=torch.tensor(0.5),
        channel_mask=torch.ones(1, dtype=torch.bool),
        score_time_mask=torch.ones(4),
        sfreq=4.0,
        tmin_ms=0.0,
        n_time=4,
        source_n_trials=8,
    )
    innovation = CausalInnovationOutput(
        history_correction=torch.zeros(8, 2, 1, 4),
        log_variance_scale=torch.zeros(8, 2, 4, 1),
        factor_scale=torch.zeros(8, 2, 4, 1, 2),
    )
    report = audit_prequential_model(
        torch.zeros(8, 1, 4),
        innovation,
        profile,
        torch.tensor([0, 1] * 4, dtype=torch.float32),
        torch.arange(4).repeat_interleave(2),
    )

    checks = report.checks_by_variant["m3_low_rank_dynamic"]
    assert not checks["incremental_predictive_skill_vs_parent"]
    assert "m3_low_rank_dynamic" not in report.eligible_variants


def test_audit_fails_closed_when_runtime_mask_differs_from_fold_profile() -> None:
    training = torch.randn(24, 2, 16)
    training_labels = torch.tensor([0, 1] * 12, dtype=torch.float32)
    profile = estimate_generative_profile(
        training,
        training_labels,
        sfreq=16.0,
        tmin_ms=0.0,
        score_interval_ms=(0.0, 1000.0),
        ar_order=2,
    )
    observation = torch.randn(16, 2, 16)
    labels = torch.tensor([0, 1] * 8, dtype=torch.float32)
    subjects = torch.arange(4).repeat_interleave(4)
    mask = torch.ones(16, 2, dtype=torch.bool)
    mask[0, 1] = False
    zero = CausalInnovationOutput(
        history_correction=torch.zeros(16, 2, 2, 16),
        log_variance_scale=torch.zeros(16, 2, 16, 2),
        factor_scale=torch.zeros(16, 2, 16, 2, 2),
    )
    report = audit_prequential_model(
        observation,
        zero,
        profile,
        labels,
        subjects,
        observation_mask=mask,
    )

    assert report.passed is False
    assert report.observation_pattern_supported is False
    assert report.minimum_observed_channels == 1
    assert not report.checks_by_variant["m1"]["supported_observation_pattern"]


def test_class_opposite_erp_leakage_cannot_cancel_in_background_gate() -> None:
    n_trials, channels, n_time = 32, 2, 24
    labels = torch.tensor([0, 1] * (n_trials // 2), dtype=torch.float32)
    subjects = torch.arange(4).repeat_interleave(n_trials // 4)
    sign = torch.where(labels > 0.5, 1.0, -1.0)
    observation = 0.4 * sign[:, None, None].expand(-1, channels, n_time)
    profile = GenerativeProfile(
        class_means=torch.zeros(2, channels, n_time),
        class_channel_variances=torch.ones(2, channels),
        ar_coefficients=torch.zeros(1, channels, channels),
        ar_channel_variances=torch.ones(2, channels),
        target_rate=torch.tensor(0.5),
        channel_mask=torch.ones(channels, dtype=torch.bool),
        score_time_mask=torch.ones(n_time),
        sfreq=24.0,
        tmin_ms=0.0,
        n_time=n_time,
        source_n_trials=n_trials,
    )
    zero = CausalInnovationOutput(
        history_correction=torch.zeros(n_trials, 2, channels, n_time),
        log_variance_scale=torch.zeros(n_trials, 2, n_time, channels),
        factor_scale=torch.zeros(n_trials, 2, n_time, channels, 2),
    )

    report = audit_prequential_model(
        observation,
        zero,
        profile,
        labels,
        subjects,
    )
    diagnostics = report.diagnostics_by_variant["m0"]

    assert diagnostics["standardized_mean_abs_max"] == pytest.approx(0.0, abs=1e-7)
    assert diagnostics["class_conditional_mean_rms_max"] == pytest.approx(0.4)
    assert diagnostics["complex_mean_difference_rms"] == pytest.approx(0.8)
    assert report.checks_by_variant["m0"]["standardized_zero_mean"]
    assert not report.checks_by_variant["m0"]["class_conditional_mean_neutrality"]
    assert not report.checks_by_variant["m0"]["complex_mean_difference_neutrality"]
