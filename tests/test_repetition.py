from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from baselines.calibration import fit_weighted_logit_temperature
from baselines.n2p3net import (
    N2P3NetBaseline,
    _reliability_gate_metrics,
    _seed_model_initialization,
)
from models.n2p3net import N2P3Net
from models.repetition import (
    RepetitionEvidenceModel,
    corrected_trial_evidence,
    extract_quality_features,
)
from train.contracts import SetMetadata
from train.losses import repetition_multi_k_objective


def _reference_sequence(model, evidence, quality, labels):
    hidden = evidence.new_zeros((1, 1, model.hidden_size))
    selected_steps = []
    for step in range(evidence.numel()):
        mixture, rho, clean, _ = model._class_log_prob(
            evidence[step : step + 1], quality[step : step + 1], hidden.squeeze(0)
        )
        label = labels[step : step + 1].long()
        selected = mixture.gather(1, label[:, None]).squeeze(1)
        selected_clean = clean.gather(1, label[:, None]).squeeze(1)
        posterior = torch.exp(torch.log(rho) + selected_clean - selected).clamp(0.0, 1.0)
        gru_input = torch.cat(
            (
                posterior[:, None] * (evidence[step : step + 1] / 5.0).clamp(-4.0, 4.0)[:, None],
                posterior[:, None] * model.normalize_quality(quality[step : step + 1]),
                labels[step : step + 1, None],
                posterior[:, None],
            ),
            dim=-1,
        )
        _, hidden = model.gru(gru_input[:, None], hidden)
        selected_steps.append(selected[0])
    return torch.stack(selected_steps)


def test_exact_fold_pos_weight_cancels_training_prior_offset() -> None:
    prior = 0.23
    pos_weight = 3.7
    temperature = 1.8
    logits = torch.tensor([-2.0, 0.0, 3.0])
    evidence = corrected_trial_evidence(
        logits,
        pos_weight=pos_weight,
        train_prior=prior,
        temperature=temperature,
    )
    expected = (logits - np.log(pos_weight) - np.log(prior / (1.0 - prior))) / temperature
    assert torch.allclose(evidence, expected, atol=1e-7)


def test_zero_reliability_has_exactly_zero_conditional_llr() -> None:
    model = RepetitionEvidenceModel(hidden_size=8)
    evidence = torch.tensor([-2.0, 0.5, 3.0])
    quality = torch.zeros(3, 8)
    labels = torch.tensor([0.0, 1.0, 0.0])
    output = model.forward_sequence(evidence, quality, labels, reliability_override=0.0)
    assert torch.equal(output.conditional_llr, torch.zeros_like(output.conditional_llr))


def test_zero_reliability_blocks_artifact_evidence_from_future_history() -> None:
    torch.manual_seed(8)
    model = RepetitionEvidenceModel(hidden_size=8)
    torch.nn.init.normal_(model.clean_context[-1].weight, std=0.2)
    quality = torch.zeros(3, 8)
    labels = torch.tensor([0.0, 1.0, 0.0])
    reliability = torch.tensor([0.0, 1.0, 1.0])
    first = model.forward_sequence(
        torch.tensor([-20.0, 0.5, -0.2]),
        quality,
        labels,
        reliability_override=reliability,
    )
    second = model.forward_sequence(
        torch.tensor([20.0, 0.5, -0.2]),
        quality,
        labels,
        reliability_override=reliability,
    )
    assert first.conditional_llr[0] == 0.0
    assert second.conditional_llr[0] == 0.0
    assert torch.allclose(first.conditional_llr[1:], second.conditional_llr[1:], atol=1e-7)


def test_candidate_chain_uses_target_and_nontarget_flashes() -> None:
    model = RepetitionEvidenceModel(hidden_size=8)
    model.eval()
    digits = torch.tensor([1, 2, 3, 1, 2, 3])
    true_digit = 2
    evidence = torch.where(digits == true_digit, 1.5, -1.5).float()
    quality = torch.zeros(len(digits), 8)
    scores, reliability = model.candidate_log_scores(
        evidence, quality, digits, digit_vocab=(1, 2, 3)
    )
    assert int(scores.argmax()) == 1
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]
    assert reliability.shape == (len(digits),)


def test_batched_ragged_sequence_matches_reference_values_and_gradients() -> None:
    torch.manual_seed(31)
    batched_model = RepetitionEvidenceModel(hidden_size=8).double()
    reference_model = copy.deepcopy(batched_model)
    lengths = torch.tensor([7, 4, 6])
    evidence = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
    reference_evidence = evidence.detach().clone().requires_grad_()
    quality = torch.randn(3, 7, 8, dtype=torch.float64)
    labels = torch.randint(0, 2, (3, 7), dtype=torch.float64)

    output = batched_model.forward_batched_sequences(evidence, quality, labels, lengths)
    active = torch.arange(7)[None] < lengths[:, None]
    batched_loss = output.observed_log_prob[active].sum()
    reference_parts = [
        _reference_sequence(
            reference_model,
            reference_evidence[row, :length],
            quality[row, :length],
            labels[row, :length],
        )
        for row, length in enumerate(lengths.tolist())
    ]
    reference_loss = torch.cat(reference_parts).sum()
    assert torch.allclose(
        output.observed_log_prob[active], torch.cat(reference_parts), atol=1e-12, rtol=1e-12
    )

    batched_loss.backward()
    reference_loss.backward()
    assert torch.allclose(evidence.grad[active], reference_evidence.grad[active], atol=1e-11)
    for batched_parameter, reference_parameter in zip(
        batched_model.parameters(), reference_model.parameters(), strict=True
    ):
        assert torch.allclose(
            batched_parameter.grad, reference_parameter.grad, atol=1e-10, rtol=1e-10
        )


def test_quality_features_are_finite_and_class_independent() -> None:
    generator = torch.Generator().manual_seed(4)
    signal = torch.randn(4, 3, 128, generator=generator)
    residual = 0.1 * torch.randn(4, 3, 128, generator=generator)
    signal[0, 0] = 0.0
    features = extract_quality_features(
        signal,
        sfreq=128.0,
        baseline_n=26,
        reconstruction_residual=residual,
        channel_mask=torch.tensor([True, True, False]),
    )
    assert features.shape == (4, 8)
    assert torch.isfinite(features).all()
    assert features[0, 6] == pytest.approx(0.5)
    assert torch.equal(features[1:, 6], torch.zeros(3))
    assert torch.allclose(features[:, -1], torch.full((4,), 1.0 / 3.0))


def test_quality_features_accept_explicit_trial_reference_slice() -> None:
    generator = torch.Generator().manual_seed(8)
    signal = torch.randn(4, 3, 128, generator=generator)
    features = extract_quality_features(
        signal,
        sfreq=128.0,
        baseline_n=2,
        reference_slice=(8, 24),
    )
    reference = signal[:, :, 8:24]
    expected_variance = torch.log(reference.var(dim=-1, unbiased=False).mean(dim=1))
    assert torch.allclose(features[:, 0], expected_variance)


def test_physical_scale_quality_features_preserve_artifact_contrasts() -> None:
    generator = torch.Generator().manual_seed(21)
    clean = 10e-6 * torch.randn(16, 3, 128, generator=generator)
    noisy = clean.clone()
    noisy[:, :, ::2] += 80e-6
    noisy[:, :, 1::2] -= 80e-6
    flatline = clean.clone()
    flatline[:, 0] = 0.0
    clean_quality = extract_quality_features(clean, sfreq=128.0, baseline_n=26)
    noisy_quality = extract_quality_features(noisy, sfreq=128.0, baseline_n=26)
    flatline_quality = extract_quality_features(flatline, sfreq=128.0, baseline_n=26)
    assert noisy_quality[:, 0].mean() > clean_quality[:, 0].mean()
    assert noisy_quality[:, 4].mean() > clean_quality[:, 4].mean()
    assert torch.all(flatline_quality[:, 6] >= 1.0 / 3.0)


def test_nested_chain_objective_has_density_gradients() -> None:
    model = RepetitionEvidenceModel(hidden_size=8)
    model.set_evidence_calibration(pos_weight=8.0, train_prior=1.0 / 9.0)
    digits = torch.tensor([1, 2, 3, 1, 2, 3])
    labels = (digits == 2).float()
    logits = torch.where(labels > 0.5, 1.0, -1.0).requires_grad_()
    quality = torch.zeros(6, 8)
    metadata = SetMetadata(
        stimulus_digits=digits,
        group_ids=torch.zeros(6, dtype=torch.long),
        repetition_ranks=torch.tensor([0, 0, 0, 1, 1, 1]),
        sequence_ranks=torch.arange(6),
    )
    digit_loss, conditional_nll, coverage = repetition_multi_k_objective(
        logits,
        quality,
        labels,
        metadata,
        model,
        evidence_ks=(1, 2),
        evidence_weights=(0.25, 0.75),
        digit_vocab=(1, 2, 3),
    )
    total = digit_loss + 0.1 * conditional_nll
    total.backward()
    assert coverage == {1: 1, 2: 1}
    assert torch.isfinite(total)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_temperature_is_fit_only_after_weighted_offset_removal() -> None:
    rng = np.random.default_rng(11)
    prior = 1.0 / 9.0
    pos_weight = 8.0
    latent_llr = rng.normal(0.0, 2.5, 12000)
    true_temperature = 2.0
    posterior_log_odds = latent_llr / true_temperature + np.log(prior / (1.0 - prior))
    labels = rng.binomial(1, 1.0 / (1.0 + np.exp(-posterior_log_odds)))
    weighted_logits = latent_llr + np.log(pos_weight) + np.log(prior / (1.0 - prior))
    calibration = fit_weighted_logit_temperature(
        weighted_logits,
        labels,
        pos_weight=pos_weight,
        train_prior=prior,
        source="inner_validation",
    )
    assert calibration.temperature == pytest.approx(true_temperature, abs=0.25)
    assert calibration.source == "inner_validation"


def _calibration_gate_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = (np.arange(36) % 9 == 0).astype(np.int64)
    subjects = np.repeat(np.arange(2), 18)
    clean = np.full(36, 0.9)
    corrupt = np.full((4, 36), 0.1)
    return clean, corrupt, labels, subjects


def test_reliability_gate_requires_absolute_synthetic_artifact_calibration() -> None:
    clean, corrupt, labels, subjects = _calibration_gate_fixture()
    calibrated = _reliability_gate_metrics(clean, corrupt, labels, subjects)
    assert calibrated["passed"]
    assert calibrated["failed_checks"] == []
    assert all(calibrated["checks"].values())
    assert calibrated["artifact_brier"] == pytest.approx(0.01)

    uncalibrated = _reliability_gate_metrics(
        np.full_like(clean, 0.99), np.full_like(corrupt, 0.84), labels, subjects
    )
    assert uncalibrated["artifact_auc"] == 1.0
    assert uncalibrated["mean_reliability_gap"] == pytest.approx(0.15)
    assert not uncalibrated["passed"]


def test_reliability_gate_detects_nonlinear_target_leakage_and_single_class() -> None:
    clean, corrupt, labels, subjects = _calibration_gate_fixture()
    clean.fill(0.85)
    clean[np.flatnonzero(labels)] = np.tile((0.75, 0.95), 2)
    nonlinear = _reliability_gate_metrics(clean, corrupt, labels, subjects)
    assert nonlinear["target_leakage_auc"] == pytest.approx(0.5)
    assert nonlinear["target_nonlinear_leakage_auc"] == pytest.approx(1.0)
    assert not nonlinear["passed"]

    single_class = _reliability_gate_metrics(clean, corrupt, np.zeros_like(labels), subjects)
    assert not single_class["passed"]
    assert "both_target_classes" in single_class["failure"]
    assert single_class["checks"]["both_target_classes_present"] is False
    assert "both_target_classes_present" in single_class["failed_checks"]

    too_few = _reliability_gate_metrics(clean[:4], corrupt[:, :4], labels[:4], subjects[:4])
    assert not too_few["passed"]
    assert too_few["checks"]["minimum_validation_trials"] is False
    assert "minimum_validation_trials" in too_few["failed_checks"]


def test_reliability_gate_is_invariant_to_validation_row_order() -> None:
    clean, corrupt, labels, subjects = _calibration_gate_fixture()
    reference = _reliability_gate_metrics(clean, corrupt, labels, subjects)
    permutation = np.random.default_rng(7).permutation(len(clean))
    shuffled = _reliability_gate_metrics(
        clean[permutation], corrupt[:, permutation], labels[permutation], subjects[permutation]
    )
    for key in (
        "passed",
        "artifact_auc",
        "artifact_brier",
        "artifact_ece10",
        "target_distribution_ks",
        "subject_artifact_auc_min",
    ):
        assert shuffled[key] == pytest.approx(reference[key])


def test_fold_model_initialization_seed_is_reset_before_construction() -> None:
    _seed_model_initialization(23)
    first = torch.randn(32)
    torch.randn(100)
    _seed_model_initialization(23)
    second = torch.randn(32)
    assert torch.equal(first, second)


def test_adapter_fits_inner_temperature_and_exposes_chain_decisions() -> None:
    rng = np.random.default_rng(19)
    n_subjects = 6
    repeats = 2
    digits = np.tile(np.repeat(np.arange(1, 10), repeats), n_subjects)
    subject_ids = np.repeat(np.arange(n_subjects), 9 * repeats)
    true_digits = (subject_ids % 9) + 1
    labels = (digits == true_digits).astype(np.int64)
    x = rng.normal(0.0, 0.3, (len(labels), 3, 64)).astype(np.float32)
    x[labels == 1, 2, 24:40] += 0.8

    adapter = N2P3NetBaseline(
        model_kwargs={
            "n_channels": 3,
            "channel_names": ("Fz", "Cz", "Pz"),
            "d_model": 8,
            "filters_per_scale": 2,
            "temporal_kernels": (13,),
            "encoder_depth": 0,
            "use_rereference": False,
            "baseline_mode": "trial",
            "tmin_ms": -200.0,
            "tmax_ms": 800.0,
            "sfreq": 64.0,
            "n_time": 64,
            "component_decoder": False,
            "use_repetition_evidence": True,
            "repetition_hidden_size": 8,
        },
        trainer_kwargs={
            "epochs": 1,
            "batch_size": 36,
            "lambda2": 0.0,
            "lambda3": 0.0,
            "lambda_pcw": 0.0,
            "lambda_digit": 0.2,
            "lambda_conditional_nll": 0.1,
            "digit_evidence_ks": (1, 2),
            "digit_evidence_weights": (0.25, 0.75),
            "lambda_recon": 0.0,
            "repetition_refit_epochs": 1,
            "early_stop_patience": 2,
            "augment": False,
            "seed": 2,
        },
        channel_mask=torch.ones(3, dtype=torch.bool),
        device=torch.device("cpu"),
        val_subject_frac=0.34,
        val_subjects_min=2,
        val_subjects_max=2,
    )
    adapter.fit(x, labels, subject_ids=subject_ids, digits=digits)
    assert adapter.last_pos_weight == pytest.approx(8.0)
    assert adapter.repetition_temperature_calibration_ is not None
    assert adapter.repetition_reliability_audit_ is not None
    adapter.repetition_ready_ = False
    descriptive_results = adapter.predict_repetition_candidates(
        x[: 9 * repeats],
        digits[: 9 * repeats],
        subject_ids[: 9 * repeats],
        evidence_budgets=(1, 2),
    )
    assert set(descriptive_results) == {
        "prefix_minK_chain_llr@1",
        "prefix_minK_chain_llr@2",
    }
    assert descriptive_results["prefix_minK_chain_llr@1"]["claim_eligible"] is False
    adapter.repetition_ready_ = True
    results = adapter.predict_repetition_candidates(
        x[: 9 * repeats],
        digits[: 9 * repeats],
        subject_ids[: 9 * repeats],
        evidence_budgets=(1, 2),
    )
    assert set(results) == {"prefix_minK_chain_llr@1", "prefix_minK_chain_llr@2"}
    assert len(results["prefix_minK_chain_llr@2"]["predicted"]) == 1
    assert np.isfinite(results["prefix_minK_chain_llr@2"]["scores"]).all()



def test_v12_residual_gate_smoke_and_fail_closed() -> None:
    rng = np.random.default_rng(23)
    n_subjects = 8
    repeats = 2
    digits = np.tile(np.repeat(np.arange(1, 10), repeats), n_subjects)
    subject_ids = np.repeat(np.arange(n_subjects), 9 * repeats)
    true_digits = (subject_ids % 9) + 1
    labels = (digits == true_digits).astype(np.int64)
    x = rng.normal(0.0, 0.3, (len(labels), 3, 64)).astype(np.float32)
    x[labels == 1, 2, 24:40] += 0.8

    adapter = N2P3NetBaseline(
        model_kwargs={
            "n_channels": 3,
            "channel_names": ("Fz", "Cz", "Pz"),
            "d_model": 8,
            "filters_per_scale": 2,
            "temporal_kernels": (13,),
            "encoder_depth": 0,
            "use_rereference": False,
            "baseline_mode": "trial",
            "tmin_ms": -200.0,
            "tmax_ms": 800.0,
            "sfreq": 64.0,
            "n_time": 64,
            "component_decoder": False,
            "use_repetition_evidence": True,
            "repetition_hidden_size": 8,
            "repetition_v12": True,
            "repetition_state_residual": True,
        },
        trainer_kwargs={
            "epochs": 1,
            "batch_size": 36,
            "lambda2": 0.0,
            "lambda3": 0.0,
            "lambda_pcw": 0.0,
            "lambda_digit": 0.2,
            "lambda_conditional_nll": 0.1,
            "digit_evidence_ks": (1, 2),
            "digit_evidence_weights": (0.25, 0.75),
            "lambda_recon": 0.0,
            "repetition_refit_epochs": 1,
            "early_stop_patience": 2,
            "augment": False,
            "seed": 2,
        },
        channel_mask=torch.ones(3, dtype=torch.bool),
        device=torch.device("cpu"),
        val_subject_frac=0.25,
        val_subjects_min=2,
        val_subjects_max=2,
    )
    adapter.fit(x, labels, subject_ids=subject_ids, digits=digits)
    audit = adapter.last_history["repetition_reliability_audit"]
    assert "fidelity" in audit
    assert "clean_probability" in audit
    assert audit["clean_probability"]["available"] is False
    gate = adapter.last_history["repetition_state_residual_gate"]
    assert "passed" in gate
    assert "strict_majority" in gate or "failure" in gate
    if not bool(gate["passed"]):
        assert adapter.model_.repetition_evidence.state_residual_gain_value() == 0.0
    else:
        assert adapter.model_.repetition_evidence.state_residual_gain_value() > 0.0


def test_v12_measurement_branch_smoke_and_fail_closed() -> None:
    rng = np.random.default_rng(31)
    n_subjects = 8
    repeats = 2
    digits = np.tile(np.repeat(np.arange(1, 10), repeats), n_subjects)
    subject_ids = np.repeat(np.arange(n_subjects), 9 * repeats)
    true_digits = (subject_ids % 9) + 1
    labels = (digits == true_digits).astype(np.int64)
    x = rng.normal(0.0, 0.3, (len(labels), 3, 64)).astype(np.float32)
    x[labels == 1, 2, 24:40] += 0.8

    adapter = N2P3NetBaseline(
        model_kwargs={
            "n_channels": 3,
            "channel_names": ("Fz", "Cz", "Pz"),
            "d_model": 8,
            "filters_per_scale": 2,
            "temporal_kernels": (13,),
            "encoder_depth": 0,
            "use_rereference": False,
            "baseline_mode": "trial",
            "tmin_ms": -200.0,
            "tmax_ms": 800.0,
            "sfreq": 64.0,
            "n_time": 64,
            "component_decoder": False,
            "use_repetition_evidence": True,
            "repetition_hidden_size": 8,
            "repetition_v12": True,
            "repetition_state_residual": True,
            "use_measurement_windows": True,
            "measurement_anchor_ms": 460.0,
            "measurement_refit_epochs": 1,
        },
        trainer_kwargs={
            "epochs": 1,
            "batch_size": 36,
            "lambda2": 0.0,
            "lambda3": 0.0,
            "lambda_pcw": 0.0,
            "lambda_digit": 0.2,
            "lambda_conditional_nll": 0.1,
            "digit_evidence_ks": (1, 2),
            "digit_evidence_weights": (0.25, 0.75),
            "lambda_recon": 0.0,
            "repetition_refit_epochs": 1,
            "early_stop_patience": 2,
            "augment": False,
            "seed": 2,
        },
        channel_mask=torch.ones(3, dtype=torch.bool),
        device=torch.device("cpu"),
        val_subject_frac=0.25,
        val_subjects_min=2,
        val_subjects_max=2,
    )
    adapter.fit(x, labels, subject_ids=subject_ids, digits=digits)
    gate = adapter.last_history["measurement_gate"]
    assert gate["available"] is True
    assert gate["passed"] is False
    assert gate["coefficient"] == 0.0
    assert adapter.measurement_gain_ == 0.0
    assert float(adapter.model_.measurement_gain) == 0.0
    assert adapter.measurement_posterior_ is not None

    # Measurement-window cache and fold-posterior stability.
    first_window = adapter._measurement_windows(x)
    second_window = adapter._measurement_windows(x)
    assert first_window is second_window
    full_posterior_n = adapter._measurement_posterior_n_
    assert full_posterior_n == len(x)
    _ = adapter._measurement_windows(x[:8])
    assert adapter._measurement_posterior_n_ == full_posterior_n

    # Branch identity must match the deployed v12 formula, including the
    # zero measurement/prequential contributions for this fail-closed fold.
    branches = adapter.predict_branches(x)
    expected = (
        float(branches["prequential_base_intercept"])
        + float(branches["prequential_base_slope"])
        * (branches["pcw"] + branches["measurement_contribution"])
        + branches["prequential_contribution"]
    )
    assert np.allclose(branches["final"], expected, atol=1e-10)

    adapter.set_clean_probability_pool(
        x,
        np.where(labels == 1, 1, 0),
        subject_ids,
        calibration_prior=0.2,
        deployment_prior=0.9,
    )
    clean_report = adapter._evaluate_clean_probability_pool()
    assert clean_report["available"] is True
    assert clean_report["passed"] is False
    assert clean_report["failure"] == "missing_digit_chain_nll_or_true_candidates"


def test_measurement_branch_fails_closed_for_heterogeneous_trial_masks() -> None:
    adapter = N2P3NetBaseline(
        model_kwargs={
            "n_channels": 3,
            "channel_names": ("Fz", "Cz", "Pz"),
            "d_model": 8,
            "filters_per_scale": 2,
            "temporal_kernels": (13,),
            "encoder_depth": 0,
            "use_rereference": False,
            "baseline_mode": "trial",
            "tmin_ms": -200.0,
            "tmax_ms": 800.0,
            "sfreq": 64.0,
            "n_time": 64,
            "use_measurement_windows": True,
        },
        trainer_kwargs={"epochs": 1, "batch_size": 4},
        channel_mask=torch.ones(3, dtype=torch.bool),
        device=torch.device("cpu"),
    )
    adapter.model_ = N2P3Net(**adapter.model_kwargs)
    X_train = np.zeros((4, 3, 64), dtype=np.float32)
    X_val = np.zeros((2, 3, 64), dtype=np.float32)
    train_mask = np.ones((4, 3), dtype=bool)
    train_mask[1, 2] = False
    val_mask = np.ones((2, 3), dtype=bool)

    gate = adapter._fit_measurement_branch(
        X_train,
        np.array([0, 1, 0, 1]),
        X_val,
        np.array([0, 1]),
        np.array(["a", "a", "b", "b"]),
        np.array(["c", "c"]),
        train_trial_channel_mask=train_mask,
        val_trial_channel_mask=val_mask,
    )

    assert gate == {
        "available": True,
        "passed": False,
        "failure": "measurement_branch_requires_homogeneous_channel_masks",
    }
    assert adapter.measurement_estimator_ is None
    assert adapter._measurement_channel_mask_ is None
