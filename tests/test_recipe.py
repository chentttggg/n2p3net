"""Canonical Neural-RIDE recipe and task-capability tests."""

from __future__ import annotations

import pytest

from models.component_window import (
    GTN_CHILD_SIGMA_BOUNDS,
    GTN_CHILD_TAU0_BOUNDS,
    GTN_CHILD_TAU0_MS,
    PCW_CANONICAL_DTAU_BOUNDS,
    PCW_CANONICAL_SIGMA_BOUNDS,
    PCW_CANONICAL_TAU0_BOUNDS,
    PCW_CANONICAL_TAU0_MS,
)
from train.recipe import (
    BINARY_ODDBALL_TASK,
    GTN_DIGIT_TASK,
    NEURAL_RIDE_V11,
    NEURAL_RIDE_V11_LEGACY,
    NEURAL_RIDE_V11_TRANSFER,
    NEURAL_RIDE_V12,
    NEURAL_RIDE_V12_STRICT_PAST_RESEARCH,
    NEURAL_RIDE_V12_Z2_AUX_REPLACE_RESEARCH,
    NEURAL_RIDE_V12_Z2_AUX_RESEARCH,
    NeuralRideRecipe,
)


def test_v12_core_training_semantics_do_not_depend_on_task_entry() -> None:
    gtn = NEURAL_RIDE_V12.trainer_config(GTN_DIGIT_TASK, epochs=30, batch_size=256, seed=0)
    binary = NEURAL_RIDE_V12.trainer_config(BINARY_ODDBALL_TASK, epochs=30, batch_size=256, seed=0)
    invariant_fields = (
        "lambda2",
        "lambda3",
        "lambda_pcw",
        "lambda_amp",
        "lambda_recon",
        "early_stop_patience",
        "track_pcw_gradients",
        "lambda_morphology_l0",
        "variance_warmup_epochs",
        "variance_ramp_epochs",
        "recon_bootstrap_samples",
        "recon_split_half_repeats",
    )
    for field in invariant_fields:
        assert getattr(binary, field) == getattr(gtn, field)
    assert gtn.lambda_digit > 0.0
    assert binary.lambda_digit == 0.0
    assert gtn.pos_weight == 8.0
    assert binary.pos_weight == 5.0
    assert NEURAL_RIDE_V12.encoder_depth == 4
    assert NEURAL_RIDE_V12.encoder_type == "tcn"
    assert NEURAL_RIDE_V12.encoder_norm == "bn"
    assert NEURAL_RIDE_V12.encoder_bn_momentum == pytest.approx(1.0 - 0.9**0.25)
    assert NEURAL_RIDE_V12.tcn_pointwise_execution == "conv1d"
    assert NEURAL_RIDE_V12.tokenizer_temporal_spatial_fusion is True
    assert NEURAL_RIDE_V12.innovation_kernel_size == 9
    assert NEURAL_RIDE_V12.innovation_dilations == (1, 2, 4, 8, 16)
    assert NEURAL_RIDE_V12.innovation_covariance_rank in (1, 2)
    assert NEURAL_RIDE_V12.component_decoder is False
    assert NEURAL_RIDE_V12.use_innovation_likelihood is False
    assert NEURAL_RIDE_V12.repetition_v12 is True
    assert NEURAL_RIDE_V12.digit_evidence_ks == (1, 3, 5)
    assert NEURAL_RIDE_V11_LEGACY.repetition_v12 is False
    assert gtn.lambda_innovation == 0.0
    assert NEURAL_RIDE_V12_STRICT_PAST_RESEARCH.use_innovation_likelihood is True
    assert NEURAL_RIDE_V12_STRICT_PAST_RESEARCH.lambda_innovation == 1.0
    assert NEURAL_RIDE_V12_STRICT_PAST_RESEARCH.variance_warmup_epochs == 3
    assert NEURAL_RIDE_V12_STRICT_PAST_RESEARCH.variance_ramp_epochs == 5
    assert NEURAL_RIDE_V12.variance_warmup_epochs == 5
    assert NEURAL_RIDE_V12.variance_ramp_epochs == 10
    assert gtn.lambda_recon == 0.0
    assert gtn.recon_bootstrap_samples >= 64
    assert gtn.recon_split_half_repeats >= 16
    assert gtn.lr_schedule == "cosine"
    assert gtn.lr_warmup_fraction == pytest.approx(0.05)
    assert gtn.min_lr_ratio == pytest.approx(0.10)
    assert gtn.weight_decay == pytest.approx(2.5e-5)
    assert gtn.recalibrate_batch_norm is True


def test_direct_recipe_defaults_follow_active_v12_contract() -> None:
    recipe = NeuralRideRecipe()

    assert recipe.name == "neural_ride_v12_pcw_fail_closed"
    assert recipe.repetition_v12 is True
    assert recipe.encoder_depth == 4
    assert recipe.encoder_type == "tcn"
    assert NEURAL_RIDE_V11_LEGACY.name == "neural_ride_v11_legacy_repetition"
    assert NEURAL_RIDE_V11_LEGACY.repetition_v12 is False


def test_gtn_runner_records_research_recipe_only_when_innovation_is_enabled() -> None:
    from experiments.run_n2p3net_gtn import _recipe_for_innovation_weight

    assert _recipe_for_innovation_weight(0.0) is NEURAL_RIDE_V12
    assert _recipe_for_innovation_weight(1.0) is NEURAL_RIDE_V12_STRICT_PAST_RESEARCH


def test_binary_task_rejects_digit_set_supervision() -> None:
    with pytest.raises(ValueError, match="no digit-set labels"):
        NEURAL_RIDE_V12.trainer_config(
            BINARY_ODDBALL_TASK,
            epochs=30,
            batch_size=256,
            seed=0,
            overrides={"lambda_digit": 0.2},
        )


def test_recipe_rejects_silent_core_loss_removal() -> None:
    with pytest.raises(ValueError, match="lambda_pcw"):
        NEURAL_RIDE_V12.trainer_config(
            GTN_DIGIT_TASK,
            epochs=30,
            batch_size=256,
            seed=0,
            overrides={"lambda_pcw": 0.0},
        )


def test_v11_transfer_recipe_enables_uncertain_canonical_shared_private_model() -> None:
    kwargs = NEURAL_RIDE_V11_TRANSFER.model_kwargs(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        tmin_ms=-200.0,
        tmax_ms=1200.0,
        sfreq=256.0,
        n_time=358,
        baseline_mode="trial",
    )
    config = NEURAL_RIDE_V11_TRANSFER.trainer_config(
        BINARY_ODDBALL_TASK,
        epochs=30,
        batch_size=64,
        seed=0,
    )

    assert kwargs["canonical_channel_names"] == ("Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz")
    assert kwargs["encoder_depth"] == 4
    assert kwargs["dataset_adapter_rank"] == 8
    assert kwargs["task_head_shared_only"] is True
    assert kwargs["shared_private"] is True
    assert config.lambda_orth > 0.0
    assert config.lambda_adv > 0.0
    assert config.lambda_private > 0.0
    assert config.lambda4 == 0.0


def test_recipe_depth_override_reaches_canonical_tcn_schedule() -> None:
    from models.n2p3net import N2P3Net

    kwargs = NEURAL_RIDE_V11.model_kwargs(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        tmin_ms=-200.0,
        tmax_ms=800.0,
        sfreq=256.0,
        n_time=256,
        baseline_mode="trial",
        overrides={"encoder_depth": 6},
    )
    model = N2P3Net(**kwargs)
    assert model.encoder.depth == 6
    assert model.encoder.tcn_dilations == (1, 4, 16, 32, 64, 128)


def test_recipe_carries_trial_reference_contract() -> None:
    kwargs = NEURAL_RIDE_V12.model_kwargs(
        n_channels=8,
        channel_names=("Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"),
        tmin_ms=0.0,
        tmax_ms=1000.0,
        sfreq=256.0,
        n_time=256,
        baseline_mode="trial_reference",
        trial_reference_window_ms=(0.0, 50.0),
        trial_reference_center="median",
        trial_reference_scale="mad",
    )
    assert kwargs["trial_reference_window_ms"] == (0.0, 50.0)
    assert kwargs["trial_reference_center"] == "median"
    assert kwargs["trial_reference_scale"] == "mad"


def test_recipe_window_defaults_match_single_canonical_source() -> None:
    recipe = NeuralRideRecipe()

    assert recipe.tau0_ms == PCW_CANONICAL_TAU0_MS
    assert recipe.tau0_bounds == PCW_CANONICAL_TAU0_BOUNDS
    assert recipe.sigma_bounds == PCW_CANONICAL_SIGMA_BOUNDS
    assert recipe.dtau_bounds == PCW_CANONICAL_DTAU_BOUNDS
    assert recipe.use_z2_aux_head is False

    kwargs = recipe.model_kwargs(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        tmin_ms=-200.0,
        tmax_ms=800.0,
        sfreq=256.0,
        n_time=256,
        baseline_mode="trial",
    )
    assert kwargs["tau0_ms"] == PCW_CANONICAL_TAU0_MS
    assert kwargs["tau0_bounds"] == PCW_CANONICAL_TAU0_BOUNDS
    assert kwargs["sigma_bounds"] == PCW_CANONICAL_SIGMA_BOUNDS
    assert kwargs["dtau_bounds"] == PCW_CANONICAL_DTAU_BOUNDS
    assert kwargs["use_z2_aux_head"] is False
    assert kwargs["z2_aux_head_mode"] == "add"


def test_z2_aux_recipes_are_named_research_only_and_fail_closed() -> None:
    assert NEURAL_RIDE_V12.use_z2_aux_head is False
    assert NEURAL_RIDE_V12_Z2_AUX_RESEARCH.use_z2_aux_head is True
    assert NEURAL_RIDE_V12_Z2_AUX_RESEARCH.z2_aux_head_mode == "add"
    assert NEURAL_RIDE_V12_Z2_AUX_RESEARCH.name == "neural_ride_v12_z2_aux_research"
    assert NEURAL_RIDE_V12_Z2_AUX_REPLACE_RESEARCH.z2_aux_head_mode == "replace"
    assert NEURAL_RIDE_V12_Z2_AUX_REPLACE_RESEARCH.name == "neural_ride_v12_z2_aux_replace_research"


def test_gtn_child_override_constants_match_e7_contract() -> None:
    assert GTN_CHILD_TAU0_MS == (220.0, 300.0, 460.0)
    assert GTN_CHILD_TAU0_BOUNDS[2] == (350.0, 600.0)
    assert GTN_CHILD_SIGMA_BOUNDS[2] == (20.0, 150.0)
