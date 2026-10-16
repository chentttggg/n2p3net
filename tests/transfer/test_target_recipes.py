"""Predeclared target treatments and exposure budgets at the public interface."""

import pytest

from transfer.calibration_candidates import calibration_candidate_recipes
from transfer.recipes import (
    TargetRecipe,
    b0_target_recipes,
    e1_target_recipes,
    unified_target_recipes,
)


def test_compact_matrix_has_three_baselines_and_three_single_factor_ablations():
    import pytest

    from transfer.recipes import compact_target_recipes

    recipes = compact_target_recipes(seed=20260906)
    assert len(recipes) == 6
    assert {recipe.initialization for recipe in recipes} == {"source", "scratch", "waveform_lda"}
    assert all(recipe.batch_size == 512 and recipe.epochs == 60 for recipe in recipes)
    assert all(recipe.max_updates is None and not recipe.refit_full_prefix for recipe in recipes)
    by_name = {recipe.name: recipe.record() for recipe in recipes}
    full = by_name["compact_full_fine"]
    expected = {
        "compact_source_statistics": {"input_statistics"},
        "compact_frozen_bn_buffers": {"bn_buffers"},
        "compact_head_only": {"weights", "bn_affine"},
    }
    for name, fields in expected.items():
        differences = {
            key for key, value in by_name[name].items() if key != "name" and value != full[key]
        }
        assert differences == fields
    for batch in (32, 128):
        with pytest.raises(ValueError, match="512"):
            compact_target_recipes(batch_size=batch)


def test_e1_declares_all_positive_budget_treatments_and_deduplicates_history():
    recipes = e1_target_recipes(batch_size=128, seed=11)
    assert len(recipes) == 23
    matrix = [recipe for recipe in recipes if recipe.name.startswith("e1_i")]
    assert len(matrix) == 8
    assert all(recipe.bn_affine == "frozen" and recipe.dropout == "off" for recipe in matrix)
    assert {recipe.weights for recipe in matrix} == {"head", "non_bn"}
    assert len(e1_target_recipes(batch_size=32)) == 21
    diagnostic = next(recipe for recipe in recipes if recipe.name == "e1_stats_i1b1")
    assert diagnostic.weights == "none"
    assert diagnostic.adapter_config().trainable_weights == "none"


def test_b0_small_calibration_deduplicates_actual_batches_and_uses_fit_rows():
    recipes = b0_target_recipes(n_fit=40, seed=7)
    assert len(recipes) == 8
    fixed = [recipe for recipe in recipes if recipe.max_updates is not None]
    assert len(fixed) == 4
    assert {recipe.max_updates for recipe in fixed} == {60}
    assert {min(recipe.batch_size, 40) for recipe in fixed} == {32, 40}
    assert all(recipe.lr == 1e-3 for recipe in recipes)


def test_unified_recipe_admission_preserves_binary_identity_and_independent_scratch():
    binary = unified_target_recipes(membership_available=False, source_prior_weight=.2)
    assert len(binary) == 4 and all(recipe.candidate_loss_weight is None for recipe in binary)
    assert all(recipe.max_updates is None for recipe in binary)
    decisions = unified_target_recipes(membership_available=True, source_prior_weight=.2)
    assert len(decisions) == 5
    neural = decisions[:-1]
    assert [recipe.candidate_loss_weight for recipe in neural] == [0., 1., 1., 0.]
    assert [recipe.source_prior_weight for recipe in neural] == [0., .2, 0., 0.]
    assert all(recipe.bn_buffers == "calibration" and recipe.weights == "non_bn" for recipe in neural)
    assert "candidate_loss_weight" not in binary[0].record()
    assert neural[0].record()["candidate_loss_weight"] == 0
    assert TargetRecipe(**neural[1].record()).record() == neural[1].record()
    assert calibration_candidate_recipes("unified", transfer_policy=neural[1]) == (neural[1],)
    with pytest.raises(ValueError, match="one explicit"):
        calibration_candidate_recipes("unified")


def test_unified_ce_scratch_matches_full_fine_supervision():
    from dataclasses import replace

    recipes = {r.name: r for r in unified_target_recipes(membership_available=True, source_prior_weight=.1)}
    assert replace(recipes["unified_q0"], name="unified_scratch_ce", initialization="scratch") == recipes["unified_scratch_ce"]
    assert replace(recipes["unified_q1"], name="unified_scratch", initialization="scratch", source_prior_weight=0.) == recipes["unified_scratch"]
