"""Predeclared target treatments and exposure budgets at the public interface."""

from transfer.recipes import b0_target_recipes, e1_target_recipes


def test_compact_matrix_has_three_baselines_and_three_single_factor_ablations():
    import pytest

    from transfer.recipes import compact_target_recipes

    recipes = compact_target_recipes(seed=20260906)
    assert len(recipes) == 6
    assert {recipe.initialization for recipe in recipes} == {"source", "scratch", "waveform_lda"}
    assert all(recipe.batch_size == 512 and recipe.epochs == 60 for recipe in recipes)
    assert all(recipe.max_updates is None and not recipe.refit_full_prefix for recipe in recipes)
    full = next(recipe for recipe in recipes if recipe.name == "compact_full_fine").record()
    expected = {
        "compact_source_statistics": {"input_statistics"},
        "compact_frozen_bn_buffers": {"bn_buffers"},
        "compact_head_only": {"weights", "bn_affine"},
    }
    for recipe in recipes:
        if recipe.name in expected:
            differences = {key for key, value in recipe.record().items()
                           if key != "name" and value != full[key]}
            assert differences == expected[recipe.name]
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
