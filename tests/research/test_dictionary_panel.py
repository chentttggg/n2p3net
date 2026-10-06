from dataclasses import replace

import pytest

from experiments.run_candidate_ensemble_study import _dictionary_panel, _fixed_prefix_ar_panel
from transfer.ensemble_fit import CalibrationExpertProgram
from transfer.response_dictionary import ResponseDictionarySpec


def experts():
    return tuple(CalibrationExpertProgram(f"{family}_{endpoint}", "continuous_ar", noise_order=1, conditioning_samples=32,
        dictionary=ResponseDictionarySpec(family, .5, 5, endpoint))
        for family, endpoint in (('gaussian', 'free'), ('cosine', 'free'), ('cosine', 'hard_zero')))


def test_endpoint_panel_is_same_ambient_space_and_explicit_capacity_change():
    panel = _dictionary_panel(experts())
    assert panel["same_ambient_cosine_for_free_and_hard"] is True
    assert panel["capacity_matching_claimed"] is False
    with pytest.raises(ValueError, match="matched"):
        _dictionary_panel((replace(experts()[0], conditioning_samples=33), *experts()[1:]))


def test_ar_panel_cannot_hide_dictionary_changes_behind_equal_n_basis():
    roster = tuple(CalibrationExpertProgram(f"p{order}-{estimator}", "continuous_ar", noise_order=order,
        conditioning_samples=32, noise_estimator=estimator, n_basis=5)
        for order in (0, 1, 4, 8, 16, 32) for estimator in ("burg", "conditional_profile"))
    changed = CalibrationExpertProgram(roster[0].expert_id, "continuous_ar", noise_order=0, conditioning_samples=32,
        dictionary=ResponseDictionarySpec("cosine", 1.2, 5))
    with pytest.raises(ValueError, match="same prefix, basis"):
        _fixed_prefix_ar_panel((changed, *roster[1:]), "fixed_prefix_ar_order")
