"""Actual six-rate source gain relation and wrong-programme counterexamples."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from data.bi2014a_candidate import BI2014A_SOURCE_BIDIRECTIONAL_PROFILE
from data.epochs import preprocessing_spec_from_contract
from transfer.ensemble_views import _source_sos_family_relation
from transfer.noise_model import array_digest

ORIGINAL = json.loads((Path(__file__).parents[1] / "support/source_sos_family.json").read_bytes())
SPEC = preprocessing_spec_from_contract(BI2014A_SOURCE_BIDIRECTIONAL_PROFILE)


def relation(source, compiled, *, rate, padding, spec=SPEC, compiled_padding=None):
    return _source_sos_family_relation(source, compiled, spec, source_rate_hz=rate,
        actual_padding=padding, recompiled_padding=padding if compiled_padding is None else compiled_padding)


@pytest.mark.parametrize("rate", [1000., 256., 250., 512., 128., 2048.])
def test_actual_original_source_six_rate_relation_preserves_both_coefficient_identities(rate):
    row = ORIGINAL["six_native_rates"][str(rate)]
    source, compiled = np.asarray(row["source_sos"]), np.asarray(row["consumer_sos"])
    receipt = relation(source, compiled, rate=rate, padding=row["source_padding"])
    assert receipt["source_sos_sha256"] == array_digest(source)
    assert receipt["recompiled_sos_sha256"] == array_digest(compiled)
    assert receipt["coefficient_replacement_performed"] is False
    if rate in {250., 512.}:
        assert receipt["policy"] == "exact_coefficients/1"
    else:
        assert receipt["scalar_gain_ulp_distance"] == (3 if rate == 128. else 1)
        assert receipt["gain_section"] == 1 and receipt["all_other_coefficients_exact"]


def test_scalar_family_policy_accepts_its_declared_four_ulp_boundary():
    row = ORIGINAL["six_native_rates"]["128.0"]
    compiled = np.asarray(row["consumer_sos"])
    source = compiled.copy()
    gain = compiled[1, 0]
    for _ in range(4):
        gain = np.nextafter(gain, np.inf)
    source[1, :3] = [gain, 2 * gain, gain]
    receipt = relation(source, compiled, rate=128., padding=row["consumer_padding"])
    assert receipt["scalar_gain_ulp_distance"] == receipt["maximum_scalar_gain_ulp_distance"] == 4


@pytest.mark.parametrize("violation", ["gain_five_ulp", "pole", "gain_shape", "notch", "padding", "notch_stack", "unsupported_order"])
def test_wrong_source_programme_cannot_use_the_scalar_gain_family_policy(violation):
    row = ORIGINAL["six_native_rates"]["128.0"]
    compiled = np.asarray(row["consumer_sos"])
    source, spec, padding = compiled.copy(), SPEC, row["consumer_padding"]
    if violation == "gain_five_ulp":
        gain = compiled[1, 0]
        for _ in range(5):
            gain = np.nextafter(gain, np.inf)
        source[1, :3] = [gain, 2 * gain, gain]
    elif violation == "pole":
        source[1, 4] = np.nextafter(source[1, 4], np.inf)
    elif violation == "gain_shape":
        source[1, 1] = np.nextafter(source[1, 1], np.inf)
    elif violation == "notch":
        source[0, 0] = np.nextafter(source[0, 0], np.inf)
    elif violation == "padding":
        padding += 1
    else:
        source = np.asarray(row["source_sos"])
        spec = replace(SPEC, notch_freqs_hz=()) if violation == "notch_stack" else replace(SPEC, filter_order=2)
    with pytest.raises(ValueError, match="source finite SOS programme"):
        relation(source, compiled, rate=128., padding=padding, spec=spec, compiled_padding=row["consumer_padding"])
