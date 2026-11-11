import numpy as np
import pytest

from models.candidate_evidence import CandidateExpertIdentity
from research.target_recognition import (
    REQUIRED_EXPERTS,
    compare_common_scored_decisions,
    score_nine_choice_recording,
)


def _score(missing=False):
    codes = np.tile(np.arange(1, 10).astype(str), 5)
    logits = {name: np.where(codes == "7", 2., 0.) for name in REQUIRED_EXPERTS}
    logits["nn_scratch"][:] = 0.  # A tied distribution supplies no unique target choice.
    if missing:
        logits["nn_source"][0] = np.nan
    identities = {name: CandidateExpertIdentity(name, "a"*64, "b"*64, None, "c"*64, "membership_llr_sum/1", False)
                  for name in REQUIRED_EXPERTS}
    return score_nine_choice_recording(session_id="S1", participant_key="fixture:p1",
        event_ids=[f"e{i}" for i in range(45)], candidate_ids=codes, event_logits=logits,
        expert_identities=identities, raw_support_digest="d"*64, revealed_target="7")


def test_missing_one_flash_changes_r1_and_r5_coverage_without_turning_unscored_into_errors():
    result = _score(missing=True)["panels"]
    r1 = result["R1"]["nn_source"]["metrics"]
    assert r1["requested_decisions"] == 5 and r1["scored_decisions"] == 4
    assert r1["accuracy"] == 1 and r1["coverage"] == .8
    assert result["R5"]["nn_source"]["metrics"]["accuracy"] is None
    assert result["R5"]["nn_source"]["metrics"]["coverage"] == 0
    assert result["R1"]["nn_scratch"]["metrics"]["accuracy"] is None
    assert result["R1"]["three_experts"]["metrics"]["scored_decisions"] == 4
    assert set(result["R1"]) == {"nn_source", "nn_scratch", "xdawn_erp", "three_experts"}


def test_common_scored_comparison_keeps_both_arms_on_identical_physical_requests():
    basic = _score()["panels"]["R1"]["nn_source"]["decisions"]
    qc = _score(missing=True)["panels"]["R1"]["nn_source"]["decisions"]
    comparison = compare_common_scored_decisions(basic, qc)
    assert comparison["requested_decisions"] == 5 and comparison["common_scored_decisions"] == 4
    assert comparison["left"]["accuracy"] == comparison["right"]["accuracy"] == 1
    with pytest.raises(ValueError, match="same unique"):
        compare_common_scored_decisions(basic, qc[1:])
