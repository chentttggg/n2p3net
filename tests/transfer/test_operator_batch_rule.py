import pytest

from transfer.target_experiments import operator_fixed_target_batch_rule, resolve_target_batch_rule


def test_operator_configuration_cannot_claim_user_instruction_or_score_selection():
    rule = operator_fixed_target_batch_rule(batch_size=512, rationale="Bound small calibration fits")
    assert resolve_target_batch_rule({"batch_rule": rule}) == (512, rule)
    assert not rule["selection_evidence"]
    for patch in ({"provenance": "user_explicit"}, {"selection_evidence": True}, {"rationale": ""}):
        with pytest.raises(ValueError, match="operator-fixed"):
            resolve_target_batch_rule({"batch_rule": {**rule, **patch}})
    with pytest.raises(ValueError, match="frozen operator-fixed"):
        resolve_target_batch_rule({"batch_rule": rule}, requested_batch=32, dry_run=True)
