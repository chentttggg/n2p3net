from experiments.verify_pool_matrix import final_status, reusable

VERIFIER = {"runtime_snapshot": {"sha256": "a" * 64}, "verification_script_sha256": "b" * 64}


def test_verdict_requires_every_completed_record_to_be_verified():
    passed = {"r1": {"status": "passed"}}
    assert final_status(passed, {"r1": "completed"}, "completed") == "passed"
    assert final_status({}, {}, "completed") == "failed"
    assert final_status(passed, {"r1": "completed", "r2": "completed"}, "completed") == "failed"
    assert final_status(passed, {"r1": "completed", "r2": "failed"}, "failed") == "failed"
    assert final_status(passed, {"r1": "completed"}, "running") == "incomplete"


def test_reused_pass_must_come_from_the_same_verifier():
    previous = {"record_sha256": "c" * 64, "status": "passed", "verifier": VERIFIER}
    assert reusable(previous, "c" * 64, VERIFIER)
    assert not reusable(previous, "c" * 64, {**VERIFIER, "verification_script_sha256": "d" * 64})
    assert not reusable({**previous, "verifier": None}, "c" * 64, VERIFIER)
