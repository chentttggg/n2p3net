"""Resume keeps existing source caches while new studies may use another disk."""
from experiments.prepare_consistent_source_cohorts import _study_output


def test_existing_partial_and_recorded_external_paths_are_never_redirected(tmp_path):
    ledger_root = tmp_path / "run"
    partial = ledger_root / "basic" / "GTN"
    partial.mkdir(parents=True)
    external = tmp_path / "cache"
    assert _study_output(ledger_root, pipeline="basic", study="GTN", ledger={}, payload_root=external) == partial
    new = _study_output(ledger_root, pipeline="QC", study="GTN", ledger={}, payload_root=external)
    assert new == external / "QC" / "GTN"
    ledger = {"study_outputs": {"QC:GTN": str(new)}}
    assert _study_output(ledger_root, pipeline="QC", study="GTN", ledger=ledger,
        payload_root=tmp_path / "another") == new
