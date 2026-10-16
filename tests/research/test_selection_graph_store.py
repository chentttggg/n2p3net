"""Complete scope storage preserves logical graphs and independent ingress."""
import json

import pytest

from research.contracts import semantic_sha256
from research.frozen_metadata import freeze_metadata


def graph_fixture():
    scope = freeze_metadata({"actual_K": 12, "input": {"budget": 12,
        "target_contract": {"source_sha256": "a" * 64},
        "complete_groups": [{"membership": [[True, False] * 18] * 12,
            "native_labels": [True, False] * 6}]}})
    return freeze_metadata({"schema": "n2p3_whole_prefix_selection_replay/1",
        "selection": {"parent_scope": scope, "final_refit_scope": scope,
            "outer_requested": ["one", "two"], "failures": [{"kind": "incomplete"}]},
        "nodes": [{"fit_key": str(i), "scope": scope} for i in range(15)]})


def test_complete_scope_pool_roundtrip_shares_scope_and_keeps_logical_hash(tmp_path):
    from research.selection_graph_store import SelectionGraphStore, save_selection_graph

    graph = graph_fixture()
    path = tmp_path / "prefix_selection.json"
    save_selection_graph(path, graph)
    raw = json.loads(path.read_bytes())
    assert raw["schema"] == "n2p3_selection_graph_store/2" and len(raw["scopes"]) == 1
    owner = SelectionGraphStore(path, expected_graph_sha256=semantic_sha256(graph),
        expected_selection_sha256=semantic_sha256(graph["selection"]))
    restored = owner.graph
    assert semantic_sha256(restored) == semantic_sha256(graph)
    assert restored == graph
    assert restored["nodes"][0]["scope"] is restored["nodes"][-1]["scope"]
    assert restored["nodes"][0]["scope"] is restored["selection"]["parent_scope"]
    assert restored["selection"]["outer_requested"] == ("one", "two")
    assert restored["selection"]["failures"] == graph["selection"]["failures"]
    with pytest.raises(AttributeError):
        owner.graph = {"caller": "cannot replace admitted state"}
    with pytest.raises(TypeError, match="frozen"):
        owner.graph["selection"]["outer_requested"] = []


@pytest.mark.parametrize("change", ["scope_bytes", "missing_scope", "unknown_ref", "drop_request", "external_sha"])
def test_fresh_scope_ingress_rejects_changed_complete_values_or_denominator(tmp_path, change):
    from research.selection_graph_store import SelectionGraphStore, save_selection_graph

    graph = graph_fixture()
    path = tmp_path / "prefix_selection.json"
    save_selection_graph(path, graph)
    payload = json.loads(path.read_bytes())
    scope_path = tmp_path / next(iter(payload["scopes"].values()))["path"]
    if change == "scope_bytes":
        original = scope_path.read_bytes()
        # Same file size and no filename/mtime heuristics can authorize changed source.
        scope_path.write_bytes(original.replace(b'a' * 64, b'b' * 64))
    elif change == "missing_scope":
        scope_path.unlink()
    elif change == "unknown_ref":
        payload["record"]["nodes"][0]["scope"] = {"$n2p3_scope": "f" * 64}
    elif change == "drop_request":
        payload["record"]["selection"]["outer_requested"].pop()
    path.write_text(json.dumps(payload), encoding="utf-8")
    expected = "0" * 64 if change == "external_sha" else semantic_sha256(graph)
    with pytest.raises((ValueError, FileNotFoundError)):
        SelectionGraphStore(path, expected_graph_sha256=expected)


def test_scope_owner_reuses_only_original_immutable_authority_and_checks_headers(tmp_path, monkeypatch):
    from dataclasses import replace

    from research import prefix_inner_replay as owner
    from tests.support.independent_prefix import independent_prefix_case
    from train.runtime import cpu_thread_budget

    with cpu_thread_budget(1):
        views = independent_prefix_case(tmp_path)[-1]
        prefix = views.calibration[200]
        scopes = owner._PrefixScopeOwner(views.prefix_compiler)
        first = scopes.scope(prefix)
        assert first["input"] == freeze_metadata(owner.prefix_input_identity(prefix))
        foreign = replace(prefix, target_contract={**prefix.target_contract,
            "source_relative_path": "same-K-foreign-source.mat"})
        with pytest.raises(ValueError, match="original source"):
            scopes.scope(foreign)
        original = owner.prefix_input_identity

        def no_rehash(value):
            raise AssertionError("same owned immutable prefix was hashed again")

        monkeypatch.setattr(owner, "prefix_input_identity", no_rehash)
        assert scopes.scope(prefix) is first
        array = prefix.raw.samples
        shape = array.shape
        array.shape = (array.size,)
        try:
            with pytest.raises(ValueError, match="array header|authority changed"):
                scopes.scope(prefix)
        finally:
            array.shape = shape
        monkeypatch.setattr(owner, "prefix_input_identity", original)
        assert scopes.scope(views.calibration[0]) is not first


def test_seed_admission_decodes_actual_checkpoint_once_and_rejects_another_seed(monkeypatch):
    import io
    from types import SimpleNamespace

    from research.prefix_inner_replay import _check_seeds, _SourceSeedAdmission
    from transfer.ensemble_fit import FrozenSourceCheckpoint

    reads = []

    def load(stream):
        assert isinstance(stream, io.BytesIO)
        value = stream.read()
        reads.append(value)
        return {"training_contract": {"seed": 4 if value == b"first" else 5}}

    monkeypatch.setattr("transfer.checkpoint.load_checkpoint_payload", load)
    spec = SimpleNamespace(recipe=SimpleNamespace(seed=9))
    sources = {"source": FrozenSourceCheckpoint(b"first")}
    admission = _SourceSeedAdmission()
    expected = _check_seeds((spec,), sources, {"source_seed": 4, "target_seed": 9}, admission=admission)
    assert _check_seeds((spec,), sources, {"source_seed": 4, "target_seed": 9}, admission=admission) == expected
    copied = FrozenSourceCheckpoint(bytes(bytearray(b"first")))
    assert copied.content is not sources["source"].content
    assert _check_seeds((spec,), {"source": copied}, {"source_seed": 4, "target_seed": 9}, admission=admission) == expected
    assert reads == [b"first"]
    with pytest.raises(ValueError, match="source seed"):
        _check_seeds((spec,), {"source": FrozenSourceCheckpoint(b"second")},
            {"source_seed": 4, "target_seed": 9}, admission=admission)
    assert reads == [b"first", b"second"]


def test_legacy_import_keeps_original_bytes_and_logical_seal_in_a_new_namespace(tmp_path):
    import hashlib

    from research.frozen_metadata import plain_metadata
    from research.selection_graph_store import SelectionGraphStore, import_legacy_selection_graph

    graph = graph_fixture()
    old = tmp_path / "original" / "prefix_selection.json"
    old.parent.mkdir()
    old.write_text(json.dumps({"record": plain_metadata(graph), "record_sha256": semantic_sha256(graph)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    original = old.read_bytes()
    receipt = {"path": str(old.resolve()), "sha256": hashlib.sha256(original).hexdigest(), "bytes": len(original)}
    new = tmp_path / "consumer" / "prefix_selection.json"
    imported = import_legacy_selection_graph(receipt, new, expected_graph_sha256=semantic_sha256(graph))
    assert old.read_bytes() == original and imported["new_fits"] == imported["EEG_PT_decodes"] == 0
    restored = SelectionGraphStore(new, expected_graph_sha256=semantic_sha256(graph))
    assert restored.graph == graph and len(restored.scope_receipts) == 1
    assert imported["token_cache"]["bytes"] < len(original)
    assert new.stat().st_size < len(original)
    with pytest.raises(ValueError, match="separate consumer"):
        import_legacy_selection_graph(receipt, old.parent / "another.json", expected_graph_sha256=semantic_sha256(graph))


def test_legacy_scanner_keeps_escape_unicode_and_whitespace_across_block_boundary(tmp_path):
    import hashlib

    from research.selection_graph_store import _legacy_minified_chunks

    # A literal's escaped quote crosses the exact one-MiB source block boundary.
    prefix = b'{"long":"'
    value = prefix + b'x' * (1024 * 1024 - len(prefix) - 1) + b'\\" tail with spaces ", "unicode": "\xe4\xb8\xad\xe6\x96\x87"}\n'
    path = tmp_path / "edge.json"
    path.write_bytes(value)
    receipt = {"path": str(path), "sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    minified = b''.join(_legacy_minified_chunks(receipt))
    assert json.loads(minified) == json.loads(value)
    assert b'tail with spaces' in minified
    receipt["sha256"] = 'f' * 64
    with pytest.raises(ValueError, match="original external seal"):
        b''.join(_legacy_minified_chunks(receipt))


def test_prepared_store_copies_complete_scopes_from_original_locator_once(tmp_path):
    import hashlib

    from research.selection_graph_store import (
        SelectionGraphStore,
        prepare_selection_graph_store,
        save_selection_graph,
    )

    graph = graph_fixture()
    original = tmp_path / "original" / "prefix_selection.json"
    save_selection_graph(original, graph)
    before = original.read_bytes()
    receipt = {"path": str(original.resolve()), "sha256": hashlib.sha256(before).hexdigest(), "bytes": len(before)}
    destination = tmp_path / "prepared" / "graph.json"
    owned, imported = prepare_selection_graph_store(receipt, destination,
        expected_graph_sha256=semantic_sha256(graph))
    assert owned.graph == graph and len(owned.scope_receipts) == 1
    owned.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = owned.graph
    assert original.read_bytes() == before and imported["consumer_store"]["path"] == str(destination.resolve())
    # The prepared compact graph owns its complete scope pool, not relative
    # sidecars accidentally still borrowed from the original package directory.
    next((original.parent / "graph_scopes").iterdir()).unlink()
    assert SelectionGraphStore(destination).graph == graph


@pytest.mark.parametrize("failure", ("header_receipt", "cache_receipt"))
def test_post_publication_receipt_failure_releases_actual_resident_lease(tmp_path, monkeypatch, failure):
    import multiprocessing
    from pathlib import Path

    from research import selection_graph_store as owner
    from research.source_execution import SourceInputAdmission

    graph = graph_fixture()
    source, destination = tmp_path / "original" / "graph.json", tmp_path / "prepared" / "graph.json"
    owner.save_selection_graph(source, graph)
    before = source.read_bytes()
    receipt = {"path": str(source.resolve()), **owner._artifact(source)}
    actual_artifact = owner._artifact
    target = destination if failure == "header_receipt" else destination.with_name("graph.legacy-minified.json")

    def receipt_failure(path):
        if Path(path) == target.resolve():
            raise OSError("post-publication receipt read failed")
        return actual_artifact(path)

    with multiprocessing.Manager() as manager:
        admission = SourceInputAdmission(manager, capacity_bytes=64 * 2**20, load_workers=1)
        prepared, _ = owner.prepare_selection_graph_store(receipt, tmp_path / "positive" / "graph.json",
            expected_graph_sha256=semantic_sha256(graph), input_admission=admission)
        assert admission.record()["resident_input_bytes"] > 0
        prepared.close()
        assert admission.record()["resident_input_bytes"] == 0
        monkeypatch.setattr(owner, "_artifact", receipt_failure)
        with pytest.raises(OSError, match="post-publication receipt"):
            owner.prepare_selection_graph_store(receipt, destination,
                expected_graph_sha256=semantic_sha256(graph), input_admission=admission)
        assert admission.record()["resident_input_bytes"] == 0
    assert source.read_bytes() == before and destination.exists() and target.exists()
