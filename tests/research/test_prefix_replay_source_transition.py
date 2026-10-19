"""New replay transport remains an exact reviewed source transition."""
import hashlib
import json

import pytest

from research.contracts import semantic_sha256
from research.prefix_recovery import _reviewed_transition, prefix_programme_authority
from research.source_execution import with_cloud_resource_floor

TRANSPORT_ROLES = {
    "src/research/source_execution.py": "resource_transport",
    "src/research/prefix_replay_dispatch.py": "resource_transport",
    "src/research/prefix_replay_execution.py": "load_replay_dispatch",
    "src/research/prefix_replay_inputs.py": "immutable_input_transport",
    "src/transfer/ensemble_views.py": "immutable_input_transport",
}


def transition(tmp_path, changes):
    producer = {"src/transfer/continuous_decision.py": "a" * 64,
                "src/research/source_execution.py": "b" * 64,
                "src/transfer/ensemble_views.py": "c" * 64}
    consumer = {**producer, **{name: "d" * 64 for name in changes}}
    rows = [{"path": name, "producer_sha256": producer.get(name), "consumer_sha256": consumer[name],
             "role": role} for name, role in changes.items()]
    record = {"schema": "n2p3_prefix_reviewed_source_transition/1", "review_status": "accepted",
        "producer_source_digest": semantic_sha256(producer), "consumer_source_digest": semantic_sha256(consumer),
        "changes": rows}
    path = tmp_path / "transition.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    receipt = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return receipt, producer, consumer


def test_exact_new_replay_transport_roster_is_admitted(tmp_path):
    receipt, producer, consumer = transition(tmp_path, TRANSPORT_ROLES)
    accepted = _reviewed_transition(receipt, producer, consumer)
    assert {row["path"] for row in accepted["changes"]} == set(TRANSPORT_ROLES)


@pytest.mark.parametrize("name", ["src/transfer/continuous_decision.py", "src/models/n2p3net.py"])
def test_transport_changes_do_not_grant_scientific_model_edits(tmp_path, name):
    receipt, producer, consumer = transition(tmp_path, {**TRANSPORT_ROLES, name: "resource_transport"})
    with pytest.raises(ValueError, match="reviewed admission/resource modules"):
        _reviewed_transition(receipt, producer, consumer)


def test_replay_resource_knobs_preserve_original_scientific_authority():
    original = {"budgets": [0, 200, 800, 2000], "seed_pairs": [[20260918, 20260918]],
                "prefix_selection": {"minimum_selection_rows": 3}}
    consumer = {**original, "prefix_replay_chunk_size": 2, "prefix_replay_graph_memory_gib": 8,
                "prefix_replay_worker_memory_gib": 4}
    assert prefix_programme_authority(consumer) == prefix_programme_authority(original)
    changed = {**consumer, "prefix_selection": {"minimum_selection_rows": 1}}
    assert prefix_programme_authority(changed) != prefix_programme_authority(original)


@pytest.mark.parametrize("floor", ["omitted", None])
def test_inherited_cuda_floor_keeps_the_original_scientific_programme(floor):
    original = {"budgets": [0, 200, 800, 2000], "seed_pairs": [[20260918, 20260918]]}
    if floor != "omitted":
        original["cloud_resource_floor"] = floor
    consumer = with_cloud_resource_floor(original, "cuda")
    assert consumer["cloud_resource_floor"] == {"cpu_threads": 25, "gpu_workers": 4}
    assert prefix_programme_authority(original) == prefix_programme_authority(consumer)
    assert prefix_programme_authority({**consumer, "budgets": [0, 800]}) != prefix_programme_authority(original)
