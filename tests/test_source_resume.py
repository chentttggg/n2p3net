import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from baselines.deep import DeepConfig
from experiments.run_source_training_chain import fit_or_reuse_source_trajectory
from research.evaluation import verified_source_snapshot
from research.source_resume import SourceChainResume
from tests.test_pretrain_supervised import _write_source_snapshot_manifest
from tests.test_source_training_chain import _source_dataset


def test_resume_reuses_original_bytes_and_rejects_changed_inputs_or_artifacts(tmp_path):
    dataset = _source_dataset()
    manifest, _, _ = _write_source_snapshot_manifest(tmp_path)
    snapshot = verified_source_snapshot(manifest)
    root = tmp_path / "original"
    config = DeepConfig(fixed_updates=1, batch_size=4, sampling="uniform_rows", compile_mode=None)
    acquired = fit_or_reuse_source_trajectory(registry={}, dataset=dataset, rows=np.arange(4), config=config,
        destination=root / "task", task_id="task", source_cache_sha256="a" * 64,
        source_snapshot_sha256=snapshot["archive_sha256"], excluded_source_datasets=["outer"],
        device="cpu", evidence_scope="software_correctness")
    record = {"source_snapshot": snapshot, "tasks": [{"key": "task", "status": "completed",
        "config": config.__dict__, "fit_ledger": acquired["fit_ledger"], "source_cache_sha256": "a" * 64}]}
    (root / "source_chain_record.json").write_text(json.dumps(record))
    inventory = {"schema": "n2p3_source_resume_inventory/1", "source_root": str(root),
                 "source_snapshot_manifest": str(manifest),
                 "files": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in root.rglob("*") if path.is_file()}}
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory))
    resume = SourceChainResume({"inventory": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    reused = resume.acquire("task", config, acquired["fit_ledger"], "a" * 64)
    assert reused["checkpoints"] == acquired["checkpoints"]
    assert reused["physical_optimizer_updates"] == 0
    with pytest.raises(ValueError, match="configuration, rows or cache changed"):
        resume.acquire("task", replace(config, seed=99), acquired["fit_ledger"], "a" * 64)
    next_root = tmp_path / "continuation"
    next_root.mkdir()
    record["tasks"][0]["checkpoints"] = {str(key): value for key, value in acquired["checkpoints"].items()}
    (next_root / "source_chain_record.json").write_text(json.dumps(record))
    next_inventory = {**inventory, "source_root": str(next_root),
        "files": {"source_chain_record.json": hashlib.sha256((next_root / "source_chain_record.json").read_bytes()).hexdigest()},
        "external_files": {Path(value).resolve().as_posix(): hashlib.sha256(Path(value).read_bytes()).hexdigest()
                           for value in acquired["checkpoints"].values()},
        "source_snapshot_manifests": {snapshot["archive_sha256"]: str(manifest)}}
    next_path = tmp_path / "next_inventory.json"
    next_path.write_text(json.dumps(next_inventory))
    continued = SourceChainResume({"inventory": str(next_path), "sha256": hashlib.sha256(next_path.read_bytes()).hexdigest()})
    assert continued.acquire("task", config, acquired["fit_ledger"], "a" * 64)["checkpoints"] == acquired["checkpoints"]
    group = next_root / "groups/00_outer_six"
    group.mkdir(parents=True)
    child_record = group / "source_chain_record.json"
    child_record.write_text(json.dumps(record))
    (next_root / "source_chain_record.json").write_text(json.dumps({"source_snapshot": snapshot,
        "tasks": [], "selection": [], "groups": [{"directory": str(group), "status": "running"}]}))
    next_inventory["files"] = {p.relative_to(next_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in next_root.rglob("*") if p.is_file()}
    next_path.write_text(json.dumps(next_inventory))
    interrupted = SourceChainResume({"inventory": str(next_path), "sha256": hashlib.sha256(next_path.read_bytes()).hexdigest()})
    assert interrupted.acquire("task", config, acquired["fit_ledger"], "a" * 64)["physical_optimizer_updates"] == 0
    original_path = root / "task/source.u1.pt"
    payload = torch.load(original_path, weights_only=False)
    next(iter(payload["trunk_state_dict"].values())).add_(1)
    torch.save(payload, original_path)
    with pytest.raises(ValueError, match="artifact changed"):
        resume.acquire("task", config, acquired["fit_ledger"], "a" * 64)
