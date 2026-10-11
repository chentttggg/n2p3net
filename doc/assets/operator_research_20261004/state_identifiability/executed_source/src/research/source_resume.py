"""Resume verified completed source artifacts without changing their identities."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from research.contracts import semantic_sha256
from research.evaluation import verified_source_snapshot
from transfer.checkpoint import checkpoint_training_contract, load_checkpoint_payload


class SourceChainResume:
    def __init__(self, declaration):
        self.inventory = None
        self.record = {}
        self.tasks = {}
        self._verified = {}
        if declaration is None:
            return
        path = Path(declaration["inventory"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != declaration["sha256"]:
            raise ValueError("source resume inventory changed")
        self.inventory = json.loads(path.read_text(encoding="utf-8"))
        if self.inventory["schema"] != "n2p3_source_resume_inventory/1":
            raise ValueError("unsupported source resume inventory")
        self.root = Path(self.inventory["source_root"])
        self.record_path = self.root / "source_chain_record.json"
        self.verify(self.record_path)
        self.record = json.loads(self.record_path.read_text(encoding="utf-8"))
        snapshot = verified_source_snapshot(self.inventory["source_snapshot_manifest"])
        if snapshot["archive_sha256"] != self.record["source_snapshot"]["archive_sha256"]:
            raise ValueError("resumed source snapshot disagrees with the original run")
        self._source_snapshots = {snapshot["archive_sha256"]}
        for digest, manifest in self.inventory.get("source_snapshot_manifests", {}).items():
            if verified_source_snapshot(manifest)["archive_sha256"] != digest:
                raise ValueError("resumed source snapshot allowlist changed")
            self._source_snapshots.add(digest)
        if self.record.get("groups") and not self.record.get("tasks"):
            for group in self.record["groups"]:
                path = Path(group["directory"]) / "source_chain_record.json"
                relative = path.resolve().relative_to(self.root.resolve()).as_posix()
                if relative not in self.inventory["files"]:
                    continue
                self.verify(path)
                child = json.loads(path.read_text(encoding="utf-8"))
                if child["source_snapshot"]["archive_sha256"] != snapshot["archive_sha256"]:
                    raise ValueError("resumed group snapshot disagrees with its parent run")
                self.record.setdefault("tasks", []).extend(child.get("tasks", []))
                self.record.setdefault("selection", []).extend(child.get("selection", []))
        self.tasks = {task["key"]: task for task in self.record["tasks"] if task["status"] == "completed"}

    def verify(self, path):
        path = Path(path)
        key = path.resolve().as_posix()
        try:
            relative = path.resolve().relative_to(self.root.resolve()).as_posix()
            expected = self.inventory["files"].get(relative)
        except ValueError:
            expected = self.inventory.get("external_files", {}).get(key)
        if expected is None:
            raise ValueError("resumed artifact is not present in the frozen inventory")
        stat = path.stat()
        version = (stat.st_size, stat.st_mtime_ns)
        if self._verified.get(key) != version:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError("resumed source artifact changed")
            self._verified[key] = version
        return expected

    def acquire(self, key, config, ledger, cache_sha256, *, refit_key=None, architecture_record=None):
        task = (self.tasks.get(key) if refit_key is None else next((task for task in self.tasks.values()
                if task.get("refit_key") == list(refit_key)), None))
        if task is None:
            return None
        if (semantic_sha256(task["config"]) != semantic_sha256(asdict(config))
                or semantic_sha256(task["fit_ledger"]) != semantic_sha256(ledger)
                or task.get("source_cache_sha256", cache_sha256) != cache_sha256):
            raise ValueError("resumed source task configuration, rows or cache changed")
        selecting = config.global_early_stopping is not None
        updates = (sorted(int(value) for value in task["checkpoints"]) if selecting else
                   sorted(set(config.snapshot_updates) | {config.fixed_updates}))
        paths = {update: (Path(task["checkpoints"][str(update)]) if task.get("checkpoints") else
                          self.root / key / f"source.u{update}.pt") if refit_key is None
                 else Path(task["checkpoint"]) for update in updates}
        for path in paths.values():
            self.verify(path)
        payload = load_checkpoint_payload(paths[updates[0] if selecting else config.fixed_updates])
        if selecting and payload.get("global_early_stopping") != config.global_early_stopping.record():
            raise ValueError("resumed checkpoint has a different global early-stopping policy")
        contract = checkpoint_training_contract(payload)
        if architecture_record is not None and semantic_sha256(contract.architecture) != semantic_sha256(architecture_record):
            raise ValueError("resumed checkpoint architecture differs from the requested source model")
        if (contract.source_cache_sha256 != cache_sha256
                or contract.source_snapshot_sha256 not in self._source_snapshots
                or set(contract.training_participant_keys) != set(ledger["training_participant_keys"])):
            raise ValueError("resumed checkpoint no longer matches the requested fit ledger")
        return {"checkpoints": {update: str(path) for update, path in paths.items()}, "fit_ledger": ledger,
                "physical_optimizer_updates": 0, "physical_trial_draws": 0,
                "source_training_reuse": {"reused": True, "kind": "completed_previous_run",
                    "origin_record": str(self.record_path), "origin_task": task["key"],
                    "origin_source_snapshot_sha256": payload["source_snapshot_sha256"],
                    "origin_checkpoints": {str(update): {"path": str(path), "sha256": self.verify(path)}
                                           for update, path in paths.items()}}}

    def selection(self, outer, source_seed, target_seed, axis):
        return next((row for row in self.record.get("selection", [])
                     if (row["outer"], row["source_seed"], row["target_seed"], row["axis"])
                     == (outer, source_seed, target_seed, axis) and row.get("outcomes")), None)
