"""Resume verified completed source artifacts without changing their identities."""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path

from research.contracts import semantic_sha256
from research.evaluation import verified_source_snapshot
from research.source_chain_report import load_source_chain_report, source_chain_report_reference
from transfer.checkpoint import checkpoint_training_contract, load_checkpoint_payload
from transfer.result_reuse import ResultIntegrityError, read_verified_bytes


class SourceChainResume:
    def __init__(self, declaration):
        self.inventory = None
        self.record = {}
        self.tasks = {}
        if declaration is None:
            return
        path = Path(declaration["inventory"])
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != declaration["sha256"]:
            raise ValueError("source resume inventory changed")
        self.inventory = json.loads(content)
        if self.inventory["schema"] != "n2p3_source_resume_inventory/1":
            raise ValueError("unsupported source resume inventory")
        self.root = Path(self.inventory["source_root"])
        report_path = self.root / "source_chain_record.json"
        self.record_path = source_chain_report_reference(report_path)
        self.record = load_source_chain_report(report_path, verify=self.verify)
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
                relative = source_chain_report_reference(path).resolve().relative_to(self.root.resolve()).as_posix()
                if relative not in self.inventory["files"]:
                    continue
                child = load_source_chain_report(path, verify=self.verify)
                if child["source_snapshot"]["archive_sha256"] != snapshot["archive_sha256"]:
                    raise ValueError("resumed group snapshot disagrees with its parent run")
                self.record.setdefault("tasks", []).extend(child.get("tasks", []))
                self.record.setdefault("selection", []).extend(child.get("selection", []))
        self.tasks = {task["key"]: task for task in self.record["tasks"] if task["status"] == "completed"}

    def _expected_digest(self, path):
        path = Path(path)
        key = path.resolve().as_posix()
        try:
            relative = path.resolve().relative_to(self.root.resolve()).as_posix()
            expected = self.inventory["files"].get(relative)
        except ValueError:
            expected = self.inventory.get("external_files", {}).get(key)
        if expected is None:
            raise ValueError("resumed artifact is not present in the frozen inventory")
        return expected

    def verify(self, path):
        path = Path(path)
        expected = self._expected_digest(path)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("resumed source artifact changed")
        return expected

    def acquire(self, key, config, ledger, cache_sha256, *, refit_key=None, architecture_record=None):
        if refit_key is None:
            task = self.tasks.get(key)
        else:
            candidates = [task for task in self.tasks.values() if task.get("refit_key") == list(refit_key)]
            matches = [task for task in candidates
                       if semantic_sha256(task["config"]) == semantic_sha256(asdict(config))
                       and semantic_sha256(task["fit_ledger"]) == semantic_sha256(ledger)
                       and task.get("source_cache_sha256", cache_sha256) == cache_sha256]
            if candidates and not matches:
                raise ValueError("resumed source task configuration, rows or cache changed")
            if len(matches) > 1:
                identities = {self._expected_digest(Path(task["checkpoint"])) for task in matches}
                if len(identities) > 1:
                    raise ValueError("resumed refit key has multiple distinct checkpoint producers")
                matches.sort(key=lambda task: task["key"])
            task = matches[0] if matches else None
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
        selected_update = updates[0] if selecting else config.fixed_updates
        selected_path = paths[selected_update]
        selected_object = selected_path.resolve()
        verified_paths = {}
        for path in paths.values():
            resolved = path.resolve()
            if resolved != selected_object and resolved not in verified_paths:
                verified_paths[resolved] = self.verify(path)
        expected = self._expected_digest(selected_path)
        try:
            content = read_verified_bytes(selected_path, expected)
        except ResultIntegrityError as error:
            raise ValueError("resumed source artifact changed or is unavailable") from error
        payload = load_checkpoint_payload(io.BytesIO(content))
        del content
        verified_paths[selected_object] = expected
        verified = {update: verified_paths[path.resolve()] for update, path in paths.items()}
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
                    "origin_checkpoints": {str(update): {"path": str(path), "sha256": verified[update]}
                                           for update, path in paths.items()}}}

    def prepared_source_view(self, outer, input_bindings):
        """Reuse the frozen whole-fold input even when execution budgets change."""
        if self.inventory is None:
            return None
        tasks = [task for task in self.record["tasks"]
                 if task["key"].startswith(f"{outer}/six/") and "source_input_bindings" in task]
        if not tasks:
            return None
        first = tasks[0]
        identity = (first["source_cache"], first["source_cache_sha256"], first["source_input_bindings"])
        if any((task["source_cache"], task["source_cache_sha256"], task["source_input_bindings"]) != identity
               for task in tasks) or first["source_input_bindings"] != input_bindings:
            raise ValueError("resumed whole-fold source view differs from admitted preparation inputs")
        path = Path(first["source_cache"])
        if self.verify(path) != first["source_cache_sha256"]:
            raise ValueError("resumed whole-fold source view differs from its recorded producer bytes")
        self.verify(path.with_suffix(".record.json"))
        return {"source_cache": str(path.resolve()),
                "expected_source_cache_sha256": first["source_cache_sha256"]}

    def selection(self, outer, source_seed, target_seed, axis):
        return next((row for row in self.record.get("selection", [])
                     if (row["outer"], row["source_seed"], row["target_seed"], row["axis"])
                     == (outer, source_seed, target_seed, axis) and row.get("outcomes")), None)
