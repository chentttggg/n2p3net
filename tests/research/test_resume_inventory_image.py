"""Inventory admission must parse the same bytes that matched its declaration."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from tests import PROJECT_ROOT


def test_inventory_parse_keeps_the_verified_image_when_the_path_changes(tmp_path, monkeypatch):
    relative = "src/research/source_resume.py"
    candidate = os.environ.get("N2P3_INVENTORY_REVIEW_SOURCE")
    source = Path(candidate) if candidate else PROJECT_ROOT / relative
    spec = importlib.util.spec_from_file_location("inventory_image_resume", source)
    owner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = owner
    spec.loader.exec_module(owner)
    real_sha256 = hashlib.sha256

    def view(name):
        root = tmp_path / name
        root.mkdir()
        report = root / "source_chain_record.json"
        report.write_text(json.dumps({"tasks": [], "source_snapshot": {"archive_sha256": "a" * 64},
                                      "original_producer": name}), encoding="utf-8")
        return {"schema": "n2p3_source_resume_inventory/1", "source_root": str(root),
                "source_snapshot_manifest": "original-source.json",
                "files": {report.name: real_sha256(report.read_bytes()).hexdigest()}}

    verified = view("verified")
    substituted = view("substituted")
    path = tmp_path / "inventory.json"
    raw = json.dumps(verified).encode()
    path.write_bytes(raw)
    declaration = {"inventory": str(path), "sha256": real_sha256(raw).hexdigest()}
    swapped = []

    def swap_after_inventory_read(content=b"", *args, **kwargs):
        digest = real_sha256(content, *args, **kwargs)
        if content == raw and not swapped:
            path.write_text(json.dumps(substituted), encoding="utf-8")
            swapped.append(True)
        return digest

    monkeypatch.setattr(owner.hashlib, "sha256", swap_after_inventory_read)
    monkeypatch.setattr(owner, "verified_source_snapshot", lambda _: {"archive_sha256": "a" * 64})
    resume = owner.SourceChainResume(declaration)
    assert swapped and json.loads(path.read_bytes()) == substituted
    assert resume.inventory == verified
    assert resume.root == tmp_path / "verified"
    assert resume.record["original_producer"] == "verified"
