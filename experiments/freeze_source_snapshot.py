"""Freeze current backend bytes and optionally materialize an isolated checkout."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from research.evaluation import verified_executed_snapshot  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402


def freeze_source(output: Path, *, root: Path = ROOT, materialize: bool = False) -> Path:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    relative_paths = {"AGENTS.md", "pyproject.toml", "uv.lock"}
    if (root / "conftest.py").is_file():
        relative_paths.add("conftest.py")
    relative_paths.update(path.relative_to(root).as_posix() for path in (root / "src").rglob("*.py"))
    relative_paths.update(path.relative_to(root).as_posix() for path in (root / "tests").rglob("*.py"))
    for directory, pattern in (("experiments", "*.py"), ("tests", "*.py"), ("configs", "*.json"), ("doc", "*.md")):
        relative_paths.update(path.relative_to(root).as_posix() for path in (root / directory).glob(pattern))
    files = {name: (root / name).read_bytes() for name in sorted(relative_paths)}
    archive = output / "source.source-only.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            stream.addfile(info, io.BytesIO(content))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines()
    manifest = output / "source.manifest.json"
    atomic_write_json(manifest, {
        "schema": "n2p3_source_freeze/1", "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "byte_size": archive.stat().st_size, "member_count": len(files),
        "source_commit": commit, "working_tree_snapshot": True, "source_commit_role": "base_commit_only",
        "git_status": status, "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
    })
    verified_executed_snapshot(manifest, root=root)
    if materialize:
        checkout = output / "checkout"
        with tarfile.open(archive, "r:gz") as stream:
            stream.extractall(checkout, filter="data")
        verified_executed_snapshot(manifest, root=checkout)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args(argv)
    manifest = freeze_source(args.output, materialize=args.materialize)
    print(json.dumps({"manifest": str(manifest), "materialized": args.materialize}))


if __name__ == "__main__":
    main()
