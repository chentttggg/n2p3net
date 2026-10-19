"""Freeze current backend bytes and optionally materialize an isolated checkout."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from research.evaluation import (  # noqa: E402
    ExecutedSourceMismatchError,
    verified_executed_snapshot,
    verified_source_snapshot,
)
from research.execution import atomic_write_json  # noqa: E402


def freeze_source(output: Path, *, root: Path = ROOT, materialize: bool = False) -> Path:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    relative_paths = {"AGENTS.md", "pyproject.toml", "uv.lock"}
    for name in (".gitattributes", "conftest.py"):
        if (root / name).is_file():
            relative_paths.add(name)
    relative_paths.update(path.relative_to(root).as_posix() for path in (root / "src").rglob("*.py"))
    relative_paths.update(path.relative_to(root).as_posix() for path in (root / "tests").rglob("*.py"))
    for directory in ("src", "tests", "configs"):
        for pattern in ("*.json", "*.html", "*.css", "*.js", "*.LICENSE"):
            relative_paths.update(path.relative_to(root).as_posix()
                                  for path in (root / directory).rglob(pattern))
    for pattern in ("*.html", "*.css", "*.js"):
        relative_paths.update(path.relative_to(root).as_posix() for path in (root / "experiments").glob(pattern))
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


def capture_executed_run_source(output: Path, *, declared_manifest: Path, root: Path,
        entrypoint: Path) -> dict:
    """Keep one actual source archive beside new outputs, preserving old origin.

    An original archive that already matches the complete runtime is copied
    byte-for-byte, retaining its producer identity. Only a typed mismatch of
    current files triggers a new freeze; corrupt original archives still fail.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    entrypoint = Path(entrypoint).resolve()
    if not entrypoint.is_relative_to(root):
        raise ValueError('the executing entrypoint must belong to the declared runtime root')
    required = (entrypoint.relative_to(root).as_posix(),)
    declared_manifest = Path(declared_manifest).resolve()
    original = verified_source_snapshot(declared_manifest)
    declared = {'manifest': str(declared_manifest), **original}
    try:
        verified_executed_snapshot(declared_manifest, root=root, required_files=required)
    except ExecutedSourceMismatchError:
        manifest = freeze_source(output, root=root)
        mode = 'current_runtime_frozen'
    else:
        output.mkdir(parents=True, exist_ok=False)
        document = json.loads(declared_manifest.read_bytes())
        original_archive = (declared_manifest.parent/document['archive']).resolve()
        relative_archive = original_archive.relative_to(declared_manifest.parent)
        archive = output/relative_archive
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_archive, archive)
        manifest = output/'source.manifest.json'
        if Path(document['archive']).is_absolute():
            atomic_write_json(manifest, {**document, 'archive': relative_archive.as_posix()})
        else:
            shutil.copyfile(declared_manifest, manifest)
        mode = 'verified_original_source_archive_copied'
    actual = verified_executed_snapshot(manifest, root=root, required_files=required)
    result = {'schema': 'n2p3_executed_run_source/1', 'mode': mode,
        'manifest': str(manifest), **actual, 'declared_source': declared,
        'entrypoint': required[0], 'entrypoint_sha256': hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
        'source_role': 'actual_new_run_executing_runtime', 'historical_producer_rewritten': False}
    atomic_write_json(output/'execution_source.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args(argv)
    manifest = freeze_source(args.output, materialize=args.materialize)
    print(json.dumps({"manifest": str(manifest), "materialized": args.materialize}))


if __name__ == "__main__":
    main()
