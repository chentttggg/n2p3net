"""Freeze the active source contract and full local proof closure for one check."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LEAN = REPO / "formal/lean"
CHECKPOINT = REPO / "doc/assets/lean_gaussian_20261016/active_checkpoint.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def local_closure(start: str) -> dict[str, str]:
    pending = [start]
    result = {}
    while pending:
        name = pending.pop()
        if name in result:
            continue
        path = LEAN / name
        result[name] = sha(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"\s*import\s+(.+)", line)
            if match:
                # One Lean import statement may contain several module names.
                for imported in re.findall(r"[A-Za-z_][\w.]*", match[1].split("--", 1)[0]):
                    dependency = imported.replace(".", "/") + ".lean"
                    if (LEAN / dependency).is_file():
                        pending.append(dependency)
    return result


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("file", "target"))
    parser.add_argument("target")
    parser.add_argument("artifact")
    parser.add_argument("--pin", action="append", default=[], help="Extra repository contract or evidence file to freeze")
    args = parser.parse_args()
    assert re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.artifact)
    if args.mode == "target":
        assert re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", args.target)
        start = args.target.replace(".", "/") + ".lean"
    else:
        assert re.fullmatch(r"[A-Za-z_]\w*(?:/[A-Za-z_]\w*)*\.lean", args.target)
        start = args.target
    output = LEAN / "artifacts" / args.artifact
    assert not output.exists(), "Use a new artifact name; previous execution evidence is immutable"
    checkpoint = json.loads(CHECKPOINT.read_bytes())
    selected = checkpoint["active_source_contract"]
    source_path = REPO / selected["path"]
    assert sha(source_path) == selected["sha256"]
    source = json.loads(source_path.read_bytes())
    manifest_path = REPO / checkpoint["frozen_manifest"]
    manifest_hash = sha(manifest_path)
    inherited = json.loads(manifest_path.read_bytes())["members_sha256"]
    closure = local_closure(start)
    pins = {str(REPO / name): expected for name, expected in source["source_sha256"].items()}
    pins.update({str(LEAN / name): expected for name, expected in inherited.items()})
    pins.update({str(LEAN / name): expected for name, expected in closure.items()})
    pins.update({str(source_path): selected["sha256"], str(manifest_path): manifest_hash,
                 str(Path(__file__)): sha(Path(__file__))})
    for name in args.pin:
        path = (REPO / name).resolve()
        assert path.is_relative_to(REPO.resolve()) and path.is_file()
        pins[str(path)] = sha(path)

    def observe() -> dict:
        assert json.loads(CHECKPOINT.read_bytes())["active_source_contract"] == selected
        image = {name: sha(Path(name)) for name in pins}
        assert image == pins
        ordinary = {p.relative_to(REPO).as_posix() for p in (REPO / "src").rglob("*.py")} | {
            p.relative_to(REPO).as_posix() for p in (REPO / "experiments").glob("*.py")}
        assert ordinary == set(source["baseline_runtime_source_paths"])
        assert ordinary | set(source["additional_runtime_source_paths"]) == set(source["runtime_source_paths"])
        return {"utc": datetime.now(UTC).isoformat(), "all_pins_match": True,
                "observed_sha256": image, "runtime_count": len(source["runtime_source_paths"])}

    before = observe()
    records = REPO / "doc/assets/lean_current_checks_20261018"
    records.mkdir(exist_ok=True)
    driver = records / (args.artifact + "_driver.py")
    with driver.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    assert sha(driver) == pins[str(Path(__file__))]
    extra_snapshot = None
    if args.pin:
        extra_snapshot = records / (args.artifact + "_extra_inputs.zip")
        with zipfile.ZipFile(extra_snapshot, "x", compression=zipfile.ZIP_DEFLATED) as stream:
            for name in args.pin:
                path = (REPO / name).resolve()
                stream.write(path, path.relative_to(REPO).as_posix())
        with zipfile.ZipFile(extra_snapshot) as stream:
            assert stream.testzip() is None
            assert all(hashlib.sha256(stream.read(Path(n).as_posix())).hexdigest() == pins[str((REPO / n).resolve())]
                       for n in args.pin)
    snapshot = records / (args.artifact + "_source.zip")
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as stream:
        for name in closure:
            stream.writestr(name, (LEAN / name).read_bytes())
    with zipfile.ZipFile(snapshot) as stream:
        assert stream.testzip() is None
        assert all(hashlib.sha256(stream.read(n)).hexdigest() == h for n, h in closure.items())
    plan_path = records / (args.artifact + "_plan.json")
    write_new(plan_path, {"mode": args.mode, "target": args.target, "artifact": args.artifact,
        "all_pins_sha256": pins, "local_import_closure_sha256": closure,
        "source_contract": selected, "accepted_manifest": checkpoint["frozen_manifest"],
        "extra_pins": args.pin,
        "driver_snapshot": driver.relative_to(REPO).as_posix(), "driver_snapshot_sha256": sha(driver),
        "extra_snapshot": extra_snapshot.relative_to(REPO).as_posix() if extra_snapshot else None,
        "extra_snapshot_sha256": sha(extra_snapshot) if extra_snapshot else None,
        "accepted_manifest_sha256": manifest_hash, "accepted_members": len(inherited),
        "source_count": len(source["source_sha256"]), "runtime_count": len(source["runtime_source_paths"]),
        "source_snapshot": snapshot.relative_to(REPO).as_posix(), "source_snapshot_sha256": sha(snapshot)})
    command = [sys.executable, "-B", str(LEAN / "scripts/run_lean.py"),
               "--" + args.mode, args.target, "--output", str(output)]
    process = subprocess.run(command, cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=False)
    after = observe()
    report = output / "report.json"
    log = output / "lean.log"
    measurement = json.loads(report.read_bytes())
    write_new(records / (args.artifact + "_result.json"), {
        "schema": "n2p3_current_source_check/1", "plan_sha256": sha(plan_path), "command": command,
        "before": before, "after": after, "report": report.relative_to(REPO).as_posix(),
        "report_sha256": sha(report), "log_sha256": sha(log), "log_bytes": log.stat().st_size,
        "measurement": measurement, "exit_code": process.returncode, "not_component_acceptance": True})
    print(json.dumps({"exit_code": process.returncode, "log_bytes": log.stat().st_size,
                      "wall_seconds": measurement["wall_seconds"], "local_imports": len(closure),
                      "source_count": len(source["source_sha256"]), "accepted_members": len(inherited)}), flush=True)
    if log.stat().st_size:
        print(log.read_text(encoding="utf-8"), flush=True)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
