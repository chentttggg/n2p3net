"""Read-only cleanup inventory; writes evidence beside this script, never deletes."""

import hashlib
import json
import os
import stat
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("D:/BCI/n2p3-net")
OUT = Path(__file__).resolve().parent
QUALITY = ROOT / "experiments/runs/brainsync_quality_20261030/implementation"
OLD = Path("D:/BCI/artifacts/apps/BrainSync_Release_20260920")
NEW = Path("D:/BCI/artifacts/apps/BrainSync_Release_20261102")
TEMP = Path("D:/BCI/tmp/brainsync_basic_entry_20261102")
TARGETS = {
    "BCI-001": ROOT / "build",
    "BCI-002": QUALITY / "final",
    "BCI-003": OLD / "_internal",
    "BCI-004": OLD / "BrainSync_Standalone.exe",
    "BCI-005": TEMP,
}
EXPECTED = {
    "BCI-001": (285, 6584163),
    "BCI-002": (3, 57936),
    "BCI-003": (486, 308676124),
    "BCI-004": (1, 120236664),
    "BCI-005": (1136, 717280797),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def is_reparse(info):
    return bool(info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def check_ancestors(path):
    checked = []
    for item in reversed([path, *path.parents]):
        if not os.path.lexists(item):
            continue
        assert not is_reparse(item.lstat()), f"Reparse ancestor: {item}"
        checked.append(str(item))
    assert path.resolve() == path, f"Physical path mismatch: {path}"
    return checked


def snapshot(path, excluded=()):
    check_ancestors(path)
    if not os.path.lexists(path):
        return {"path": str(path), "exists": False}
    records = []
    pending = [path]
    while pending:
        item = pending.pop()
        if item in excluded:
            continue
        info = item.lstat()
        relative = "." if item == path else item.relative_to(path).as_posix()
        if is_reparse(info):
            records.append({"relative": relative, "type": "reparse", "target": os.readlink(item)})
        elif stat.S_ISDIR(info.st_mode):
            records.append({"relative": relative, "type": "directory"})
            pending.extend(sorted(item.iterdir(), reverse=True))
        else:
            assert stat.S_ISREG(info.st_mode), f"Unexpected file type: {item}"
            records.append({"relative": relative, "type": "file", "bytes": info.st_size,
                            "sha256": digest(item)})
    files = [item for item in records if item["type"] == "file"]
    return {"path": str(path), "exists": True, "files": len(files),
            "logical_bytes": sum(item["bytes"] for item in files),
            "reparse_points": sum(item["type"] == "reparse" for item in records),
            "entries": records}


def protected():
    paths = {
        "accepted_quality": QUALITY / "accepted",
        "first_failure_evidence": QUALITY / "attempts",
        "validated_wheel_distribution": QUALITY / "distribution",
        "quality_delivery_index": QUALITY / "index.html",
        "editable_metadata": ROOT / "src/n2p3_net.egg-info",
        "paused_checkpoint": ROOT / "doc/assets/research_resume_20261012/pause_checkpoint_current.json",
        "basic_entry_evidence": ROOT / "doc/assets/brainsync_basic_entry_20261102",
        "quality_original_receipt": ROOT / "doc/assets/build_cleanup/brainsync_quality_20261030.receipt.json",
        "basic_original_receipt": ROOT / "doc/assets/build_cleanup/brainsync_basic_entry_20261102.receipt.json",
        "replacement_release": NEW,
        "old_release_retained": OLD,
    }
    result = {}
    for name, path in paths.items():
        result[name] = snapshot(path, (TARGETS["BCI-003"], TARGETS["BCI-004"]))
        assert result[name]["exists"], f"Required preservation missing: {path}"
    return result


def identity_checks():
    profile = QUALITY / "accepted/continuous_reviewed_native7.json"
    wheel = QUALITY / "distribution/n2p3_net-0.1.0-py3-none-any.whl"
    expected_exe = json.loads((ROOT / "doc/assets/brainsync_basic_entry_20261102/packaged_code_verification.json").read_text(encoding="utf-8"))["exe_sha256"]
    assert digest(profile) == "c52a6acaf0f18b090960056a38fd54f81b4c438876bab00192b1668535cba665"
    assert digest(wheel) == "17beceb5e64412575bd6c3d1381ce7aae9d83bff3481b38f0451631b7e6a84ac"
    assert digest(NEW / "BrainSync.exe") == expected_exe
    assert digest(ROOT / "doc/assets/build_cleanup/brainsync_quality_20261030.receipt.json") == "faeb4cd2151835cec5ecbc018be4e97c54ee58c8f084ddfeb115eb5a88cf5844"
    release_manifest = json.loads((NEW / "release-manifest.json").read_text(encoding="utf-8"))
    for entry in release_manifest["files"]:
        path = NEW / entry["path"]
        assert path.resolve().is_relative_to(NEW)
        assert path.stat().st_size == entry["bytes"], f"Manifest size: {path}"
        assert digest(path) == entry["sha256"], f"Manifest hash: {path}"
    with zipfile.ZipFile(wheel) as archive:
        assert archive.testzip() is None
    return {"profile_sha256_matches": True, "wheel_sha256_matches": True,
            "wheel_zip_integrity": True, "replacement_exe_sha256_matches": True,
            "original_quality_receipt_sha256_matches": True,
            "replacement_manifest_files_verified": len(release_manifest["files"])}


def main():
    phase = sys.argv[1]
    assert phase in ("before", "after")
    result = {"schema": "bci_cleanup_inventory/1", "phase": phase,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "targets": {name: snapshot(path) for name, path in TARGETS.items()},
              "protected": protected(), "replacement_checks": identity_checks()}
    if phase == "before":
        for name, item in result["targets"].items():
            assert item["exists"] and item["reparse_points"] == 0
            assert (item["files"], item["logical_bytes"]) == EXPECTED[name], f"Inventory drift: {name}"
        staged = snapshot(TEMP / "dist_current/BrainSync")
        release = result["protected"]["replacement_release"]
        staged_entries = {entry["relative"]: entry for entry in staged["entries"]}
        release_entries = {entry["relative"]: entry for entry in release["entries"]}
        differences = [{"relative": name, "staged": staged_entries.get(name),
                        "release": release_entries.get(name)}
                       for name in sorted(staged_entries.keys() | release_entries.keys())
                       if staged_entries.get(name) != release_entries.get(name)]
        (OUT / "stage_comparison.json").write_text(json.dumps(differences, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        allowed_additions = {"BrainSync 软件使用手册.docx", "RELEASE.md", "release-manifest.json"}
        for difference in differences:
            name = difference["relative"]
            if name in allowed_additions:
                assert difference["staged"] is None and difference["release"]["type"] == "file"
            else:
                assert name == "config/decoding.json", f"Unreviewed staged/release difference: {difference}"
                staged_config = json.loads((TEMP / "dist_current/BrainSync/config/decoding.json").read_text(encoding="utf-8"))
                release_config = json.loads((NEW / name).read_text(encoding="utf-8"))
                assert staged_config == {"schema": "brainsync-decoding-ui/1", "backend_root": "D:/BCI/n2p3-net", "method": "source", "data_root": str(TEMP / "dist_current/BrainSync/data")}
                assert release_config == {"schema": "brainsync-decoding-ui/1", "backend_root": "D:/BCI/n2p3-net", "method": "source", "device": "cpu", "data_root": "D:/BCI/BrainSync_Tool_newest/BrainSync_Tool/data"}
                (OUT / "staged_decoding.json").write_bytes((TEMP / "dist_current/BrainSync/config/decoding.json").read_bytes())
        assert digest(TEMP / "BrainSync.spec") == digest(ROOT / "doc/assets/brainsync_basic_entry_20261102/BrainSync.spec")
        result["staged_payload_matches_with_reviewed_release_additions_and_configuration"] = True
        result["build_spec_preserved"] = True
        accepted_digest = digest(QUALITY / "accepted/continuous_reviewed_native7.json")
        result["duplicate_profile_files"] = [entry["relative"] for entry in result["targets"]["BCI-002"]["entries"] if entry.get("sha256") == accepted_digest]
        assert result["duplicate_profile_files"], "Duplicate final profile not found"
    else:
        baseline = json.loads((OUT / "before.json").read_text(encoding="utf-8"))
        result["all_targets_absent"] = all(not item["exists"] for item in result["targets"].values())
        result["preservation_checks"] = {name: item == baseline["protected"][name] for name, item in result["protected"].items()}
        result["all_protected_unchanged"] = all(result["preservation_checks"].values())
        (OUT / "after.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        assert result["all_protected_unchanged"], f"Preserved files changed: {result['preservation_checks']}"
    (OUT / f"{phase}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("targets", "protected")}, ensure_ascii=False))
    for name, item in result["targets"].items():
        print(json.dumps({"id": name, **{key: value for key, value in item.items() if key != "entries"}}, ensure_ascii=False))
    for name, item in result["protected"].items():
        print(json.dumps({"protected": name, "files": item["files"], "logical_bytes": item["logical_bytes"]}))


if __name__ == "__main__":
    main()
