"""Correct explicitly identified BrainSync reference metadata without changing EEG samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.contract import brainsync_reference_record  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _correct_reference_fields(value, previous, replacement):
    if isinstance(value, dict):
        return {key: replacement if key in {"ref_label", "reference", "EEGReference"} and item == previous
                else _correct_reference_fields(item, previous, replacement) for key, item in value.items()}
    if isinstance(value, list):
        return [_correct_reference_fields(item, previous, replacement) for item in value]
    return value


def prepare_session_correction(root: Path, *, previous: str, authority: str, allow_incomplete: bool = False) -> dict[Path, bytes]:
    root = root.resolve()
    session_path = root / "session.json"
    original = session_path.read_bytes()
    session = json.loads(original.decode("utf-8-sig"))
    if not str(session.get("schema", "")).startswith("brainsync-gtn-session/"):
        raise ValueError("Reference correction only accepts an explicit BrainSync session.")
    montage = session.get("montage", {})
    if montage.get("ref_label") != previous:
        raise ValueError(f"{root.name}: declared REF differs from the explicit previous label")
    reference = brainsync_reference_record()
    replacement = reference["label"]
    correction = {"schema": "brainsync-reference-correction/1", "authority": authority,
                  "previous_label": previous, "corrected_reference": reference,
                  "original_session_sha256": _sha(original), "signal_samples_changed": False,
                  "software_rereference_changed": False, "missing_declared_sidecars": []}
    corrected = _correct_reference_fields(session, previous, replacement)
    corrected["montage"]["reference_details"] = reference
    corrected["reference_correction"] = correction
    changes = {session_path: (json.dumps(corrected, indent=2, ensure_ascii=False) + "\n").encode("utf-8")}
    recording = session.get("recording", {})
    bids = recording.get("bids", {})
    for key in ("eeg_json", "channels_tsv"):
        relative = bids.get(key)
        if not relative:
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("BIDS reference sidecar must remain inside the session")
        if not path.is_file():
            if allow_incomplete:
                correction["missing_declared_sidecars"].append(relative)
                continue
            raise ValueError(f"Missing declared BIDS reference sidecar: {path}")
        if key == "eeg_json":
            sidecar = json.loads(path.read_text(encoding="utf-8-sig"))
            if sidecar.get("EEGReference") != previous:
                raise ValueError("BIDS and session reference must agree before correction")
            sidecar = _correct_reference_fields(sidecar, previous, replacement)
            sidecar["EEGReferenceDetails"] = reference
            sidecar["ReferenceCorrection"] = correction
            changes[path] = (json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        else:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                fields, rows = reader.fieldnames, list(reader)
            if "reference" in fields:
                for row in rows:
                    if row["reference"] == previous:
                        row["reference"] = replacement
                output = io.StringIO(newline="")
                writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
                changes[path] = output.getvalue().encode("utf-8")
    changes[session_path] = (json.dumps(corrected, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return changes


def correct_sessions(roots, *, previous: str, authority: str, output: Path, allow_incomplete: bool = False):
    roots = [Path(path).resolve() for path in roots]
    if len(roots) != len(set(roots)):
        raise ValueError("Reference correction sessions must be unique")
    changes = {}
    for root in roots:
        changes.update(prepare_session_correction(root, previous=previous, authority=authority, allow_incomplete=allow_incomplete))
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "original_reference_metadata.zip"
    if archive.exists():
        raise FileExistsError(archive)
    old = {path: path.read_bytes() for path in changes}
    signals = {path: _sha(path.read_bytes()) for root in roots for path in root.rglob("*.edf")}
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as zipped:
        for index, (path, content) in enumerate(old.items()):
            zipped.writestr(f"{index:04}/{path.name}", content)
    with zipfile.ZipFile(archive) as zipped:
        for index, (path, content) in enumerate(old.items()):
            if zipped.read(f"{index:04}/{path.name}") != content:
                raise ValueError("Reference metadata archive verification failed")
    # All sessions are prevalidated and archived before any active file is replaced.
    try:
        for path, value in changes.items():
            if path.read_bytes() != old[path]:
                raise ValueError(f"Concurrent reference metadata change: {path}")
            path.write_bytes(value)
    except Exception:
        for path, value in old.items():
            if path.read_bytes() == changes[path]:
                path.write_bytes(value)
        raise
    if any(_sha(path.read_bytes()) != digest for path, digest in signals.items()):
        raise ValueError("EDF changed during metadata correction")
    manifest = {"schema": "brainsync-reference-correction-manifest/1", "authority": authority,
                "reference": brainsync_reference_record(), "archive": archive.name,
                "archive_sha256": _sha(archive.read_bytes()),
                "sessions": [str(root) for root in roots],
                "files": [{"path": str(path), "before_sha256": _sha(old[path]), "after_sha256": _sha(value),
                           "archive_member": f"{index:04}/{path.name}"}
                          for index, (path, value) in enumerate(changes.items())],
                "unchanged_edf_sha256": {str(path): digest for path, digest in signals.items()},
                "raw_header_policy": "historical binary headers and event logs remain original; corrected BIDS/session metadata is authoritative"}
    atomic_write_json(output / "manifest.json", manifest)
    return manifest


def amend_continuous_quality_reference(source: Path, session_roots, *, previous: str, authority: str, output: Path):
    """Rebind a metadata-only correction after verifying the original fit EDF bytes."""
    from data.brainsync_contract import validate_brainsync_bids_session
    from data.continuous_qc import continuous_qc_from_record, load_continuous_qc, save_continuous_qc

    if output.exists():
        raise FileExistsError(output)
    original = load_continuous_qc(source)
    if original.fit_provenance.get("source_reference") != previous:
        raise ValueError("Quality model does not declare the explicit previous acquisition reference")
    sessions = {item.session_id: item for item in map(validate_brainsync_bids_session, session_roots)}
    for recorded in original.fit_provenance.get("sources", []):
        session = sessions.get(recorded["session_id"])
        if session is None or session.bids.source_reference != brainsync_reference_record()["label"]:
            raise ValueError("Quality correction requires all original fit sessions with corrected REF")
        if _sha(session.bids.raw_path.read_bytes()) != recorded["raw_sha256"]:
            raise ValueError("Original quality fit EEG bytes changed; metadata correction cannot reuse this fit")
        correction = session.manifest.get("reference_correction", {})
        if correction.get("previous_label") != previous or correction.get("authority") != authority:
            raise ValueError("Quality and raw metadata corrections need the same explicit authority")
    if not original.fit_provenance.get("sources"):
        raise ValueError("Quality model lacks raw fit identities")
    updated = original.record()
    updated["fit_provenance"]["source_reference"] = brainsync_reference_record()["label"]
    updated["fit_provenance"]["reference_metadata_correction"] = {
        "authority": authority, "previous_label": previous,
        "corrected_reference": brainsync_reference_record(),
        "original_model_sha256": _sha(source.read_bytes()), "parameters_refitted": False,
    }
    if updated.get("epoch_recovery") is not None:
        updated["epoch_recovery"]["fit_provenance"]["source_reference"] = brainsync_reference_record()["label"]
        updated["epoch_recovery"]["fit_provenance"]["reference_metadata_correction"] = updated["fit_provenance"]["reference_metadata_correction"]
    amended = continuous_qc_from_record(updated)
    for key in set(original.record()) - {"fit_provenance", "epoch_recovery"}:
        if amended.record()[key] != original.record()[key]:
            raise ValueError("Metadata correction changed a numerical quality parameter")
    if original.epoch_recovery is not None:
        for key in set(original.epoch_recovery.record()) - {"fit_provenance"}:
            if amended.epoch_recovery.record()[key] != original.epoch_recovery.record()[key]:
                raise ValueError("Metadata correction changed an epoch recovery parameter")
    save_continuous_qc(output, amended)
    return amended


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, action="append", required=True)
    parser.add_argument("--previous-label", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="Correct existing metadata without creating missing sidecars or changing admission.")
    args = parser.parse_args()
    record = correct_sessions(args.session_dir, previous=args.previous_label, authority=args.authority, output=args.output,
                              allow_incomplete=args.allow_incomplete)
    print(json.dumps({"sessions": len(record["sessions"]), "corrected_files": len(record["files"]),
                      "unchanged_edf": len(record["unchanged_edf_sha256"]), "manifest": str(args.output / "manifest.json")}))


if __name__ == "__main__":
    main()
