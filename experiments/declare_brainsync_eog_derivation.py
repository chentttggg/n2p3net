"""Materialize the user's confirmed EOG metadata without editing original inputs."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from data.brainsync_contract import validate_brainsync_bids_session  # noqa: E402
from data.source_records import file_sha256  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402

AUTHORITY = "2026-10-31 user: 如果也是EOG，那就相同; actual EDF/BIDS EOG channel and left-earlobe EEG reference verified before applying this declaration"
DECLARATION = {"EOG": {"kind": "single_ended", "placement": "left_eye_corner", "reference": "left earlobe"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-manifest", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    quality = json.loads(args.quality_manifest.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    selected = {name for name in plan["calibration_order"] if "202609" in name}
    derived = copy.deepcopy(quality)
    receipt = {"schema": "n2p3_explicit_EOG_declaration_overlay/1", "authority": AUTHORITY,
        "original_quality_manifest_sha256": file_sha256(args.quality_manifest), "records": []}
    for row in derived["sessions"]:
        if row["directory_id"] not in selected:
            continue
        session = validate_brainsync_bids_session(row["input_root"])
        try:
            if ([n for n, kind in zip(session.bids.channel_names, session.bids.channel_types, strict=True)
                 if kind == "EOG"] != ["EOG"] or session.bids.source_reference != "left earlobe"
                    or session.bids.eog_derivation):
                raise ValueError("conditional user declaration does not match the actual source channel/reference metadata")
            original_root = session.root
            raw_path = session.bids.raw_path.relative_to(original_root)
            sidecar_path = session.bids.eeg_json_path.relative_to(original_root)
            original_raw_sha = file_sha256(session.bids.raw_path)
            original_sidecar_sha = file_sha256(session.bids.eeg_json_path)
            target = output / "inputs" / session.session_id
            shutil.copytree(original_root, target)
            sidecar = json.loads((target / sidecar_path).read_text(encoding="utf-8"))
            sidecar["EOGDerivation"] = DECLARATION
            sidecar["EOGDerivationAuthority"] = AUTHORITY
            atomic_write_json(target / sidecar_path, sidecar)
            if file_sha256(target / raw_path) != original_raw_sha:
                raise RuntimeError("derived declaration changed EDF bytes")
            if file_sha256(session.bids.eeg_json_path) != original_sidecar_sha:
                raise RuntimeError("original EOG sidecar changed during derivation")
            row["input_root"] = str(target)
            item = {"session_id": session.session_id, "original_root": str(original_root),
                "derived_root": str(target), "raw_sha256": original_raw_sha,
                "original_sidecar_sha256": original_sidecar_sha,
                "derived_sidecar_sha256": file_sha256(target / sidecar_path), "before": {}, "after": DECLARATION}
        finally:
            session.bids.close()
        restored = validate_brainsync_bids_session(target)
        try:
            if dict(restored.bids.eog_derivation) != DECLARATION:
                raise RuntimeError("derived recording does not validate the confirmed EOG declaration")
        finally:
            restored.bids.close()
        receipt["records"].append(item)
        print(item["session_id"], "verified EDF unchanged", flush=True)
    if len(receipt["records"]) != len(selected):
        raise ValueError("the derived EOG declaration did not cover the complete requested September roster")
    derived["EOG_declaration_overlay"] = {"receipt": str(output / "receipt.json"), "authority": AUTHORITY}
    atomic_write_json(output / "receipt.json", receipt)
    atomic_write_json(output / "quality_manifest.json", derived)


if __name__ == "__main__":
    main()
