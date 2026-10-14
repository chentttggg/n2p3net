"""Huebner registered input dispatched through the original shared prefix issuer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data.huebner2017_candidate import huebner_dataset_adapter
from transfer.ensemble_views import RecordCalibrationPrefixCompiler


def prepare_huebner_independent_prefixes(declarations, **arguments):
    return RecordCalibrationPrefixCompiler(huebner_dataset_adapter(declarations), **arguments).independent_views()


def huebner_candidate_source_declarations(readout_path, *, nemar_root):
    """Bind every original census record, retaining unavailable EEG in its roster.

    The exact audit lists original marker members; the adapter verifies them
    again when reading. Whole EEG hashes are streamed once at declaration,
    without EEG inflation or preload. Rest records have another task and are
    explicitly outside this declared copy-spelling programme.
    """
    readout = json.loads(Path(readout_path).read_bytes())
    participants, statuses = {}, []
    nemar_root = Path(nemar_root)
    for row in readout["records"]:
        if "subject" not in row:
            statuses.append({"member": row["member"]["name"], "status": "other_task_original_rest"})
            continue
        subject, session, run = str(row["subject"]), row["block"]-1, row["run"]-1
        basename = f"sub-{subject}_ses-{session}_task-p300_run-{run}"
        prefix = nemar_root/f"sub-{subject}/ses-{session}/eeg"/basename
        declaration = {"subject": subject, "block": session, "run": run,
            "marker": {"archive_path": row["original_zip"], "member": row["member"]}}
        bdf = Path(str(prefix)+"_eeg.bdf")
        if bdf.exists():
            for kind, suffix in (("bdf", "_eeg.bdf"), ("channels", "_channels.tsv"),
                                 ("events", "_events.tsv"), ("eeg", "_eeg.json")):
                path = Path(str(prefix)+suffix)
                with path.open("rb") as stream:
                    sha = hashlib.file_digest(stream, "sha256").hexdigest()
                declaration[kind] = {"path": str(path.resolve()), "sha256": sha, "bytes": path.stat().st_size}
            status = "original_marker_and_NEMAR_EEG_available"
        else:
            declaration["bdf"] = None
            status = "original_marker_EEG_unavailable"
        participants.setdefault(subject, []).append(declaration)
        statuses.append({"subject": subject, "session": session, "run": run, "status": status,
                         "member": row["member"]["name"]})
    for _subject, sources in participants.items():
        sources.sort(key=lambda x: (x["block"], x["run"]))
        if len({(x["block"], x["run"]) for x in sources}) != len(sources):
            raise ValueError("original census contains two competing payloads for one acquisition record")
    return {"schema": "n2p3_Huebner_candidate_input_roster/1", "dataset_id": "Huebner2017",
        "source_census_sha256": hashlib.sha256(Path(readout_path).read_bytes()).hexdigest(),
        "participants": {subject: {"dataset_id": "Huebner2017", "source_format": "original_vmrk_plus_NEMAR_BDF",
                                   "record_sources": sources} for subject, sources in participants.items()},
        "record_statuses": statuses, "public_training_fits": 0, "EEG_member_extractions": 0}


def write_huebner_candidate_declarations(readout_path, output, *, nemar_root):
    roster = huebner_candidate_source_declarations(readout_path, nemar_root=nemar_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    participants = []
    for subject, declaration in roster["participants"].items():
        path = output/f"participant_{subject}.json"
        path.write_text(json.dumps(declaration, sort_keys=True, indent=2)+"\n", encoding="utf-8")
        participants.append({"dataset_id": "Huebner2017", "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "subject": subject})
    record = {**roster, "participant_declarations": participants}
    (output/"roster.json").write_text(json.dumps(record, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    return record
