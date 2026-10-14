"""Huebner registered input dispatched through the original shared prefix issuer."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from data.huebner2017_candidate import huebner_dataset_adapter
from transfer.ensemble_views import RecordCalibrationPrefixCompiler


def prepare_huebner_independent_prefixes(declarations, **arguments):
    return RecordCalibrationPrefixCompiler(huebner_dataset_adapter(declarations), **arguments).independent_views()


def huebner_candidate_plan(records, source_entry, *, budgets=(0, 200, 800, 2000), seed=20261014,
                           nn_epochs=60, baseline_name="scratch_nn_fixed", minimum_selection_rows=1,
                           evidence_scope="public_calibration_development", device="cpu"):
    """Opt in to the original public quartet and its actual prefix selector.

    This builder declares target work only. The original outer-Huebner source
    checkpoint is supplied by its accepted catalog entry, never trained here.
    """
    from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
    from data.epochs import preprocessing_spec_from_contract
    from research.calibration_selection import transfer_scratch_prefix_plan
    from transfer.calibration_candidates import waveform_lda_recipe
    from transfer.ensemble import EnsembleProgram
    from transfer.ensemble_fit import CalibrationExpertProgram
    from transfer.ensemble_views import SelectionEpochProfile
    from transfer.recipes import TargetRecipe, erp_target_recipe

    preprocessing = replace(preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT), baseline_mode="none")
    profile = SelectionEpochProfile.from_preprocessing(preprocessing, source_sample_rate_hz=1000.,
        channel_names=("FZ", "P3", "PZ", "P4", "PO8", "OZ"), reference="nose",
        reference_channel="CZ", allow_missing_channels=True)
    source = TargetRecipe("NN", initialization="source", input_statistics="source", epochs=nn_epochs,
        batch_size=64, bn_buffers="calibration", dropout="off", seed=seed,
        trial_loss_reduction="calibration_balanced_fixed_mass")
    experts = (CalibrationExpertProgram("NN", "trial", source, profile),
        CalibrationExpertProgram("scratch", "trial", replace(source, name="scratch", initialization="scratch",
            input_statistics="target_prefix"), profile),
        CalibrationExpertProgram("xdawn", "trial", erp_target_recipe("xdawn_erp", bin_width=16, seed=seed),
            replace(profile, preprocessing=replace(preprocessing, baseline_mode="mean_only"))),
        CalibrationExpertProgram("waveform", "trial", waveform_lda_recipe(seed=seed), profile))
    mixtures = (EnsembleProgram.equal_weights("source_nn_fixed", ("NN",)),
        EnsembleProgram.equal_weights("scratch_nn_fixed", ("scratch",)),
        EnsembleProgram.equal_weights("source_nn_xdawn_fixed", ("NN", "xdawn")),
        EnsembleProgram.equal_weights("scratch_nn_xdawn_fixed", ("scratch", "xdawn")),
        EnsembleProgram.equal_weights("waveform_control", ("waveform",)))
    plan = {"schema": "n2p3_candidate_ensemble_study/1", "dataset_id": "Huebner2017",
        "evaluation_design": "independent_prefix", "evidence_scope": evidence_scope,
        "records": list(records), "budgets": list(budgets), "seed_pairs": [[source_entry["source_seed"], seed]],
        "source_seed_catalog": [source_entry], "source_checkpoints": {},
        "expected_source_config": {"pos_weight": 1., "val_groups_min": 2, "val_groups_max": None},
        "experts": [expert.record() for expert in experts], "mixtures": [mixture.record() for mixture in mixtures],
        "baseline_mixture": baseline_name, "calibration_strategies": ["fixed"],
        "authorization_preprocessing": preprocessing.record(), "observation_response_seconds": 1.203125,
        "pre_roll_seconds": .2, "device": device, "workers": 1, "cpu_threads_per_worker": 1}
    return transfer_scratch_prefix_plan(plan, source_expert_id="NN", scratch_expert_id="scratch", xdawn_expert_id="xdawn",
        baseline_name=baseline_name, output_name="selected_fixed_family", minimum_selection_rows=minimum_selection_rows)


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


def write_huebner_original_manifest_declarations(manifest_path, output, *, evaluation_session_ids=("2",)):
    """Freeze exact full original metadata envelopes and stream each available BDF once."""
    from data.huebner2017_candidate import _file_identity

    manifest_path = Path(manifest_path)
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    participants, records = [], []
    if manifest["schema"] != "n2p3_huebner_participant_envelope_manifest/1":
        raise ValueError("unsupported original metadata participant envelope manifest")
    for row in manifest["records"]:
        original = Path(row["path"])
        payload = original.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError("original participant envelope differs from its exact metadata manifest")
        declaration = json.loads(payload)
        declaration["evaluation_session_ids"] = list(evaluation_session_ids)
        for record in declaration["record_sources"]:
            bdf = record.get("bdf")
            if bdf is not None:
                path = Path(bdf["path"])
                before = _file_identity(path.stat())
                with path.open("rb") as stream:
                    sha = hashlib.file_digest(stream, "sha256").hexdigest()
                if _file_identity(path.stat()) != before:
                    raise ValueError("original EEG file changed during declaration byte hashing")
                record["bdf"] = {"path": str(path.resolve()), "sha256": sha, "bytes": path.stat().st_size}
            records.append({"subject": record["subject"], "session": record["block"], "run": record["run"],
                "status": "original_marker_EEG_unavailable" if bdf is None else "original_marker_and_NEMAR_EEG_available",
                "marker_sha256": record["marker"]["sha256"]})
        path = output/f"participant_{declaration['source_subject_id']}.json"
        path.write_text(json.dumps(declaration, sort_keys=True, indent=2)+"\n", encoding="utf-8")
        participants.append({"dataset_id": "Huebner2017", "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "subject": declaration["source_subject_id"]})
    roster = {"schema": "n2p3_Huebner_candidate_input_roster/2", "dataset_id": "Huebner2017",
        "original_metadata_manifest": {"path": str(manifest_path.resolve()),
            "sha256": hashlib.sha256(manifest_payload).hexdigest()},
        "evaluation_session_ids": list(evaluation_session_ids), "participant_declarations": participants,
        "record_statuses": records, "requested_last_session_characters": 13*9*7,
        "EEG_payload_hashes": sum(r["status"] == "original_marker_and_NEMAR_EEG_available" for r in records),
        "EEG_preload": 0, "model_fits": 0, "original_archive_writes": 0}
    (output/"roster.json").write_text(json.dumps(roster, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    return roster
