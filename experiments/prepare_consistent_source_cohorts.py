"""Prepare both full source cohorts from the same verified original record roster."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from data.eog_preprocessing import EOGPreprocessingPolicy  # noqa: E402
from data.epochs import load_preprocessing_spec  # noqa: E402
from data.exclusions import source_participant_excluded  # noqa: E402
from data.source_admission import verify_source_preparation  # noqa: E402
from data.source_records import file_sha256, verify_prepared_cache  # noqa: E402
from experiments.admit_source_qc_outcomes import validate_roster  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402

STUDIES = ("BI2014a", "BI2015a", "BNCI2014-008", "Huebner2017", "BI2013a", "EPFLP300",
           "Chailloux2020", "Speier2017", "GTN", "BI2012", "Cattan2019_VR", "BrainForm")


def _study_output(output, *, pipeline, study, ledger, payload_root):
    key = pipeline + ":" + study
    recorded = ledger.get("study_outputs", {}).get(key)
    if recorded is not None:
        return Path(recorded)
    local = output / pipeline / study
    if local.exists() or payload_root is None:
        return local
    return Path(payload_root).resolve() / pipeline / study


def publish_progress(path, ledger, task_checkpoint=None):
    ledger["updated_utc"] = datetime.now(UTC).isoformat()
    ledger["controller_pid"] = os.getpid()
    atomic_write_json(path, ledger)
    if task_checkpoint is not None:
        task_checkpoint = Path(task_checkpoint)
        task = json.loads(task_checkpoint.read_text(encoding="utf-8"))
        if task.get("state") == "paused_by_user":
            return
        task.update(updated_utc=ledger["updated_utc"],
            state="source_inputs_ready_GPU_unavailable" if ledger["status"] == "source_studies_prepared" else
                  "source_preparation_failed_GPU_unavailable" if ledger["status"] in {"preparation_failed", "insufficient_disk"} else
                  "source_preparation_running_GPU_unavailable",
            source_preparation_state_owner=str(path),
            background_processes_running=ledger["status"] in {"preparing", "merging_pools"})
        task["source_preparation_summary"] = {"status": ledger["status"], "active": ledger.get("active"),
            "completed_study_views": len(ledger["completed"]),
            "no_evidence_records": sum(row.get("no_evidence_records", 0) for row in ledger["completed"].values())}
        atomic_write_json(task_checkpoint, task)


def prepare(*, manifests, raw_root, output, resume=False, exclusion_evidence=None, payload_root=None,
            study_runtimes=None, workers=1, task_checkpoint=None, pool_output_root=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=resume)
    ledger_path = output / "checkpoint.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {
        "schema": "n2p3_matched_source_preparation/1", "status": "preparing",
        "scope": "fresh basic and QC raw preparation; no neural fits",
        "source_pool_policy": "same twelve studies and same original recordings in both arms; only actual recorded EOG is regressed",
        "studies": {}, "commands": [], "completed": {}, "failures": {}}
    ledger.update(status="preparing", resume_requires_explicit_user_instruction=False)
    ledger.setdefault("study_outputs", {})
    if type(workers) is not int or workers < 1:
        raise ValueError("source preparation worker limit must be a positive integer")
    ledger["requested_record_workers"] = workers
    study_runtimes = dict(study_runtimes or {})
    spec = load_preprocessing_spec(ROOT / "configs/brainsync_bidirectional_preprocessing.json")
    for folder in ("manifests", "policies", "logs"):
        (output / folder).mkdir(exist_ok=True)
    exclusion_path = (ROOT / "experiments/runs/unified12_20260918/BI2012_invalid_record.json"
                      if exclusion_evidence is None else Path(exclusion_evidence).resolve())
    exclusion_text = exclusion_path.read_text(encoding="utf-8")
    # The original captured stdout contains one MNE banner before its JSON.
    banner = "NOTE: pick_types() is a legacy function. New code should use inst.pick(...).\n"
    if exclusion_text.startswith(banner):
        exclusion_text = exclusion_text[len(banner):]
    excluded_record = json.loads(exclusion_text)["record"]
    ledger["historical_invalid_record_evidence"] = {"path": str(exclusion_path), "sha256": file_sha256(exclusion_path)}
    for study in STUDIES:
        source = Path(manifests) / (study + ".json")
        declaration = json.loads(source.read_text(encoding="utf-8"))
        retained, excluded = [], []
        for row in declaration["records"]:
            known_invalid = all(row.get(key) == excluded_record.get(key)
                for key in ("study", "subject", "session", "run", "sha256"))
            author_excluded = source_participant_excluded(row["study"], row["subject"])
            if known_invalid or author_excluded:
                excluded.append({"record": row, "reason": "historical_verified_constant_Cz" if known_invalid else "source_author_exclusion"})
            else:
                retained.append(row)
        manifest = {**declaration, "remote_root": str(Path(raw_root).resolve()), "records": retained}
        path = output / "manifests" / (study + ".json")
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("prepared source roster changed during resume")
        atomic_write_json(path, manifest)
        ledger["studies"][study] = {"original_manifest_sha256": file_sha256(source),
            "matched_record_count": len(retained), "excluded": excluded}
    for pipeline, enabled in (("basic", False), ("QC", True)):
        policy = (EOGPreprocessingPolicy.corrected() if enabled else
            EOGPreprocessingPolicy("basic_eog_ols_v1", False, "record_unavailable"))
        path = output / "policies" / (pipeline + ".json")
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != policy.record():
            raise ValueError("source preprocessing policy changed during resume")
        atomic_write_json(path, policy.record())
        for study in STUDIES:
            key = pipeline + ":" + study
            runtime = Path(study_runtimes.get(study, ROOT)).resolve()
            ledger.setdefault("study_runtime_roots", {})[key] = str(runtime)
            study_output = _study_output(output, pipeline=pipeline, study=study,
                ledger=ledger, payload_root=payload_root)
            study_output.mkdir(parents=True, exist_ok=True)
            ledger["study_outputs"][key] = str(study_output.resolve())
            admission_path = study_output / "qc_disposition_view" / "admission.json"
            preparation_path = study_output / "preparation.json"
            prepared = json.loads(preparation_path.read_text(encoding="utf-8")) if preparation_path.exists() else {}
            if admission_path.exists():
                admitted = json.loads(admission_path.read_text(encoding="utf-8"))
                if (admitted.get("status") != "completed"
                        or admitted.get("manifest_sha256") != file_sha256(output / "manifests" / (study + ".json"))
                        or admitted.get("original_preparation_sha256") != file_sha256(preparation_path)):
                    raise ValueError("source QC admission is partial or its original inputs changed")
                verify_prepared_cache(admitted["cache"], cache_sha256=admitted["cache_sha256"],
                    cache_record_sha256=admitted["cache_record_sha256"])
                for rejection in admitted["no_evidence_records"]:
                    if file_sha256(rejection["witness"]) != rejection["witness_sha256"]:
                        raise ValueError("saved source QC zero-evidence witness changed")
                ledger["completed"][key] = {"preparation": str(admission_path), "cache": admitted["cache"],
                    "cache_sha256": admitted["cache_sha256"], "records": admitted["requested_records"],
                    "records_with_EEG": admitted["records_with_EEG"],
                    "no_evidence_records": len(admitted["no_evidence_records"])}
                ledger["failures"].pop(key, None)
                publish_progress(ledger_path, ledger, task_checkpoint)
                print("VERIFIED", key, "including explicit zero-evidence records", flush=True)
                continue
            if prepared.get("status") == "completed":
                if prepared["manifest_sha256"] != file_sha256(output / "manifests" / (study + ".json")):
                    raise ValueError("completed source manifest changed")
                verify_source_preparation(prepared, study=study, preprocessing=spec)
                ledger["completed"][key] = {"preparation": str(preparation_path), "cache": prepared["cache"],
                    "cache_sha256": prepared["cache_sha256"], "records": len(prepared["records"])}
                publish_progress(ledger_path, ledger, task_checkpoint)
                print("VERIFIED", key, len(prepared["records"]), flush=True)
                continue
            if shutil.disk_usage(study_output).free < 3 * 2**30:
                ledger.update(status="insufficient_disk", active=key)
                publish_progress(ledger_path, ledger, task_checkpoint)
                raise RuntimeError("source preparation needs at least 3 GiB free before the next study")
            cmd = [sys.executable, "-X", "utf8", str(runtime / "experiments/prepare_source_records.py"),
                "--manifest", str(output / "manifests" / (study + ".json")),
                "--output", str(study_output), "--workers", str(workers),
                "--preprocessing-config", str(runtime / "configs/brainsync_bidirectional_preprocessing.json"),
                "--eog-policy", str(path)]
            # A previous exhaustive QC attempt can already contain only the
            # known no-evidence outcomes. Replay and publish that explicit view.
            disposition = False
            if pipeline == "QC" and prepared.get("status") == "failed":
                try:
                    validate_roster(json.loads((output / "manifests" / (study + ".json")).read_text(encoding="utf-8")), prepared)
                except ValueError:
                    pass
                else:
                    disposition = True
            ledger["active"] = key
            ledger["commands"].append(cmd)
            publish_progress(ledger_path, ledger, task_checkpoint)
            with (output / "logs" / (pipeline + "_" + study + ".log")).open("a", encoding="utf-8") as stream:
                result = None if disposition else subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
            prepared = json.loads(preparation_path.read_text(encoding="utf-8")) if preparation_path.exists() else {}
            if pipeline == "QC" and prepared.get("status") == "failed":
                try:
                    validate_roster(json.loads((output / "manifests" / (study + ".json")).read_text(encoding="utf-8")), prepared)
                except ValueError:
                    pass
                else:
                    backend_manifest = runtime.parent / "source.manifest.json"
                    replay_cmd = [sys.executable, "-X", "utf8", str(ROOT / "experiments/admit_source_qc_outcomes.py"),
                        "--manifest", str(output / "manifests" / (study + ".json")),
                        "--preparation", str(preparation_path), "--backend-root", str(runtime),
                        "--backend-manifest", str(backend_manifest), "--output", str(admission_path.parent)]
                    ledger["commands"].append(replay_cmd)
                    publish_progress(ledger_path, ledger, task_checkpoint)
                    with (output / "logs" / (pipeline + "_" + study + "_disposition.log")).open("a", encoding="utf-8") as stream:
                        result = subprocess.run(replay_cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
                    if result.returncode == 0 and admission_path.exists():
                        admitted = json.loads(admission_path.read_text(encoding="utf-8"))
                        if admitted.get("status") == "completed":
                            ledger["completed"][key] = {"preparation": str(admission_path), "cache": admitted["cache"],
                                "cache_sha256": admitted["cache_sha256"], "records": admitted["requested_records"],
                                "records_with_EEG": admitted["records_with_EEG"],
                                "no_evidence_records": len(admitted["no_evidence_records"])}
                            ledger["failures"].pop(key, None)
                            publish_progress(ledger_path, ledger, task_checkpoint)
                            print("PREPARED", key, "including explicit zero-evidence records", flush=True)
                            continue
            if result is None or result.returncode or prepared.get("status") != "completed":
                ledger["failures"][key] = {"returncode": None if result is None else result.returncode, "errors": prepared.get("errors", []),
                    "log": str(output / "logs" / (pipeline + "_" + study + ".log"))}
                ledger["status"] = "preparation_failed"
                publish_progress(ledger_path, ledger, task_checkpoint)
                raise RuntimeError(f"matched cohort preparation stopped at {key}; preserve the complete failure ledger")
            ledger["completed"][key] = {"preparation": str(preparation_path), "cache": prepared["cache"],
                "cache_sha256": prepared["cache_sha256"], "records": len(prepared["records"])}
            ledger["failures"].pop(key, None)
            publish_progress(ledger_path, ledger, task_checkpoint)
            print("PREPARED", key, len(prepared["records"]), flush=True)
    if pool_output_root is not None:
        ledger.update(status="merging_pools", active="source_pools")
        publish_progress(ledger_path, ledger, task_checkpoint)
        pool_binding = output / "source_pools.json"
        with (output / "logs/source_pools.log").open("a", encoding="utf-8") as stream:
            completed = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "experiments/merge_consistent_source_pools.py"),
                "--queue", str(ledger_path), "--output", str(pool_output_root), "--binding", str(pool_binding)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            ledger.update(status="preparation_failed", active="source_pools")
            publish_progress(ledger_path, ledger, task_checkpoint)
            raise RuntimeError("complete source pool merge failed; inspect source_pools.log")
        ledger["source_pools"] = str(pool_binding)
    ledger.update(status="source_studies_prepared", active=None)
    publish_progress(ledger_path, ledger, task_checkpoint)
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exclusion-evidence", type=Path)
    parser.add_argument("--payload-root", type=Path,
                        help="Alternate cache storage for new studies; existing preparation paths stay bound")
    parser.add_argument("--study-runtime", action="append", default=[], metavar="STUDY=CHECKOUT",
                        help="Explicit parser producer for one study; other prepared studies keep their original runtime")
    parser.add_argument("--workers", type=int, default=1,
                        help="Record preparation workers; native threads remain one and memory admission caps the actual count")
    parser.add_argument("--task-checkpoint", type=Path,
                        help="Update the task checkpoint when the source queue advances, completes or fails")
    parser.add_argument("--pool-output-root", type=Path,
                        help="After all source records settle, build the two pooled inputs in this directory")
    args = parser.parse_args(argv)
    study_runtimes = {}
    for value in args.study_runtime:
        study, separator, path = value.partition("=")
        if not separator or study not in STUDIES or not path or study in study_runtimes:
            raise ValueError("study runtime must be one unique known STUDY=CHECKOUT")
        study_runtimes[study] = path
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    try:
        prepare(manifests=args.manifests, raw_root=args.raw_root, output=args.output, resume=args.resume,
            exclusion_evidence=args.exclusion_evidence, payload_root=args.payload_root, study_runtimes=study_runtimes,
            workers=args.workers, task_checkpoint=args.task_checkpoint, pool_output_root=args.pool_output_root)
    except Exception as error:
        path = args.output.resolve() / "checkpoint.json"
        if path.exists():
            ledger = json.loads(path.read_text(encoding="utf-8"))
            ledger.update(status="preparation_failed", failure={"type": type(error).__name__, "message": str(error)})
            publish_progress(path, ledger, args.task_checkpoint)
        raise


if __name__ == "__main__":
    main()
