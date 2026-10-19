"""Prepare the complete eight-person BNCI C600/R10 cohort without training."""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.bnci2014_008_candidate import (  # noqa: E402
    BNCI2014_008_DATASET_ID,
    BNCI2014_008_SUBJECT_IDS,
    read_bnci2014_008_candidate_record,
)
from data.bnci_trial_preparation import (  # noqa: E402
    BNCI_TRIAL_CHANNELS,
    BNCI_TRIAL_INPUT_PROFILE,
    BNCI_TRIAL_REFERENCE,
    build_bnci_trial_prepared_participant,
    load_bnci_trial_prepared_participant,
)
from data.contract import assert_p300_input_contract  # noqa: E402
from data.epochs import (  # noqa: E402
    PreprocessingSpec,
    parse_preprocessing_spec,
    save_epoch_dataset,
)
from data.identity import DatasetIdentityTable  # noqa: E402
from data.study import study_ids_for_identity  # noqa: E402


def _file_receipt(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def source_profile_from_checkpoint(path, *, expected_sha256, seed=20260918):
    """Admit the actual excluded-study producer and all its input profile fields."""
    from research.contracts import semantic_sha256
    from transfer.checkpoint import load_checkpoint_payload

    receipt = _file_receipt(path)
    if receipt["sha256"] != expected_sha256:
        raise ValueError("BNCI source profile checkpoint differs from the declared producer")
    payload = load_checkpoint_payload(path)
    spec = parse_preprocessing_spec(payload["input_preprocessing"])
    declared = payload["training_contract"]["preprocessing"]
    channels = tuple(payload["input_channel_names"])
    reference = payload["input_source_reference"]
    if (parse_preprocessing_spec(declared["epoch"]) != spec
            or tuple(declared["channel_names"]) != channels or declared["source_reference"] != reference
            or channels != BNCI_TRIAL_CHANNELS or reference != BNCI_TRIAL_REFERENCE
            or payload["n_times"] != spec.n_times or payload["input_sample_rate_hz"] != spec.sfreq
            or payload["n_channels"] != len(channels) or payload["input_tmin_s"] != spec.tmin_ms / 1000):
        raise ValueError("BNCI source checkpoint geometry, preprocessing or reference owners differ")
    assert_p300_input_contract(spec, BNCI_TRIAL_INPUT_PROFILE)
    identity = DatasetIdentityTable.from_payload(payload["training_identity_ledger"])
    studies = study_ids_for_identity(identity, payload["source_study_membership"])
    if BNCI2014_008_DATASET_ID in studies or payload["training_contract"]["seed"] != seed:
        raise ValueError("BNCI source producer violates the declared study exclusion or source seed")
    receipt.update(preprocessing=spec.record(), input_channel_names=list(channels), input_source_reference=reference,
        source_seed=seed, source_studies=list(studies), architecture_sha256=semantic_sha256(payload["architecture"]),
        source_snapshot_sha256=payload["training_contract"]["source_snapshot_sha256"])
    return spec, receipt


def prepare_episode_subject(arguments):
    """One executor-owned actor produces an immutable pair and returns receipts."""
    from research.execution import atomic_write_json
    from research.source_execution import current_input_admission

    raw_path = Path(arguments["raw_file"]["path"])
    output = Path(arguments["output"])
    output.mkdir(parents=True, exist_ok=False)
    with raw_path.open("rb") as stream:
        record = read_bnci2014_008_candidate_record(stream, source_relative_path=raw_path.name,
                                                   source_sha256=arguments["raw_file"]["sha256"])
    prepared = build_bnci_trial_prepared_participant(record, preprocessing=PreprocessingSpec(**arguments["preprocessing"]))
    for dataset in (prepared.calibration, prepared.future):
        dataset.provenance["source_profile_checkpoint"] = arguments["source_profile_checkpoint"]
    c_path = save_epoch_dataset(output / "calibration.npz", prepared.calibration, compressed=False)
    f_path = save_epoch_dataset(output / "future.npz", prepared.future, compressed=False)
    ledger = {**prepared.ledger, "raw_file": arguments["raw_file"],
        "source_profile_checkpoint": arguments["source_profile_checkpoint"],
        "calibration_cache": _file_receipt(c_path), "future_cache": _file_receipt(f_path),
        "preparation_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_module_sha256": hashlib.sha256((ROOT / "src/data/bnci_trial_preparation.py").read_bytes()).hexdigest()}
    atomic_write_json(output / "preparation.json", ledger, input_admission=current_input_admission())
    return {"subject": record.subject_id, "calibration_labels": prepared.calibration.n_epochs,
        "calibration_class_counts": ledger["actual_calibration_class_counts"],
        "requested_future_episodes": len(prepared.future_decisions),
        "requested_future_flashes": prepared.future.event_timeline.n_events,
        "retained_future_epochs": prepared.future.n_epochs, "preparation": _file_receipt(output / "preparation.json"),
        "calibration_cache": ledger["calibration_cache"], "future_cache": ledger["future_cache"]}


def main():
    from research.cloud_execution import CloudExecutor, CloudTask, PhaseResources
    from research.execution import atomic_write_json
    from research.source_execution import (
        HostMemoryAdmission,
        SourceInputAdmission,
        resolve_source_resources,
    )
    from train.runtime import cpu_thread_budget

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--cpu-budget", type=int, default=25)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--native-threads", type=int)
    args = parser.parse_args()
    spec, profile = source_profile_from_checkpoint(args.source_checkpoint,
        expected_sha256=args.source_checkpoint_sha256, seed=args.seed)
    files = [args.root / f"{subject}.mat" for subject in BNCI2014_008_SUBJECT_IDS]
    raw_receipts = [_file_receipt(path) for path in files]
    resources = resolve_source_resources({"cpu_budget": args.cpu_budget, "source_workers": 1,
        "source_worker_memory_gib": 2, "target_worker_memory_gib": 2, "host_memory_fraction": .95,
        "shared_staging_memory_gib": 0}, 1, "cpu")
    workers = min(len(files), resources["worker_cpu_budget"]) if args.workers is None else args.workers
    if not 1 <= workers <= min(len(files), resources["worker_cpu_budget"]):
        raise ValueError("BNCI preparation concurrency exceeds ready participants or the actual CPU budget")
    native_threads = (max(1, resources["worker_cpu_budget"] // workers)
                      if args.native_threads is None else args.native_threads)
    if native_threads < 1 or workers * native_threads > resources["worker_cpu_budget"]:
        raise ValueError("BNCI actual native claims exceed the declared total CPU budget")
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = tuple(CloudTask(f"bnci-trial-prepare:{subject}",
        "experiments.prepare_bnci_trial_candidate:prepare_episode_subject",
        {"raw_file": raw_file, "output": str((args.output / subject).resolve()),
         "preprocessing": spec.record(), "source_profile_checkpoint": profile},
        str(args.output / subject), PhaseResources("preparation", native_threads, 2 * 2**30, native_threads, "cpu"))
        for subject, raw_file in zip(BNCI2014_008_SUBJECT_IDS, raw_receipts, strict=True))
    accepted = []
    with cpu_thread_budget(1), multiprocessing.get_context("spawn").Manager() as manager:
        inputs = SourceInputAdmission(manager, capacity_bytes=resources["host_memory_budget_bytes"], load_workers=workers)
        admission = HostMemoryAdmission(manager, resources, input_admission=inputs)

        def finished(outcome):
            if outcome.status == "completed":
                for key in ("preparation", "calibration_cache", "future_cache"):
                    if _file_receipt(outcome.value[key]["path"]) != outcome.value[key]:
                        raise ValueError("BNCI preparation output changed before parent admission")
                loaded = load_bnci_trial_prepared_participant(Path(outcome.value["preparation"]["path"]).parent)
                if loaded.ledger["source_profile_checkpoint"] != profile:
                    raise ValueError("BNCI preparation has another source profile producer")
                accepted.append(outcome.value)
            atomic_write_json(args.output / "progress.json", {"completed": len(accepted), "requested": len(tasks),
                "last_subject": outcome.task_id, "last_status": outcome.status}, input_admission=inputs)
            return ()

        run = CloudExecutor(admission, max_workers=workers, diagnostic_directory=args.output / "worker_diagnostics",
            identity={"programme": "BNCI2014-008 C600/R10 independent raw preparation", "source_profile_checkpoint": profile}).run(
                tasks, on_finished=finished,
                on_observation=lambda row: atomic_write_json(args.output / "resources.live.json", row, input_admission=inputs))
        report = {"schema": "n2p3_bnci_trial_preparation_cohort/1", "status": run.status,
            "requested_subjects": list(BNCI2014_008_SUBJECT_IDS), "calibration_episode_count": 5,
            "primary_horizon": 10, "source_profile_checkpoint": profile,
            "participants": sorted(accepted, key=lambda row: row["subject"]),
            "resources": resources, "actual_worker_limit": workers, "actual_native_threads": native_threads,
            "worker_results": [{"task_id": row.task_id, "status": row.status, "worker": row.worker}
                               for row in run.results.values()],
            "failures": [{"task_id": row.task_id, "status": row.status, "failure": row.failure,
                          "supplementary_failures": row.supplementary_failures}
                         for row in run.results.values() if row.status != "completed"],
            "host_admission": admission.record(), "input_admission": inputs.record(), "public_training_fits": 0}
        atomic_write_json(args.output / "cohort.json", report, input_admission=inputs)
    if report["status"] != "completed":
        raise RuntimeError("BNCI preparation did not complete its full declared cohort; failures preserved")


if __name__ == "__main__":
    main()
