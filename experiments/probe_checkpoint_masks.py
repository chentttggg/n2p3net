"""Run the five fixed-model S1 deletion probes on fully observed BNCI six-slot data."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch

from data.epochs import load_epoch_dataset, loaded_epoch_cache_attestation
from data.study import load_study_membership_proof
from research.evaluation import verified_executed_snapshot
from research.execution import atomic_write_json
from train.runtime import cpu_thread_budget
from transfer.checkpoint import load_checkpoint_payload
from transfer.mask_probe import run_checkpoint_mask_probe


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-snapshot-manifest", type=Path)
    parser.add_argument("--study-membership-proof", type=Path, action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if not args.dry_run and args.evaluation_snapshot_manifest is None:
        parser.error("execution requires --evaluation-snapshot-manifest")
    snapshot = (
        verified_executed_snapshot(
            args.evaluation_snapshot_manifest,
            root=ROOT,
            required_files=("src/transfer/mask_probe.py", "experiments/probe_checkpoint_masks.py"),
        )
        if args.evaluation_snapshot_manifest is not None
        else None
    )
    dataset = load_epoch_dataset(args.cache, require_labels=True, validation="attested")
    payload = load_checkpoint_payload(args.checkpoint)
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    cache_sha = loaded_epoch_cache_attestation(dataset)["sha256"]
    requested = (
        list(dataset.identity_table.local_subject_ids)
        if args.subjects == "all"
        else args.subjects.split(",")
    )
    if len(set(requested)) != len(requested):
        raise ValueError("probe requests must contain unique participants")
    requested_identity = dataset.identity_table.subset(requested)
    study_proofs = tuple(load_study_membership_proof(path) for path in args.study_membership_proof)
    records, failures = [], []
    with cpu_thread_budget(args.cpu_threads):
        for subject in requested:
            try:
                records.append(
                    run_checkpoint_mask_probe(
                        dataset,
                        payload,
                        target_subject=subject,
                        checkpoint_sha256=checkpoint_sha,
                        target_cache_sha256=cache_sha,
                        evaluation_snapshot_sha256=snapshot["archive_sha256"]
                        if snapshot
                        else "0" * 64,
                        device=torch.device(args.device),
                        batch_size=args.batch_size,
                        dry_run=args.dry_run,
                        study_membership_proofs=study_proofs,
                    )
                )
            except ValueError as error:
                failures.append(
                    {
                        "participant_key": requested_identity.record_for(subject).authority_key(),
                        "local_subject_id": subject,
                        "stage": "mask_probe_admission",
                        "reason": str(error),
                        "requested_arms": 5,
                    }
                )
    result = {
        "schema": "n2p3_fixed_checkpoint_mask_probe_run/1",
        "dry_run": args.dry_run,
        "execution_snapshot": snapshot,
        "execution_snapshot_status": "verified" if snapshot else "required_before_execution",
        "unbound_plan_contract_digest_placeholder": "0" * 64 if snapshot is None else None,
        "checkpoint_sha256": checkpoint_sha,
        "target_cache_sha256": cache_sha,
        "requested_participant_keys": list(requested_identity.authority_keys()),
        "requested_tasks": len(requested) * 5,
        "admitted_tasks": len(records) * 5,
        "fit_rows": 0,
        "optimizer_updates": 0,
        "records": records,
        "failures": failures,
    }
    atomic_write_json(args.output, result)
    print(
        f"mask probe: {len(records) * 5}/{len(requested) * 5} admitted tasks; "
        f"{len(failures)} participant failures; dry_run={args.dry_run}"
    )
    return result


if __name__ == "__main__":
    main()
