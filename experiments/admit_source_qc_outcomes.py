"""Publish a source QC view while preserving explicitly verified zero-evidence records.

The original preparation ledger and its frozen numerical implementation remain
unchanged. Only the known whole-record QC rejection is admitted after replay;
other preparation errors keep the study unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

NO_EVIDENCE = "source QC left no supported EEG epochs; preserve failure instead of silently dropping the record"


def record_key(row):
    return json.dumps(row, sort_keys=True)


def validate_roster(manifest, preparation):
    policy = preparation.get("preparation_method", {}).get("artifact_preprocessing_policy", {})
    if (preparation.get("status") != "failed" or not preparation.get("errors")
            or policy.get("enabled") is not True or not policy.get("continuous_qc")):
        raise ValueError("explicitly failed QC source preparation is required")
    requested = [record_key(row) for row in manifest["records"]]
    good = [record_key(row["record"]) for row in preparation["records"]]
    rejected = [record_key(row["record"]) for row in preparation["errors"]]
    if (len(set(requested)) != len(requested) or len(set(good + rejected)) != len(good + rejected)
            or set(requested) != set(good + rejected) or not good):
        raise ValueError("every unique requested record must have exactly one explicit preparation outcome")
    if any(e.get("error_type") != "ValueError" or e.get("message") != NO_EVIDENCE for e in preparation["errors"]):
        raise ValueError("ordinary preparation errors cannot be admitted as zero EEG evidence")
    if any(row.get("preparation", {}).get("method") != preparation["preparation_method"] for row in preparation["records"]):
        raise ValueError("retained shards disagree with the original preparation method")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--preparation", required=True, type=Path)
    parser.add_argument("--backend-root", required=True, type=Path)
    parser.add_argument("--backend-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    sys.path[:0] = [str(args.backend_root / "src"), str(args.backend_root)]
    import numpy as np

    import data.source_qc as owner
    from data.channel import canonical_channel_name
    from data.eog_preprocessing import EOGPreprocessingPolicy
    from data.epochs import load_epoch_dataset, parse_preprocessing_spec
    from data.source_records import (
        file_sha256,
        merge_source_shards,
        prepare_source_record,
        verify_prepared_cache,
    )
    from research.evaluation import verified_executed_snapshot
    from research.execution import atomic_write_json

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    preparation = json.loads(args.preparation.read_text(encoding="utf-8"))
    validate_roster(manifest, preparation)
    if file_sha256(args.manifest) != preparation["manifest_sha256"]:
        raise ValueError("original source roster no longer matches the preparation ledger")
    source = verified_executed_snapshot(args.backend_manifest, root=args.backend_root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    accepted = {"schema": "n2p3_source_QC_disposition_admission/1", "status": "replaying_rejections",
        "study": manifest["study"], "original_preparation": str(args.preparation.resolve()),
        "original_preparation_sha256": file_sha256(args.preparation),
        "manifest": str(args.manifest.resolve()), "manifest_sha256": file_sha256(args.manifest), "original_runtime": source,
        "consumer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "thresholds_changed": False, "new_NN_fits": 0, "retained_records": preparation["records"],
        "no_evidence_records": []}
    for row in preparation["records"]:
        verify_prepared_cache(row["cache"], cache_sha256=row["cache_sha256"],
            cache_record_sha256=row["cache_record_sha256"])
    spec = parse_preprocessing_spec(preparation["preparation_method"]["preprocessing"])
    policy = EOGPreprocessingPolicy.from_record(preparation["preparation_method"]["artifact_preprocessing_policy"])
    original = owner.apply_source_epoch_masks
    for error in preparation["errors"]:
        observed = []

        def capture(dataset, *, raw, result, recovery, model, observed=observed):
            try:
                return original(dataset, raw=raw, result=result, recovery=recovery, model=model)
            except ValueError as rejection:
                if str(rejection) != NO_EVIDENCE:
                    raise
                timeline = dataset.event_timeline
                samples = np.asarray(timeline.onset_samples) - int(raw.first_samp)
                native = {canonical_channel_name(name): name for name in result.channel_names}
                names = tuple(native[name] for name in dataset.channel_names)
                quality = result.epoch_channel_mask(samples, channels=names,
                    tmin_s=spec.tmin_ms/1000, tmax_s=spec.tmax_ms/1000)
                boundary, details = recovery.assess_epochs(raw, samples, boundary_action=model.spec.boundary_action)
                quality &= boundary[:, [recovery.channel_names.index(name) for name in names]]
                minimum = int(np.ceil(dataset.n_channels * model.spec.minimum_channel_fraction))
                rows = np.flatnonzero(timeline.evidence_indices >= 0)
                if (quality[rows].sum(axis=1) >= minimum).any():
                    raise RuntimeError("zero-evidence rejection disagrees with independent support recomputation") from rejection
                observed.append({"scheduled": timeline.n_events, "prefilter_retained_epochs": dataset.n_epochs,
                    "retained_epochs": 0, "event_ids": timeline.event_ids.tolist(),
                    "event_evidence_indices_before_QC": timeline.evidence_indices.tolist(),
                    "observed_channels_per_event": quality.sum(axis=1).tolist(), "minimum_channels": minimum,
                    "continuous_QC_model": model.record(), "continuous_QC_intervals": result.record(),
                    "boundary_recovery": details, "labels_used_for_disposition": False})
                raise

        owner.apply_source_epoch_masks = capture
        try:
            prepare_source_record(error["record"], root=manifest["remote_root"], output=output / "replay",
                preprocessing=spec, eog_policy=policy)
        except ValueError as rejection:
            if str(rejection) != NO_EVIDENCE or len(observed) != 1:
                raise
        else:
            raise RuntimeError("original rejected source now has evidence; original disposition cannot be reused")
        finally:
            owner.apply_source_epoch_masks = original
        detail_path = output / (f"rejection_{len(accepted['no_evidence_records']) + 1}.json")
        atomic_write_json(detail_path, {"record": error["record"], "reason": NO_EVIDENCE, **observed[0]})
        accepted["no_evidence_records"].append({"record": error["record"], "reason": NO_EVIDENCE,
            "scheduled": observed[0]["scheduled"], "retained_epochs": 0, "witness": str(detail_path),
            "witness_sha256": file_sha256(detail_path)})
        atomic_write_json(output / "admission.json", accepted)
        print(manifest["study"], "verified zero-evidence source record", error["record"]["subject"], flush=True)
    paths = [row["cache"] for row in preparation["records"]]
    merged = merge_source_shards(paths, output / "epochs.npz")
    dataset = load_epoch_dataset(merged, validation="attested")
    accepted.update(status="completed", cache=str(merged), cache_sha256=file_sha256(merged),
        cache_record_sha256=file_sha256(merged.with_suffix(".record.json")), n_epochs=dataset.n_epochs,
        requested_records=len(manifest["records"]), records_with_EEG=len(paths),
        scheduled_with_EEG_cache=dataset.event_timeline.n_events,
        scheduled_requested=dataset.event_timeline.n_events + sum(row["scheduled"] for row in accepted["no_evidence_records"]))
    atomic_write_json(output / "admission.json", accepted)
    print(manifest["study"], "QC disposition view published", len(paths), "with EEG;", len(accepted["no_evidence_records"]), "without EEG", flush=True)


if __name__ == "__main__":
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    main()
