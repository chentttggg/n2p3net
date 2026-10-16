"""Resource-aware, resumable preparation of explicitly manifested public raw records."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.epochs import parse_preprocessing_spec  # noqa: E402
from data.preprocess import RawSampleScope  # noqa: E402
from data.source_parsers import validate_source_manifest  # noqa: E402
from data.source_records import (  # noqa: E402
    SOURCE_EOG_PREPROCESSING,
    SOURCE_PREPARATION_SCHEMA,
    file_sha256,
    merge_source_shards,
    prepare_source_record,
    prepare_source_record_prefixes,
    prepare_source_subject_prefixes,
    source_preparation_method,
    verify_prepared_cache,
    verify_prepared_prefixes,
    verify_source_record_inputs,
)
from research.execution import atomic_write_json  # noqa: E402
from research.source_execution import available_host_memory_bytes  # noqa: E402
from train.runtime import available_cpu_threads  # noqa: E402


def worker(arguments):
    from threadpoolctl import threadpool_limits

    item, root, output, method, preprocessing, prefixes = arguments
    with threadpool_limits(limits=1):
        if prefixes is not None:
            if isinstance(item, list):
                return prepare_source_subject_prefixes(item, root=root, output=output,
                    preparation_method=method, preprocessing=preprocessing, **prefixes)
            return prepare_source_record_prefixes(item, root=root, output=output,
                preparation_method=method, preprocessing=preprocessing, **prefixes)
        return prepare_source_record(item, root=root, output=output, preparation_method=method,
                                     preprocessing=preprocessing)


def _manifest_preprocessing(manifest, config_path):
    declarations = []
    if config_path is not None:
        declarations.append(parse_preprocessing_spec(json.loads(config_path.read_text(encoding="utf-8"))))
    if "preprocessing" in manifest:
        declarations.append(parse_preprocessing_spec(manifest["preprocessing"]))
    declarations.extend(parse_preprocessing_spec(item["preprocessing"])
                        for item in manifest["records"] if "preprocessing" in item)
    if declarations and any(value != declarations[0] for value in declarations[1:]):
        raise ValueError("source manifest and config must declare the same preprocessing profile")
    return declarations[0] if declarations else SOURCE_EOG_PREPROCESSING


def _row_key(row):
    return json.dumps(row["records"] if "records" in row else row["record"], sort_keys=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--preprocessing-config", type=Path,
                        help="Complete PreprocessingSpec JSON; must agree with manifest declarations")
    parser.add_argument("--target-prefix-budgets", type=int, nargs="+",
                        help="Prepare independent raw prefixes for these original-label allocations")
    parser.add_argument("--early-stopping-inner", action="store_true",
                        help="Also prepare independent inner raw views for target early-stopping research")
    parser.add_argument("--minimum-test-trials", type=int, default=200)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_source_manifest(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "preparation.json"
    spec = _manifest_preprocessing(manifest, args.preprocessing_config)
    prefixes = None
    requests = manifest["records"]
    if args.target_prefix_budgets is not None:
        from transfer.binary import validate_calibration_budgets

        validate_calibration_budgets(tuple(args.target_prefix_budgets))
        prefixes = {"budgets": args.target_prefix_budgets, "minimum_test_trials": args.minimum_test_trials,
                    "prepare_inner": args.early_stopping_inner}
        protocol = manifest.get("target_protocol")
        if protocol is not None:
            if protocol.get("kind") == "last_session_holdout":
                prefixes["protocol"] = protocol
                groups = {}
                for item in requests:
                    groups.setdefault((item["study"], str(item["subject"])), []).append(item)
                requests = list(groups.values())
            elif protocol.get("kind") == "single_recording_prefix":
                prefixes["minimum_test_trials"] = protocol.get("minimum_test_trials", args.minimum_test_trials)
            else:
                raise ValueError("unsupported explicit target protocol in source manifest")
    method = source_preparation_method() if spec == SOURCE_EOG_PREPROCESSING else source_preparation_method(spec)
    result = json.loads(result_path.read_text()) if result_path.exists() else {
        "schema": SOURCE_PREPARATION_SCHEMA, "study": manifest["study"],
        "preparation_method": method,
        "target_prefixes": prefixes,
        "manifest_sha256": file_sha256(args.manifest), "records": [], "errors": []}
    if result["manifest_sha256"] != file_sha256(args.manifest):
        raise ValueError("cannot resume preparation under different source evidence")
    if result.get("schema") != SOURCE_PREPARATION_SCHEMA or result.get("preparation_method") != method:
        raise ValueError("source preparation method changed or is unrecorded; preserve this run and use a new output directory")
    if result.get("target_prefixes") != prefixes:
        raise ValueError("source preparation prefix allocation changed; use a new output directory")
    completed = {_row_key(row): row for row in result["records"]}
    requested = [json.dumps(item, sort_keys=True) for item in requests]
    if len(set(requested)) != len(requested) or len(completed) != len(result["records"]) or set(completed) - set(requested):
        raise ValueError("source preparation requires unique, requested record identities")
    pending = []
    for item in requests:
        found = completed.get(json.dumps(item, sort_keys=True))
        if found:
            if prefixes is not None:
                verify_prepared_prefixes(found, item=item, root=manifest["remote_root"], method=method, **prefixes)
                continue
            current_inputs = verify_source_record_inputs(item, root=manifest["remote_root"])
            preparation = found.get("preparation", {})
            if preparation.get("method") != method or preparation.get("consumed_inputs") != current_inputs:
                raise ValueError("source preparation inputs changed or lack evidence; preserve this run and use a new output directory")
            declared_scope = RawSampleScope.from_record(item["raw_scope"]).record() if "raw_scope" in item else None
            if preparation.get("raw_scope") != declared_scope:
                raise ValueError("source preparation raw scope changed or lacks execution evidence")
            verify_prepared_cache(found["cache"], cache_sha256=found["cache_sha256"],
                                  cache_record_sha256=found.get("cache_record_sha256"))
            continue
        pending.append((item, manifest["remote_root"], str(args.output / "shards"), method, spec, prefixes))
    if result.get("status") == "completed" and not pending:
        if prefixes is not None:
            final = Path(result["target_prefix_manifest"])
            if not final.is_file() or file_sha256(final) != result["target_prefix_manifest_sha256"]:
                raise ValueError("completed target prefix manifest changed or is missing")
            saved = json.loads(final.read_text(encoding="utf-8"))
            if {_row_key(row): row for row in saved["records"]} != completed:
                raise ValueError("completed target prefix ledger differs from the published manifest")
            print(json.dumps({"status": "completed", "reused_records": len(completed), "target_prefixes": str(final)}), flush=True)
            return
        final = Path(result["cache"])
        verify_prepared_cache(final, cache_sha256=result["cache_sha256"],
                              cache_record_sha256=result.get("cache_record_sha256"))
        print(json.dumps({"status": "completed", "reused_records": len(completed), "cache": str(final)}), flush=True)
        return
    if (args.output / "epochs.npz").exists():
        raise FileExistsError("uncommitted merged cache already exists; preserve it and use a new output directory")
    if result["errors"]:
        result["previous_errors"] = result["errors"]
        result["errors"] = []
    workers = max(1, min(len(pending) or 1, available_cpu_threads(),
        int(available_host_memory_bytes() * .7 // (2 * 2**30)), args.workers or available_cpu_threads()))
    result.update(workers=workers, status="running")
    atomic_write_json(result_path, result)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, item): item[0] for item in pending}
        for future in as_completed(futures):
            try:
                row = future.result()
                completed[_row_key(row)] = row
            except Exception as error:
                result["errors"].append({"record": futures[future], "error_type": type(error).__name__, "message": str(error)})
            result["records"] = list(completed.values())
            atomic_write_json(result_path, result)
            print(json.dumps({"study": manifest["study"], "prepared": len(completed), "errors":len(result["errors"]),
                              "requested":len(manifest["records"])}), flush=True)
    if result["errors"]:
        result["status"] = "failed"
        atomic_write_json(result_path, result)
        raise RuntimeError("source preparation has unresolved records; see complete failure ledger")
    rows = sorted(result["records"], key=_row_key)
    if prefixes is not None:
        final = args.output / "target_prefixes.json"
        atomic_write_json(final, {"schema": "n2p3_target_raw_prefix_collection/1",
            "preprocessing": spec.record(), "preparation_method": method, "records": rows})
        result.update(status="completed", target_prefix_manifest=str(final.resolve()),
                      target_prefix_manifest_sha256=file_sha256(final))
        atomic_write_json(result_path, result)
        print(json.dumps({"status": "completed", "target_prefixes": str(final), "records": len(rows)}), flush=True)
        return
    final = merge_source_shards([row["cache"] for row in rows], args.output / "epochs.npz")
    result.update(status="completed", cache=str(final), cache_sha256=file_sha256(final),
        cache_record_sha256=file_sha256(final.with_suffix(".record.json")),
        n_epochs=sum(row["n_epochs"] for row in rows), scheduled=sum(row["scheduled"] for row in rows),
        participants=len({row["record"]["subject"] for row in rows}))
    atomic_write_json(result_path, result)
    print(json.dumps({k:v for k,v in result.items() if k not in {"records","errors"}}), flush=True)


if __name__ == "__main__":
    main()
