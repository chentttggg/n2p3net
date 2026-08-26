"""Run development or one-use confirmatory GTN evaluation across seeds.

This wrapper deliberately launches each seed as an isolated process, so global
PyTorch RNG and device state cannot leak across seeds. It does not change model
architecture; remaining arguments are forwarded to the selected runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.experiment_protocol import (  # noqa: E402
    EvaluationProtocol,
    aggregate_seed_subject_hits,
    canonical_sha256,
    confirmatory_units_from_manifest,
    external_assets_sha256,
    parse_seeds,
    reserve_confirmatory_lock,
    runtime_environment_sha256,
    source_tree_sha256,
    validate_eligibility_manifest,
)


def _score_path(output: Path, runner: str, model: str) -> Path:
    name = "n2p3net" if runner == "n2p3net" else model
    return output / "scores" / f"{name}.json"


def _require_fresh_score_path(path: Path, *, seed: int) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing stale score file before seed {seed}: {path}.")


_LOCKED_RUNNER_OPTIONS = frozenset(
    {
        "--seed",
        "--subjects",
        "--primary-decision",
        "--save-scores-dir",
        "--run-dir",
        "--run-name",
        "--cache-dir",
        "--epoch-tmax",
        "--epoch-tmax-ms",
        "--no-cache",
        "--prepare-cache-only",
        "--benchmark",
        "--fold-offset",
        "--max-folds",
        "--evaluation-mode",
        "--protocol-sha256",
        "--dataset-sha256",
        "--device",
        "--deep-jobs",
        "--cohort-manifest",
        "--confirmatory-lock",
        "--source-sha256",
        "--runtime-sha256",
        "--external-assets-sha256",
    }
)


def _validate_forwarded_args(values: list[str]) -> list[str]:
    forwarded = values[1:] if values and values[0] == "--" else list(values)
    conflicts: set[str] = set()
    for token in forwarded:
        option = token.split("=", 1)[0]
        if not option.startswith("--"):
            continue
        if any(locked == option or locked.startswith(option) for locked in _LOCKED_RUNNER_OPTIONS):
            conflicts.add(option)
    if conflicts:
        raise ValueError(f"runner_args cannot override locked options: {sorted(conflicts)}.")
    return forwarded


def _forwarded_file_option(values: list[str], option: str) -> str | None:
    found: list[str] = []
    for index, token in enumerate(values):
        if token == option:
            if index + 1 >= len(values) or values[index + 1].startswith("--"):
                raise ValueError(f"{option} requires a file path.")
            found.append(values[index + 1])
        elif token.startswith(f"{option}="):
            found.append(token.split("=", 1)[1])
    if len(found) > 1:
        raise ValueError(f"{option} may be supplied only once.")
    if not found:
        return None
    path = Path(found[0])
    return str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())


def _validate_score_payload(
    payload: dict,
    *,
    path: Path,
    seed: int,
    mode: str,
    primary_metric: str,
    model: str,
    dataset_sha256: str,
    protocol_sha256: str,
    source_sha256: str | None = None,
    runtime_sha256: str | None = None,
    external_assets_sha256: str | None = None,
    confirmatory_id: str | None = None,
    confirmatory_lock_sha256: str | None = None,
) -> tuple[dict[str, float], dict[str, bool], tuple[str, ...], str]:
    if payload.get("schema") != "n2p3net_subject_scores/2":
        raise ValueError(f"{path} is not a schema-v2 complete ITT score file.")
    expected = {
        "seed": seed,
        "evaluation_mode": mode,
        "primary_decision_metric": primary_metric,
        "model": model,
        "dataset_sha256": dataset_sha256,
        "protocol_sha256": protocol_sha256,
        "source_sha256": source_sha256,
        "runtime_sha256": runtime_sha256,
        "external_assets_sha256": external_assets_sha256,
    }
    if mode == "confirmatory":
        expected["confirmatory_id"] = confirmatory_id
        expected["confirmatory_lock_sha256"] = confirmatory_lock_sha256
    mismatched = {
        key: (value, payload.get(key))
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatched:
        raise ValueError(f"{path} does not match its frozen run identity: {mismatched}.")
    stored_score_hash = payload.get("score_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "score_sha256"}
    if stored_score_hash != canonical_sha256(unsigned):
        raise ValueError(f"{path} score_sha256 does not match its contents.")
    units = tuple(str(unit) for unit in payload.get("evaluation_units", []))
    records = payload.get("primary_records", [])
    record_units = tuple(str(record.get("subject")) for record in records)
    if not units or len(set(units)) != len(units) or record_units != units:
        raise ValueError(f"{path} does not contain exactly one ordered record per frozen unit.")
    if int(payload.get("primary_n_subjects", -1)) != len(units):
        raise ValueError(f"{path} primary denominator differs from its frozen universe.")
    hits: dict[str, float] = {}
    availability: dict[str, bool] = {}
    for record in records:
        subject = str(record["subject"])
        available_raw = record.get("available")
        if not isinstance(available_raw, bool):
            raise ValueError(f"{path} has non-boolean availability for {subject!r}.")
        available = available_raw
        hit = float(record["hit"])
        predicted = record.get("predicted")
        true = record.get("true")
        if (
            hit not in (0.0, 1.0)
            or (not available and (hit != 0.0 or predicted is not None))
            or (available and predicted is None)
            or (predicted is not None and true is not None and hit != float(predicted == true))
        ):
            raise ValueError(f"{path} has an invalid ITT record for {subject!r}.")
        hits[subject] = hit
        availability[subject] = available
    recomputed = sum(hits.values()) / len(units)
    if abs(recomputed - float(payload["primary_hit_rate"])) > 1e-12:
        raise ValueError(f"{path} primary ITT rate does not match its complete records.")
    cohort_sha256 = str(payload.get("cohort_sha256", ""))
    if not cohort_sha256:
        raise ValueError(f"{path} lacks a cohort fingerprint.")
    return hits, availability, units, cohort_sha256


def main() -> None:
    parser = argparse.ArgumentParser(
        description="locked multi-seed GTN evaluation", allow_abbrev=False
    )
    parser.add_argument("--runner", choices=("n2p3net", "baseline"), required=True)
    parser.add_argument(
        "--model",
        default="eegnet",
        help="baseline runner model; 'all' is intentionally unsupported",
    )
    parser.add_argument("--mode", choices=("development", "confirmatory"), default="development")
    parser.add_argument(
        "--seeds", default=None, help="comma-separated; confirmatory default=0,1,2,3,4"
    )
    parser.add_argument("--subjects", type=int, default=None)
    parser.add_argument("--primary-decision", default="exact_llr@3")
    parser.add_argument("--confirmatory-id", default=None)
    parser.add_argument(
        "--cache-sha256",
        default=None,
        help="locked cohort hash; default hashes the standard full GTN cache",
    )
    parser.add_argument(
        "--cohort-manifest",
        default=str(ROOT / "experiments" / "protocols" / "gtn_confirmatory_cohort_v1.json"),
        help="independently frozen eligible-unit manifest",
    )
    parser.add_argument("--output-dir", default="experiments/runs/multiseed")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        forwarded = _validate_forwarded_args(args.runner_args)
        external_assets = {
            "frozen_erp_prior": _forwarded_file_option(forwarded, "--frozen-erp-prior"),
            "pretrained_checkpoint": _forwarded_file_option(
                forwarded, "--pretrained-checkpoint"
            ),
            "pretrained_mapping": _forwarded_file_option(forwarded, "--pretrained-mapping"),
        }
        assets_sha256 = external_assets_sha256(external_assets)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    seeds = parse_seeds(args.seeds, mode=args.mode)
    protocol = EvaluationProtocol(
        args.mode,
        seeds,
        args.primary_decision,
        args.subjects,
        benchmark=False,
    ).validate()
    if args.runner == "baseline" and args.model == "all":
        parser.error("multi-seed aggregation requires one baseline model at a time.")

    output_root = Path(args.output_dir)
    run_root = (
        output_root / "confirmatory" / str(args.confirmatory_id)
        if args.mode == "confirmatory" and args.confirmatory_id
        else output_root
    )
    frozen_config = {
        "schema": "n2p3net_multiseed/2",
        "runner": args.runner,
        "model": args.model if args.runner == "baseline" else "n2p3net",
        "protocol": {
            "mode": protocol.mode,
            "seeds": list(protocol.seeds),
            "primary_metric": protocol.primary_metric,
            "subject_limit": protocol.subject_limit,
        },
        "runner_args": forwarded,
        "external_assets_sha256": assets_sha256,
    }
    cohort_manifest_path = Path(args.cohort_manifest).resolve()
    if not cohort_manifest_path.is_file():
        parser.error(f"missing frozen cohort manifest: {cohort_manifest_path}")
    frozen_config["cohort_manifest"] = str(cohort_manifest_path)
    frozen_config["cohort_manifest_sha256"] = hashlib.sha256(
        cohort_manifest_path.read_bytes()
    ).hexdigest()
    # Confirmatory model comparisons share one preprocessing/event ledger.
    cache_path = (
        ROOT
        / "experiments"
        / "cache"
        / "gtn_events_v2_3ch_sf256_lf0.1_tm-0.2_tx1.2_nall.npz"
    )
    if not cache_path.is_file():
        parser.error(f"cannot lock cohort: missing shared cache {cache_path}")
    actual_dataset_sha256 = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    if args.cache_sha256 is not None and args.cache_sha256 != actual_dataset_sha256:
        parser.error(
            "--cache-sha256 does not match the shared tx1.2 cache: "
            f"{args.cache_sha256} != {actual_dataset_sha256}"
        )
    dataset_sha256 = actual_dataset_sha256
    from experiments.run_gtn_baseline import _load_gtn_cache

    _, _, _, _, truth, _, _ = _load_gtn_cache(cache_path)
    eligibility_manifest = validate_eligibility_manifest(
        cohort_manifest_path,
        dataset="gtn",
        truth_by_unit=truth,
    )
    if args.mode == "confirmatory":
        try:
            confirmatory_units_from_manifest(eligibility_manifest, truth)
        except ValueError as exc:
            parser.error(str(exc))
    if args.runner == "n2p3net" or args.model in {"eegnet", "inception", "conformer"}:
        import torch

        if not torch.cuda.is_available():
            parser.error("locked neural evaluation requires an available CUDA device")
    source_sha256 = source_tree_sha256(ROOT)
    runtime_sha256 = runtime_environment_sha256()
    frozen_config["dataset_sha256"] = dataset_sha256
    frozen_config["source_sha256"] = source_sha256
    frozen_config["runtime_sha256"] = runtime_sha256
    protocol_sha256 = canonical_sha256(frozen_config)
    manifest = {
        **frozen_config,
        "protocol_sha256": protocol_sha256,
        "started_utc": datetime.now(UTC).isoformat(),
    }
    if args.mode == "confirmatory" and run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"Confirmatory output directory is not empty: {run_root}.")
    confirmatory_lock_path: Path | None = None
    confirmatory_lock_sha256: str | None = None
    if args.mode == "confirmatory" and not args.dry_run:
        if not args.confirmatory_id:
            parser.error("--confirmatory-id is required in confirmatory mode.")
        confirmatory_lock_path = reserve_confirmatory_lock(
            output_root / "confirmatory_locks",
            confirmatory_id=args.confirmatory_id,
            dataset_sha256=dataset_sha256,
            protocol=protocol,
            frozen_config=frozen_config,
        )
        confirmatory_lock_sha256 = hashlib.sha256(confirmatory_lock_path.read_bytes()).hexdigest()
        manifest["confirmatory_lock"] = str(confirmatory_lock_path)
        manifest["confirmatory_lock_sha256"] = confirmatory_lock_sha256

    commands = []
    for seed in seeds:
        seed_dir = run_root / f"seed_{seed}"
        if args.runner == "n2p3net":
            command = [
                sys.executable,
                str(ROOT / "experiments" / "run_n2p3net_gtn.py"),
                "--seed",
                str(seed),
                "--primary-decision",
                args.primary_decision,
                "--run-dir",
                str(run_root),
                "--run-name",
                f"seed_{seed}",
                "--save-scores-dir",
                str(seed_dir / "scores"),
                "--evaluation-mode",
                args.mode,
                "--protocol-sha256",
                protocol_sha256,
                "--dataset-sha256",
                dataset_sha256,
                "--epoch-tmax-ms",
                "1200",
                "--device",
                "cuda",
                "--cohort-manifest",
                str(cohort_manifest_path),
                "--source-sha256",
                source_sha256,
                "--runtime-sha256",
                runtime_sha256,
                "--external-assets-sha256",
                assets_sha256,
            ]
        else:
            command = [
                sys.executable,
                str(ROOT / "experiments" / "run_gtn_baseline.py"),
                "--model",
                args.model,
                "--seed",
                str(seed),
                "--primary-decision",
                args.primary_decision,
                "--save-scores-dir",
                str(seed_dir / "scores"),
                "--evaluation-mode",
                args.mode,
                "--protocol-sha256",
                protocol_sha256,
                "--dataset-sha256",
                dataset_sha256,
                "--epoch-tmax",
                "1.2",
                "--cohort-manifest",
                str(cohort_manifest_path),
                "--source-sha256",
                source_sha256,
                "--runtime-sha256",
                runtime_sha256,
                "--external-assets-sha256",
                assets_sha256,
            ]
            if args.model in {"eegnet", "inception", "conformer"}:
                command.extend(("--device", "cuda", "--deep-jobs", "1"))
        if args.subjects is not None:
            command.extend(("--subjects", str(args.subjects)))
        if confirmatory_lock_path is not None:
            command.extend(("--confirmatory-lock", str(confirmatory_lock_path)))
        command.extend(forwarded)
        commands.append(command)

    manifest["commands"] = commands
    if args.dry_run:
        for command in commands:
            print(subprocess.list2cmdline(command))
        return
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    hits_by_seed = {}
    availability_by_seed = {}
    per_seed = []
    frozen_units: tuple[str, ...] | None = None
    cohort_sha256: str | None = None
    for seed, command in zip(seeds, commands, strict=True):
        if source_tree_sha256(ROOT) != source_sha256:
            raise RuntimeError("Executable source changed after the protocol was frozen.")
        if runtime_environment_sha256() != runtime_sha256:
            raise RuntimeError("Dependency runtime changed after the protocol was frozen.")
        if external_assets_sha256(external_assets) != assets_sha256:
            raise RuntimeError("External checkpoint/calibration assets changed after freezing.")
        score_path = _score_path(run_root / f"seed_{seed}", args.runner, args.model)
        _require_fresh_score_path(score_path, seed=seed)
        subprocess.run(command, cwd=ROOT, check=True)
        payload = json.loads(score_path.read_text(encoding="utf-8"))
        hits, availability, units, cohort = _validate_score_payload(
            payload,
            path=score_path,
            seed=seed,
            mode=args.mode,
            primary_metric=args.primary_decision,
            model="n2p3net" if args.runner == "n2p3net" else args.model,
            dataset_sha256=dataset_sha256,
            protocol_sha256=protocol_sha256,
            source_sha256=source_sha256,
            runtime_sha256=runtime_sha256,
            external_assets_sha256=assets_sha256,
            confirmatory_id=args.confirmatory_id if args.mode == "confirmatory" else None,
            confirmatory_lock_sha256=confirmatory_lock_sha256,
        )
        if frozen_units is None:
            frozen_units = units
            cohort_sha256 = cohort
        elif units != frozen_units or cohort != cohort_sha256:
            raise ValueError("Seed outputs changed the frozen cohort identity or ordering.")
        hits_by_seed[seed] = hits
        availability_by_seed[seed] = availability
        primary_metric_gate = dict(payload.get("primary_metric_gate") or {})
        per_seed.append(
            {
                "seed": seed,
                "primary_hit_rate": payload.get("primary_hit_rate", payload["hit_rate_mean"]),
                "n_subjects": len(units),
                "coverage": sum(availability.values()) / len(units),
                "primary_metric_gate": primary_metric_gate,
                "primary_claim_eligible": bool(
                    primary_metric_gate.get("claim_eligible", False)
                ),
                "score_path": str(score_path),
            }
        )

    uncertainty = aggregate_seed_subject_hits(
        hits_by_seed,
        subject_universe=frozen_units,
        availability_by_seed=availability_by_seed,
    )
    aggregate = uncertainty.__dict__
    aggregate["primary_claim_eligible"] = all(
        bool(entry["primary_claim_eligible"]) for entry in per_seed
    )
    aggregate["primary_claim_eligible_seed_fraction"] = (
        sum(bool(entry["primary_claim_eligible"]) for entry in per_seed) / len(per_seed)
        if per_seed
        else 0.0
    )
    aggregate["primary_metric_gate_by_seed"] = {
        str(entry["seed"]): entry["primary_metric_gate"] for entry in per_seed
    }
    result = {
        **manifest,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    result["finished_utc"] = datetime.now(UTC).isoformat()
    (run_root / "record.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
