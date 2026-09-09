"""Analyze declared target-stage contrasts without treating seeds as participants."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for search_path in (ROOT, ROOT / "src"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from research.execution import atomic_write_json  # noqa: E402
from stats.transfer import (  # noqa: E402
    ANALYSIS_SEED,
    BOOTSTRAP_ITERATIONS,
    analyze_transfer_contrasts,
    standard_s2_contrasts,
    standard_training_contrasts,
)


def _record_evidence_scope(payload: dict) -> tuple[str, list[str]]:
    markers = set()
    software = {"software_correctness", "synthetic", "contract", "contract_test", "synthetic_contract",
                "declared_mock", "simulated", "software_plan_no_device_or_accuracy_evidence"}
    public = {"development", "public_development"}

    def inspect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"evidence_scope", "stage", "simulation_status", "evidence_level"} and isinstance(child, str):
                    if child in software | public:
                        markers.add(child)
                if key == "synthetic" and child is True:
                    markers.add("synthetic")
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(payload)
    scope = "software_correctness" if markers & software else "public_development" if markers & public else "undeclared"
    return scope, sorted(markers)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, action="append", required=True)
    parser.add_argument("--stage", choices=("E0", "E1", "E3", "S2", "compact", "ES"), required=True)
    parser.add_argument("--studies", nargs="+", required=True)
    parser.add_argument("--source-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--target-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--standard-budgets", type=int, nargs="+", default=(200, 800, 2000))
    parser.add_argument("--contrast-json", type=Path)
    parser.add_argument("--arm-field")
    parser.add_argument("--group-field", action="append")
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--analysis-seed", type=int, default=ANALYSIS_SEED)
    parser.add_argument("--allow-software-correctness", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("analysis output exists; choose a new version")
    paths = tuple(path.resolve() for path in args.record)
    if len(set(paths)) != len(paths):
        parser.error("record paths must be unique")
    cells, sources, requests, scopes = [], [], [], set()
    for path in paths:
        data = path.read_bytes()
        payload = json.loads(data)
        if not isinstance(payload, dict) or not isinstance(payload.get("cells"), list):
            parser.error(f"stage record must contain a cells list: {path}")
        scope, markers = _record_evidence_scope(payload)
        if scope != "public_development" and not args.allow_software_correctness:
            parser.error(f"record evidence is {scope}; --allow-software-correctness retains a software-only analysis: {path}")
        scopes.add("software_correctness" if scope == "undeclared" else scope)
        for item in payload["cells"]:
            row = dict(item)
            for field in ("source_arm", "source_seed", "target_seed"):
                if field not in row and field in payload:
                    row[field] = payload[field]
            cells.append(row)
        manifest = payload.get("task_manifest", {})
        requested_budgets = manifest.get("requested_calibration_budgets")
        for scope in manifest.get("scopes", {}).values():
            if not scope.get("participant_key"):
                parser.error("task scope lacks canonical participant_key for the requested denominator")
            requests.append({"study": scope.get("study", payload.get("target_study")),
                             "participant_key": scope["participant_key"], "requested_Ks": requested_budgets,
                             "status": "reachable_scope", "actual_K": scope.get("K")})
        for exclusion in manifest.get("protocol_exclusions", []):
            if not exclusion.get("participant_key"):
                parser.error("protocol exclusion lacks canonical participant_key for the requested denominator")
            requests.append({"study": exclusion.get("study", payload.get("target_study")),
                             "participant_key": exclusion["participant_key"], "requested_Ks": requested_budgets,
                             "status": "protocol_excluded", "reason": exclusion.get("reason")})
        sources.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                        "schema": payload.get("schema"), "status": payload.get("status"),
                        "evidence_scope": scope, "evidence_scope_markers": markers})
    if len(scopes) != 1:
        parser.error("analyze public development and software correctness records separately")
    arm_field = args.arm_field or ("source_arm" if args.stage == "S2" else "arm")
    group_fields = tuple(args.group_field) if args.group_field is not None else (("arm",) if args.stage == "S2" else ())
    if args.contrast_json is not None:
        contrasts = json.loads(args.contrast_json.read_text(encoding="utf-8"))
    elif args.stage == "S2":
        by_study = {study: sorted({row[arm_field] for row in cells if row["study"] == study})
                    for study in args.studies}
        contrasts = standard_s2_contrasts(by_study)
    else:
        contrasts = standard_training_contrasts(args.stage)
    report = analyze_transfer_contrasts(
        cells, contrasts, expected_studies=args.studies, source_seeds=args.source_seeds,
        target_seeds=args.target_seeds, standard_budgets=args.standard_budgets,
        iterations=args.bootstrap_iterations, seed=args.analysis_seed, arm_field=arm_field,
        group_fields=group_fields,
        requested_participants=requests,
        evidence_scope=next(iter(scopes)),
    )
    report.update(stage=args.stage, source_records=sources)
    if args.contrast_json is not None:
        report["contrast_declaration"] = {"path": str(args.contrast_json.resolve()),
                                          "sha256": hashlib.sha256(args.contrast_json.read_bytes()).hexdigest()}
    atomic_write_json(args.output, report)
    print(f"Wrote {len(report['groups'])} analysis groups to {args.output.resolve()}")


if __name__ == "__main__":
    main()
