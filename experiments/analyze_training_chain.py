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
    analyze_source_objective_contrasts,
    analyze_transfer_contrasts,
    resolve_seed_pairs,
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
    parser.add_argument("--stage", choices=("E0", "E1", "E3", "S2", "compact", "ES", "P1", "P2"), required=True)
    parser.add_argument("--studies", nargs="+", required=True)
    parser.add_argument("--source-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--target-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--seed-pair", type=int, nargs=2, action="append", metavar=("SOURCE", "TARGET"),
                        help="Explicit complete-flow seed pairs, checked against execution declarations")
    parser.add_argument("--seed-design", choices=("paired", "crossed"))
    parser.add_argument("--source-arms", nargs="+")
    parser.add_argument("--reference-source-arm", default="P2_source_ce")
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
    cells, sources, requests, scopes, declarations = [], [], [], set(), []
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
        declarations.append({field: payload.get(field, manifest.get(field)) for field in
                             ("analysis_seed_pairs", "analysis_source_arms")})
        requested_budgets = manifest.get("requested_calibration_budgets")
        for task_scope in manifest.get("scopes", {}).values():
            if not task_scope.get("participant_key"):
                parser.error("task scope lacks canonical participant_key for the requested denominator")
            requests.append({"study": task_scope.get("study", payload.get("target_study")),
                             "participant_key": task_scope["participant_key"], "requested_Ks": requested_budgets,
                             "status": "reachable_scope", "actual_K": task_scope.get("K"),
                             **{field: task_scope.get(field, payload.get(field, payload.get("plan", {}).get(field)))
                                for field in ("source_arm", "source_layout", "source_selector")
                                if task_scope.get(field, payload.get(field, payload.get("plan", {}).get(field))) is not None}})
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

    def declaration(field):
        values = [record[field] for record in declarations]
        encoded = {json.dumps(value, sort_keys=True) for value in values}
        if len(encoded) != 1:
            raise ValueError(f"records disagree on {field} declaration, including absent declarations")
        return values[0]

    recorded_pairs = declaration("analysis_seed_pairs")
    seed_pairs = resolve_seed_pairs(args.source_seeds, args.target_seeds,
        seed_pairs=args.seed_pair if args.seed_pair is not None else recorded_pairs,
        mode=args.seed_design or "crossed")
    if recorded_pairs is not None:
        recorded = resolve_seed_pairs(args.source_seeds, args.target_seeds, seed_pairs=recorded_pairs)
        if set(seed_pairs) != set(recorded):
            raise ValueError("CLI seed pair declaration differs from execution declaration")
        if args.seed_design and set(recorded) != set(resolve_seed_pairs(
                args.source_seeds, args.target_seeds, mode=args.seed_design)):
            raise ValueError("CLI seed design declaration differs from execution declaration")
        seed_pairs = recorded
    elif args.stage in {"P1", "P2"} and not args.seed_pair and not args.seed_design and len(seed_pairs) > 1:
        parser.error("P1/P2 require execution analysis_seed_pairs or explicit --seed-pair/--seed-design")
    arm_field = args.arm_field or ("source_arm" if args.stage == "S2" else "arm")
    group_fields = tuple(args.group_field) if args.group_field is not None else (
        ("arm",) if args.stage == "S2" else ("source_arm",) if args.stage == "P2" else ())
    if args.contrast_json is not None:
        contrasts = json.loads(args.contrast_json.read_text(encoding="utf-8"))
    elif args.stage == "S2":
        by_study = {study: sorted({row[arm_field] for row in cells if row["study"] == study})
                    for study in args.studies}
        contrasts = standard_s2_contrasts(by_study)
    else:
        contrasts = standard_training_contrasts(args.stage)
    analysis_options = dict(expected_studies=args.studies, source_seeds=args.source_seeds,
        target_seeds=args.target_seeds, standard_budgets=args.standard_budgets,
        seed_pairs=seed_pairs, iterations=args.bootstrap_iterations, seed=args.analysis_seed,
        requested_participants=requests,
        evidence_scope=next(iter(scopes)))
    report = analyze_transfer_contrasts(cells, contrasts, arm_field=arm_field,
                                       group_fields=group_fields, **analysis_options)
    if args.stage == "P2" and args.arm_field is None and args.contrast_json is None:
        recorded_arms = declaration("analysis_source_arms")
        if args.source_arms is not None and recorded_arms is not None and set(args.source_arms) != set(recorded_arms):
            raise ValueError("CLI source arm declaration differs from execution declaration")
        source_arms = args.source_arms if args.source_arms is not None else recorded_arms
        if source_arms is None:
            parser.error("P2 source comparisons require analysis_source_arms or --source-arms")
        report["source_objective_comparisons"] = analyze_source_objective_contrasts(
            cells, source_arms=source_arms, reference_source_arm=args.reference_source_arm,
            group_fields=tuple(field for field in group_fields if field != "source_arm"), **analysis_options)
    report.update(stage=args.stage, source_records=sources)
    if args.contrast_json is not None:
        report["contrast_declaration"] = {"path": str(args.contrast_json.resolve()),
                                          "sha256": hashlib.sha256(args.contrast_json.read_bytes()).hexdigest()}
    atomic_write_json(args.output, report)
    print(f"Wrote {len(report['groups'])} analysis groups to {args.output.resolve()}")


if __name__ == "__main__":
    main()
