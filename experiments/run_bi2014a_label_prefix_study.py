"""Prepare and run an opt-in BI2014a label-K programme from original files.

This entry uses chronological label budgets and the common independent-prefix
owner. The existing first-five-episode preparation and trial programme retain
their separately frozen input, dose and requested-decision contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.bi2014a_candidate import BI2014A_CHANNELS  # noqa: E402
from data.bi2014a_recording import BI2014A_ORIGINAL_FORMATS  # noqa: E402
from data.exclusions import select_allowed_subjects  # noqa: E402
from research.calibration_selection import transfer_scratch_prefix_plan  # noqa: E402
from research.source_catalog import paired_pipeline_seeds  # noqa: E402


def _artifact(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(Path(path).resolve()), "sha256": digest}


def prepare_bi2014a_label_prefix_plan(base_plan_path, output_plan, *, raw_root, subjects,
        source_format, source_expert_id, scratch_expert_id, xdawn_expert_id,
        baseline_name, output_name, minimum_selection_rows):
    """Bind the exact original cohort and derive the accepted four fixed family.

    Source checkpoint/profile/seed admission remains in the common public owner;
    this builder preserves its original catalog fields and normalizes paths.
    It creates only small source declarations and the new experiment plan.
    """
    base_plan_path, output_plan, raw_root = map(Path, (base_plan_path, output_plan, raw_root))
    plan = json.loads(base_plan_path.read_bytes())
    if (plan.get("schema") != "n2p3_candidate_ensemble_study/1"
            or plan.get("evaluation_design") != "independent_prefix"
            or plan.get("dataset_id", "BI2014a") != "BI2014a"
            or source_format not in BI2014A_ORIGINAL_FORMATS):
        raise ValueError("BI label-K requires an explicit original-format independent-prefix plan")
    pairs = paired_pipeline_seeds(plan["seed_pairs"])
    if {(row["outer_target_study"], row["source_seed"]) for row in plan["source_seed_catalog"]} != {
            ("BI2014a", pair[0]) for pair in pairs}:
        raise ValueError("BI label-K source catalog must declare the actual BI-excluded source seed producers")
    admitted = select_allowed_subjects("BI2014a", subjects, explicit=True)
    if not admitted or len(set(admitted)) != len(admitted):
        raise ValueError("BI label-K requires an explicit distinct admitted participant cohort")
    if output_plan.exists():
        raise FileExistsError("BI label-K plan output already belongs to another experiment")
    plan = deepcopy(plan)
    plan["dataset_id"] = "BI2014a"
    native_channels = {name.upper() for name in BI2014A_CHANNELS}
    for expert in plan["experts"]:
        if expert["family"] == "trial" and {name.upper() for name in expert["profile"]["channel_names"]} - native_channels:
            expert["profile"]["allow_missing_channels"] = True
    plan["source_seed_catalog"] = [{**entry, "path": str((base_plan_path.parent / entry["path"]).resolve())}
                                   for entry in plan["source_seed_catalog"]]
    if plan.get("source_snapshot_manifest"):
        plan["source_snapshot_manifest"] = str((base_plan_path.parent / plan["source_snapshot_manifest"]).resolve())
    plan = transfer_scratch_prefix_plan(plan, source_expert_id=source_expert_id,
        scratch_expert_id=scratch_expert_id, xdawn_expert_id=xdawn_expert_id,
        baseline_name=baseline_name, output_name=output_name,
        minimum_selection_rows=minimum_selection_rows)
    kind = "mat" if source_format.startswith("original_mat") else "csv"
    declarations = []
    # Read-only admission precedes writing any requested participant declaration.
    for subject in admitted:
        name = f"subject_{subject:02d}"
        path = raw_root / (name + (".zip" if source_format.endswith("_zip") else f"/{name}.{kind}"))
        if not source_format.endswith("_zip") and not path.is_file():
            path = raw_root / f"{name}.{kind}"
        source = {**_artifact(path), "subject": f"{subject:02d}"}
        if source_format.endswith("_zip"):
            source["member"] = f"{name}.{kind}"
        declarations.append((name, {"dataset_id": "BI2014a", "source_format": source_format,
                                    "record_sources": [source]}))
    directory = output_plan.parent / (output_plan.stem + "_original_sources")
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    for name, declaration in declarations:
        path = directory / (name + ".json")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(declaration, stream, indent=2, ensure_ascii=False)
        records.append({**_artifact(path), "dataset_id": "BI2014a", "subject": name.removeprefix("subject_")})
    plan["records"] = records
    with output_plan.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, ensure_ascii=False)
    return plan


def run_bi2014a_label_prefix_study(plan_path, output, *, execution="cloud"):
    """Run the same public caller and source gates with the BI original adapter."""
    from experiments.run_candidate_ensemble_study import run_study

    plan = json.loads(Path(plan_path).read_bytes())
    if (plan.get("dataset_id") != "BI2014a" or plan.get("evaluation_design") != "independent_prefix"
            or any(row.get("dataset_id") != "BI2014a" for row in plan["records"])):
        raise ValueError("BI label-K entry requires its own original-source independent-prefix cohort")
    return run_study(plan_path, output, execution=execution)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare")
    prepare.add_argument("--base-plan", type=Path, required=True)
    prepare.add_argument("--plan-output", type=Path, required=True)
    prepare.add_argument("--raw-root", type=Path, required=True)
    prepare.add_argument("--subjects", default=",".join(str(subject) for subject in range(1, 62)))
    prepare.add_argument("--source-format", choices=sorted(BI2014A_ORIGINAL_FORMATS), default="original_mat_zip")
    prepare.add_argument("--source-expert-id", required=True)
    prepare.add_argument("--scratch-expert-id", required=True)
    prepare.add_argument("--xdawn-expert-id", required=True)
    prepare.add_argument("--baseline-name", choices=("source_nn_fixed", "scratch_nn_fixed",
        "source_nn_xdawn_fixed", "scratch_nn_xdawn_fixed"), required=True)
    prepare.add_argument("--output-name", required=True)
    prepare.add_argument("--minimum-selection-rows", type=int, required=True)
    run = subcommands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--execution", choices=("cloud", "inline"), default="cloud")
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare_bi2014a_label_prefix_plan(arguments.base_plan, arguments.plan_output,
            raw_root=arguments.raw_root, subjects=[int(value) for value in arguments.subjects.split(",")],
            source_format=arguments.source_format, source_expert_id=arguments.source_expert_id,
            scratch_expert_id=arguments.scratch_expert_id, xdawn_expert_id=arguments.xdawn_expert_id,
            baseline_name=arguments.baseline_name, output_name=arguments.output_name,
            minimum_selection_rows=arguments.minimum_selection_rows)
    else:
        run_bi2014a_label_prefix_study(arguments.plan, arguments.output, execution=arguments.execution)


if __name__ == "__main__":
    main()
