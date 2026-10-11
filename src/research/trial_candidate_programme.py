"""Run frozen trial programmes through one scientific and resource owner."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path

from data.candidate_task import MembershipDecision
from data.digests import file_sha256
from data.epochs import load_epoch_dataset
from models.candidate_evidence import CandidateEvidence
from research.candidate_metrics import (
    compare_candidate_probability_metrics,
    evaluate_candidate_evidence,
)
from research.cloud_execution import CloudExecutor, CloudTask, target_phase_resources
from research.contracts import (
    DecisionPlanContract,
    DecisionPlanEntry,
    semantic_sha256,
)
from research.evaluation import verified_executed_snapshot
from research.execution import atomic_write_json
from research.source_execution import (
    HostMemoryAdmission,
    SourceInputAdmission,
    resolve_source_resources,
)
from research.trial_candidate_execution import (
    TrialCandidateDecision,
    execute_trial_candidate_participant,
)
from train.runtime import cpu_thread_budget
from transfer.ensemble_fit import FrozenSourceCheckpoint
from transfer.outcomes import CandidateCoverage, DecisionKey
from transfer.recipes import TargetRecipe

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "n2p3_trial_candidate_programme/1"


def _load_prepared(directory, dataset_id):
    if dataset_id == "BI2014a":
        from data.bi2014a_candidate import load_bi2014a_prepared_participant
        return load_bi2014a_prepared_participant(directory)
    if dataset_id == "BNCI2014-008":
        from data.bnci_trial_preparation import load_bnci_trial_prepared_participant
        return load_bnci_trial_prepared_participant(directory)
    raise ValueError("trial programme has no admitted dataset preparation owner")


def _verify_file(receipt):
    path = Path(receipt["path"])
    if file_sha256(path) != receipt["sha256"]:
        raise ValueError("trial programme input differs from its frozen file receipt")
    return path


def trial_participant_adapter(arguments):
    """Actor sees calibration labels and physical future schedules, without truth."""
    for name in ("calibration_cache", "future_cache", "source_checkpoint"):
        _verify_file(arguments[name])
    calibration = load_epoch_dataset(arguments["calibration_cache"]["path"], require_labels=True)
    future = load_epoch_dataset(arguments["future_cache"]["path"], require_labels=False)
    requests = tuple(TrialCandidateDecision.from_arguments(row) for row in arguments["requests"])
    result = execute_trial_candidate_participant(
        calibration=calibration, future=future, requests=requests,
        source=FrozenSourceCheckpoint.from_path(arguments["source_checkpoint"]["path"]),
        recipes=tuple(TargetRecipe(**row) for row in arguments["recipes"]),
        local_subject=arguments["local_subject"], output=arguments["output"],
        source_snapshot_sha256=arguments["source_snapshot_sha256"],
        execution_source_sha256=arguments["execution_source_sha256"], device=arguments["device"],
        excluded_source_datasets=(arguments["dataset_id"],),
    )
    path = Path(arguments["output"]) / "result.json"
    return {"participant_key": arguments["participant_key"], "result_path": str(path),
            "result_sha256": file_sha256(path), "arms": [row["status"] for row in result["arms"]]}


def _admit_participant(value, arguments):
    path = Path(value["result_path"])
    if path.resolve() != (Path(arguments["output"]) / "result.json").resolve():
        raise ValueError("trial result belongs to another participant output")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != value["result_sha256"]:
        raise ValueError("trial participant result changed before parent admission")
    result = json.loads(content)
    requests = tuple(TrialCandidateDecision.from_arguments(row) for row in arguments["requests"])
    if (value["participant_key"] != arguments["participant_key"] or result["local_subject"] != arguments["local_subject"]
            or result["source_checkpoint_sha256"] != arguments["source_checkpoint"]["sha256"]
            or result["source_snapshot_sha256"] != arguments["source_snapshot_sha256"]
            or result["execution_source_sha256"] != arguments["execution_source_sha256"]
            or result["recipes"] != arguments["recipes"]
            or semantic_sha256(result["requested"]) != semantic_sha256([row.authority_record() for row in requests])
            or [row["arm"] for row in result["arms"]] != ["source_zero_shot", *[row["name"] for row in arguments["recipes"]]]):
        raise ValueError("trial participant source, arm or execution authority differs")
    for arm in result["arms"]:
        if arm["status"] == "completed":
            model = Path(arguments["output"]) / arm["model_artifact"]["path"]
            if model.parent.resolve() != Path(arguments["output"]).resolve() or file_sha256(model) != arm["model_artifact"]["sha256"]:
                raise ValueError("trial participant model artifact differs from its producing arm")
            evidence = tuple(CandidateEvidence.from_record(row) for row in arm["predictions"])
            if len(evidence) != len(requests):
                raise ValueError("trial completed arm lacks its full original requested prediction roster")
            for prediction, request in zip(evidence, requests, strict=True):
                if (prediction.participant_key != request.participant_key or prediction.decision_id != request.decision_id
                        or prediction.evidence_level != request.evidence_level
                        or prediction.physical_support_digest != request.physical_support_digest
                        or prediction.expected_event_counts != tuple(request.expected_event_counts)
                        or prediction.candidate_vocabulary != request.decision.candidate_vocabulary):
                    raise ValueError("trial arm prediction differs from its physical request authority")
    return result


def compile_participants(plan, execution_source, output):
    """Freeze every original request and retain censored episodes in the denominator."""
    cohort = json.loads(_verify_file(plan["preparation_cohort"]).read_bytes())
    if cohort["status"] != "completed" or len(cohort["participants"]) != len(cohort["requested_subjects"]):
        raise ValueError("trial training requires the complete declared prepared cohort")
    entries, actors, horizons, coverage, request_rows = [], [], {}, {}, []
    for row in cohort["participants"]:
        directory = _verify_file(row["preparation"]).parent
        prepared = _load_prepared(directory, plan["dataset_id"])
        if prepared.ledger["dataset_id"] != plan["dataset_id"]:
            raise ValueError("prepared cohort belongs to another declared dataset")
        ledger, calibration = prepared.ledger, prepared.calibration
        if ledger["calibration_episode_count"] != plan["calibration_episode_count"]:
            raise ValueError("prepared episode budget differs from the frozen training programme")
        if ledger["source_profile_checkpoint"]["sha256"] != plan["source_checkpoint"]["sha256"]:
            raise ValueError("prepared input profile differs from the admitted source checkpoint")
        subject = ledger["subject"]
        participant = calibration.identity_table.record_for(subject).authority_key("source_or_global")
        requests = []
        for descriptor in ledger["future_decisions"]:
            episode = descriptor["episode_id"]
            decision = MembershipDecision.from_record(descriptor["membership_decision"])
            level = min(descriptor["complete_raw_repetitions"], plan["maximum_horizon"])
            if level < 1:
                raise ValueError("trial requested episode has no declared complete raw horizon")
            target = ledger["parent_scoring_truth"][episode]
            if target is None:
                raise ValueError("trial parent truth is missing for an original requested episode")
            entries.append(DecisionPlanEntry(participant, episode, decision.candidate_vocabulary[target]))
            key = DecisionKey(participant, episode)
            horizons[key] = level
            # The first complete raw repetition belongs to the same physical
            # schedule owner. It supplies each stimulus pattern once; its
            # declared repeats supply expected coverage even when the raw tail
            # ends early. No missing event identities or EEG rows are invented.
            first = decision.prefix(1)
            if not first.schedule_complete:
                raise ValueError("trial expected schedule lacks its complete physical first repetition")
            expected_counts = tuple(int(value) for value in first.membership.sum(axis=0) * level)
            expected = dict(zip(first.candidate_vocabulary, expected_counts, strict=True))
            coverage[key] = CandidateCoverage.from_mappings(expected, {})
            support = semantic_sha256({"participant": participant, "source_raw_sha256": calibration.provenance["source_sha256"],
                "episode": episode, "membership_decision": decision.record(), "horizon": level})
            requests.append({"participant_key": participant, "decision_id": episode,
                "decision": decision.record(), "physical_support_digest": support, "evidence_level": level,
                "expected_event_counts": expected_counts,
                "secondary_evidence_level": decision.repetitions if descriptor["terminal_censor"] else None,
                "unavailable_reason": None})
            request_rows.append({"participant_key": participant, **descriptor, "evidence_level": level,
                "physical_support_digest": support, "expected_event_counts": expected_counts,
                "primary_endpoint": "complete_raw_episode_depth"})
        actors.append({"dataset_id": plan["dataset_id"], "participant_key": participant, "local_subject": subject,
            "calibration_cache": ledger["calibration_cache"], "future_cache": ledger["future_cache"],
            "source_checkpoint": plan["source_checkpoint"], "requests": requests,
            "recipes": plan["recipes"], "source_snapshot_sha256": plan["source_snapshot_sha256"],
            "execution_source_sha256": execution_source, "device": plan["device"],
            "output": str(output / "participants" / directory.name)})
    decision_plan = DecisionPlanContract(plan["preparation_cohort"]["sha256"],
        semantic_sha256([row["participant_key"] for row in actors]),
        tuple(row["participant_key"] for row in actors), tuple(entries))
    return actors, decision_plan, horizons, coverage, request_rows


def run_programme(plan_path, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = json.loads(plan_path.read_bytes())
    if plan["schema"] == "n2p3_bi2014a_trial_programme/1":
        plan = {**plan, "schema": SCHEMA, "dataset_id": "BI2014a"}
    if plan["schema"] != SCHEMA or plan["seed_pairs"] != [[20260918, 20260918]]:
        raise ValueError("trial daily programme requires its frozen one-seed pipeline declaration")
    recipes = tuple(TargetRecipe(**row) for row in plan["recipes"])
    if (tuple(row.initialization for row in recipes) != ("source", "scratch")
            or any(row.seed != plan["seed_pairs"][0][1] or row.trial_loss_reduction != "calibration_balanced_fixed_mass"
                   for row in recipes)
            or {key: value for key, value in recipes[0].record().items() if key not in {"name", "initialization"}}
            != {key: value for key, value in recipes[1].record().items() if key not in {"name", "initialization"}}):
        raise ValueError("trial source/scratch require the same declared target procedure and fixed class risk")
    execution_source = verified_executed_snapshot(plan["execution_manifest"], root=ROOT)["archive_sha256"]
    _verify_file(plan["source_checkpoint"])
    output.mkdir(parents=True, exist_ok=True)
    plan_digest = semantic_sha256(plan)
    bound_path = output / "plan.json"
    if bound_path.exists() and semantic_sha256(json.loads(bound_path.read_bytes())) != plan_digest:
        raise ValueError("trial output belongs to another frozen programme")
    atomic_write_json(bound_path, plan)
    actors, decision_plan, horizons, coverage, requests = compile_participants(plan, execution_source, output)
    atomic_write_json(output / "requested.json", {"decision_plan": decision_plan.record(), "requests": requests})
    resources = resolve_source_resources(plan["resources"], len(actors), plan["device"])
    tasks = tuple(CloudTask(f"{plan['dataset_id']}:{row['local_subject']}",
        "research.trial_candidate_programme:trial_participant_adapter", row, row["output"],
        target_phase_resources(resources, plan["device"])) for row in actors)
    task_arguments = {task.task_id: task.arguments for task in tasks}
    completed, failures, admitted = [], [], {}
    with cpu_thread_budget(resources["coordinator_cpu_threads"]), multiprocessing.get_context("spawn").Manager() as manager:
        inputs = SourceInputAdmission(manager, capacity_bytes=resources["host_memory_budget_bytes"],
            load_workers=resources["target_workers"])
        admission = HostMemoryAdmission(manager, resources, input_admission=inputs)

        def finished(outcome):
            if outcome.status == "completed":
                value = outcome.value
                admitted[value["participant_key"]] = _admit_participant(value, task_arguments[outcome.task_id])
                completed.append(value)
            else:
                failures.append({"task_id": outcome.task_id, "status": outcome.status, "failure": outcome.failure})
            atomic_write_json(output / "progress.json", {"requested_participants": len(tasks),
                "completed": completed, "failures": failures}, input_admission=inputs)
            return ()

        run = CloudExecutor(admission, max_workers=resources["target_workers"],
            diagnostic_directory=output / "worker_diagnostics", identity={"plan_digest": plan_digest,
                "execution_source_sha256": execution_source}).run(tasks, on_finished=finished,
            on_observation=lambda row: atomic_write_json(output / "resources.live.json", row, input_admission=inputs))
        seed = plan["seed_pairs"][0][1]
        arm_names = ("source_zero_shot", *[row.name for row in recipes])
        by_arm = {name: [] for name in arm_names}
        fit_failures = {name: {} for name in arm_names}
        results = []
        for value in completed:
            record = admitted[value["participant_key"]]
            for entry in record["arms"]:
                if entry["status"] == "fit_failure":
                    failure = entry["failure"]
                    fit_failures[entry["arm"]][value["participant_key"]] = (
                        f"{failure['stage']}:{failure['error_code']}:{failure['detail']}")
                else:
                    by_arm[entry["arm"]].extend(CandidateEvidence.from_record(row) for row in entry["predictions"])
            results.append(value)
        vocabulary = tuple(actors[0]["requests"][0]["decision"]["candidate_vocabulary"])
        metrics = {name: evaluate_candidate_evidence({seed: predictions}, decision_plan=decision_plan,
            seeds=(seed,), candidate_vocabulary=vocabulary, requested_evidence_levels=horizons,
            requested_coverage=coverage, participant_failures_by_seed={seed: fit_failures[name]})
            for name, predictions in by_arm.items()}
        report = {"schema": "n2p3_trial_candidate_results/1", "status": run.status,
            "evidence_scope": "public_calibration_development", "plan_digest": plan_digest,
            "execution_source_sha256": execution_source, "source_checkpoint": plan["source_checkpoint"],
            "seed_pairs": plan["seed_pairs"], "decision_plan": decision_plan.record(),
            "participants": results, "execution_failures": failures,
            "metrics": {name: row.record() for name, row in metrics.items()},
            "contrasts": {name: compare_candidate_probability_metrics(metrics["source_zero_shot"], row).record()
                for name, row in metrics.items() if name != "source_zero_shot"},
            "resources": resources, "host_admission": admission.record(), "input_admission": inputs.record(),
            "worker_results": [{"task_id": row.task_id, "status": row.status, "worker": row.worker}
                for row in run.results.values()]}
        report["contrasts"]["source_full_fine_minus_matched_scratch"] = compare_candidate_probability_metrics(
            metrics[recipes[1].name], metrics[recipes[0].name]).record()
        report["secondary_terminal_partial"] = [{"participant_key": value["participant_key"],
            "arm": arm["arm"], **diagnostic} for value in completed
            for arm in admitted[value["participant_key"]]["arms"]
            if arm["status"] == "completed" for diagnostic in arm["secondary_diagnostics"]]
        atomic_write_json(output / "report.json", report, input_admission=inputs)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_programme(args.plan, args.output)
    if report["status"] != "completed":
        raise RuntimeError("trial programme has preserved its complete request roster and partial artifacts")


if __name__ == "__main__":
    main()
