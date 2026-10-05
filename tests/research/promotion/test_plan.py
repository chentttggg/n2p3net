"""Promotion plan declarations and their generated task arguments."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from experiments import run_candidate_promotion_matrix as matrix
from research.contracts import MIN_PROMOTION_HIT_AT_R
from tests.research.promotion.support import (
    _argument,
    _load,
    _plan_record,
    _single_multidomain_arm_record,
    _write_plan,
)
from tests.support.io import write_json


def test_plan_rejects_hit_at_r_below_promotion_floor(tmp_path: Path) -> None:
    record = _plan_record()
    record["test_reps"] = 7
    record["statistical_design"]["evidence_level"] = 7
    with pytest.raises(ValueError, match="at least 8"):
        _load(tmp_path, record)


def test_omitted_evidence_floor_keeps_promotion_default(tmp_path: Path) -> None:
    """Predeclared plans that never mention the floor keep the promotion depth."""

    assert "evidence_floor" not in _plan_record()
    assert _load(tmp_path).evidence_floor == MIN_PROMOTION_HIT_AT_R


def test_plan_accepts_declared_evidence_floor_below_promotion_default(tmp_path: Path) -> None:
    """A development estimand may target a shallower hit@R, but must say so."""

    record = _plan_record()
    record["evidence_floor"] = 2
    record["test_reps"] = 2
    record["statistical_design"]["evidence_level"] = 2
    plan = _load(tmp_path, record)
    assert plan.evidence_floor == 2
    assert plan.test_reps == 2
    assert plan.statistical_design.evidence_level == 2


def test_plan_rejects_evidence_floor_below_one(tmp_path: Path) -> None:
    for floor in (0, -3):
        record = _plan_record()
        record["evidence_floor"] = floor
        with pytest.raises(ValueError, match="at least 1"):
            _load(tmp_path, record)


def test_plan_metric_name_defaults_requested_and_validates(tmp_path: Path) -> None:
    default = _load(tmp_path, _plan_record())
    assert default.metric_name == "requested_participant_operational_hit_rate"

    record = _plan_record()
    record["metric_name"] = "valid_subject_operational_hit_rate"
    plan = _load(tmp_path, record)
    assert plan.metric_name == "valid_subject_operational_hit_rate"
    assert "metric_name" in plan.raw_record

    record["metric_name"] = "made_up_metric"
    with pytest.raises(ValueError, match="metric_name must be one of"):
        _load(tmp_path, record)


@pytest.mark.parametrize("floor", [True, 2.5, "2"])
def test_plan_rejects_non_integer_evidence_floor(tmp_path: Path, floor) -> None:
    record = _plan_record()
    record["evidence_floor"] = floor
    with pytest.raises(ValueError, match="must be an integer"):
        _load(tmp_path, record)


@pytest.mark.parametrize(
    "floor,test_reps,evidence_level,message",
    [
        pytest.param(4, 3, 3, "at least 4", id="test-below-floor"),
        pytest.param(4, 8, 2, "at least 4", id="evidence-below-floor"),
        pytest.param(2, 4, 5, r"2 <= R <= 4 \(got 5\)", id="evidence-above-test"),
    ],
)
def test_plan_rejects_inconsistent_evidence_budget(
    tmp_path: Path,
    floor,
    test_reps,
    evidence_level,
    message,
) -> None:
    record = _plan_record()
    record["evidence_floor"] = floor
    record["test_reps"] = test_reps
    record["statistical_design"]["evidence_level"] = evidence_level
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, record)


def test_training_rejects_even_kernel_but_accepts_explicit_k65(tmp_path: Path) -> None:
    even = _plan_record()
    even["training"]["temporal_kernel_size"] = 34
    with pytest.raises(ValueError, match="must be odd"):
        _load(tmp_path, even)

    broad = _plan_record()
    broad["training"]["temporal_kernel_size"] = 65
    assert _load(tmp_path, broad).training.temporal_kernel_size == 65


def test_training_precision_is_explicit_and_validated(tmp_path: Path) -> None:
    record = _plan_record()
    record["training"]["precision"] = "bf16"
    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    checkpoint = next(task for task in dag.tasks if task.kind == "checkpoint_batch")

    assert _argument(checkpoint.argv, "--precision") == "bf16"

    record["training"]["precision"] = "tf32"
    with pytest.raises(ValueError, match="training.precision"):
        _load(tmp_path, record)


@pytest.mark.parametrize(
    "training,source_overrides",
    [
        pytest.param(
            {
                "feature_transfer_invariant_bottleneck": 2,
                "feature_transfer_private_channels": 3,
                "feature_transfer_target_bottleneck": 2,
                "feature_transfer_target_kernel": 5,
                "feature_transfer_max_invariant_residual": 0.2,
                "feature_transfer_max_target_residual": 0.3,
                "feature_alignment_weight": 0.1,
                "private_domain_weight": 0.05,
                "orthogonality_weight": 0.01,
            },
            {"source_domain_mass": None, "risk_within_domain_unit": None, "selection_domain": None},
            id="feature-transfer",
        ),
        pytest.param(
            {
                "feature_transfer_invariant_bottleneck": 2,
                "feature_alignment_weight": 0.1,
                "feature_alignment_mode": "conditional_rbf_mmd",
            },
            {"source_domain_mass": None, "risk_within_domain_unit": None, "selection_domain": None},
            id="conditional-rbf-mmd",
        ),
        pytest.param(
            {
                "feature_transfer_invariant_bottleneck": 2,
                "feature_alignment_weight": 0.1,
                "feature_alignment_mode": "conditional_rbf_mmd",
                "feature_alignment_min_cell_count": 6,
                "feature_alignment_inactive_policy": "fail_closed",
            },
            {},
            id="alignment-cell-policy",
        ),
    ],
)
def test_training_forwards_feature_contract_to_pretraining(
    tmp_path: Path,
    training,
    source_overrides,
) -> None:
    record = _plan_record()
    record["source_arms"][1].update(source_overrides)
    record["training"].update(training)
    plan = _load(tmp_path, record)
    checkpoint = matrix.build_dag(plan).task_by_id["checkpoint_batch/source_a/rep01"]
    for field, expected in training.items():
        assert getattr(plan.training, field) == expected
        flag = "--" + field.replace("_", "-")
        assert _argument(checkpoint.argv, flag) == str(expected)
    assert plan.source_arms[1].selection_domain == record["source_arms"][1]["selection_domain"]


def test_plan_round_trips_named_splitbn_and_source_residual_contract(
    tmp_path: Path,
) -> None:
    record = _single_multidomain_arm_record()
    record["source_arms"][0]["splitbn_source_weights"] = {"B": 0.7, "A": 0.3}
    record["training"].update(
        {
            "feature_transfer_invariant_bottleneck": 2,
            "splitbn_target_buffer_source": "source_weighted",
            "source_residual_drop_p": 0.2,
            "source_residual_drop_mode": "batch",
        }
    )

    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    checkpoint = next(task for task in dag.tasks if task.kind == "checkpoint_batch")

    assert plan.raw_record == record
    assert plan.training.splitbn_target_buffer_source == "source_weighted"
    assert plan.training.source_residual_drop_p == 0.2
    assert plan.training.source_residual_drop_mode == "batch"
    assert dict(plan.source_arms[0].splitbn_source_weights) == {"A": 0.3, "B": 0.7}
    assert _argument(checkpoint.argv, "--splitbn-target-buffer-source") == "source_weighted"
    weights = [
        value for flag, value in pairwise(checkpoint.argv) if flag == "--splitbn-source-weight"
    ]
    assert weights == ["A=0.3", "B=0.7"]
    assert _argument(checkpoint.argv, "--source-residual-drop-p") == "0.2"
    assert _argument(checkpoint.argv, "--source-residual-drop-mode") == "batch"


def test_plan_round_trips_source_selection_record_and_forwards_to_pretraining(
    tmp_path: Path,
) -> None:
    record = _plan_record()
    record["source_arms"][0]["source_selection_record"] = "selection/source-a.json"

    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)

    expected = (tmp_path / "selection" / "source-a.json").resolve()
    assert plan.source_arms[0].source_selection_record == expected
    assert plan.source_arms[1].source_selection_record is None
    selected = dag.task_by_id["checkpoint_batch/source_a/rep01"]
    unselected = dag.task_by_id["checkpoint_batch/source_ab/rep01"]
    assert _argument(selected.argv, "--source-selection-record") == str(expected)
    assert "--source-selection-record" not in unselected.argv

    selected_result_batch = dag.task_by_id["result_batch/source_a/rep01/p01"]
    unselected_result_batch = dag.task_by_id["result_batch/source_ab/rep01/p01"]
    for job in dag.result_batch_job_records[selected_result_batch.task_id]["jobs"]:
        argv = list(job["argv"])
        assert _argument(argv, "--source-selection-record") == str(expected)
    for job in dag.result_batch_job_records[unselected_result_batch.task_id]["jobs"]:
        assert "--source-selection-record" not in job["argv"]


def test_plan_rejects_invalid_source_selection_record(tmp_path: Path) -> None:
    record = _plan_record()
    record["source_arms"][0]["source_selection_record"] = 7

    with pytest.raises(ValueError, match="source_selection_record must be a non-empty string"):
        _load(tmp_path, record)


def test_default_training_omits_p0_p1_cli_flags(tmp_path: Path) -> None:
    plan = _load(tmp_path)
    checkpoint = next(
        task for task in matrix.build_dag(plan).tasks if task.kind == "checkpoint_batch"
    )

    assert plan.training.splitbn_target_buffer_source is None
    assert plan.training.source_residual_drop_p is None
    assert plan.training.source_residual_drop_mode is None
    assert not {
        "--splitbn-target-buffer-source",
        "--splitbn-source-weight",
        "--source-residual-drop-p",
        "--source-residual-drop-mode",
    } & set(checkpoint.argv)


@pytest.mark.parametrize(
    "buffer_source,weights,message",
    [
        pytest.param("implicit_target", None, "splitbn_target_buffer_source", id="unknown-buffer"),
        pytest.param("source_weighted", None, "requires domain-keyed", id="missing-weights"),
        pytest.param(
            "source_weighted", [0.5, 0.5], "must be a JSON mapping", id="positional-weights"
        ),
        pytest.param(None, {"A": 0.5, "B": 0.5}, "only valid", id="inactive-weights"),
        pytest.param("source_weighted", {"A": 0.6, "B": 0.5}, "must sum to one", id="weight-sum"),
        pytest.param(
            "source_weighted", {"A": 0.5, "C": 0.5}, "source_domain_mass keys", id="domain-mismatch"
        ),
    ],
)
def test_plan_rejects_invalid_splitbn_configuration(tmp_path, buffer_source, weights, message):
    record = _single_multidomain_arm_record()
    record["training"]["splitbn_target_buffer_source"] = buffer_source
    record["source_arms"][0]["splitbn_source_weights"] = weights
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, record)


@pytest.mark.parametrize(
    "bottleneck,drop_p,drop_mode,message",
    [
        pytest.param(2, 0.1, "sample", "preregistered", id="drop-gap"),
        pytest.param(
            0, 0.2, "sample", "feature_transfer_invariant_bottleneck", id="no-feature-transfer"
        ),
        pytest.param(2, 0.2, None, "requires an explicit", id="missing-mode"),
        pytest.param(
            0, None, "sample", "requires source_residual_drop_p", id="missing-probability"
        ),
        pytest.param(2, 0.2, "epoch", "source_residual_drop_mode", id="unknown-mode"),
    ],
)
def test_plan_rejects_invalid_source_residual_configuration(
    tmp_path,
    bottleneck,
    drop_p,
    drop_mode,
    message,
):
    record = _single_multidomain_arm_record()
    record["training"].update(
        feature_transfer_invariant_bottleneck=bottleneck,
        source_residual_drop_p=drop_p,
        source_residual_drop_mode=drop_mode,
    )
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, record)


@pytest.mark.parametrize("drop_p", [0.0, 0.2, 0.5])
def test_plan_accepts_preregistered_source_residual_drop_arms(
    tmp_path: Path,
    drop_p: float,
) -> None:
    record = _single_multidomain_arm_record()
    record["training"].update(
        {
            "feature_transfer_invariant_bottleneck": 2,
            "source_residual_drop_p": drop_p,
            "source_residual_drop_mode": "sample",
        }
    )

    plan = _load(tmp_path, record)

    assert plan.training.source_residual_drop_p == drop_p


@pytest.mark.parametrize("buffer_source", ["pooled_source", "target_calibration"])
def test_plan_accepts_unweighted_splitbn_buffer_sources(
    tmp_path: Path,
    buffer_source: str,
) -> None:
    record = _single_multidomain_arm_record()
    record["training"]["splitbn_target_buffer_source"] = buffer_source

    plan = _load(tmp_path, record)
    checkpoint = next(
        task for task in matrix.build_dag(plan).tasks if task.kind == "checkpoint_batch"
    )

    assert plan.training.splitbn_target_buffer_source == buffer_source
    assert "--splitbn-source-weight" not in checkpoint.argv


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_domain_mass", {"A": 0.7, "B": 0.2}, "summing to one"),
        ("risk_within_domain_unit", "row_repeat", "epoch or participant"),
        ("selection_domain", "UNKNOWN", "weighted domain"),
    ],
)
def test_plan_rejects_invalid_multisource_risk(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    record = _plan_record()
    record["source_arms"][1][field] = value

    with pytest.raises(ValueError, match=match):
        _load(tmp_path, record)


def test_plan_supports_natural_ce_with_target_domain_selection(tmp_path: Path) -> None:
    record = _plan_record()
    arm = record["source_arms"][1]
    arm["source_domain_mass"] = None
    arm["risk_within_domain_unit"] = None
    arm["selection_domain"] = "A"

    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    checkpoint = dag.task_by_id["checkpoint_batch/source_ab/rep01"]

    assert "--source-domain-mass" not in checkpoint.argv
    assert "--risk-within-domain-unit" not in checkpoint.argv
    assert _argument(checkpoint.argv, "--selection-domain") == "A"


def test_plan_supports_candidate_loss_ablation_interface(tmp_path: Path) -> None:
    record = _plan_record()
    record["evaluation_arms"][1]["candidate_loss_weight"] = 0.25
    record["evaluation_arms"][1]["candidate_loss_temperature"] = 2.0
    plan = _load(tmp_path, record)
    assert plan.evaluation_arms[1].candidate_loss_weight == 0.25
    assert plan.evaluation_arms[1].candidate_loss_temperature == 2.0
    dag = matrix.build_dag(plan)
    batch_task = dag.task_by_id["result_batch/source_a/rep01/p01"]
    argv = next(
        list(job["argv"])
        for job in dag.result_batch_job_records[batch_task.task_id]["jobs"]
        if job["arm_name"] == "source_a__linear_time_split"
    )
    assert "--candidate-loss-weight" in argv
    assert "--candidate-loss-temperature" in argv
    assert _argument(argv, "--candidate-loss-weight") == "0.25"
    assert _argument(argv, "--candidate-loss-temperature") == "2.0"


def test_plan_rejects_candidate_loss_on_zero_shot_arm(tmp_path: Path) -> None:
    record = _plan_record()
    record["evaluation_arms"][0]["candidate_loss_weight"] = 0.1
    with pytest.raises(ValueError, match="candidate_loss_weight must be 0"):
        _load(tmp_path, record)


@pytest.mark.parametrize("head", ["linear", "mlp16", "classifier_fine", "full_fine", "adapter"])
def test_plan_supports_current_fitted_runner_heads(tmp_path: Path, head: str) -> None:
    record = _plan_record()
    record["evaluation_arms"][1]["head"] = head
    assert _load(tmp_path, record).evaluation_arms[1].head == head


def test_plan_supports_checkpoint_deterministic_auto_head(tmp_path: Path) -> None:
    record = _plan_record()
    record["evaluation_arms"][0]["head"] = "auto"
    assert _load(tmp_path, record).evaluation_arms[0].head == "auto"


@pytest.mark.parametrize(
    ("arm_index", "field", "value", "match"),
    [
        (0, "normalization", "target_prefix", "zero-shot combination"),
        (0, "epoch_selection", "target_time_split", "zero-shot combination"),
        (0, "epochs", 1, "zero-shot combination"),
        (0, "batch_size", 64, "zero-shot combination"),
        (0, "lr", 1e-3, "zero-shot combination"),
        (0, "fold_local_qc", True, "zero-shot combination"),
        (1, "epochs", None, "requires explicit values"),
        (1, "target_stat_weight", 0.25, "unless normalization is shrinkage"),
    ],
)
def test_plan_rejects_invalid_evaluation_arm_combinations(
    tmp_path: Path,
    arm_index: int,
    field: str,
    value: object,
    match: str,
) -> None:
    record = _plan_record()
    record["evaluation_arms"][arm_index][field] = value
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, record)


@pytest.mark.parametrize(
    "field,message",
    [
        ("target_subject", "target subject mappings"),
        ("source_holdout_subject", "source holdout mappings"),
    ],
)
def test_plan_rejects_duplicate_target_holdout_mapping(tmp_path, field, message):
    record = _plan_record()
    first, second = record["target_subjects"]
    second[field] = first[field]
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, record)


def test_plan_requires_source_holdout_mapping(tmp_path):
    record = _plan_record()
    del record["target_subjects"][0]["source_holdout_subject"]
    with pytest.raises(ValueError, match="missing required fields"):
        _load(tmp_path, record)


@pytest.mark.parametrize(
    "section,field,message",
    [
        ("source_arms", "name", "source arm names"),
        ("training_replicates", "seed", "training replicate seeds"),
    ],
)
def test_plan_rejects_duplicate_arm_or_seed(tmp_path, section, field, message):
    record = _plan_record()
    first, second = record[section]
    second[field] = first[field]
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, record)


def test_plan_rejects_output_component_escape(tmp_path: Path) -> None:
    record = _plan_record()
    record["target_subjects"][0]["partition_key"] = "../../outside"
    with pytest.raises(ValueError, match="portable path component"):
        _load(tmp_path, record)


def test_plan_rejects_output_root_escape_from_explicit_root(tmp_path: Path) -> None:
    record = _plan_record(output_root="../outside")
    plan_path = _write_plan(tmp_path, record)
    with pytest.raises(ValueError, match="explicit --root boundary"):
        matrix.load_plan(plan_path, root=tmp_path)


def test_dry_run_emits_full_matrix_without_writing_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = _write_plan(tmp_path, _plan_record())
    resolved_inputs = {
        "source_snapshot": {"archive_sha256": "a" * 64},
        "digest": "b" * 64,
    }
    monkeypatch.setattr(matrix, "_verify_inputs", lambda _dag: resolved_inputs)
    matrix.main(["--plan", str(plan_path), "--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert output["task_counts"] == {
        "checkpoint_batch": 4,
        "result": 0,
        "result_batch": 8,
        "manifest": 1,
        "analysis": 1,
    }
    assert output["task_count"] == 14
    assert output["resolved_inputs"] == resolved_inputs
    assert all(task["argv"][0] == "$PYTHON" for task in output["tasks"])
    assert len(output["result_batch_records"]) == 8
    batch_record = next(
        record["record"]
        for record in output["result_batch_records"].values()
        if record["record"]["job_count"] == 2
    )
    assert batch_record["jobs"][0]["argv"][0] == "--dataset-cache"
    assert not (tmp_path / "outputs").exists()


def test_dag_passes_explicit_time_split_linear_configuration(tmp_path: Path) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    batch_task = dag.task_by_id["result_batch/source_a/rep01/p01"]
    assert batch_task.kind == "result_batch"
    assert dag.result_batch_job_paths[batch_task.task_id].is_absolute()
    jobs = {
        job["arm_name"]: list(job["argv"])
        for job in dag.result_batch_job_records[batch_task.task_id]["jobs"]
    }
    assert set(jobs) == {"source_a__zero_shot", "source_a__linear_time_split"}
    linear = jobs["source_a__linear_time_split"]
    expected = {
        "--head": "linear",
        "--normalization": "source",
        "--epoch-selection": "target_time_split",
        "--epochs": "30",
        "--batch-size": "64",
        "--lr": "0.001",
        "--target-stat-weight": "0.0",
    }
    for flag, value in expected.items():
        assert _argument(linear, flag) == value
    assert "--no-fold-local-qc" in linear

    weighted_checkpoint = dag.task_by_id["checkpoint_batch/source_ab/rep01"]
    checkpoint_argv = list(weighted_checkpoint.argv)
    masses = [value for flag, value in pairwise(checkpoint_argv) if flag == "--source-domain-mass"]
    assert masses == ["A=0.8", "B=0.2"]
    assert _argument(checkpoint_argv, "--risk-within-domain-unit") == "participant"
    assert _argument(checkpoint_argv, "--selection-domain") == "A"
    assert "--subject-prefix-repeat" not in checkpoint_argv

    zero_shot = jobs["source_a__zero_shot"]
    assert "--zero-shot-feature-cache" in zero_shot
    assert not {
        "--epochs",
        "--batch-size",
        "--lr",
        "--fold-local-qc",
        "--no-fold-local-qc",
        "--adapt-batchnorm",
        "--epoch-selection",
        "--target-stat-weight",
    } & set(zero_shot)


def test_plan_supports_chunked_checkpoint_batches(tmp_path: Path) -> None:
    record = _plan_record()
    record["checkpoint_batch_size"] = 1
    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    checkpoints = [task for task in dag.tasks if task.kind == "checkpoint_batch"]
    assert len(checkpoints) == 8
    assert {
        "checkpoint_batch/source_a/rep01/chunk-00",
        "checkpoint_batch/source_a/rep01/chunk-01",
    } <= {task.task_id for task in checkpoints}
    result_batch = dag.task_by_id["result_batch/source_a/rep01/p01"]
    assert result_batch.dependencies == ("checkpoint_batch/source_a/rep01/chunk-00",)
    zero_shot_argv = next(
        list(job["argv"])
        for job in dag.result_batch_job_records[result_batch.task_id]["jobs"]
        if job["arm_name"] == "source_a__zero_shot"
    )
    assert "--zero-shot-feature-cache" in zero_shot_argv


@pytest.mark.parametrize(
    "options,expected_threads,expected_cli_threads",
    [
        pytest.param(
            {"result_max_workers": 16, "result_threads": 1}, 1, "1", id="explicit-threads"
        ),
        pytest.param({"result_max_workers": 8}, None, "18", id="resource-aware-threads"),
    ],
)
def test_plan_supports_result_worker_and_thread_tuning(
    tmp_path,
    monkeypatch,
    options,
    expected_threads,
    expected_cli_threads,
):
    monkeypatch.setattr(matrix, "resolve_cpu_threads", lambda workers: workers + 10)
    record = _plan_record()
    record.update(options)
    plan = _load(tmp_path, record)
    workers = options["result_max_workers"]
    assert plan.result_max_workers == workers
    assert plan.result_threads == expected_threads
    dag = matrix.build_dag(plan, result_max_workers=workers)
    jobs = dag.result_batch_job_records["result_batch/source_a/rep01/p01"]["jobs"]
    argv = next(job["argv"] for job in jobs if job["arm_name"] == "source_a__zero_shot")
    assert _argument(argv, "--cpu-threads") == expected_cli_threads


def test_plan_round_trips_euclidean_alignment_acw_state_and_sampler_kind(
    tmp_path: Path,
) -> None:
    record = _plan_record()
    arm = record["source_arms"][0]
    arm["euclidean_alignment"] = {
        "mode": "source",
        "cache": "source-a.npz",
        "reference_mode": "native_preserve",
    }
    arm["acw_state"] = "offline/source-a.acw.json"
    arm["sampler_kind"] = "uniform"

    plan = _load(tmp_path, record)
    parsed = plan.source_arms[0]

    assert parsed.sampler_kind == "uniform"
    assert parsed.acw_state == (tmp_path / "offline" / "source-a.acw.json").resolve()
    assert parsed.euclidean_alignment is not None
    assert parsed.euclidean_alignment.mode == "source"
    assert parsed.euclidean_alignment.cache == (tmp_path / "source-a.npz").resolve()
    assert parsed.euclidean_alignment.reference_mode == "native_preserve"

    # An omitted alignment cache defaults to the source arm cache.
    arm["euclidean_alignment"] = {"mode": "source", "reference_mode": "common_average"}
    plan = _load(tmp_path, record)
    alignment = plan.source_arms[0].euclidean_alignment
    assert alignment is not None
    assert alignment.cache == plan.source_arms[0].cache
    assert alignment.reference_mode == "common_average"

    arm["euclidean_alignment"] = {"mode": "none"}
    plan = _load(tmp_path, record)
    alignment = plan.source_arms[0].euclidean_alignment
    assert alignment is not None
    assert alignment.mode == "none"
    assert alignment.cache is None
    assert alignment.reference_mode is None


@pytest.mark.parametrize(
    ("alignment", "match"),
    [
        ({"mode": "bogus"}, "euclidean_alignment.mode"),
        ({"mode": "source", "cache": "source-ab.npz"}, "must resolve to the source arm cache"),
        ({"mode": "source", "reference_mode": "identity"}, "reference_mode must be one of"),
        ({"mode": "none", "cache": "source-a.npz"}, "cache must be null/absent"),
        (
            {"mode": "none", "reference_mode": "native_preserve"},
            "reference_mode must be null/absent",
        ),
    ],
)
def test_plan_rejects_invalid_euclidean_alignment_fields(
    tmp_path: Path,
    alignment: dict[str, Any],
    match: str,
) -> None:
    record = _plan_record()
    record["source_arms"][0]["euclidean_alignment"] = alignment
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, record)


def test_sampler_kind_is_cross_checked_against_existing_selection_record(
    tmp_path: Path,
) -> None:
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    selection_path = selection_dir / "source-a.json"
    write_json(
        selection_path,
        {
            "schema": "n2p3_source_selection/2",
            "sampler_kind": "target_similarity_diverse",
        },
    )
    record = _plan_record()
    arm = record["source_arms"][0]
    arm["source_selection_record"] = "selection/source-a.json"
    arm["sampler_kind"] = "target_similarity_diverse"

    assert _load(tmp_path, record).source_arms[0].sampler_kind == ("target_similarity_diverse")

    write_json(selection_path, {"schema": "n2p3_source_selection/2", "sampler_kind": "dense_pds"})
    with pytest.raises(ValueError, match="disagrees with declared"):
        _load(tmp_path, record)

    write_json(
        selection_path,
        {"schema": "n2p3_source_selection/1", "sampler_kind": "target_similarity_diverse"},
    )
    with pytest.raises(ValueError, match="unsupported schema"):
        _load(tmp_path, record)


def test_sampler_kind_record_combinations_and_missing_files(tmp_path: Path) -> None:
    record = _plan_record()
    arm = record["source_arms"][0]
    arm["sampler_kind"] = "uniform"
    arm["source_selection_record"] = "selection/source-a.json"
    with pytest.raises(ValueError, match="cannot be combined"):
        _load(tmp_path, record)

    arm.pop("source_selection_record")
    arm["sampler_kind"] = "dense_pds"
    with pytest.raises(ValueError, match="requires source_selection_record"):
        _load(tmp_path, record)

    arm["source_selection_record"] = "selection/missing.json"
    plan = _load(tmp_path, record)
    assert plan.source_arms[0].sampler_kind == "dense_pds"
    assert (
        plan.source_arms[0].source_selection_record
        == (tmp_path / "selection" / "missing.json").resolve()
    )


def test_plan_parses_as_mmd_source_stream_full_fine(tmp_path: Path) -> None:
    record = _plan_record()
    arm = record["evaluation_arms"][1]
    arm["head"] = "full_fine"
    arm["as_mmd"] = {
        "lambda0": 0.25,
        "warmup_epochs": 3,
        "target_weight_min": 2.0,
        "target_weight_max": 8.0,
        "input": "source_stream",
    }

    plan = _load(tmp_path, record)
    parsed = plan.evaluation_arms[1]

    assert parsed.as_mmd_lambda0 == 0.25
    assert parsed.as_mmd_warmup_epochs == 3
    assert parsed.as_mmd_target_weight_min == 2.0
    assert parsed.as_mmd_target_weight_max == 8.0
    assert parsed.as_mmd_input == "source_stream"
    assert parsed.as_mmd_source_logits is None

    arm["as_mmd"] = {"lambda0": 0.1}
    parsed = _load(tmp_path, record).evaluation_arms[1]
    assert parsed.as_mmd_warmup_epochs == 1
    assert parsed.as_mmd_target_weight_min == 1.0
    assert parsed.as_mmd_target_weight_max == 6.0
    assert parsed.as_mmd_input == "source_stream"


def test_plan_parses_frozen_logits_as_mmd_and_requires_source_logits(
    tmp_path: Path,
) -> None:
    record = _plan_record()
    arm = record["evaluation_arms"][1]
    arm["as_mmd"] = {
        "lambda0": 0.5,
        "input": "frozen_logits",
        "source_logits": "offline/logits.npz",
    }

    plan = _load(tmp_path, record)
    parsed = plan.evaluation_arms[1]
    assert parsed.head == "linear"
    assert parsed.as_mmd_input == "frozen_logits"
    assert parsed.as_mmd_source_logits == (tmp_path / "offline" / "logits.npz").resolve()

    arm["as_mmd"].pop("source_logits")
    with pytest.raises(ValueError, match="source_logits is required"):
        _load(tmp_path, record)

    arm["as_mmd"] = {"lambda0": 0.5, "source_logits": "offline/logits.npz"}
    with pytest.raises(ValueError, match="source_logits must be null/absent"):
        _load(tmp_path, record)


def test_as_mmd_zero_lambda_forbids_input_and_source_logits(tmp_path: Path) -> None:
    record = _plan_record()
    arm = record["evaluation_arms"][1]
    arm["head"] = "full_fine"
    arm["as_mmd"] = {"lambda0": 0.0, "input": "source_stream"}
    with pytest.raises(ValueError, match="input must be null/absent when lambda0 is 0"):
        _load(tmp_path, record)

    arm["as_mmd"] = {"lambda0": 0.0, "source_logits": "offline/logits.npz"}
    with pytest.raises(ValueError, match="source_logits must be null/absent when lambda0 is 0"):
        _load(tmp_path, record)

    arm["as_mmd"] = {"lambda0": 0.0}
    parsed = _load(tmp_path, record).evaluation_arms[1]
    assert parsed.as_mmd_lambda0 == 0.0
    assert parsed.as_mmd_input is None
    assert parsed.as_mmd_source_logits is None


def test_as_mmd_source_stream_requires_full_fine(tmp_path: Path) -> None:
    record = _plan_record()
    record["evaluation_arms"][1]["as_mmd"] = {"lambda0": 0.1}
    with pytest.raises(ValueError, match="requires head='full_fine'"):
        _load(tmp_path, record)


def test_as_mmd_splitbn_target_buffer_source_must_match_training(
    tmp_path: Path,
) -> None:
    record = _plan_record()
    record["training"]["splitbn_target_buffer_source"] = "pooled_source"
    arm = record["evaluation_arms"][1]
    arm["head"] = "full_fine"
    arm["as_mmd"] = {
        "lambda0": 0.1,
        "splitbn_target_buffer_source": "target_calibration",
    }
    with pytest.raises(ValueError, match="must equal training.splitbn_target_buffer_source"):
        _load(tmp_path, record)

    arm["as_mmd"]["splitbn_target_buffer_source"] = "pooled_source"
    plan = _load(tmp_path, record)
    assert plan.training.splitbn_target_buffer_source == "pooled_source"


def test_build_dag_forwards_as_mmd_and_source_selection_to_result_argv(
    tmp_path: Path,
) -> None:
    record = _plan_record()
    source_arm = record["source_arms"][0]
    source_arm["source_selection_record"] = "selection/source-a.json"
    evaluation_arm = record["evaluation_arms"][1]
    evaluation_arm["head"] = "full_fine"
    evaluation_arm["as_mmd"] = {
        "lambda0": 0.25,
        "warmup_epochs": 5,
        "target_weight_min": 2.0,
        "target_weight_max": 8.0,
    }

    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    batch_task = dag.task_by_id["result_batch/source_a/rep01/p01"]
    jobs = dag.result_batch_job_records[batch_task.task_id]["jobs"]
    argv = next(
        list(job["argv"]) for job in jobs if job["arm_name"] == "source_a__linear_time_split"
    )

    assert _argument(argv, "--as-mmd-lambda0") == "0.25"
    assert _argument(argv, "--as-mmd-warmup-epochs") == "5"
    assert _argument(argv, "--as-mmd-target-weight-min") == "2.0"
    assert _argument(argv, "--as-mmd-target-weight-max") == "8.0"
    assert _argument(argv, "--source-stream-cache") == str(plan.source_arms[0].cache)
    assert _argument(argv, "--source-selection-record") == str(
        plan.source_arms[0].source_selection_record
    )
    assert argv.index("--as-mmd-lambda0") < argv.index("--source-stream-cache")
    assert argv.index("--source-stream-cache") < argv.index("--source-selection-record")
    assert argv.index("--source-selection-record") < argv.index("--seed")

    zero_shot_argv = next(
        list(job["argv"]) for job in jobs if job["arm_name"] == "source_a__zero_shot"
    )
    assert not {
        "--as-mmd-lambda0",
        "--as-mmd-warmup-epochs",
        "--as-mmd-target-weight-min",
        "--as-mmd-target-weight-max",
        "--source-stream-cache",
        "--as-mmd-source-logits",
    } & set(zero_shot_argv)
    assert "--source-selection-record" in zero_shot_argv


def test_build_dag_forwards_frozen_logits_as_mmd_source_logits(tmp_path: Path) -> None:
    record = _plan_record()
    arm = record["evaluation_arms"][1]
    arm["as_mmd"] = {
        "lambda0": 0.5,
        "input": "frozen_logits",
        "source_logits": "offline/logits.npz",
    }

    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    batch_task = dag.task_by_id["result_batch/source_a/rep01/p01"]
    argv = next(
        list(job["argv"])
        for job in dag.result_batch_job_records[batch_task.task_id]["jobs"]
        if job["arm_name"] == "source_a__linear_time_split"
    )
    assert "--source-stream-cache" not in argv
    assert _argument(argv, "--as-mmd-source-logits") == str(
        plan.evaluation_arms[1].as_mmd_source_logits
    )


def _selected_arm_record(initial_checkpoint_arm: str | None = "source_a") -> dict[str, Any]:
    record = _plan_record()
    record["source_arms"][1] = {
        "name": "source_sel",
        "cache": "source-a.npz",
        "source_domain_mass": None,
        "risk_within_domain_unit": None,
        "selection_domain": None,
        "splitbn_source_weights": None,
        "euclidean_alignment": {"mode": "none"},
        "sampler_kind": "target_similarity_diverse",
        "source_selection_record": "selection-does-not-need-to-exist.json",
        "initial_checkpoint_arm": initial_checkpoint_arm,
    }
    record["statistical_design"]["planned_contrasts"] = [
        ["source_a__zero_shot", "source_sel__zero_shot"],
        ["source_a__linear_time_split", "source_sel__linear_time_split"],
    ]
    return record


def test_selection_arm_declares_uniform_pretrain_checkpoint_dependency(tmp_path: Path) -> None:
    plan = _load(tmp_path, _selected_arm_record())
    assert plan.source_arms[1].initial_checkpoint_arm == "source_a"
    dag = matrix.build_dag(plan)
    selected = next(
        task for task in dag.tasks if task.task_id.startswith("checkpoint_batch/source_sel/")
    )
    initial = next(
        task for task in dag.tasks if task.task_id.startswith("checkpoint_batch/source_a/")
    )
    assert selected.dependencies == (initial.task_id,)
    assert "--initial-checkpoint" in selected.argv
    assert "--source-selection-record" in selected.argv


def test_selection_initial_checkpoint_requires_earlier_uniform_same_cache_arm(
    tmp_path: Path,
) -> None:
    record = _selected_arm_record()
    record["source_arms"][1]["initial_checkpoint_arm"] = "source_sel"
    with pytest.raises(ValueError, match="earlier source arm"):
        _load(tmp_path, record)

    record = _selected_arm_record()
    record["source_arms"][1]["initial_checkpoint_arm"] = "missing"
    with pytest.raises(ValueError, match="unknown arm"):
        _load(tmp_path, record)

    record = _selected_arm_record()
    record["source_arms"][1]["cache"] = "source-ab.npz"
    with pytest.raises(ValueError, match="share its source cache"):
        _load(tmp_path, record)


def test_plan_rejects_outer_qc_with_inner_epoch_selection(tmp_path: Path) -> None:
    record = _plan_record()
    record["evaluation_arms"][1]["fold_local_qc"] = True
    with pytest.raises(ValueError, match="separate inner-training fit scopes"):
        _load(tmp_path, record)
