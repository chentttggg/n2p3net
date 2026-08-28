from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import run_receptive_field_comparison as runner
from models.n2p3net import N2P3ArchitectureConfig


def _architecture(**overrides) -> N2P3ArchitectureConfig:
    values = {
        "temporal_kernel_size": 65,
        "st_temporal_dilation": 1,
        "mst_kernel_sizes": (5, 17),
        "mst_dilations": (1, 1),
    }
    values.update(overrides)
    return N2P3ArchitectureConfig(**values)


def _architectures() -> dict[str, N2P3ArchitectureConfig]:
    return {
        "A": _architecture(),
        "B": _architecture(temporal_kernel_size=35),
        "C": _architecture(temporal_kernel_size=33),
        "D": _architecture(temporal_kernel_size=33, st_temporal_dilation=2),
        "E": _architecture(temporal_kernel_size=33, mst_kernel_sizes=(13, 25)),
    }


def _fold(loss: float, auc: float, *, fold: int = 0) -> dict[str, object]:
    return {
        "anchor_fold_index": fold,
        "anchor_subject": str(fold + 1),
        "best_task_val_loss": loss,
        "final_task_val_auc": auc,
        "epochs_ran": 2,
        "batch_size": 32,
        "fused_adam_requested": False,
        "compile_mode_requested": None,
        "fused_adam": False,
        "compile_mode": None,
        "compile_scope": None,
        "optimizer_fallback_reason": None,
        "oom_retries": 0,
    }


def _arm_record(arm: str, loss: float, auc: float) -> dict[str, object]:
    seed_records = []
    for seed in (11, 12):
        seed_records.append(
            {
                "seed": seed,
                "per_fold": [_fold(loss, auc, fold=0), _fold(loss + 0.02, auc - 0.01, fold=1)],
            }
        )
    return {
        "arm": arm,
        "mean_inner_best_task_val_loss": loss + 0.01,
        "mean_inner_final_task_val_auc": auc - 0.005,
        "per_seed": seed_records,
    }


def test_fixed_architecture_registry_accepts_only_the_preregistered_a_to_e_geometry() -> None:
    architectures = runner._validate_rf_architectures(_architectures())

    assert tuple(architectures) == runner.ARM_ORDER
    assert architectures["D"].st_temporal_dilation == 2
    assert architectures["D"].mst_dilations == (1, 1)


def test_fixed_architecture_registry_rejects_missing_or_drifted_arms() -> None:
    missing = _architectures()
    missing.pop("E")
    with pytest.raises(RuntimeError, match="exactly the fixed arms"):
        runner._validate_rf_architectures(missing)

    drifted = _architectures()
    drifted["D"] = _architecture(temporal_kernel_size=33, st_temporal_dilation=1)
    with pytest.raises(RuntimeError, match="arm D geometry drifted"):
        runner._validate_rf_architectures(drifted)


def test_missing_model_registry_fails_with_an_actionable_message(monkeypatch) -> None:
    monkeypatch.setattr(runner.importlib, "import_module", lambda _: SimpleNamespace())

    with pytest.raises(RuntimeError, match="source revision that provides the fixed A-E"):
        runner._load_rf_architectures()


def test_integer_list_parsers_require_unique_nonnegative_values() -> None:
    assert runner._parse_fold_indices("0, 7,63") == (0, 7, 63)
    assert runner._parse_seeds("20260828,20260829") == (20260828, 20260829)
    with pytest.raises(argparse.ArgumentTypeError, match="unique non-negative"):
        runner._parse_seeds("1,1")
    with pytest.raises(argparse.ArgumentTypeError, match="unique non-negative"):
        runner._parse_fold_indices("-1")


def test_cpu_qc_defaults_use_runtime_resource_detection() -> None:
    args = runner._build_parser(runner.DeepConfig()).parse_args(
        ["--dataset-cache", "dataset.npz"]
    )

    assert args.cpu_threads is None
    assert args.artifact_qc_jobs is None


def test_run_name_must_remain_below_the_run_directory() -> None:
    assert runner._validate_run_name("rf/seed_1") == "rf/seed_1"
    with pytest.raises(argparse.ArgumentTypeError, match="safe relative"):
        runner._validate_run_name("../outside")
    with pytest.raises(argparse.ArgumentTypeError, match="safe relative"):
        runner._validate_run_name("/absolute")


def test_source_commit_prefers_explicit_environment() -> None:
    assert runner._resolve_source_commit(environ={"SOURCE_COMMIT": "archive-b0d58f9"}) == (
        "archive-b0d58f9",
        "SOURCE_COMMIT",
    )


def test_run_fingerprint_is_canonical_and_protocol_sensitive() -> None:
    first = runner._fingerprint({"folds": [0, 1], "seed": 7})
    reordered = runner._fingerprint({"seed": 7, "folds": [0, 1]})
    changed = runner._fingerprint({"seed": 8, "folds": [0, 1]})

    assert first == reordered
    assert first != changed


def test_seed_record_whitelists_inner_metrics_and_drops_outer_results() -> None:
    result = SimpleNamespace(
        best_task_val_loss=0.42,
        final_task_val_auc=0.73,
        epochs_ran=2,
        batch_size=32,
        fused_adam_requested=True,
        compile_mode_requested="reduce-overhead",
        fused_adam=True,
        compile_mode="reduce-overhead",
        compile_scope="train_step",
        optimizer_fallback_reason=None,
        oom_retries=0,
        auc=0.999,
        balanced_acc=0.999,
        scores=[999.0],
    )

    record = runner._seed_record(
        arm="A",
        seed=11,
        source_commit="abc123",
        cache_sha256="0" * 64,
        fold_indices=(0,),
        anchor_subjects=("1",),
        model_record={"name": "n2p3net_full_unfold"},
        fold_results=(result,),
        wall_seconds=1.5,
        run_fingerprint="f" * 64,
    )

    assert record["outer_test_metrics_persisted"] is False
    assert record["selection_metric"] == runner.ARM_SELECTION_METRIC
    assert record["mean_inner_best_task_val_loss"] == pytest.approx(0.42)
    assert record["mean_inner_final_task_val_auc"] == pytest.approx(0.73)
    assert not {"auc", "balanced_acc", "scores"} & set(record["per_fold"][0])


def test_resume_rejects_a_seed_record_from_another_run_contract() -> None:
    record = {
        "schema": runner.SEED_RECORD_SCHEMA,
        "arm": "A",
        "seed": 11,
        "source_commit": "abc123",
        "cache_sha256": "0" * 64,
        "run_fingerprint": "old",
        "fold_indices": [0],
        "outer_test_metrics_persisted": False,
    }

    with pytest.raises(RuntimeError, match="run_fingerprint"):
        runner._validate_resumed_seed(
            record,
            arm="A",
            seed=11,
            source_commit="abc123",
            cache_sha256="0" * 64,
            fold_indices=(0,),
            run_fingerprint="new",
        )


def test_screening_ranks_by_loss_and_emits_only_the_three_planned_contrasts() -> None:
    records = {
        "A": _arm_record("A", 0.50, 0.90),
        "B": _arm_record("B", 0.40, 0.60),
        "C": _arm_record("C", 0.45, 0.80),
        "D": _arm_record("D", 0.55, 0.75),
        "E": _arm_record("E", 0.60, 0.70),
    }

    screening = runner._screening_record(records)

    assert [entry["arm"] for entry in screening["ranking"]] == ["B", "C", "A", "D", "E"]
    assert screening["selection_metric"] == runner.SCREENING_SELECTION_METRIC
    assert screening["outer_test_metrics_persisted"] is False
    assert screening["outer_test_metrics_used_for_selection"] is False
    assert [entry["contrast"] for entry in screening["planned_contrasts"]] == [
        "C-D",
        "D-A",
        "E-A",
    ]
    assert all("B" not in entry["contrast"] for entry in screening["planned_contrasts"])
    assert screening["planned_contrasts"][0]["mean_best_task_val_loss_delta"] == pytest.approx(
        -0.10
    )


def test_main_writes_complete_inner_only_comparison_artifacts(monkeypatch, tmp_path) -> None:
    architectures = _architectures()

    class FakeQC:
        def subset(self, _mask):
            return self

    class FakeDataset:
        X = np.zeros((4, 2, 8), dtype=np.float32)
        y = np.asarray([0, 1, 0, 1], dtype=np.int64)
        subject_ids = np.asarray(["1", "1", "2", "2"])
        channel_mask = np.ones(2, dtype=bool)
        trial_channel_mask = None
        qc_features = FakeQC()
        preprocessing = SimpleNamespace(sfreq=128.0)

        @staticmethod
        def record(*, validate=False):
            assert validate is False
            return {"schema": "n2p3net_epoch_dataset/4", "shape": [4, 2, 8]}

    class FakeModel:
        def __init__(self, arm: str) -> None:
            self.arm = arm

        @staticmethod
        def configure_epoch_progress(_path) -> None:
            return None

    def fake_model_for_arm(architecture, *_args, **_kwargs):
        arm = next(name for name, candidate in architectures.items() if candidate is architecture)
        return FakeModel(arm)

    def fake_describe(_name, model):
        architecture = architectures[model.arm]
        return {
            "name": "n2p3net_full_unfold",
            "parameter_count": {"A": 1506, "B": 1266, "C": 1250, "D": 1250, "E": 1506}[
                model.arm
            ],
            "architecture": {
                "st_temporal_kernel_samples": architecture.temporal_kernel_size,
                "st_temporal_dilation": architecture.st_temporal_dilation,
                "mst_total_receptive_field_samples": {
                    "A": [84, 132],
                    "B": [54, 102],
                    "C": [52, 100],
                    "D": [84, 132],
                    "E": [84, 132],
                }[model.arm],
                "mst_total_receptive_span_ms": [1.0, 2.0],
            },
        }

    outer_loud_result = SimpleNamespace(
        best_task_val_loss=0.4,
        final_task_val_auc=0.7,
        epochs_ran=1,
        batch_size=4,
        fused_adam_requested=False,
        compile_mode_requested=None,
        fused_adam=False,
        compile_mode=None,
        compile_scope=None,
        optimizer_fallback_reason=None,
        oom_retries=0,
        auc=0.999,
        balanced_acc=0.999,
        scores=[999.0],
    )

    monkeypatch.setattr(runner, "_load_rf_architectures", lambda: architectures)
    monkeypatch.setattr(runner, "_resolve_source_commit", lambda: ("abc123", "git"))
    monkeypatch.setattr(runner, "load_epoch_dataset", lambda *_args, **_kwargs: FakeDataset())
    monkeypatch.setattr(
        runner,
        "read_epoch_cache_attestation",
        lambda _path: {"sha256": "0" * 64, "byte_size": 1},
    )
    monkeypatch.setattr(runner, "assert_p300_input_contract", lambda *_args: None)
    monkeypatch.setattr(runner, "assert_p300_source_provenance", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "resolve_fold_local_artifact_models",
        lambda *_args, **_kwargs: ({0: object()}, {"enabled": True, "hit": False}),
    )
    monkeypatch.setattr(runner, "_resolve_device", lambda _choice: torch.device("cpu"))
    monkeypatch.setattr(runner, "_model_for_arm", fake_model_for_arm)
    monkeypatch.setattr(runner, "describe_binary_model", fake_describe)
    monkeypatch.setattr(
        runner,
        "evaluate_binary",
        lambda *_args, **_kwargs: SimpleNamespace(per_fold=[outer_loud_result]),
    )

    runner.main(
        [
            "--dataset-cache",
            "fake.npz",
            "--run-dir",
            str(tmp_path),
            "--run-name",
            "rf_test",
            "--subjects",
            "2",
            "--fold-indices",
            "0",
            "--seeds",
            "11",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--early-stop-patience",
            "1",
            "--fold-jobs",
            "1",
            "--cpu-threads",
            "1",
            "--artifact-qc-jobs",
            "1",
            "--compile-mode",
            "none",
            "--no-fused-adam",
            "--device",
            "cpu",
        ]
    )

    root = tmp_path / "rf_test"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    screening = json.loads((root / "screening.json").read_text(encoding="utf-8"))
    assert manifest["source_commit"] == "abc123"
    assert manifest["cache_attestation"]["sha256"] == "0" * 64
    assert manifest["arm_order"] == list(runner.ARM_ORDER)
    assert manifest["fold_indices"] == [0]
    assert manifest["seeds"] == [11]
    assert manifest["outer_test_metrics_persisted"] is False
    assert [entry["contrast"] for entry in screening["planned_contrasts"]] == [
        "C-D",
        "D-A",
        "E-A",
    ]

    for arm in runner.ARM_ORDER:
        seed_record = json.loads(
            (root / arm / "seed_11" / "record.json").read_text(encoding="utf-8")
        )
        arm_record = json.loads((root / arm / "record.json").read_text(encoding="utf-8"))
        assert seed_record["outer_test_metrics_persisted"] is False
        assert arm_record["outer_test_metrics_persisted"] is False
        assert not {"auc", "balanced_acc", "scores"} & set(seed_record["per_fold"][0])
