from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baselines.n2p3net import N2P3NetBaseline as ContractN2P3NetBaseline
from data.contract import SOURCE_COHORT_DATA_CONTRACTS
from data.epochs import preprocessing_spec_from_contract
from data.events import observed_only_timeline
from data.identity import (
    DatasetIdentityTable,
    training_identity_ledger_from_rows,
)
from experiments.run_pretrain_supervised import (
    _SOURCE_DATASET_CACHE,
    _SOURCE_SNAPSHOT_CACHE,
    SOURCE_PRETRAIN_BATCH_SCHEMA,
    parse_source_domain_mass,
)
from experiments.run_pretrain_supervised import (
    main as run_pretrain_supervised_main,
)
from models.adapters import FeatureDomainTransfer
from models.n2p3net import N2P3Net
from research.contracts import training_procedure_record
from train.source_sampling import PDSSelection, array_digest
from transfer.checkpoint import (
    checkpoint_architecture_record,
    checkpoint_training_contract,
    checkpoint_training_identity_ledger,
    load_n2p3_trunk_checkpoint,
)


def _write_source_snapshot_manifest(
    directory: Path,
) -> tuple[Path, str, Path]:
    source = directory / "snapshot_source.py"
    source.write_text("SNAPSHOT_VALUE = 1\n", encoding="utf-8")
    archive = directory / "source_snapshot.tar.gz"
    with tarfile.open(archive, mode="w:gz") as stream:
        stream.add(source, arcname="snapshot_source.py")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = directory / "source_snapshot.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "n2p3_source_freeze/1",
                "archive": archive.name,
                "archive_sha256": digest,
                "source_commit": "a" * 40,
                "member_count": 1,
                "byte_size": archive.stat().st_size,
            }
        ),
        encoding="utf-8",
    )
    return manifest, digest, archive


def _supervised_source_dataset() -> SimpleNamespace:
    contract = SOURCE_COHORT_DATA_CONTRACTS["causal"]
    subjects = np.repeat(np.asarray(["s1", "s2", "s3", "s4", "s5", "s6"]), 250)
    identity = DatasetIdentityTable.from_source_rows(
        subjects.tolist(),
        ["synthetic-source"] * len(subjects),
    )
    return SimpleNamespace(
        X=np.zeros((len(subjects), 3, contract.n_times), dtype=np.float32),
        y=np.tile(np.asarray([0, 1], dtype=np.int64), len(subjects) // 2),
        subject_ids=subjects,
        identity_table=identity,
        n_epochs=len(subjects),
        n_channels=3,
        n_times=contract.n_times,
        preprocessing=preprocessing_spec_from_contract(contract),
        channel_names=("Fz", "Cz", "Pz"),
        provenance={"source_reference": "synthetic-source"},
        name="synthetic_supervised_source",
    )


def _direct_supervised_source_dataset() -> SimpleNamespace:
    contract = SOURCE_COHORT_DATA_CONTRACTS["causal"]
    decision_count = 14
    candidate_count = 9
    repetitions = 8
    rows_per_decision = candidate_count * repetitions
    n_rows = decision_count * rows_per_decision
    decision_ids = np.repeat(
        np.asarray([f"decision-{index:02d}" for index in range(decision_count)]),
        rows_per_decision,
    )
    subjects = np.repeat(
        np.asarray([f"s{index:02d}" for index in range(decision_count)]),
        rows_per_decision,
    )
    candidate_block = np.repeat(np.arange(candidate_count), repetitions)
    repetition_block = np.tile(np.arange(repetitions), candidate_count)
    candidate_ids = np.tile(candidate_block, decision_count).astype(str)
    repetition_indices = np.tile(repetition_block, decision_count)
    target_by_decision = np.arange(decision_count) % candidate_count
    target_ids = np.repeat(target_by_decision, rows_per_decision).astype(str)
    labels = (candidate_ids == target_ids).astype(np.int64)
    timeline = observed_only_timeline(
        dataset_id="synthetic-direct",
        subject_ids=subjects,
        stimulus_ids=np.arange(n_rows, dtype=np.int64),
        onset_times_s=np.arange(n_rows, dtype=float),
        evidence_available_times_s=np.arange(n_rows, dtype=float) + 0.8,
        group_ids=decision_ids,
        selection_ids=decision_ids,
        candidate_ids=candidate_ids,
        target_candidate_ids=target_ids,
        repetition_indices=repetition_indices,
        online_causal=True,
        timing_source="synthetic_direct_contract",
    )
    timeline = replace(timeline, complete=True).validate(n_epochs=n_rows)
    identity = DatasetIdentityTable.from_source_rows(
        subjects.tolist(),
        ["synthetic-direct"] * n_rows,
    )
    return SimpleNamespace(
        X=np.zeros((n_rows, 3, contract.n_times), dtype=np.float32),
        y=labels,
        subject_ids=subjects,
        identity_table=identity,
        event_timeline=timeline,
        n_epochs=n_rows,
        n_channels=3,
        n_times=contract.n_times,
        preprocessing=preprocessing_spec_from_contract(contract),
        channel_names=("Fz", "Cz", "Pz"),
        provenance={
            "source_reference": "synthetic-source",
            "event_contract": "preserve",
        },
        name="synthetic_direct_supervised_source",
    )


class _ExecutionRecord:
    def record(self) -> dict[str, object]:
        return {"optimizer": "test-double", "compiled": False}


class _FastSupervisedBaseline:
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        sfreq: float,
        *,
        tmin_s: float,
        pooling_mode: str,
        architecture,
        **_,
    ) -> None:
        self.model_ = N2P3Net(
            n_channels=n_channels,
            n_times=n_times,
            sfreq=sfreq,
            tmin_s=tmin_s,
            pooling_mode=pooling_mode,
            **architecture.model_kwargs(),
        )
        self.optimizer_execution = _ExecutionRecord()
        self.training_pos_weight_ = 1.0
        self.training_prior_ = 0.5
        self.calibration_logits_ = None
        self.calibration_labels_ = None
        self.calibration_source_ = None
        self.last_history: dict[str, object] = {}
        self.last_source_risk = None
        self.last_runtime = {
            "device": "cpu",
            "precision": "fp32",
            "batch_size": 16,
            "oom_retries": 0,
        }
        self._input_mean = np.zeros((1, n_channels, 1), dtype=np.float32)
        self._input_std = np.ones((1, n_channels, 1), dtype=np.float32)

    def architecture_record(self) -> dict[str, object]:
        return self.model_.architecture_record()

    def source_domain_axis_record(self) -> None:
        return None

    def fit(self, X, y, *, group_ids, **_) -> None:
        del X, y
        self.last_history = {
            "best_epoch": 0 if group_ids is not None else None,
            "final_task_val_auc": 0.5 if group_ids is not None else None,
        }


def test_parse_source_domain_mass_preserves_exact_domains() -> None:
    assert parse_source_domain_mass(["BI=0.8", "BNCI=0.2"]) == {
        "BI": 0.8,
        "BNCI": 0.2,
    }
    assert parse_source_domain_mass([]) == {}


@pytest.mark.parametrize(
    "value",
    [
        ["BI"],
        ["=0.8", "BNCI=0.2"],
        ["BI=0", "BNCI=1"],
        ["BI=0.8", "BI=0.2"],
        ["BI=0.7", "BNCI=0.2"],
    ],
)
def test_parse_source_domain_mass_rejects_invalid_contracts(value: list[str]) -> None:
    with pytest.raises(ValueError):
        parse_source_domain_mass(value)


def test_training_identity_ledger_uses_only_rows_retained_after_all_gates() -> None:
    subjects = np.asarray(["s1", "s1", "s2", "s2", "s3"])
    table = DatasetIdentityTable.from_source_rows(
        subjects.tolist(),
        ["source"] * len(subjects),
    )
    post_holdout_and_qc = np.asarray([True, False, False, False, True])

    ledger = training_identity_ledger_from_rows(
        table,
        subjects,
        post_holdout_and_qc,
    )

    assert ledger.local_subject_ids == ("s1", "s3")
    assert "s2" not in ledger.local_subject_ids


def test_training_identity_ledger_rejects_index_list_as_ambiguous_mask() -> None:
    subjects = np.asarray(["s1", "s2"])
    table = DatasetIdentityTable.from_source_rows(subjects.tolist(), ["source"] * 2)

    with pytest.raises(ValueError, match="boolean mask"):
        training_identity_ledger_from_rows(table, subjects, [0, 1])


def test_supervised_runner_records_verified_physical_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _supervised_source_dataset()
    source_manifest, source_digest, _ = _write_source_snapshot_manifest(tmp_path)
    checkpoint = tmp_path / "supervised.pt"
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.load_epoch_dataset",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.loaded_epoch_cache_attestation",
        lambda *_args, **_kwargs: {"sha256": "1" * 64},
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.N2P3NetBaseline",
        _FastSupervisedBaseline,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain_supervised.py",
            "--source-cache",
            str(tmp_path / "synthetic.npz"),
            "--source-snapshot-manifest",
            str(source_manifest),
            "--checkpoint",
            str(checkpoint),
            "--epochs",
            "1",
            "--device",
            "cpu",
        ],
    )

    run_pretrain_supervised_main()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = checkpoint_training_contract(payload)
    assert contract.source_snapshot_sha256 == source_digest
    assert payload["source_snapshot_sha256"] == source_digest
    assert payload["source_snapshot_manifest"] == str(source_manifest.resolve())
    assert contract.optimizer["selection_config"]["precision"] == "auto"
    assert contract.optimizer["refit_config"]["precision"] == "auto"
    assert contract.optimizer["selection_runtime"]["precision"] == "fp32"
    assert contract.optimizer["refit_runtime"]["precision"] == "fp32"
    assert payload["runtime"]["selection"]["precision"] == "fp32"
    assert payload["runtime"]["refit"]["precision"] == "fp32"


def test_supervised_runner_binds_pds_source_selection_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _supervised_source_dataset()
    source_manifest, _, _ = _write_source_snapshot_manifest(tmp_path)
    checkpoint = tmp_path / "pds.pt"
    cache_sha256 = "1" * 64
    selected_indices = np.arange(250, len(dataset.X), dtype=np.int64)
    selection = PDSSelection(
        selected_indices=selected_indices,
        source_count=len(dataset.X),
        target_count=64,
        selection_scope="target_calibration",
        seed=7,
        target_temperature=1.0,
        diversity_weight=1.0,
        bandwidth=1.0,
        source_digest="2" * 64,
        target_digest="3" * 64,
        selected_digest=array_digest(selected_indices),
    )
    selection_record = selection.record()
    selection_record["source_cache_sha256"] = cache_sha256
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection_record), encoding="utf-8")

    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.load_epoch_dataset",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.loaded_epoch_cache_attestation",
        lambda *_args, **_kwargs: {"sha256": cache_sha256},
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.N2P3NetBaseline",
        _FastSupervisedBaseline,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain_supervised.py",
            "--source-cache",
            str(tmp_path / "synthetic.npz"),
            "--source-snapshot-manifest",
            str(source_manifest),
            "--source-selection-record",
            str(selection_path),
            "--checkpoint",
            str(checkpoint),
            "--epochs",
            "1",
            "--device",
            "cpu",
        ],
    )

    run_pretrain_supervised_main()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = checkpoint_training_contract(payload)
    assert payload["n_source_epochs_used"] == len(dataset.X) - 250
    assert payload["pds_dropped_source_epochs"] == 250
    assert payload["pds_dropped_source_epochs_by_label"] == [125, 125]
    assert payload["source_selection_contract"] == selection_record
    assert contract.objective["source_selection_contract"] == selection_record
    procedure = training_procedure_record(contract)
    expected_procedure_selection = {
        key: value for key, value in selection_record.items() if key != "seed"
    }
    assert (
        procedure["objective"]["source_selection_contract"]
        == expected_procedure_selection
    )
    ledger = checkpoint_training_identity_ledger(payload)
    assert ledger.local_subject_ids == ("s2", "s3", "s4", "s5", "s6")


def test_supervised_runner_rejects_selection_from_another_source_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _supervised_source_dataset()
    source_manifest, _, _ = _write_source_snapshot_manifest(tmp_path)
    checkpoint = tmp_path / "wrong-selection.pt"
    selected_indices = np.arange(0, len(dataset.X), dtype=np.int64)
    selection = PDSSelection(
        selected_indices=selected_indices,
        source_count=len(dataset.X),
        target_count=2,
        selection_scope="target_calibration",
        seed=0,
        target_temperature=1.0,
        diversity_weight=1.0,
        bandwidth=1.0,
        source_digest="2" * 64,
        target_digest="3" * 64,
        selected_digest=array_digest(selected_indices),
    )
    selection_record = selection.record()
    selection_record["source_cache_sha256"] = "f" * 64
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection_record), encoding="utf-8")

    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.load_epoch_dataset",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.loaded_epoch_cache_attestation",
        lambda *_args, **_kwargs: {"sha256": "1" * 64},
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.N2P3NetBaseline",
        _FastSupervisedBaseline,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain_supervised.py",
            "--source-cache",
            str(tmp_path / "synthetic.npz"),
            "--source-snapshot-manifest",
            str(source_manifest),
            "--source-selection-record",
            str(selection_path),
            "--checkpoint",
            str(checkpoint),
            "--epochs",
            "1",
            "--device",
            "cpu",
        ],
    )

    with pytest.raises(ValueError, match="source_cache_sha256"):
        run_pretrain_supervised_main()


def test_supervised_batch_loads_source_once_and_attests_each_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _supervised_source_dataset()
    source_manifest, source_digest, _ = _write_source_snapshot_manifest(tmp_path)
    checkpoints = [tmp_path / "first.pt", tmp_path / "second.pt"]
    batch_record = tmp_path / "batch.json"
    load_calls = 0

    def load_once(*_args, **_kwargs):
        nonlocal load_calls
        load_calls += 1
        return dataset

    _SOURCE_DATASET_CACHE.clear()
    _SOURCE_SNAPSHOT_CACHE.clear()
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.load_epoch_dataset",
        load_once,
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.loaded_epoch_cache_attestation",
        lambda *_args, **_kwargs: {"sha256": "1" * 64},
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.N2P3NetBaseline",
        _FastSupervisedBaseline,
    )

    run_pretrain_supervised_main(
        [
            "--source-cache",
            str(tmp_path / "synthetic.npz"),
            "--source-snapshot-manifest",
            str(source_manifest),
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--job",
            "",
            str(checkpoints[0]),
            "--job",
            "",
            str(checkpoints[1]),
            "--batch-record",
            str(batch_record),
        ]
    )

    assert load_calls == 1
    record = json.loads(batch_record.read_text(encoding="utf-8"))
    assert record["schema"] == SOURCE_PRETRAIN_BATCH_SCHEMA
    assert record["completed"] is True
    assert record["job_count"] == 2
    assert record["source_snapshot_sha256"] == source_digest
    assert [Path(job["checkpoint"]) for job in record["jobs"]] == [
        path.resolve() for path in checkpoints
    ]
    for job, checkpoint in zip(record["jobs"], checkpoints, strict=True):
        assert job["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert job["training_contract_digest"] == checkpoint_training_contract(
            torch.load(checkpoint, map_location="cpu", weights_only=False)
        ).digest()

    from experiments import run_candidate_promotion_matrix as matrix

    task = matrix.MatrixTask(
        task_id="checkpoint_batch/source/seed",
        kind="checkpoint_batch",
        argv=(
            "python",
            "run_pretrain_supervised.py",
            "--job",
            "",
            str(checkpoints[0]),
            "--job",
            "",
            str(checkpoints[1]),
            "--batch-record",
            str(batch_record),
        ),
        output=batch_record,
        dependencies=(),
    )
    embedded = matrix._embedded_artifact_record(task)
    assert embedded["schema"] == SOURCE_PRETRAIN_BATCH_SCHEMA
    assert embedded["job_count"] == 2

    checkpoints[0].write_bytes(checkpoints[0].read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="bytes changed"):
        matrix._embedded_artifact_record(task)


def test_supervised_runner_round_trips_p0_p1_domain_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _supervised_source_dataset()
    domains = np.repeat(np.asarray(["domain-a", "domain-b"]), len(dataset.X) // 2)
    dataset.provenance = {
        **dataset.provenance,
        "source_domain_axis": {
            "schema": "n2p3_source_domain_axis/1",
            "column": "source_domain",
            "domains": ["domain-b", "domain-a"],
        },
    }
    source_manifest, _, _ = _write_source_snapshot_manifest(tmp_path)
    checkpoint = tmp_path / "p0-p1.pt"
    init_records: list[dict[str, object]] = []
    fit_records: list[dict[str, object]] = []

    class ContractOnlyBaseline(_FastSupervisedBaseline):
        def __init__(self, n_channels, n_times, sfreq, **kwargs) -> None:
            super().__init__(n_channels, n_times, sfreq, **kwargs)
            contract = ContractN2P3NetBaseline(
                n_channels,
                n_times,
                sfreq,
                **kwargs,
            )
            self.model_ = contract._make_model()
            self._architecture_record = contract.architecture_record()
            self._axis_record = contract.source_domain_axis_record()
            init_records.append(dict(kwargs))

        def architecture_record(self) -> dict[str, object]:
            return dict(self._architecture_record)

        def source_domain_axis_record(self) -> dict[str, object] | None:
            return None if self._axis_record is None else dict(self._axis_record)

        def fit(self, X, y, *, group_ids, **kwargs) -> None:
            del X, y
            fit_records.append({"group_ids": group_ids, **kwargs})
            residual = {
                "schema": "n2p3_source_residual_ledger/1",
                "active": True,
                "status": "valid",
                "valid": True,
                "epoch_trial_loss_m0": [0.7],
                "epoch_trial_loss_m1": [0.6],
                "run_max_ratio": 0.1,
            }
            self.last_history = {
                "best_epoch": 0 if group_ids is not None else None,
                "final_task_val_auc": 0.5 if group_ids is not None else None,
                "source_residual": residual,
            }

    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.load_epoch_dataset",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.loaded_epoch_cache_attestation",
        lambda *_args, **_kwargs: {"sha256": "1" * 64},
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.source_domain_vocabulary",
        lambda _dataset: ("domain-b", "domain-a"),
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.source_domain_ids",
        lambda _dataset: domains,
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.N2P3NetBaseline",
        ContractOnlyBaseline,
    )

    run_pretrain_supervised_main(
        [
            "--source-cache",
            str(tmp_path / "synthetic.npz"),
            "--source-snapshot-manifest",
            str(source_manifest),
            "--checkpoint",
            str(checkpoint),
            "--epochs",
            "1",
            "--feature-transfer-invariant-bottleneck",
            "2",
            "--feature-alignment-weight",
            "0.05",
            "--private-domain-weight",
            "0.05",
            "--source-residual-drop-p",
            "0.3",
            "--source-residual-drop-mode",
            "sample",
            "--splitbn-target-buffer-source",
            "pooled_source",
            "--device",
            "cpu",
        ]
    )

    assert len(init_records) == 2
    assert all(
        record["source_domain_vocabulary"] == ("domain-b", "domain-a")
        for record in init_records
    )
    assert all(record["splitbn_domain_count"] == 2 for record in init_records)
    assert len(fit_records) == 2
    assert all(np.array_equal(record["source_domain_ids"], domains) for record in fit_records)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint_architecture_record(payload)
    assert architecture["normalization"]["mode"] == "splitbn"
    assert architecture["feature_transfer"]["source_domain_count"] == 2
    assert payload["source_domain_axis"]["vocabulary"] == [
        "domain-b",
        "domain-a",
    ]
    contract = checkpoint_training_contract(payload)
    assert contract.objective["source_domain_axis"] == payload["source_domain_axis"]
    assert contract.objective["source_residual_diagnostics"]["refit"] == payload[
        "source_residual_refit"
    ]
    restored, _ = load_n2p3_trunk_checkpoint(payload, dataset)
    assert isinstance(restored, FeatureDomainTransfer)

    forged = dict(payload)
    forged_refit = dict(forged["source_residual_refit"])
    forged_refit.update({"status": "evaluation_failed", "valid": False})
    forged["source_residual_refit"] = forged_refit
    forged_contract = replace(
        checkpoint_training_contract(forged),
        objective={
            **checkpoint_training_contract(forged).objective,
            "source_residual_diagnostics": {
                "selection": forged["source_residual_selection"],
                "refit": forged_refit,
            },
        },
    )
    forged["training_contract"] = forged_contract.record()
    forged["training_contract_digest"] = forged_contract.digest()
    with pytest.raises(ValueError, match="source residual ledger is not a valid"):
        load_n2p3_trunk_checkpoint(forged, dataset)


def test_supervised_runner_round_trips_direct_decision_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _direct_supervised_source_dataset()
    source_manifest, _, _ = _write_source_snapshot_manifest(tmp_path)
    checkpoint = tmp_path / "p2.pt"
    fit_records: list[dict[str, object]] = []

    class DecisionContractBaseline(_FastSupervisedBaseline):
        def fit(self, X, y, *, group_ids, **kwargs) -> None:
            metadata = kwargs["decision_metadata"]
            config = kwargs["decision_loss_config"]
            assert metadata.n_rows == len(X) == len(y)
            fit_records.append(
                {
                    "group_ids": group_ids,
                    "metadata": metadata,
                    "config": config,
                }
            )
            ledger = {
                "schema": "n2p3_decision_training_ledger/1",
                "status": "valid",
                "valid": True,
                "config": config.record(),
                "metadata": metadata.record(include_rows=False),
                "fit_completed": True,
            }
            self.last_history = {
                "best_epoch": 0 if group_ids is not None else None,
                "final_task_val_auc": 0.5 if group_ids is not None else None,
                "decision_loss": ledger,
            }

    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.load_epoch_dataset",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.loaded_epoch_cache_attestation",
        lambda *_args, **_kwargs: {"sha256": "1" * 64},
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.N2P3NetBaseline",
        DecisionContractBaseline,
    )

    run_pretrain_supervised_main(
        [
            "--source-cache",
            str(tmp_path / "synthetic.npz"),
            "--source-snapshot-manifest",
            str(source_manifest),
            "--checkpoint",
            str(checkpoint),
            "--epochs",
            "1",
            "--candidate-loss-weight",
            "0.2",
            "--candidate-loss-temperature",
            "0.7",
            "--candidate-loss-prefix-mode",
            "uniform",
            "--candidate-loss-max-repetitions",
            "8",
            "--device",
            "cpu",
        ]
    )

    assert len(fit_records) == 2
    assert all(record["metadata"].n_rows == 1008 for record in fit_records)
    assert all(record["config"].lambda_candidate == 0.2 for record in fit_records)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = checkpoint_training_contract(payload)
    assert contract.objective["decision_loss"]["temperature"] == 0.7
    assert contract.objective["decision_loss"]["prefix_weights"] == [0.125] * 8
    assert len(payload["decision_metadata"]["requested_decision_ids"]) == 14
    restored, _ = load_n2p3_trunk_checkpoint(payload, dataset)
    assert isinstance(restored, N2P3Net)

    unattested = dict(payload)
    forged_ledger = {
        "schema": "n2p3_source_residual_ledger/1",
        "active": True,
        "status": "valid",
        "valid": True,
        "run_max_ratio": 0.1,
    }
    unattested["source_residual_selection"] = forged_ledger
    unattested["source_residual_refit"] = forged_ledger
    unattested_contract = replace(
        checkpoint_training_contract(unattested),
        objective={
            **checkpoint_training_contract(unattested).objective,
            "source_residual_diagnostics": {
                "selection": forged_ledger,
                "refit": forged_ledger,
            },
        },
    )
    unattested["training_contract"] = unattested_contract.record()
    unattested["training_contract_digest"] = unattested_contract.digest()
    with pytest.raises(ValueError, match="active but architecture disables"):
        load_n2p3_trunk_checkpoint(unattested, dataset)


def test_supervised_runner_rejects_tampered_source_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest, _, archive = _write_source_snapshot_manifest(tmp_path)
    tampered = bytearray(archive.read_bytes())
    tampered[4] ^= 1
    archive.write_bytes(tampered)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain_supervised.py",
            "--source-cache",
            str(tmp_path / "unused.npz"),
            "--source-snapshot-manifest",
            str(source_manifest),
            "--checkpoint",
            str(tmp_path / "must-not-exist.pt"),
        ],
    )

    with pytest.raises(ValueError, match="archive hash disagrees"):
        run_pretrain_supervised_main()


def test_supervised_runner_rejects_legacy_naked_snapshot_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain_supervised.py",
            "--source-cache",
            str(tmp_path / "unused.npz"),
            "--source-snapshot-sha256",
            "a" * 64,
            "--checkpoint",
            str(tmp_path / "unused.pt"),
        ],
    )

    with pytest.raises(SystemExit):
        run_pretrain_supervised_main()


@pytest.mark.parametrize(
    "obsolete_option",
    ["--subject-prefix-repeat", "--input-stats-subject-prefix"],
)
def test_supervised_runner_rejects_obsolete_prefix_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    obsolete_option: str,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain_supervised.py",
            "--source-cache",
            str(tmp_path / "unused.npz"),
            "--source-snapshot-manifest",
            str(tmp_path / "unused.json"),
            "--checkpoint",
            str(tmp_path / "unused.pt"),
            obsolete_option,
            "BI=3",
        ],
    )

    with pytest.raises(SystemExit):
        run_pretrain_supervised_main()
