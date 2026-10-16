"""Plan and physical input builders shared by promotion tests."""

from __future__ import annotations

import hashlib
import tarfile
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.channel import build_channel_identity
from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.domain import SOURCE_DOMAIN_AXIS_SCHEMA, SOURCE_DOMAIN_COLUMN
from data.epochs import EpochDataset, preprocessing_spec_from_contract, save_epoch_dataset
from data.events import observed_only_timeline
from data.identity import DatasetIdentityTable, ParticipantIdentityRecord, origin_subject_key
from experiments import run_candidate_promotion_matrix as matrix
from tests.support.io import write_json


def _argument(argv, flag):
    return next(value for option, value in pairwise(argv) if option == flag)


def _plan_record(*, output_root: str = "outputs") -> dict[str, Any]:
    return {
        "schema": matrix.PLAN_SCHEMA,
        "stage": "development",
        "source_snapshot_manifest": "source-freeze.manifest.json",
        "target_cache": "target.npz",
        "target_subjects": [
            {
                "target_subject": "TARGET::01",
                "source_holdout_subject": "SOURCE::01",
                "partition_key": "p01",
            },
            {
                "target_subject": "TARGET::02",
                "source_holdout_subject": "SOURCE::02",
                "partition_key": "p02",
            },
        ],
        "source_arms": [
            {
                "name": "source_a",
                "cache": "source-a.npz",
                "source_domain_mass": None,
                "risk_within_domain_unit": None,
                "selection_domain": None,
            },
            {
                "name": "source_ab",
                "cache": "source-ab.npz",
                "source_domain_mass": {"A": 0.8, "B": 0.2},
                "risk_within_domain_unit": "participant",
                "selection_domain": "A",
            },
        ],
        "training_replicates": [
            {"key": "rep01", "seed": 101},
            {"key": "rep02", "seed": 202},
        ],
        "evaluation_arms": [
            {
                "name": "zero_shot",
                "head": "zero_shot",
                "normalization": "source",
                "epoch_selection": "fixed_budget",
                "epochs": None,
                "batch_size": None,
                "lr": None,
                "target_stat_weight": 0.0,
                "fold_local_qc": False,
            },
            {
                "name": "linear_time_split",
                "head": "linear",
                "normalization": "source",
                "epoch_selection": "target_time_split",
                "epochs": 30,
                "batch_size": 64,
                "lr": 1e-3,
                "target_stat_weight": 0.0,
                "fold_local_qc": False,
            },
        ],
        "training": {
            "pooling_mode": "full_unfold",
            "temporal_kernel_size": 35,
            "epochs": 2,
            "batch_size": 16,
            "precision": "auto",
        },
        "calibration_selections": 5,
        "test_reps": 8,
        "identity_exclusion_policy": "source_or_global",
        "output_root": output_root,
        "statistical_design": {
            "inference_scope": "conditional_frozen_models",
            "planned_contrasts": [
                ["source_a__zero_shot", "source_ab__zero_shot"],
                ["source_a__linear_time_split", "source_ab__linear_time_split"],
            ],
            "evidence_level": 8,
            "bootstrap_iterations": 100,
            "bootstrap_seed": 303,
            "confidence_level": 0.95,
        },
    }


def _single_multidomain_arm_record() -> dict[str, Any]:
    record = _plan_record()
    record["source_arms"] = [record["source_arms"][1]]
    record["statistical_design"]["planned_contrasts"] = [
        ["source_ab__linear_time_split", "source_ab__zero_shot"]
    ]
    return record


def _write_plan(tmp_path: Path, record: dict[str, Any]) -> Path:
    path = tmp_path / "plan.json"
    write_json(path, record)
    return path


def _load(tmp_path: Path, record: dict[str, Any] | None = None) -> matrix.PromotionPlan:
    return matrix.load_plan(_write_plan(tmp_path, record or _plan_record()))


def _cache_dataset(
    *,
    name: str,
    local_subjects: tuple[str, ...],
    authority_subjects: tuple[str, ...],
    position_offset: float = 0.0,
    source_domains: tuple[str, ...] | None = None,
    data_alignment: dict[str, Any] | None = None,
) -> EpochDataset:
    identity = build_channel_identity(("Fz", "Cz", "Pz"), allow_missing_positions=False)
    positions = np.asarray(identity.coords, dtype=float).copy()
    positions[0, 0] += position_offset
    subjects = np.asarray(local_subjects)
    timeline = observed_only_timeline(
        dataset_id=name,
        subject_ids=subjects,
        stimulus_ids=np.arange(len(subjects), dtype=np.int64),
        onset_times_s=np.arange(len(subjects), dtype=float),
        evidence_available_times_s=np.arange(len(subjects), dtype=float) + 0.8,
        group_ids=subjects,
        online_causal=True,
        timing_source="promotion_preflight_fixture",
    )
    return EpochDataset(
        name=name,
        X=np.zeros(
            (
                len(subjects),
                len(identity.names),
                SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.n_times,
            ),
            dtype=np.float32,
        ),
        y=np.zeros(len(subjects), dtype=np.int64),
        subject_ids=subjects,
        channel_names=identity.names,
        channel_positions_m=positions,
        channel_mask=np.ones(len(identity.names), dtype=bool),
        preprocessing=preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT),
        event_timeline=timeline,
        metadata=(
            pd.DataFrame(
                {
                    "subject": subjects,
                    SOURCE_DOMAIN_COLUMN: np.asarray(source_domains),
                }
            )
            if source_domains is not None
            else pd.DataFrame({"subject": subjects})
        ),
        provenance={
            "source": "unit_test",
            "source_reference": "average",
            **(
                {
                    "source_domain_axis": {
                        "schema": SOURCE_DOMAIN_AXIS_SCHEMA,
                        "column": SOURCE_DOMAIN_COLUMN,
                        "domains": sorted(set(source_domains)),
                    }
                }
                if source_domains is not None
                else {}
            ),
            **({"data_alignment": data_alignment} if data_alignment is not None else {}),
        },
        identity_table=DatasetIdentityTable(
            tuple(
                ParticipantIdentityRecord(
                    local_subject_id=local,
                    origin_subject_keys=(origin_subject_key("shared", authority),),
                    identity_status="source_verified",
                )
                for local, authority in zip(local_subjects, authority_subjects, strict=True)
            )
        ),
    )


def _write_preflight_inputs(tmp_path: Path) -> None:
    source_member = tmp_path / "source.txt"
    source_member.write_text("source snapshot", encoding="utf-8")
    source_archive = tmp_path / "source.tar.gz"
    with tarfile.open(source_archive, "w:gz") as archive:
        archive.add(source_member, arcname="source.txt")
    write_json(
        tmp_path / "source-freeze.manifest.json",
        {
            "schema": "n2p3_source_freeze/1",
            "archive": source_archive.name,
            "archive_sha256": hashlib.sha256(source_archive.read_bytes()).hexdigest(),
            "source_commit": "a" * 40,
            "member_count": 1,
            "byte_size": source_archive.stat().st_size,
        },
    )
    target_subjects = ("TARGET::01", "TARGET::02")
    source_subjects = ("SOURCE::01", "SOURCE::02")
    authority_subjects = ("01", "02")
    save_epoch_dataset(
        tmp_path / "target.npz",
        _cache_dataset(
            name="target",
            local_subjects=target_subjects,
            authority_subjects=authority_subjects,
        ),
    )
    for cache_name, domains in (("source-a.npz", None), ("source-ab.npz", ("A", "B"))):
        save_epoch_dataset(
            tmp_path / cache_name,
            _cache_dataset(
                name=cache_name,
                local_subjects=source_subjects,
                authority_subjects=authority_subjects,
                source_domains=domains,
            ),
        )
