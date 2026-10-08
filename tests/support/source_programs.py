"""Small source-only inputs and budgets for selection/refit contract tests."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from baselines.deep import DeepConfig
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.domain import annotate_source_domain, namespace_epoch_dataset
from data.epochs import preprocessing_spec_from_contract
from data.events import observed_only_timeline
from data.identity import DatasetIdentityTable
from data.source_records import (
    SOURCE_PREPARATION_SCHEMA,
    file_sha256,
    merge_source_shards,
    prepare_source_record,
    source_preparation_method,
)
from tests.support.datasets import epoch_dataset
from train.early_stopping import GlobalEarlyStoppingPolicy
from transfer.target_experiments import user_fixed_target_batch_rule


def source_early_stopping_data():
    data = epoch_dataset()
    data.subject_ids = np.repeat(["s1", "s2", "s3", "s4", "s5"], 8)
    data.y = np.tile([0, 1], 20)
    data.X = np.random.default_rng(31).normal(size=(40, 3, 100)).astype(np.float32) * 1e-6
    data.metadata = pd.DataFrame({"subject": data.subject_ids, "run": ["1"] * 40})
    data.event_timeline = observed_only_timeline(
        dataset_id="source", subject_ids=data.subject_ids, group_ids=data.subject_ids,
        stimulus_ids=np.arange(40), onset_times_s=np.arange(40) * 2.,
        evidence_available_times_s=np.arange(40) * 2. + 1.2,
        online_causal=False, timing_source="synthetic_schedule",
    )
    data.identity_table = DatasetIdentityTable.from_source_rows(data.subject_ids.tolist(), ["source"] * 40)
    return annotate_source_domain(namespace_epoch_dataset(data, "source"), "source")


def source_early_stopping_config():
    return DeepConfig(global_early_stopping=GlobalEarlyStoppingPolicy(
        patience=2, min_delta=100., source_max_epochs=5, target_max_epochs=5),
        batch_size=16, sampling="uniform_rows", compile_mode=None, val_group_frac=None, seed=7)


def prepared_finite_source(tmp_path):
    """Exercise the actual BDF reader, filter, measured-Cz and cache producers."""
    import edfio

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    prepared = tmp_path / "prepared" / "Huebner2017"
    spec = replace(preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT),
                   baseline_mode="none")
    method = source_preparation_method(spec)
    channels = ("Fz", "Cz", "P3", "Pz", "P4", "PO8", "Oz")
    items, rows = [], []
    for person in range(1, 5):
        path = raw_root / f"person{person}_eeg.bdf"
        values = np.random.default_rng(person).normal(0, 4, (len(channels), 12 * 250))
        edfio.Bdf([edfio.BdfSignal(value, sampling_frequency=250, label=name,
            physical_dimension="uV", physical_range=(-100, 100))
            for name, value in zip(channels, values, strict=True)]).write(path)
        (raw_root / f"person{person}_channels.tsv").write_text(
            "name\ttype\tstatus\n" + "".join(f"{name}\tEEG\tgood\n" for name in channels),
            encoding="utf-8")
        (raw_root / f"person{person}_events.tsv").write_text(
            "sample\tonset\ttrial_type\n" + "".join(
                f"{(i + 2) * 250}\t{i + 2}\t{'Target' if i % 2 else 'NonTarget'}\n"
                for i in range(8)), encoding="utf-8")
        item = dict(study="Huebner2017", subject=str(person), path=path.name,
            sha256=file_sha256(path), sample_rate_hz=250., reference="native common",
            session="s1", run="r1")
        items.append(item)
        rows.append(prepare_source_record(item, root=raw_root, output=prepared / "shards",
            preprocessing=spec, preparation_method=method))
    manifest = raw_root / "manifest.json"
    manifest.write_text(json.dumps({"schema": "n2p3_source_records/1", "study": "Huebner2017",
        "preprocessing": spec.record(),
        "remote_root": str(raw_root), "records": items}), encoding="utf-8")
    cache = merge_source_shards([row["cache"] for row in rows], prepared / "epochs.npz")
    record = dict(schema=SOURCE_PREPARATION_SCHEMA, study="Huebner2017", status="completed",
        preparation_method=method, target_prefixes=None, manifest_sha256=file_sha256(manifest),
        records=rows, errors=[], cache=str(cache), cache_sha256=file_sha256(cache),
        cache_record_sha256=file_sha256(cache.with_suffix(".record.json")), n_epochs=32,
        scheduled=32, participants=4)
    (prepared / "preparation.json").write_text(json.dumps(record), encoding="utf-8")
    return cache, spec, manifest


def independent_source_plan(cache, spec, snapshot):
    return {"schema": "n2p3_source_training_chain/1",
        "source_program": "independent_selection_refit", "source_preprocessing": spec.record(),
        "source_config": {"pos_weight": 8., "val_groups_min": 2, "val_groups_max": 12},
        "source_objectives": {"ce": {}}, "source_layouts": ["six"],
        "source_snapshot_manifest": str(snapshot), "evidence_scope": "software_correctness",
        "global_early_stopping": {"patience": 1, "min_delta": 100., "source_max_epochs": 2,
                                  "target_max_epochs": 2, "source_validation_fraction": .2},
        "source_seeds": [11, 12, 13], "target_seeds": [11, 12, 13],
        "paired_source_target_seeds": True, "compile_mode": None,
        "cpu_budget": 1, "source_workers": 1, "target_workers": 1,
        "source_worker_memory_gib": 1., "target_worker_memory_gib": .25, "host_memory_fraction": .95,
        "batch_rule": user_fixed_target_batch_rule(),
        "outer_folds": [{"name": "leave_BNCI", "target_study": "BNCI2014-008",
            "source_cache": str(cache), "source_studies": ["Huebner2017"],
            "excluded_source_datasets": ["BNCI2014-008"]}]}
