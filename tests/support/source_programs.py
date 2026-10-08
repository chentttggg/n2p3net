"""Small source-only inputs and budgets for selection/refit contract tests."""

import numpy as np
import pandas as pd

from baselines.deep import DeepConfig
from data.domain import annotate_source_domain, namespace_epoch_dataset
from data.events import observed_only_timeline
from data.identity import DatasetIdentityTable
from tests.support.datasets import epoch_dataset
from train.early_stopping import GlobalEarlyStoppingPolicy


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
