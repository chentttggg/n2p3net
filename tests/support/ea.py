"""Physical EEG fixture shared by EA artifact and CLI contract tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.channel import build_channel_identity
from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.domain import SOURCE_DOMAIN_AXIS_SCHEMA, SOURCE_DOMAIN_COLUMN
from data.epochs import EpochDataset, preprocessing_spec_from_contract
from data.events import observed_only_timeline


def make_ea_dataset(
    name: str,
    subjects: tuple[str, ...],
    *,
    domains: tuple[str, ...] | None = None,
) -> EpochDataset:
    identity = build_channel_identity(("Fz", "Cz", "Pz"), allow_missing_positions=False)
    n_epochs = len(subjects)
    rng = np.random.default_rng(20260904)
    scale = np.asarray([[2.0, 0.3, 0.1], [0.0, 1.8, 0.2], [0.1, 0.0, 2.2]])
    z = rng.normal(
        size=(n_epochs, scale.shape[0], SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.n_times)
    )
    x = np.einsum("cp,npt->nct", scale, z)
    timeline = observed_only_timeline(
        dataset_id=name,
        subject_ids=np.asarray(subjects),
        stimulus_ids=np.arange(n_epochs, dtype=np.int64),
        group_ids=np.asarray(subjects),
        online_causal=True,
        timing_source="ea_cache_cli_fixture",
    )
    metadata = pd.DataFrame({"subject": np.asarray(subjects)})
    provenance: dict[str, object] = {
        "source": "unit_test",
        "source_reference": "average",
    }
    if domains is not None:
        metadata[SOURCE_DOMAIN_COLUMN] = np.asarray(domains)
        provenance["source_domain_axis"] = {
            "schema": SOURCE_DOMAIN_AXIS_SCHEMA,
            "column": SOURCE_DOMAIN_COLUMN,
            "domains": sorted(set(domains)),
        }
    return EpochDataset(
        name=name,
        X=np.ascontiguousarray(x, dtype=np.float32),
        y=np.zeros(n_epochs, dtype=np.int64),
        subject_ids=np.asarray(subjects),
        channel_names=identity.names,
        channel_positions_m=np.asarray(identity.coords, dtype=np.float32),
        channel_mask=np.ones(len(identity.names), dtype=bool),
        preprocessing=preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT),
        event_timeline=timeline,
        metadata=metadata,
        provenance=provenance,
    )

