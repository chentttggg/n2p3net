from __future__ import annotations

import pytest

from train.batch import DEFAULT_TRAINING_BATCH, TrainingBatchConfig


def test_default_batch_contract() -> None:
    assert DEFAULT_TRAINING_BATCH.record() == {
        "physical_batch_size": 256,
        "effective_batch_size": 256,
        "accumulation_steps": 1,
    }


def test_batch_contract_derives_accumulation() -> None:
    config = TrainingBatchConfig.from_cli(physical_batch_size=256, effective_batch_size=4096)
    assert config.accumulation_steps == 16
    assert config.trainer_overrides() == {"batch_size": 256, "accum_steps": 16}


def test_legacy_accum_steps_is_compatible() -> None:
    config = TrainingBatchConfig.from_cli(physical_batch_size=256, accum_steps=8)
    assert config.effective_batch_size == 2048


@pytest.mark.parametrize(
    "physical,effective",
    [(0, 2048), (256, 128), (256, 2000)],
)
def test_batch_contract_rejects_invalid_sizes(physical: int, effective: int) -> None:
    with pytest.raises(ValueError):
        TrainingBatchConfig(physical, effective)
