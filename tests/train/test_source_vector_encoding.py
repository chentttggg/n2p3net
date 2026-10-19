import copy

import numpy as np
import pytest
import torch

from baselines.deep import DeepConfig
from baselines.n2p3net import N2P3NetBaseline
from train.source_sampling import (
    decode_source_vector,
    encode_source_vector,
    sample_source_rows,
    source_sampling_cdf,
)


@pytest.mark.parametrize("values", [np.arange(4096), np.full(4096, 1 / 4096),
    np.tile([.125, .5, .25], 1366), np.r_[np.arange(2000), np.arange(4000, 6100)],
    np.tile([0., -0.], 1025), np.full(2050, -0.)])
def test_source_vector_is_lossless_and_digest_bound(values):
    encoded = encode_source_vector(values)
    np.testing.assert_array_equal(decode_source_vector(encoded), values)
    assert decode_source_vector(encoded).tobytes() == values.tobytes()
    bad = copy.deepcopy(encoded)
    bad["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        decode_source_vector(bad)


def test_compact_source_ledgers_do_not_change_updates_or_parameters(monkeypatch):
    import baselines.deep as deep
    import train.source_sampling as sampling

    values = np.random.default_rng(5).normal(size=(1050, 3, 100)).astype(np.float32)
    labels = np.arange(len(values)) % 2
    config = DeepConfig(fixed_updates=3, batch_size=8, seed=7, compile_mode=None, sampling="uniform_rows")
    before = N2P3NetBaseline(3, 100, 100, config=config, device=torch.device("cpu"))
    with monkeypatch.context() as legacy:
        legacy.setattr(deep, "encode_source_vector", lambda x: np.asarray(x).tolist())
        legacy.setattr(sampling, "encode_source_vector", lambda x: np.asarray(x).tolist())
        legacy.setattr(deep, "sample_source_rows", lambda rng, probability, size, **kw:
                       rng.choice(len(probability), size=size, replace=True, p=probability))
        before.fit(values, labels)
    after = N2P3NetBaseline(3, 100, 100, config=config, device=torch.device("cpu"))
    after.fit(values, labels)
    for name, value in before.model_.state_dict().items():
        torch.testing.assert_close(value, after.model_.state_dict()[name], rtol=0, atol=0)
    assert before.last_history["batch_sequence_sha256"] == after.last_history["batch_sequence_sha256"]
    for name in ("row_probability", "target_risk_mass", "importance_weight"):
        np.testing.assert_array_equal(before.sampling_ledger_[name], decode_source_vector(after.sampling_ledger_[name]))


@pytest.mark.parametrize("probability", [np.full(100003, 1 / 100003),
    np.arange(1, 100004, dtype=float) / np.arange(1, 100004, dtype=float).sum(),
    np.array([0., .125, 0., .25, .625, 0.])])
def test_precomputed_cdf_preserves_every_draw_and_rng_state(probability):
    original, optimized = np.random.default_rng(19), np.random.default_rng(19)
    cumulative = source_sampling_cdf(probability)
    for _ in range(16):
        expected = original.choice(len(probability), size=512, replace=True, p=probability)
        actual = sample_source_rows(optimized, probability, 512, cumulative=cumulative)
        np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(original.random(32), optimized.random(32))
