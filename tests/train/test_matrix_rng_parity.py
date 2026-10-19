"""The same declared row stream survives residency and shuffle materialization."""
from __future__ import annotations

import pytest
import torch

from train.runtime import GpuPerformanceScheduler, MatrixBatchSource


def test_shuffle_uses_explicit_cpu_device_even_under_another_default_device():
    source = MatrixBatchSource(torch.arange(12).float().reshape(6, 2), None,
        GpuPerformanceScheduler(torch.device("cpu")), preload=False)
    generator = source.make_generator(13)
    expected = torch.randperm(6, generator=torch.Generator(device="cpu").manual_seed(13), device="cpu")
    with torch.device("meta"):
        actual = source.random_permutation(generator)
    assert actual.device.type == "cpu"
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA residency required")
@pytest.mark.parametrize("materialize", [False, True])
def test_actual_cuda_residency_preserves_rng_state_and_every_row_axis(materialize):
    runtime = GpuPerformanceScheduler(torch.device("cuda"))
    n = 31
    values = torch.arange(n * 3 * 5).float().reshape(n, 3, 5)
    labels = torch.arange(n, dtype=torch.long) * 3
    weights = torch.arange(1, n + 1).float()
    domains = torch.arange(n, dtype=torch.long) % 3
    masks = torch.arange(n * 3).reshape(n, 3) % 4 != 0
    sources = [MatrixBatchSource(values, labels, runtime, preload=preload,
        row_weights=weights, row_domains=domains, row_channel_mask=masks,
        release_cpu_after_preload=preload) for preload in (False, True)]
    assert not sources[0].preloaded and sources[1].preloaded
    sources[1].materialize_shuffled_epoch = materialize
    assert sources[1].cpu_X is None
    generators = [source.make_generator(71) for source in sources]
    assert all(generator.device.type == "cpu" for generator in generators)
    expected_generator = torch.Generator(device="cpu").manual_seed(71)
    for _epoch in range(2):
        expected = torch.randperm(n, generator=expected_generator, device="cpu")
        for source, generator in zip(sources, generators, strict=True):
            batches = list(source.iter_batches(7, generator=generator))
            for axis, original in enumerate((values, labels, weights, domains, masks)):
                actual = torch.cat([batch[axis].cpu() for batch in batches])
                assert torch.equal(actual, original[expected]), (materialize, axis)
            assert torch.equal(generator.get_state(), expected_generator.get_state())
    with pytest.raises(ValueError, match="CPU generator"):
        sources[1].random_permutation(torch.Generator(device="cuda").manual_seed(71))
