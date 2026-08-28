from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from baselines.validation import group_disjoint_validation_split
from data.channel import build_channel_identity
from data.epochs import EpochDataset, PreprocessingSpec
from data.events import ScheduledEventTimeline
from models.n2p3net import N2P3Net
from transfer.evaluation import hit_at_repetition
from transfer.heads import SubjectProbeHead, WaveDecoderHead
from transfer.losses import (
    DEFAULT_BANDS_HZ,
    ReconstructionLossConfig,
    band_balanced_spectral_loss,
    band_magnitudes,
    estimate_band_weights,
    reconstruction_loss,
)
from transfer.masking import MaskingConfig, apply_time_mask, make_temporal_mask
from transfer.pretraining import PretrainingConfig, PretrainingTask
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig
from transfer.within_subject import causal_prefix_suffix_split


def _trunk() -> N2P3Net:
    return N2P3Net(
        n_channels=3,
        n_times=128,
        sfreq=128.0,
        tmin_s=-0.2,
        pooling_mode="ms_flatten",
    )


def test_n2p3net_exposes_trunk_features_without_changing_logits() -> None:
    trunk = _trunk()
    x = torch.randn(2, 3, 128)
    features = trunk.forward_features(x)
    assert features.shape == (2, 4, 32)
    logits = trunk(x)
    assert logits.shape == (2, 2)
    # forward_features is part of forward, so repeated calls are finite.
    assert torch.isfinite(features).all()


def test_temporal_mask_is_deterministic_and_nonempty() -> None:
    config = MaskingConfig(mask_fraction=0.5, min_block_samples=12, max_block_samples=32)
    gen = torch.Generator().manual_seed(7)
    first = make_temporal_mask(128, config=config, generator=gen)
    gen = torch.Generator().manual_seed(7)
    second = make_temporal_mask(128, config=config, generator=gen)
    assert torch.equal(first, second)
    fraction = float(first.to(torch.float32).mean())
    assert 0.35 <= fraction <= 0.65
    x = torch.randn(4, 3, 128)
    masked, keep = apply_time_mask(x, first)
    assert masked.shape == x.shape
    assert torch.all(masked[:, :, first] == 0.0)
    assert keep.dtype == x.dtype


def test_band_weights_sum_to_one_and_zero_reconstruction_loss() -> None:
    x = torch.randn(8, 3, 128)
    weights = estimate_band_weights(x, sfreq=128.0)
    assert weights.shape == (4,)
    assert abs(float(weights.sum()) - 1.0) < 1e-6
    config = ReconstructionLossConfig()
    losses = reconstruction_loss(x, x, sfreq=128.0, config=config, weights=weights)
    assert float(losses["waveform"]) < 1e-6
    assert float(losses["spectral"]) < 1e-5


def test_band_balanced_loss_is_lower_for_exact_waveform() -> None:
    x = torch.randn(4, 3, 128)
    bad = torch.randn(4, 3, 128)
    good = band_balanced_spectral_loss(x, x, sfreq=128.0)
    worse = band_balanced_spectral_loss(x, bad, sfreq=128.0)
    assert float(good) < float(worse)


def test_band_balanced_loss_is_invariant_to_repeated_batch_rows() -> None:
    x = torch.randn(3, 3, 128)
    x_hat = x + 0.1 * torch.randn_like(x)
    single = band_balanced_spectral_loss(x, x_hat, sfreq=128.0)
    repeated = band_balanced_spectral_loss(
        x.repeat(4, 1, 1),
        x_hat.repeat(4, 1, 1),
        sfreq=128.0,
    )
    assert torch.allclose(single, repeated, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [(3, 128), (2, 3, 128)])
def test_cached_band_projection_matches_direct_slices_and_has_gradient(shape) -> None:
    x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    actual = torch.stack(
        band_magnitudes(x, sfreq=128.0, bands_hz=DEFAULT_BANDS_HZ)
    )
    window = torch.hann_window(128, periodic=False, dtype=x.dtype)
    spectrum = torch.fft.rfft((x - x.mean(dim=-1, keepdim=True)) * window, dim=-1).abs()
    freqs = torch.fft.rfftfreq(128, d=1.0 / 128.0, dtype=x.dtype)
    expected = torch.stack(
        [spectrum[..., (freqs >= start) & (freqs < end)].mean(dim=-1) for start, end in DEFAULT_BANDS_HZ]
    )
    assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-12)
    actual.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_wave_decoder_matches_contract_and_is_small() -> None:
    decoder = WaveDecoderHead(trunk_channels=4, output_channels=3, st_pool_size=4)
    features = torch.randn(2, 4, 32)
    mask = torch.ones(2, 128)
    out = decoder(features, mask)
    assert out.shape == (2, 3, 128)
    assert decoder.parameter_count() < 200


def test_pretraining_task_loss_and_subject_probe() -> None:
    trunk = _trunk()
    config = PretrainingConfig(subject_probe_subjects=4)
    task = PretrainingTask(trunk, config)
    x = torch.randn(8, 3, 128)
    ids = torch.randint(0, 4, (8,))
    components = task.loss_components(x, subject_ids=ids)
    assert set(components) == {"waveform", "spectral", "total", "subject_probe"}
    assert torch.isfinite(components["total"])
    task.discard_decoder()
    assert task.decoder is None and task.probe is None


def test_subject_probe_shape() -> None:
    probe = SubjectProbeHead(16, 5)
    out = probe(torch.randn(3, 16))
    assert out.shape == (3, 5)


def test_subject_adapter_fits_and_predicts() -> None:
    rng = np.random.default_rng(0)
    trunk = _trunk()
    adapter = SubjectAdapter(
        trunk,
        config=SubjectAdapterConfig(
            head_kind="linear",
            epochs=3,
            batch_size=16,
            val_group_fraction=None,
        ),
    )
    X = rng.normal(size=(32, 3, 128)).astype(np.float32)
    y = np.tile(np.array([0, 1], dtype=np.int64), 16)
    X[y == 1, 2, 60:80] += 0.5
    adapter.fit(X, y)
    logits = adapter.predict_logit(X)
    assert logits.shape == (32,)
    assert np.isfinite(logits).all()
    # Frozen-trunk adapter should train only the 2-class linear head.
    assert adapter.parameter_count() == 34


def test_frozen_subject_adapter_keeps_trunk_batchnorm_state() -> None:
    rng = np.random.default_rng(4)
    trunk = _trunk()
    before = {
        name: value.detach().clone()
        for name, value in trunk.state_dict().items()
        if "running_" in name or "num_batches_tracked" in name
    }
    adapter = SubjectAdapter(
        trunk,
        config=SubjectAdapterConfig(
            head_kind="linear",
            epochs=2,
            batch_size=8,
            val_group_fraction=None,
        ),
    )
    X = rng.normal(size=(16, 3, 128)).astype(np.float32)
    y = np.tile(np.array([0, 1], dtype=np.int64), 8)
    adapter.fit(X, y)
    after = trunk.state_dict()
    assert all(torch.equal(value, after[name]) for name, value in before.items())


def test_subject_adapter_statistics_exclude_validation_groups() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(24, 3, 128)).astype(np.float32)
    y = np.tile(np.array([0, 1], dtype=np.int64), 12)
    groups = np.repeat(np.array(["g0", "g1", "g2", "g3"]), 6)
    config = SubjectAdapterConfig(
        head_kind="linear",
        epochs=1,
        batch_size=8,
        val_group_fraction=0.25,
        val_groups_min=1,
        val_groups_max=1,
        early_stop_patience=1,
    )
    split = group_disjoint_validation_split(
        groups,
        fraction=config.val_group_fraction,
        min_groups=config.val_groups_min,
        max_groups=config.val_groups_max,
        seed=config.seed,
    )
    adapter = SubjectAdapter(_trunk(), config=config)
    adapter.fit(X, y, group_ids=groups)
    expected = X[split.train_mask].mean(axis=(0, 2), keepdims=True)
    assert np.allclose(adapter._input_mean, expected)
    assert isinstance(adapter.calibration_logits_, np.ndarray)


def test_hit_at_repetition_sums_target_logits() -> None:
    # Two groups, digits 1..4, two repetitions per digit. group-a truth=2,
    # group-b truth=3; positive logits are placed on those target digits.
    digit_block = np.array([1, 2, 3, 4], dtype=np.int64)
    digits = np.concatenate([np.repeat(digit_block, 2), np.repeat(digit_block, 2)])
    groups = np.concatenate([np.repeat("a", 8), np.repeat("b", 8)])
    repetitions = np.tile(np.array([0, 1], dtype=np.int64), 8)
    logits = np.zeros(16, dtype=float)
    logits[(groups == "a") & (digits == 2)] = 1.0
    logits[(groups == "b") & (digits == 3)] = 1.5
    hits = hit_at_repetition(
        logits,
        digits,
        groups,
        {"a": 2, "b": 3},
        repetitions,
        aggregation="sum",
        max_repetitions=2,
    )
    assert hits[1] == 1.0
    assert hits[2] == 1.0


def test_hit_at_repetition_rejects_nonfinite_logits() -> None:
    with pytest.raises(ValueError, match="NaN"):
        hit_at_repetition(
            [1.0, float("nan")],
            [1, 2],
            ["g", "g"],
            {"g": 1},
            [0, 0],
        )


def _causal_candidate_dataset() -> EpochDataset:
    digits = np.array([1, 2, 3, 4], dtype=np.int64)
    n_epochs = 3 * len(digits) * 4
    groups = np.repeat(np.array(["g1", "g2", "g3"]), n_epochs // 3)
    candidates = np.tile(np.repeat(digits, 4), 3)
    targets = np.repeat(np.array([2, 3, 1], dtype=np.int64), n_epochs // 3)
    repetitions = np.tile(np.arange(4, dtype=np.int64), 3 * len(digits))
    identity = build_channel_identity(("Fz", "Cz", "Pz"), allow_missing_positions=False)
    timeline = ScheduledEventTimeline(
        event_ids=np.asarray([f"e{i}" for i in range(n_epochs)]),
        group_ids=groups,
        subject_ids=groups,
        stimulus_ids=candidates,
        onset_samples=np.arange(n_epochs, dtype=np.int64) * 100,
        onset_times_s=np.arange(n_epochs, dtype=float),
        evidence_available_times_s=np.arange(n_epochs, dtype=float) + 0.8,
        evidence_indices=np.arange(n_epochs, dtype=np.int64),
        statuses=np.repeat("available", n_epochs),
        status_details=np.repeat("", n_epochs),
        dataset_ids=np.repeat("synthetic", n_epochs),
        session_ids=np.repeat("", n_epochs),
        run_ids=groups,
        selection_ids=groups,
        complete=True,
        online_causal=True,
        timing_source="synthetic_causal",
        candidate_ids=candidates.astype(str),
        target_candidate_ids=targets.astype(str),
        repetition_indices=repetitions,
    )
    return EpochDataset(
        name="synthetic_causal_candidates",
        X=np.zeros((n_epochs, 3, 128), dtype=np.float32),
        y=(candidates == targets).astype(np.int64),
        subject_ids=groups.astype(str),
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=np.ones(3, dtype=bool),
        preprocessing=PreprocessingSpec(
            name="p300_single_subject_causal_v1",
            sfreq=128.0,
            l_freq=2.0,
            h_freq=30.0,
            tmin_ms=-200.0,
            tmax_ms=800.0,
            n_times=128,
            baseline_mode="mean_only",
            filter_phase="forward",
        ),
        event_timeline=timeline,
        metadata=pd.DataFrame({"subject": groups.astype(str)}),
        provenance={"source": "unit_test", "source_reference": "average", "source_sample_rate_hz": 128.0},
    )


def test_causal_prefix_suffix_split_is_chronological_and_complete() -> None:
    dataset = _causal_candidate_dataset()
    split = causal_prefix_suffix_split(dataset, prefix_repetitions=2, test_repetitions=2)

    assert set(split.usable_groups) == {"g1", "g2", "g3"}
    assert int(split.prefix_mask.sum()) == int(split.suffix_mask.sum()) == 3 * 4 * 2
    for group in split.usable_groups:
        rows = np.flatnonzero(split.group_ids == group)
        assert int(split.prefix_mask[rows].sum()) == 8
        assert int(split.suffix_mask[rows].sum()) == 8
        suffix_reps = split.suffix_repetition_indices[rows][split.suffix_mask[rows]]
        assert set(np.unique(suffix_reps).tolist()) == {0, 1}


def test_causal_prefix_suffix_split_rejects_zero_phase_cache() -> None:
    dataset = _causal_candidate_dataset()
    dataset.preprocessing = PreprocessingSpec(
        name="offline",
        sfreq=128.0,
        l_freq=2.0,
        h_freq=30.0,
        tmin_ms=-200.0,
        tmax_ms=800.0,
        n_times=128,
        baseline_mode="mean_only",
        filter_phase="zero",
    )
    # EpochDataset.validate now refuses a forward contract whose timeline is
    # still marked acausal; this is the first leakage gate.
    with pytest.raises(ValueError, match="online_causal"):
        causal_prefix_suffix_split(dataset, prefix_repetitions=2, test_repetitions=2)
