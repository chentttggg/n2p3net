from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from baselines.validation import group_disjoint_validation_split
from data.channel import build_channel_identity
from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.domain import namespace_epoch_dataset
from data.epochs import (
    EpochDataset,
    PreprocessingSpec,
    materialize_dataset_identity,
    preprocessing_spec_from_contract,
    save_epoch_dataset,
)
from data.events import ScheduledEventTimeline, candidate_repetition_indices
from data.identity import (
    DatasetIdentityTable,
    ParticipantIdentityRecord,
    origin_subject_key,
)
from experiments.run_pretrain import _source_training_rows, _subject_probe_validation_mask
from experiments.run_pretrain import main as run_pretrain_main
from experiments.run_within_subject_transfer import (
    _load_trunk,
    _target_fixed_budget_calibration_payload,
)
from experiments.run_within_subject_transfer import main as run_transfer_main
from models.n2p3net import N2P3Net
from research.contracts import TrainingRunContract
from transfer.checkpoint import (
    CHECKPOINT_SCHEMA,
    checkpoint_architecture_record,
    checkpoint_classifier_is_trained,
    checkpoint_input_stats,
    checkpoint_scores_to_llr,
    checkpoint_training_contract,
    load_n2p3_trunk_checkpoint,
)
from transfer.evaluation import candidate_evidence_endpoints, hit_at_repetition
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
from transfer.within_subject import (
    causal_prefix_suffix_split,
    chronological_time_validation_split,
)


def _trunk() -> N2P3Net:
    return N2P3Net(
        n_channels=3,
        n_times=128,
        sfreq=128.0,
        tmin_s=-0.2,
        pooling_mode="ms_flatten",
    )


def _dataset_trunk(dataset: EpochDataset) -> N2P3Net:
    return N2P3Net(
        n_channels=dataset.n_channels,
        n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq,
        tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
        pooling_mode="ms_flatten",
    )


def _identity_ledger(
    source_dataset_id: str,
    *subjects: str,
    global_key: str | None = None,
) -> DatasetIdentityTable:
    return DatasetIdentityTable(
        records=tuple(
            ParticipantIdentityRecord(
                local_subject_id=subject,
                origin_subject_keys=(origin_subject_key(source_dataset_id, subject),),
                global_person_keys=(global_key,) if global_key is not None else (),
                identity_status=(
                    "global_verified" if global_key is not None else "source_verified"
                ),
            )
            for subject in subjects
        )
    )


def _checkpoint_identity_fields(
    table: DatasetIdentityTable,
    dataset: EpochDataset,
    *,
    source_snapshot_sha256: str = "2" * 64,
) -> dict[str, object]:
    architecture = _dataset_trunk(dataset).architecture_record()
    preprocessing = {
        "epoch": asdict(dataset.preprocessing),
        "channel_names": list(dataset.channel_names),
        "source_reference": dataset.provenance["source_reference"],
    }
    contract = TrainingRunContract(
        source_cache_sha256="1" * 64,
        source_identity_digest=table.digest(),
        source_snapshot_sha256=source_snapshot_sha256,
        architecture=architecture,
        preprocessing=preprocessing,
        optimizer={"fixture": "adam"},
        validation={"fixture": "target_excluded"},
        objective={"fixture": "binary_or_reconstruction"},
        seed=0,
        training_participant_keys=table.authority_keys(),
        holdout_participant_keys=(),
    )
    return {
        "schema": CHECKPOINT_SCHEMA,
        "source_cache_sha256": "1" * 64,
        "source_snapshot_sha256": source_snapshot_sha256,
        "architecture": architecture,
        "input_preprocessing": asdict(dataset.preprocessing),
        "input_channel_names": list(dataset.channel_names),
        "input_source_reference": dataset.provenance["source_reference"],
        "classifier_trained": True,
        "input_mean": [0.0] * dataset.n_channels,
        "input_std": [1.0] * dataset.n_channels,
        "n_channels": dataset.n_channels,
        "n_times": dataset.n_times,
        "input_sample_rate_hz": dataset.preprocessing.sfreq,
        "input_tmin_s": dataset.preprocessing.tmin_ms / 1000.0,
        "training_identity_ledger": table.payload(),
        "training_identity_ledger_digest": table.digest(),
        "training_contract": contract.record(),
        "training_contract_digest": contract.digest(),
    }


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


def test_reconstruction_loss_ignores_visible_region_errors() -> None:
    x = torch.zeros(2, 3, 128)
    x_hat = x.clone()
    x_hat[..., :64] = 10.0
    masked_region = torch.zeros(1, 1, 128, dtype=torch.bool)
    masked_region[..., 64:] = True

    losses = reconstruction_loss(
        x,
        x_hat,
        sfreq=128.0,
        config=ReconstructionLossConfig(waveform_weight=1.0, spectral_weight=0.0),
        sample_mask=masked_region,
    )
    assert float(losses["total"]) == pytest.approx(0.0, abs=1e-8)


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
    assert set(components) == {
        "waveform",
        "spectral",
        "total",
        "subject_probe",
        "subject_probe_correct",
    }
    assert torch.isfinite(components["total"])
    task.zero_grad(set_to_none=True)
    components["subject_probe"].backward()
    assert all(parameter.grad is None for parameter in task.trunk.parameters())
    assert any(parameter.grad is not None for parameter in task.probe.parameters())
    task.discard_decoder()
    assert task.decoder is None and task.probe is None


def test_pretraining_holdout_must_exist_in_the_source_cache() -> None:
    rows, subjects = _source_training_rows(
        np.asarray(["s1", "s2", "s1"]),
        {"s2"},
    )
    assert subjects == {"s1", "s2"}
    assert rows.tolist() == [True, False, True]

    with pytest.raises(ValueError, match="absent from the source cache"):
        _source_training_rows(np.asarray(["s1", "s2"]), {"typo"})


def test_subject_probe_validation_split_holds_out_each_subject() -> None:
    subjects = np.repeat(np.asarray(["s1", "s2", "s3"]), 10)
    first = _subject_probe_validation_mask(subjects, seed=11)
    second = _subject_probe_validation_mask(subjects, seed=11)

    assert np.array_equal(first, second)
    assert [int(first[subjects == subject].sum()) for subject in np.unique(subjects)] == [2, 2, 2]


def test_pretraining_runner_records_trained_stop_gradient_subject_probe(
    tmp_path, monkeypatch
) -> None:
    dataset = _causal_candidate_dataset()
    cache = save_epoch_dataset(tmp_path / "pretrain.npz", dataset)
    checkpoint = tmp_path / "pretrained.pt"
    source_manifest, source_digest, _ = _write_source_snapshot_manifest(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain.py",
            "--source-cache",
            str(cache),
            "--source-snapshot-manifest",
            str(source_manifest),
            "--checkpoint",
            str(checkpoint),
            "--cohort",
            "causal",
            "--holdout-subjects",
            "g3",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--device",
            "cpu",
        ],
    )

    run_pretrain_main()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["classifier_trained"] is False
    assert payload["schema"] == CHECKPOINT_SCHEMA
    assert payload["subject_probe_audit"]["enabled"] is True
    assert payload["subject_probe_audit"]["stop_gradient"] is True
    assert payload["subject_probe_audit"]["n_subjects"] == 2
    assert payload["subject_probe_audit"]["final_validation_accuracy"] is not None
    assert "training_subject_keys" not in payload
    ledger = DatasetIdentityTable.from_payload(payload["training_identity_ledger"])
    assert ledger.local_subject_ids == ("g1", "g2")
    assert "g3" not in ledger.local_subject_ids
    assert payload["training_identity_ledger_digest"] == ledger.digest()
    contract = checkpoint_training_contract(payload)
    assert contract.source_snapshot_sha256 == source_digest
    assert payload["source_snapshot_sha256"] == source_digest
    assert payload["source_snapshot_manifest"] == str(source_manifest.resolve())
    assert set(contract.training_participant_keys) == set(
        ledger.authority_keys()
    )
    assert origin_subject_key("synthetic", "g3") in contract.holdout_participant_keys
    changed = replace(
        contract,
        objective={**contract.objective, "waveform_weight": 99.0},
    )
    assert changed.digest() != contract.digest()
    mutated_payload = {
        **payload,
        "training_contract": changed.record(),
    }
    with pytest.raises(ValueError, match="training_contract_digest"):
        checkpoint_training_contract(mutated_payload)


def test_pretraining_runner_requires_source_snapshot_manifest(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain.py",
            "--source-cache",
            str(tmp_path / "source.npz"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
        ],
    )

    with pytest.raises(SystemExit):
        run_pretrain_main()


def test_pretraining_runner_rejects_legacy_naked_snapshot_digest(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain.py",
            "--source-cache",
            str(tmp_path / "source.npz"),
            "--source-snapshot-sha256",
            "a" * 64,
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
        ],
    )

    with pytest.raises(SystemExit):
        run_pretrain_main()


def test_pretraining_runner_rejects_tampered_source_archive(
    tmp_path, monkeypatch
) -> None:
    source_manifest, _, archive = _write_source_snapshot_manifest(tmp_path)
    tampered = bytearray(archive.read_bytes())
    tampered[4] ^= 1
    archive.write_bytes(tampered)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain.py",
            "--source-cache",
            str(tmp_path / "unused.npz"),
            "--source-snapshot-manifest",
            str(source_manifest),
            "--checkpoint",
            str(tmp_path / "must-not-exist.pt"),
        ],
    )

    with pytest.raises(ValueError, match="archive hash disagrees"):
        run_pretrain_main()


def test_subject_probe_shape() -> None:
    probe = SubjectProbeHead(16, 5)
    out = probe(torch.randn(3, 16))
    assert out.shape == (3, 5)


def test_wave_decoder_restores_nondivisible_causal_length() -> None:
    decoder = WaveDecoderHead(trunk_channels=4, output_channels=3, st_pool_size=4)
    mask = torch.ones(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.n_times)
    features = torch.randn(2, 4, len(mask) // 4)

    reconstructed = decoder(features, mask)

    assert reconstructed.shape == (2, 3, len(mask))


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


def test_full_fine_batchnorm_adaptation_is_an_explicit_arm() -> None:
    frozen = SubjectAdapter(
        _trunk(),
        config=SubjectAdapterConfig(head_kind="full_fine", freeze_batchnorm_stats=True),
    )
    adaptive = SubjectAdapter(
        _trunk(),
        config=SubjectAdapterConfig(head_kind="full_fine", freeze_batchnorm_stats=False),
    )

    frozen._set_training_mode(trunk_trainable=True)
    adaptive._set_training_mode(trunk_trainable=True)

    frozen_bn = [module.training for module in frozen.trunk.modules() if "BatchNorm" in type(module).__name__]
    adaptive_bn = [module.training for module in adaptive.trunk.modules() if "BatchNorm" in type(module).__name__]
    assert frozen_bn and not any(frozen_bn)
    assert adaptive_bn and all(adaptive_bn)


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


def test_chronological_refit_uses_every_prefix_row_after_epoch_selection() -> None:
    rng = np.random.default_rng(19)
    X = rng.normal(size=(30, 3, 128)).astype(np.float32)
    y = np.tile(np.array([0, 0, 0, 0, 1], dtype=np.int64), 6)
    train = np.arange(len(X)) < 18
    validation = np.arange(len(X)) >= 22
    adapter = SubjectAdapter(
        _trunk(),
        config=SubjectAdapterConfig(epochs=2, batch_size=16, early_stop_patience=2),
    )

    adapter.fit(X, y, training_mask=train, validation_mask=validation)

    assert adapter.last_history["refit_full_prefix"] is True
    assert 1 <= int(adapter.last_history["refit_epochs"]) <= 2
    np.testing.assert_allclose(
        adapter._input_mean,
        X.mean(axis=(0, 2), keepdims=True),
        rtol=1e-6,
        atol=1e-6,
    )


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


def test_hit_at_repetition_accepts_zero_based_encoded_candidate_vocabulary() -> None:
    candidate_block = np.arange(9, dtype=np.int64)
    digits = np.tile(np.repeat(candidate_block, 2), 2)
    groups = np.repeat(np.array(["a", "b"]), 18)
    repetitions = np.tile(np.tile(np.array([0, 1], dtype=np.int64), 9), 2)
    logits = np.zeros(36, dtype=float)
    logits[(groups == "a") & (digits == 0)] = 1.0
    logits[(groups == "b") & (digits == 8)] = 1.0

    hits = hit_at_repetition(
        logits,
        digits,
        groups,
        {"a": 0, "b": 8},
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


def test_trim_two_repetitions_keeps_both_values_and_ties_abstain() -> None:
    logits = np.asarray([2.0, 1.0, 0.0, 0.0])
    digits = np.asarray([1, 1, 2, 2])
    repetitions = np.asarray([0, 1, 0, 1])
    hits = hit_at_repetition(
        logits,
        digits,
        np.repeat("g", 4),
        {"g": 1},
        repetitions,
        aggregation="trim0.2",
        max_repetitions=2,
        candidate_vocabulary=(1, 2),
    )
    tied = hit_at_repetition(
        np.zeros(4),
        digits,
        np.repeat("g", 4),
        {"g": 1},
        repetitions,
        aggregation="trim0.2",
        max_repetitions=2,
        candidate_vocabulary=(1, 2),
    )

    assert hits[2] == 1.0
    assert tied[2] == 0.0


def test_positive_source_calibration_preserves_score_order() -> None:
    payload = {
        "source_calibration": {
            "pos_weight": 8.0,
            "train_prior": 1.0 / 9.0,
            "temperature": 3.0,
            "source": "source_validation",
        },
    }
    logits = np.asarray([-2.0, 0.5, 4.0])

    llr, record = checkpoint_scores_to_llr(payload, logits)

    assert np.array_equal(np.argsort(llr), np.argsort(logits))
    assert record["order_preserving"] is True


@pytest.mark.parametrize(
    "missing_key",
    ["pos_weight", "train_prior", "temperature", "source"],
)
def test_source_calibration_rejects_missing_structured_key(missing_key: str) -> None:
    calibration = {
        "pos_weight": 8.0,
        "train_prior": 1.0 / 9.0,
        "temperature": 1.0,
        "source": "source_validation",
    }
    calibration.pop(missing_key)

    with pytest.raises(ValueError, match="lacks required keys"):
        checkpoint_scores_to_llr(
            {"source_calibration": calibration},
            np.asarray([0.0, 1.0]),
        )


def test_source_calibration_rejects_top_level_legacy_fields() -> None:
    payload = {
        "training_pos_weight": 8.0,
        "training_prior": 1.0 / 9.0,
    }

    with pytest.raises(ValueError, match="obsolete top-level calibration fields"):
        checkpoint_scores_to_llr(payload, np.asarray([0.0, 1.0]))


def test_source_calibration_rejects_coexisting_top_level_legacy_fields() -> None:
    payload = {
        "training_pos_weight": 99.0,
        "source_calibration": {
            "pos_weight": 8.0,
            "train_prior": 1.0 / 9.0,
            "temperature": 1.0,
            "source": "source_validation",
        },
    }

    with pytest.raises(ValueError, match="obsolete top-level calibration fields"):
        checkpoint_scores_to_llr(payload, np.asarray([0.0, 1.0]))


@pytest.mark.parametrize("invalid_value", [True, "8.0", None])
def test_source_calibration_rejects_non_numeric_weight(invalid_value: object) -> None:
    payload = {
        "source_calibration": {
            "pos_weight": invalid_value,
            "train_prior": 1.0 / 9.0,
            "temperature": 1.0,
            "source": "source_validation",
        },
    }

    with pytest.raises(ValueError, match="numeric fields are invalid"):
        checkpoint_scores_to_llr(payload, np.asarray([0.0, 1.0]))


def test_target_fixed_budget_adapter_uses_current_calibration_contract() -> None:
    payload = _target_fixed_budget_calibration_payload(
        pos_weight=8.0,
        train_prior=1.0 / 9.0,
    )

    llr, record = checkpoint_scores_to_llr(payload, np.asarray([0.0, 1.0]))

    assert np.isfinite(llr).all()
    assert record["source"] == "target_fixed_budget_weighted_ce_analytic"
    assert "training_pos_weight" not in payload
    assert "training_prior" not in payload


def test_current_checkpoint_helpers_reject_implicit_defaults() -> None:
    with pytest.raises(ValueError, match="explicit input_mean"):
        checkpoint_input_stats({}, 3)
    with pytest.raises(ValueError, match="explicit boolean"):
        checkpoint_classifier_is_trained({"config": {"training": "supervised"}})
    architecture = _trunk().architecture_record()
    with pytest.raises(ValueError, match="required input geometry"):
        checkpoint_architecture_record({"architecture": architecture})
    with pytest.raises(ValueError, match="requires a source_calibration mapping"):
        checkpoint_scores_to_llr({}, np.asarray([0.0, 1.0]))


def test_all_evidence_reports_raw_bias_balanced_endpoint_and_missing_coverage() -> None:
    endpoints = candidate_evidence_endpoints(
        logits=[0.5, 0.5, 0.5, 1.0, 2.0],
        digits=[0, 0, 0, 1, 0],
        group_ids=["imbalanced", "imbalanced", "imbalanced", "imbalanced", "missing"],
        truth_by_group={"imbalanced": 1, "missing": 1},
        repetition_indices=[0, 1, 2, 0, 0],
        aggregation="sum",
        candidate_vocabulary=(0, 1),
    )

    assert endpoints["raw_predictions_by_group"] == {
        "imbalanced": 0,
        "missing": None,
    }
    assert endpoints["balanced_predictions_by_group"] == {
        "imbalanced": 1,
        "missing": None,
    }
    assert endpoints["raw_all_hit_rate"] == 0.0
    assert endpoints["balanced_all_hit_rate"] == 0.5
    assert endpoints["eligible_by_repetition"][1] == 1
    assert endpoints["correct_by_repetition"][1] == 1

    tempered = candidate_evidence_endpoints(
        logits=[0.5, 0.5, 0.5, 1.0],
        digits=[0, 0, 0, 1],
        group_ids=["imbalanced"] * 4,
        truth_by_group={"imbalanced": 1},
        repetition_indices=[0, 1, 2, 0],
        aggregation="tempered_evidence",
        evidence_count_power=0.5,
        candidate_vocabulary=(0, 1),
    )
    assert tempered["raw_predictions_by_group"] == {"imbalanced": 1}
    assert tempered["raw_all_hit_rate"] == 1.0


def test_precision_aggregation_requires_predictive_variance() -> None:
    logits = np.asarray([1.0, 2.0, 0.1, 0.1])
    digits = np.asarray([1, 1, 2, 2])
    groups = np.repeat("g", 4)
    repetitions = np.asarray([0, 1, 0, 1])

    with pytest.raises(ValueError, match="predictive variances"):
        hit_at_repetition(
            logits,
            digits,
            groups,
            {"g": 1},
            repetitions,
            aggregation="precision",
        )

    hits = hit_at_repetition(
        logits,
        digits,
        groups,
        {"g": 1},
        repetitions,
        aggregation="precision",
        logit_variances=np.asarray([1.0, 1.0, 0.01, 0.01]),
    )
    assert hits[2] == 1.0


def _causal_candidate_dataset() -> EpochDataset:
    one_group_candidates = np.array(
        [1, 2, 1, 3, 4, 2, 1, 3, 4, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        dtype=np.int64,
    )
    n_epochs = 3 * len(one_group_candidates)
    groups = np.repeat(np.array(["g1", "g2", "g3"]), n_epochs // 3)
    candidates = np.tile(one_group_candidates, 3)
    targets = np.repeat(np.array([2, 3, 1], dtype=np.int64), n_epochs // 3)
    repetitions = candidate_repetition_indices(candidates.astype(str), groups)
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
        X=np.zeros(
            (n_epochs, 3, SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.n_times),
            dtype=np.float32,
        ),
        y=(candidates == targets).astype(np.int64),
        subject_ids=groups.astype(str),
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=np.ones(3, dtype=bool),
        preprocessing=preprocessing_spec_from_contract(
            SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
        ),
        event_timeline=timeline,
        metadata=pd.DataFrame({"subject": groups.astype(str)}),
        provenance={"source": "unit_test", "source_reference": "average", "source_sample_rate_hz": 128.0},
    )


def test_causal_prefix_suffix_split_is_chronological_and_complete() -> None:
    dataset = _causal_candidate_dataset()
    split = causal_prefix_suffix_split(dataset, prefix_repetitions=2, test_repetitions=2)

    assert set(split.usable_groups) == {"g1", "g2", "g3"}
    assert split.candidate_vocab == (0, 1, 2, 3)
    assert split.excluded_groups == {}
    assert int(split.prefix_mask.sum()) == int(split.suffix_mask.sum()) == 3 * 4 * 2
    for group in split.usable_groups:
        rows = np.flatnonzero(split.group_ids == group)
        assert int(split.prefix_mask[rows].sum()) == 8
        assert int(split.suffix_mask[rows].sum()) == 8
        suffix_reps = split.suffix_repetition_indices[rows][split.suffix_mask[rows]]
        assert set(np.unique(suffix_reps).tolist()) == {0, 1}
        prefix_onsets = dataset.event_timeline.onset_times_s[rows][split.prefix_mask[rows]]
        suffix_onsets = dataset.event_timeline.onset_times_s[rows][split.suffix_mask[rows]]
        assert float(np.max(prefix_onsets)) < float(np.min(suffix_onsets))
        prefix_available = dataset.event_timeline.evidence_available_times_s[rows][
            split.prefix_mask[rows]
        ]
        suffix_epoch_starts = suffix_onsets + dataset.preprocessing.tmin_ms / 1000.0
        assert float(np.max(prefix_available)) < float(np.min(suffix_epoch_starts))
        prefix_evidence_times = dataset.event_timeline.evidence_available_times_s[rows][
            split.prefix_mask[rows]
        ]
        assert float(np.max(prefix_evidence_times)) < float(np.min(suffix_onsets))


def test_causal_split_supports_session_start_zero_calibration() -> None:
    dataset = _causal_candidate_dataset()
    split = causal_prefix_suffix_split(dataset, prefix_repetitions=0, test_repetitions=2)

    assert not split.prefix_mask.any()
    assert int(split.suffix_mask.sum()) == 3 * 4 * 2
    assert all(record["prefix_available_trials"] == 0 for record in split.evidence_cost_by_group.values())


def test_operational_split_allows_unequal_available_candidate_counts() -> None:
    dataset = _causal_candidate_dataset()
    timeline = dataset.event_timeline
    scheduled_group = np.asarray(timeline.group_ids).astype(str)
    scheduled_candidate = np.asarray(timeline.candidate_ids).astype(str)
    remove_event = np.flatnonzero(
        (scheduled_group == "g1") & (scheduled_candidate == "1")
    )[-1]
    remove_epoch = int(timeline.evidence_indices[remove_event])
    evidence = np.asarray(timeline.evidence_indices, dtype=np.int64).copy()
    evidence[remove_event] = -1
    evidence[evidence > remove_epoch] -= 1
    statuses = np.asarray(timeline.statuses).astype(str).copy()
    details = np.asarray(timeline.status_details).astype(str).copy()
    available_at = np.asarray(timeline.evidence_available_times_s, dtype=float).copy()
    statuses[remove_event] = "missing"
    details[remove_event] = "synthetic_missing_candidate_occurrence"
    available_at[remove_event] = np.nan
    keep = np.arange(dataset.n_epochs) != remove_epoch
    dataset.X = dataset.X[keep]
    dataset.y = dataset.y[keep]
    dataset.subject_ids = dataset.subject_ids[keep]
    dataset.metadata = dataset.metadata.iloc[keep].reset_index(drop=True)
    dataset.event_timeline = replace(
        timeline,
        evidence_indices=evidence,
        statuses=statuses,
        status_details=details,
        evidence_available_times_s=available_at,
    )
    dataset.validate(require_labels=True)

    split = causal_prefix_suffix_split(dataset, prefix_repetitions=2, test_repetitions=2)

    assert "g1" in split.usable_groups
    selected = split.selected_scheduled_repetitions["g1"]["suffix"]
    assert len({len(values) for values in selected.values()}) == 1

    all_split = causal_prefix_suffix_split(
        dataset, prefix_repetitions=0, test_repetitions=None
    )
    all_counts = all_split.evidence_cost_by_group["g1"][
        "suffix_available_trials_by_candidate"
    ]
    assert len(set(all_counts.values())) == 2
    assert all_split.evidence_cost_by_group["g1"]["balanced_all_repetitions"] == 5
    assert set(all_split.evidence_cost_by_repetition["g1"]) == {
        "1",
        "2",
        "3",
        "4",
        "5",
    }


def test_real_time_inner_split_enforces_evidence_embargo() -> None:
    onset = np.arange(15, dtype=float)
    available = onset + 0.8
    y = np.zeros(15, dtype=np.int64)
    y[[1, 4, 7, 10, 13]] = 1

    split = chronological_time_validation_split(
        onset,
        available,
        y,
        epoch_start_offset_s=-0.2,
        min_positive_per_partition=2,
    )

    assert float(np.max(available[split.train_mask])) < float(
        np.min((onset - 0.2)[split.validation_mask])
    )
    assert split.embargo_mask is not None and split.embargo_mask.any()
    assert np.count_nonzero(y[split.train_mask]) >= 2
    assert np.count_nonzero(y[split.validation_mask]) >= 2


def test_causal_prefix_suffix_split_rejects_overlapping_epoch_evidence() -> None:
    dataset = _causal_candidate_dataset()
    dataset.event_timeline = replace(
        dataset.event_timeline,
        evidence_available_times_s=dataset.event_timeline.onset_times_s + 100.0,
    )

    with pytest.raises(ValueError, match="No selection group"):
        causal_prefix_suffix_split(dataset, prefix_repetitions=2, test_repetitions=2)


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


def test_pretrained_trunk_rejects_target_subject_overlap(tmp_path) -> None:
    dataset = _causal_candidate_dataset()
    identity = materialize_dataset_identity(dataset)
    checkpoint = tmp_path / "pretrained.pt"
    training_ledger = identity.subset(["g1"])
    payload = {
        "trunk_state_dict": _dataset_trunk(dataset).state_dict(),
        **_checkpoint_identity_fields(training_ledger, dataset),
        "source_dataset_name": dataset.name,
        "input_channel_names": list(dataset.channel_names),
        "input_preprocessing": asdict(dataset.preprocessing),
        "input_source_reference": dataset.provenance["source_reference"],
        "architecture": _dataset_trunk(dataset).architecture_record(),
        "config": {
            "pooling_mode": "ms_flatten",
            "temporal_kernel_size": 35,
            "training": "supervised",
        },
    }
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="shared source identity"):
        _load_trunk(
            checkpoint,
            dataset,
            target_subject="g1",
            identity_exclusion_policy="source",
        )

    payload.update(
        _checkpoint_identity_fields(_identity_ledger("synthetic", "other"), dataset)
    )
    torch.save(payload, checkpoint)
    loaded = _load_trunk(
        checkpoint,
        dataset,
        target_subject="g1",
        identity_exclusion_policy="source",
    )
    assert isinstance(loaded, N2P3Net)

    payload["input_channel_names"] = list(reversed(dataset.channel_names))
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="input signature disagrees"):
        _load_trunk(
            checkpoint,
            dataset,
            target_subject="g1",
            identity_exclusion_policy="source",
        )


def test_current_checkpoint_rejects_incomplete_architecture() -> None:
    dataset = _causal_candidate_dataset()
    materialize_dataset_identity(dataset)
    trunk = _dataset_trunk(dataset)
    architecture = trunk.architecture_record()
    architecture.pop("st_temporal_kernel_samples")
    payload = {
        "trunk_state_dict": trunk.state_dict(),
        **_checkpoint_identity_fields(_identity_ledger("synthetic", "other"), dataset),
        "input_channel_names": list(dataset.channel_names),
        "input_preprocessing": asdict(dataset.preprocessing),
        "input_source_reference": dataset.provenance["source_reference"],
        "architecture": architecture,
    }
    contract = checkpoint_training_contract(payload)
    contract = replace(contract, architecture=architecture)
    payload["training_contract"] = contract.record()
    payload["training_contract_digest"] = contract.digest()

    with pytest.raises(ValueError, match="lacks current fields"):
        load_n2p3_trunk_checkpoint(payload, dataset, target_subject="g1")


def test_checkpoint_rejects_missing_current_schema() -> None:
    dataset = _causal_candidate_dataset()
    materialize_dataset_identity(dataset)
    payload = {
        "trunk_state_dict": _dataset_trunk(dataset).state_dict(),
        "input_channel_names": list(dataset.channel_names),
        "input_preprocessing": asdict(dataset.preprocessing),
        "input_source_reference": dataset.provenance["source_reference"],
    }

    with pytest.raises(ValueError, match="schema is unsupported"):
        load_n2p3_trunk_checkpoint(payload, dataset, target_subject="g1")


def test_same_local_number_from_unrelated_source_is_not_false_overlap() -> None:
    dataset = _causal_candidate_dataset()
    materialize_dataset_identity(dataset)
    unrelated = _identity_ledger("independent-study", "g1")
    payload = {
        "trunk_state_dict": _dataset_trunk(dataset).state_dict(),
        **_checkpoint_identity_fields(unrelated, dataset),
        "input_channel_names": list(dataset.channel_names),
        "input_preprocessing": asdict(dataset.preprocessing),
        "input_source_reference": dataset.provenance["source_reference"],
        "architecture": _dataset_trunk(dataset).architecture_record(),
        "config": {"pooling_mode": "ms_flatten", "temporal_kernel_size": 35},
    }

    loaded, _ = load_n2p3_trunk_checkpoint(
        payload,
        dataset,
        target_subject="g1",
    )

    assert isinstance(loaded, N2P3Net)


def test_namespaced_derived_target_still_overlaps_its_source_participant() -> None:
    source = _causal_candidate_dataset()
    source_identity = materialize_dataset_identity(source)
    target = namespace_epoch_dataset(source, "DERIVED")
    training = source_identity.subset(["g1"])
    payload = {
        "trunk_state_dict": _dataset_trunk(source).state_dict(),
        **_checkpoint_identity_fields(training, target),
        "input_channel_names": list(target.channel_names),
        "input_preprocessing": asdict(target.preprocessing),
        "input_source_reference": target.provenance["source_reference"],
        "architecture": _dataset_trunk(source).architecture_record(),
        "config": {"pooling_mode": "ms_flatten", "temporal_kernel_size": 35},
    }

    with pytest.raises(ValueError, match="shared source identity"):
        load_n2p3_trunk_checkpoint(
            payload,
            target,
            target_subject="DERIVED::g1",
        )


def test_explicit_global_identity_rejects_cross_source_same_person() -> None:
    dataset = _causal_candidate_dataset()
    source_table = materialize_dataset_identity(dataset)
    records = []
    for record in source_table.records:
        if record.local_subject_id == "g1":
            records.append(
                ParticipantIdentityRecord(
                    local_subject_id="g1",
                    origin_subject_keys=record.origin_subject_keys,
                    global_person_keys=("registry:person-1",),
                    identity_status="global_verified",
                )
            )
        else:
            records.append(record)
    dataset.identity_table = DatasetIdentityTable(records=tuple(records))
    training = _identity_ledger(
        "independent-study",
        "different-local-id",
        global_key="registry:person-1",
    )
    payload = {
        "trunk_state_dict": _dataset_trunk(dataset).state_dict(),
        **_checkpoint_identity_fields(training, dataset),
        "input_channel_names": list(dataset.channel_names),
        "input_preprocessing": asdict(dataset.preprocessing),
        "input_source_reference": dataset.provenance["source_reference"],
        "architecture": _dataset_trunk(dataset).architecture_record(),
        "config": {"pooling_mode": "ms_flatten", "temporal_kernel_size": 35},
    }

    with pytest.raises(ValueError, match="shared global identity"):
        load_n2p3_trunk_checkpoint(payload, dataset, target_subject="g1")

    loaded, _ = load_n2p3_trunk_checkpoint(
        payload,
        dataset,
        target_subject="g1",
        identity_exclusion_policy="source",
    )
    assert isinstance(loaded, N2P3Net)


def test_checkpoint_identity_digest_tampering_fails_closed() -> None:
    dataset = _causal_candidate_dataset()
    materialize_dataset_identity(dataset)
    training = _identity_ledger("independent-study", "other")
    payload = {
        "trunk_state_dict": _dataset_trunk(dataset).state_dict(),
        **_checkpoint_identity_fields(training, dataset),
    }
    payload["training_identity_ledger_digest"] = "0" * 64

    with pytest.raises(ValueError, match="digest disagrees"):
        load_n2p3_trunk_checkpoint(payload, dataset, target_subject="g1")


def test_session_start_zero_shot_runner_never_requires_prefix_training(
    tmp_path, monkeypatch
) -> None:
    dataset = _causal_candidate_dataset()
    cache = save_epoch_dataset(tmp_path / "causal.npz", dataset)
    trunk = _dataset_trunk(dataset)
    checkpoint = tmp_path / "supervised.pt"
    source_manifest, source_digest, _ = _write_source_snapshot_manifest(tmp_path)
    torch.save(
        {
            "trunk_state_dict": trunk.state_dict(),
            **_checkpoint_identity_fields(
                _identity_ledger("synthetic", "other"),
                dataset,
                source_snapshot_sha256=source_digest,
            ),
            "holdout_subjects": ["g1", "g2", "g3"],
            "source_dataset_name": dataset.name,
            "input_channel_names": list(dataset.channel_names),
            "input_preprocessing": asdict(dataset.preprocessing),
            "input_source_reference": dataset.provenance["source_reference"],
            "config": {"pooling_mode": "ms_flatten", "training": "supervised"},
            "classifier_trained": True,
            "input_mean": [0.0, 0.0, 0.0],
            "input_std": [1.0, 1.0, 1.0],
            "source_calibration": {
                "pos_weight": 3.0,
                "train_prior": 0.25,
                "temperature": 1.0,
                "source": "fixture_weighted_ce_analytic",
            },
            "architecture": trunk.architecture_record(),
            "n_channels": 3,
            "n_times": dataset.n_times,
            "input_sample_rate_hz": dataset.preprocessing.sfreq,
            "input_tmin_s": -0.2,
        },
        checkpoint,
    )
    output = tmp_path / "zero-shot.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_within_subject_transfer.py",
            "--dataset-cache",
            str(cache),
            "--checkpoint",
            str(checkpoint),
            "--arm-name",
            "zero_shot_source",
            "--source-snapshot-manifest",
            str(source_manifest),
            "--identity-exclusion-policy",
            "source",
            "--head",
            "zero_shot",
            "--test-reps",
            "2",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )

    run_transfer_main()

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["prefix_reps"] == 0
    assert record["estimand"] == "target_excluded_session_start_zero_calibration"
    assert record["n_groups"] == 3
    assert record["source_snapshot_manifest"] == str(source_manifest.resolve())
    assert record["source_snapshot_sha256"] == source_digest
    assert record["evaluation_contract"]["source_snapshot_sha256"] == source_digest

    all_output = tmp_path / "zero-shot-all.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_within_subject_transfer.py",
            "--dataset-cache",
            str(cache),
            "--checkpoint",
            str(checkpoint),
            "--arm-name",
            "zero_shot_source_all",
            "--source-snapshot-manifest",
            str(source_manifest),
            "--identity-exclusion-policy",
            "source",
            "--head",
            "zero_shot",
            "--test-reps",
            "all",
            "--device",
            "cpu",
            "--output",
            str(all_output),
        ],
    )
    run_transfer_main()
    all_record = json.loads(all_output.read_text(encoding="utf-8"))
    assert all_record["test_reps"] == "all"
    assert all_record["test_repetition_mode"] == "all_post_boundary"
    assert all(item["n_suffix"] == 24 for item in all_record["records"])
    assert set(all_record["hit_mean_by_repetition"]) == {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    }
    assert "raw_hit_at_all" in all_record
    assert "hit_at_all_balanced" in all_record

    target_file = tmp_path / "targets.json"
    target_file.write_text('["g1", "absent-subject"]', encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_within_subject_transfer.py",
            "--dataset-cache",
            str(cache),
            "--checkpoint",
            str(checkpoint),
            "--arm-name",
            "invalid_target_manifest",
            "--source-snapshot-manifest",
            str(source_manifest),
            "--identity-exclusion-policy",
            "source",
            "--target-subjects-file",
            str(target_file),
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit):
        run_transfer_main()


def test_transfer_runner_rejects_checkpoint_snapshot_mismatch(
    tmp_path, monkeypatch
) -> None:
    dataset = _causal_candidate_dataset()
    cache = save_epoch_dataset(tmp_path / "causal.npz", dataset)
    source_manifest, _, _ = _write_source_snapshot_manifest(tmp_path)
    checkpoint = tmp_path / "mismatched.pt"
    torch.save(
        {
            "trunk_state_dict": _dataset_trunk(dataset).state_dict(),
            **_checkpoint_identity_fields(
                _identity_ledger("synthetic", "other"), dataset
            ),
        },
        checkpoint,
    )
    output = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_within_subject_transfer.py",
            "--dataset-cache",
            str(cache),
            "--checkpoint",
            str(checkpoint),
            "--arm-name",
            "snapshot_mismatch",
            "--source-snapshot-manifest",
            str(source_manifest),
            "--identity-exclusion-policy",
            "source",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(ValueError, match="verified physical source freeze"):
        run_transfer_main()
    assert not output.exists()


def test_transfer_runner_rejects_legacy_naked_snapshot_digest(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_within_subject_transfer.py",
            "--dataset-cache",
            str(tmp_path / "unused.npz"),
            "--arm-name",
            "legacy_digest",
            "--source-snapshot-sha256",
            "a" * 64,
            "--identity-exclusion-policy",
            "source",
            "--output",
            str(tmp_path / "unused.json"),
        ],
    )

    with pytest.raises(SystemExit):
        run_transfer_main()


def test_target_subject_file_rejects_ambiguous_offset_or_limit(
    tmp_path, monkeypatch
) -> None:
    source_manifest, _, _ = _write_source_snapshot_manifest(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_within_subject_transfer.py",
            "--dataset-cache",
            str(tmp_path / "unused.npz"),
            "--arm-name",
            "ambiguous_scope",
            "--source-snapshot-manifest",
            str(source_manifest),
            "--identity-exclusion-policy",
            "source",
            "--target-subjects-file",
            str(tmp_path / "targets.json"),
            "--max-subjects",
            "1",
            "--output",
            str(tmp_path / "unused.json"),
        ],
    )

    with pytest.raises(SystemExit):
        run_transfer_main()
