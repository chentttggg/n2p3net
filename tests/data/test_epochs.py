from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data.epochs as epochs_module
from data.contract import (
    CAUSAL_IIR_INITIAL_STATE,
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
    assert_causal_p300_input_contract,
    assert_default_p300_input_contract,
)
from data.epochs import (
    EpochDataset,
    PreprocessingSpec,
    bind_named_preprocessing_contract,
    concatenate_epoch_datasets,
    load_epoch_dataset,
    loaded_epoch_cache_attestation,
    materialize_dataset_identity,
    preprocessing_spec_from_contract,
    save_epoch_dataset,
    select_epoch_channels,
    write_epoch_dataset_record,
)
from data.identity import DatasetIdentityTable
from tests.support.datasets import epoch_dataset
from tests.support.io import read_json, write_json
from tests.support.preparation_cli import set_preparation_cli


def _substituted_cache_pair(tmp_path):
    canonical = save_epoch_dataset(
        tmp_path / "canonical.npz",
        epoch_dataset(),
        compressed=False,
    )
    with np.load(canonical, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    payload["X"] = payload["X"].copy()
    payload["X"][0, 0, 0] += np.float32(0.123)
    malicious = tmp_path / "malicious.npz"
    np.savez(malicious, **payload)
    assert malicious.stat().st_size == canonical.stat().st_size
    return canonical, malicious


def _named_contract_source() -> EpochDataset:
    from data.domain import adapt_common_channel_average_reference

    dataset = epoch_dataset()
    preprocessing = replace(
        preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT),
        name="producer_equivalent_causal_profile",
    )
    dataset.preprocessing = preprocessing
    dataset.X = np.arange(
        dataset.n_epochs * dataset.n_channels * preprocessing.n_times,
        dtype=np.float32,
    ).reshape(dataset.n_epochs, dataset.n_channels, preprocessing.n_times) * np.float32(1e-10)
    dataset.event_timeline = replace(dataset.event_timeline, online_causal=True)
    dataset.validate(require_labels=True)
    return adapt_common_channel_average_reference(
        dataset,
        dataset.channel_names,
        name="producer_equivalent_causal_car",
    )


def test_epoch_dataset_safe_round_trip(tmp_path) -> None:
    source = epoch_dataset()
    path = save_epoch_dataset(tmp_path / "epochs.npz", source)
    loaded = load_epoch_dataset(
        path,
        expected_preprocessing=source.preprocessing,
        require_labels=True,
    )
    assert loaded.name == source.name
    assert loaded.channel_names == source.channel_names
    assert np.array_equal(loaded.X, source.X)
    assert np.array_equal(loaded.y, source.y)
    assert np.allclose(loaded.channel_positions_m, source.channel_positions_m)
    assert loaded.metadata.to_dict("list") == source.metadata.to_dict("list")
    assert loaded.provenance == source.provenance
    assert np.array_equal(loaded.event_timeline.candidate_ids, source.event_timeline.candidate_ids)
    assert loaded.event_timeline.supports_full_candidate_chain is True
    assert loaded.identity_table is not None
    assert loaded.identity_table.payload() == source.identity_table.payload()


@pytest.mark.parametrize("legacy", [False, True])
def test_metadata_utf8_and_legacy_cache_preserve_decoded_identity(tmp_path, legacy):
    source = epoch_dataset()
    source.metadata["unicode_note"] = ["中文 🧠 " + "a" * 1024] * source.n_epochs
    path = save_epoch_dataset(tmp_path / "metadata.npz", source, compressed=False)
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    assert payload["metadata_json"].dtype == np.uint8
    assert payload["metadata_json"].ndim == 1
    if legacy:
        payload["metadata_json"] = np.asarray(payload["metadata_json"].tobytes().decode("utf-8"))
        np.savez(path, **payload)
        write_epoch_dataset_record(path, source)
    loaded = load_epoch_dataset(path, validation="attested")
    pd.testing.assert_frame_equal(loaded.metadata, source.metadata)
    assert epochs_module._decoded_dataset_sha256(loaded) == epochs_module._decoded_dataset_sha256(source)


def test_saved_cache_embeds_decoded_dataset_sha256_and_attested_load_reuses_it(
    tmp_path,
) -> None:
    source = epoch_dataset()
    path = save_epoch_dataset(tmp_path / "embedded-sha.npz", source)
    expected = epochs_module._decoded_dataset_sha256(source)
    with np.load(path, allow_pickle=False) as archive:
        embedded = str(np.asarray(archive["decoded_dataset_sha256"]).item())
    assert embedded == expected

    loaded = load_epoch_dataset(path, require_labels=True, validation="attested")
    attestation = loaded_epoch_cache_attestation(loaded)
    assert attestation["decoded_dataset_sha256"] == embedded
    assert attestation["decoded_dataset_sha256"] == expected


def test_legacy_cache_without_embedded_decoded_sha_recomputes_it(tmp_path) -> None:
    source = epoch_dataset()
    canonical = save_epoch_dataset(tmp_path / "canonical.npz", source)
    legacy = tmp_path / "legacy.npz"
    with np.load(canonical, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    payload.pop("decoded_dataset_sha256")
    np.savez(legacy, **payload)
    write_epoch_dataset_record(legacy, source)

    loaded = load_epoch_dataset(legacy, require_labels=True, validation="attested")
    assert loaded_epoch_cache_attestation(loaded)["decoded_dataset_sha256"] == (
        epochs_module._decoded_dataset_sha256(source)
    )


def test_attested_load_rejects_malformed_embedded_decoded_sha(tmp_path) -> None:
    source = epoch_dataset()
    canonical = save_epoch_dataset(tmp_path / "canonical.npz", source)
    malformed = tmp_path / "malformed.npz"
    with np.load(canonical, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    payload["decoded_dataset_sha256"] = np.asarray("not-a-sha256")
    np.savez(malformed, **payload)
    write_epoch_dataset_record(malformed, source)

    with pytest.raises(ValueError, match="decoded_dataset_sha256 is not a valid SHA-256"):
        load_epoch_dataset(malformed, require_labels=True, validation="attested")


def test_attested_load_rejects_a_substituted_physical_object(tmp_path: Path, monkeypatch) -> None:
    canonical, malicious = _substituted_cache_pair(tmp_path)

    real_open = epochs_module.os.open

    def substitute_open(path, flags, *args):
        selected = malicious if Path(path) == canonical else path
        return real_open(selected, flags, *args)

    monkeypatch.setattr(epochs_module.os, "open", substitute_open)

    with pytest.raises(ValueError, match="SHA-256"):
        load_epoch_dataset(canonical, validation="attested")
    assert not list(tmp_path.glob(".n2p3-attested-cache-*"))


def test_attested_load_decodes_private_snapshot_when_path_is_swapped_after_hash(
    tmp_path: Path, monkeypatch
) -> None:
    canonical, malicious = _substituted_cache_pair(tmp_path)

    original_load = epochs_module.np.load
    attack_executed = False

    def swap_path_then_decode(snapshot, *args, **kwargs):
        nonlocal attack_executed
        canonical.write_bytes(malicious.read_bytes())
        attack_executed = True
        return original_load(snapshot, *args, **kwargs)

    monkeypatch.setattr(epochs_module.np, "load", swap_path_then_decode)
    loaded = load_epoch_dataset(canonical, validation="attested")

    assert attack_executed is True
    assert loaded.X[0, 0, 0] == epoch_dataset().X[0, 0, 0]
    assert canonical.read_bytes() == malicious.read_bytes()


def test_attested_load_fails_closed_when_snapshot_disk_space_is_insufficient(
    tmp_path: Path, monkeypatch
) -> None:
    canonical = save_epoch_dataset(tmp_path / "canonical.npz", epoch_dataset())

    class NoFreeSpace:
        free = 0

    monkeypatch.setattr(epochs_module.shutil, "disk_usage", lambda _path: NoFreeSpace())

    with pytest.raises(ValueError, match="insufficient free space"):
        load_epoch_dataset(canonical, validation="attested")


def test_named_contract_binding_preserves_content_and_binds_attestation(
    tmp_path: Path,
) -> None:
    from data.domain import ensure_common_channel_average_reference

    source_path = save_epoch_dataset(
        tmp_path / "source.npz",
        _named_contract_source(),
        compressed=False,
    )
    source = load_epoch_dataset(source_path, require_labels=True, validation="attested")
    source_attestation = loaded_epoch_cache_attestation(source)
    source_lineage_head = source.lineage.entity_digest
    source_timeline_fingerprint = source.event_timeline.fingerprint()
    source_identity = source.identity_table.payload()
    source_metadata = source.metadata.copy()
    source_X = source.X.copy()
    source_y = source.y.copy()

    with pytest.raises(TypeError):
        source.verified_cache_attestation["sha256"] = "0" * 64

    rebound = bind_named_preprocessing_contract(
        source,
        SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
        name="canonical-causal-car",
    )
    second_name = bind_named_preprocessing_contract(
        source,
        SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
        name="canonical-causal-car-2",
    )

    assert rebound.preprocessing == preprocessing_spec_from_contract(
        SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
    )
    assert rebound.verified_cache_attestation is None
    assert rebound.X is source.X
    assert rebound.y is source.y
    assert rebound.event_timeline is source.event_timeline
    assert np.array_equal(rebound.X, source_X)
    assert rebound.X.tobytes() == source_X.tobytes()
    assert np.array_equal(rebound.y, source_y)
    assert rebound.event_timeline.fingerprint() == source_timeline_fingerprint
    assert rebound.event_timeline.supports_full_candidate_chain is True
    pd.testing.assert_frame_equal(rebound.metadata, source_metadata)
    assert rebound.identity_table.payload() == source_identity

    binding = rebound.provenance["preprocessing_contract_binding"]
    assert binding["source_cache_attestation"] == source_attestation
    assert binding["content_invariants"]["event_timeline_fingerprint"] == (
        source_timeline_fingerprint
    )
    assert set(binding["content_invariants"]["qc_features"]) == {
        "schema",
        "relative_ptp_sha256",
        "channel_std_v_sha256",
        "epoch_scale_v_sha256",
        "observed_mask_sha256",
    }
    assert binding["metadata_updates"] == {
        "domain_adapter.preprocessing_identity": {
            "from": "producer_equivalent_causal_profile",
            "to": SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.name,
        }
    }
    assert rebound.provenance["domain_adapter"]["preprocessing_identity"] == (
        SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.name
    )
    assert source.provenance["domain_adapter"]["preprocessing_identity"] == (
        "producer_equivalent_causal_profile"
    )
    assert rebound.lineage.entities[-1].parent_entity_digests == (source_lineage_head,)
    assert dict(rebound.lineage.entities[-1].parameters) == binding
    assert rebound.lineage.entity_digest != second_name.lineage.entity_digest
    assert ensure_common_channel_average_reference(rebound, rebound.channel_names) is rebound

    output_path = save_epoch_dataset(tmp_path / "bound.npz", rebound)
    loaded = load_epoch_dataset(output_path, require_labels=True, validation="attested")
    output_attestation = loaded_epoch_cache_attestation(loaded)
    assert output_attestation["sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert loaded.X.tobytes() == source_X.tobytes()
    assert np.array_equal(loaded.y, source_y)
    assert loaded.event_timeline.fingerprint() == source_timeline_fingerprint
    pd.testing.assert_frame_equal(loaded.metadata, source_metadata)
    assert loaded.identity_table.payload() == source_identity
    assert loaded.provenance["preprocessing_contract_binding"] == binding
    assert not list(tmp_path.glob(".n2p3-attested-cache-*"))


def test_named_contract_binding_rejects_an_existing_canonical_name(tmp_path: Path) -> None:
    dataset = _named_contract_source()
    dataset.preprocessing = preprocessing_spec_from_contract(
        SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
    )
    source = load_epoch_dataset(
        save_epoch_dataset(tmp_path / "canonical.npz", dataset),
        validation="attested",
    )

    with pytest.raises(ValueError, match="already uses"):
        bind_named_preprocessing_contract(
            source,
            SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
        )


def test_named_contract_binding_does_not_silently_ignore_an_empty_output_name(
    tmp_path: Path,
) -> None:
    source = load_epoch_dataset(
        save_epoch_dataset(tmp_path / "source.npz", _named_contract_source()),
        validation="attested",
    )

    with pytest.raises(ValueError, match="name must be a non-empty"):
        bind_named_preprocessing_contract(
            source,
            SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
            name="",
        )


def test_named_contract_binding_rejects_content_mutated_after_attested_load(
    tmp_path: Path,
) -> None:
    source = load_epoch_dataset(
        save_epoch_dataset(tmp_path / "source.npz", _named_contract_source()),
        validation="attested",
    )
    source.X[0, 0, 0] += np.float32(1e-4)

    with pytest.raises(ValueError, match="content changed after"):
        bind_named_preprocessing_contract(
            source,
            SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
        )


@pytest.mark.parametrize(
    ("field_name", "different_value", "reported_field"),
    [
        ("sample_rate_hz", 129.0, "sfreq"),
        ("l_freq", 0.2, "l_freq"),
        ("h_freq", 29.0, "h_freq"),
        ("tmin_ms", -100.0, "tmin_ms"),
        ("tmax_ms", 1100.0, "tmax_ms"),
        ("baseline_mode", "none", "baseline_mode"),
        ("signal_unit", "uV", "signal_unit"),
        ("filter_method", "fir", "filter_method"),
        ("filter_order", 2, "filter_order"),
        ("filter_phase", "zero", "filter_phase"),
        ("causal_iir_initial_state", "not_applicable", "causal_iir_initial_state"),
        ("resample_domain", "continuous", "resample_domain"),
        ("resample_method", "polyphase", "resample_method"),
        ("resample_npad", "0", "resample_npad"),
        ("resample_window", "boxcar", "resample_window"),
        ("resample_pad", "reflect", "resample_pad"),
    ],
)
def test_named_contract_binding_rejects_every_physical_contract_difference(
    tmp_path: Path,
    field_name: str,
    different_value: object,
    reported_field: str,
) -> None:
    source = load_epoch_dataset(
        save_epoch_dataset(tmp_path / "source.npz", _named_contract_source()),
        validation="attested",
    )
    target = replace(
        SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
        **{field_name: different_value},
    )

    with pytest.raises(ValueError, match=reported_field):
        bind_named_preprocessing_contract(source, target)


@pytest.mark.parametrize(
    ("field_name", "different_value"),
    [
        ("trial_reference_window_ms", (-200.0, 0.0)),
        ("trial_reference_center", "median"),
        ("trial_reference_scale", "std"),
        ("reject_threshold_v", 1e-4),
    ],
)
def test_named_contract_binding_rejects_extra_preprocessing_physical_differences(
    tmp_path: Path,
    field_name: str,
    different_value: object,
) -> None:
    dataset = _named_contract_source()
    dataset.preprocessing = replace(
        dataset.preprocessing,
        **{field_name: different_value},
    )
    source = load_epoch_dataset(
        save_epoch_dataset(tmp_path / "source.npz", dataset),
        validation="attested",
    )

    with pytest.raises(ValueError, match=field_name):
        bind_named_preprocessing_contract(
            source,
            SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
        )


def test_bind_preprocessing_contract_cli_is_canonical_and_non_destructive(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from experiments.bind_preprocessing_contract import main

    source_path = save_epoch_dataset(tmp_path / "source.npz", _named_contract_source())
    output_path = tmp_path / "bound.npz"
    set_preparation_cli(
        monkeypatch,
        "bind_preprocessing_contract.py",
        dataset_cache=source_path,
        contract="causal",
        output=output_path,
    )
    main()

    assert "[contract-bound]" in capsys.readouterr().out
    assert source_path.is_file()
    loaded = load_epoch_dataset(output_path, validation="attested")
    assert loaded.preprocessing.name == SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.name
    assert (
        loaded.provenance["preprocessing_contract_binding"]["source_cache_attestation"]["sha256"]
        == loaded_epoch_cache_attestation(load_epoch_dataset(source_path, validation="attested"))[
            "sha256"
        ]
    )

    set_preparation_cli(
        monkeypatch,
        "bind_preprocessing_contract.py",
        dataset_cache=source_path,
        contract="causal",
        output=source_path,
    )
    with pytest.raises(ValueError, match="must not overwrite"):
        main()

    existing_output = tmp_path / "existing.npz"
    existing_output.write_bytes(b"do-not-replace")
    set_preparation_cli(
        monkeypatch,
        "bind_preprocessing_contract.py",
        dataset_cache=source_path,
        contract="causal",
        output=existing_output,
    )
    with pytest.raises(FileExistsError, match="requires a new output path"):
        main()
    assert existing_output.read_bytes() == b"do-not-replace"

    set_preparation_cli(
        monkeypatch,
        "bind_preprocessing_contract.py",
        dataset_cache=source_path,
        contract="invented-contract",
        output=tmp_path / "invalid.npz",
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_attested_cache_load_skips_repeated_full_contract_scan(tmp_path, monkeypatch) -> None:
    path = save_epoch_dataset(tmp_path / "epochs.npz", epoch_dataset())
    assert path.with_suffix(".record.json").is_file()

    def fail_full_validation(*_args, **_kwargs):
        raise AssertionError("attested load must not repeat the full dataset scan")

    monkeypatch.setattr(EpochDataset, "validate", fail_full_validation)
    loaded = load_epoch_dataset(path, require_labels=True, validation="attested")

    assert loaded.X.shape == epoch_dataset().X.shape


def test_attested_cache_rejects_missing_record(tmp_path) -> None:
    path = save_epoch_dataset(tmp_path / "epochs.npz", epoch_dataset())
    path.with_suffix(".record.json").unlink()

    with pytest.raises(ValueError, match="lacks a cache attestation"):
        load_epoch_dataset(path, validation="attested")


@pytest.mark.parametrize(
    "section,field,message",
    [("cache_attestation", "sha256", "SHA-256"), ("events", "fingerprint", "events")],
)
def test_attested_cache_rejects_sidecar_tampering(tmp_path, section, field, message) -> None:
    path = save_epoch_dataset(tmp_path / "epochs.npz", epoch_dataset())
    record_path = path.with_suffix(".record.json")
    record = read_json(record_path)
    record[section][field] = "0" * 64
    write_json(record_path, record)

    with pytest.raises(ValueError, match=message):
        load_epoch_dataset(path, validation="attested")


def test_attestation_can_reuse_a_completed_full_validation(tmp_path, monkeypatch) -> None:
    path = save_epoch_dataset(tmp_path / "epochs.npz", epoch_dataset())
    dataset = load_epoch_dataset(path, require_labels=True)

    def fail_full_validation(*_args, **_kwargs):
        raise AssertionError("the full validation was already completed")

    monkeypatch.setattr(EpochDataset, "validate", fail_full_validation)
    record = write_epoch_dataset_record(path, dataset, already_validated=True)

    assert record.is_file()


def test_epoch_dataset_decompresses_signal_member_once(tmp_path, monkeypatch) -> None:
    from collections import Counter

    from numpy.lib.npyio import NpzFile

    path = save_epoch_dataset(tmp_path / "epochs.npz", epoch_dataset())
    original_getitem = NpzFile.__getitem__
    calls = Counter()

    def counted_getitem(archive, key):
        calls[key] += 1
        return original_getitem(archive, key)

    monkeypatch.setattr(NpzFile, "__getitem__", counted_getitem)
    loaded = load_epoch_dataset(path, require_labels=True)

    assert loaded.X.shape == epoch_dataset().X.shape
    assert calls["X"] == 1


def test_epoch_loader_rejects_every_retired_schema(tmp_path) -> None:
    source = epoch_dataset()
    current = save_epoch_dataset(tmp_path / "current.npz", source)
    with np.load(current, allow_pickle=False) as archive:
        payload = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key
            not in {
                "event_candidate_ids",
                "event_target_candidate_ids",
                "event_repetition_indices",
            }
        }
    for retired_schema in (
        "n2p3net_epoch_dataset/2",
        "n2p3net_epoch_dataset/3",
        "n2p3net_epoch_dataset/4",
    ):
        payload["schema"] = np.asarray(retired_schema)
        retired = tmp_path / f"retired_{retired_schema.rsplit('/', 1)[-1]}.npz"
        np.savez(retired, **payload)
        with pytest.raises(ValueError, match="Unsupported EpochDataset schema"):
            load_epoch_dataset(retired)


def test_current_cache_rejects_identity_digest_tampering(tmp_path) -> None:
    current = save_epoch_dataset(tmp_path / "current.npz", epoch_dataset())
    with np.load(current, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    identity = json.loads(str(payload["identity_json"].item()))
    identity["records"][0]["origin_subject_keys"] = ["forged-origin"]
    payload["identity_json"] = np.asarray(json.dumps(identity))
    tampered = tmp_path / "tampered.npz"
    np.savez(tampered, **payload)

    with pytest.raises(ValueError, match="identity digest"):
        load_epoch_dataset(tampered)


def test_identity_materialization_requires_producer_source_dataset_ids() -> None:
    dataset = epoch_dataset()
    dataset.event_timeline = replace(
        dataset.event_timeline,
        dataset_ids=np.repeat("", dataset.event_timeline.n_events),
    )

    with pytest.raises(ValueError, match="source_dataset_id must be non-empty"):
        materialize_dataset_identity(dataset)


def test_identity_table_must_cover_exactly_the_epoch_subjects() -> None:
    dataset = epoch_dataset()
    dataset.identity_table = DatasetIdentityTable.from_source_rows(["s1"], ["source"])

    with pytest.raises(ValueError, match="cover exactly"):
        dataset.validate()


def test_epoch_dataset_rejects_candidate_truth_label_disagreement() -> None:
    dataset = epoch_dataset()
    dataset.y[0] = 1

    with pytest.raises(ValueError, match="candidate_id == target_candidate_id"):
        dataset.validate()


def test_candidate_decision_support_is_derived_from_event_contract() -> None:
    dataset = epoch_dataset()
    assert dataset.event_timeline.supports_full_candidate_chain is True

    dataset.event_timeline = replace(dataset.event_timeline, complete=False)
    assert dataset.event_timeline.supports_full_candidate_chain is False


def test_epoch_dataset_trial_masks_and_qc_features_round_trip(tmp_path) -> None:
    source = epoch_dataset()
    source.preprocessing = replace(source.preprocessing, baseline_mode="none")
    source.X = np.random.default_rng(44).normal(size=source.X.shape).astype(np.float32) * 1e-6
    trial_mask = np.ones(source.X.shape[:2], dtype=bool)
    trial_mask[1, source.channel_names.index("FZ")] = False
    trial_mask[4, source.channel_names.index("PZ")] = False
    source.trial_channel_mask = trial_mask
    source.X[~trial_mask] = 0.0
    loaded = load_epoch_dataset(save_epoch_dataset(tmp_path / "masked.npz", source), require_labels=True)
    np.testing.assert_array_equal(loaded.trial_channel_mask, trial_mask)
    assert loaded.record()["has_trial_channel_mask"] is True
    assert source.qc_features is not None and loaded.qc_features is not None
    for field in ("relative_ptp", "channel_std_v", "epoch_scale_v"):
        np.testing.assert_allclose(getattr(loaded.qc_features, field), getattr(source.qc_features, field))
    np.testing.assert_array_equal(loaded.qc_features.observed_mask, trial_mask)

def test_epoch_dataset_rejects_nonzero_trial_masked_channel() -> None:
    dataset = epoch_dataset()
    dataset.trial_channel_mask = np.ones(dataset.X.shape[:2], dtype=bool)
    dataset.trial_channel_mask[0, 1] = False

    with pytest.raises(ValueError, match="Trial-masked channels"):
        dataset.validate()


def test_epoch_dataset_rejects_legacy_npz(tmp_path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez(path, X=np.zeros((2, 3, 100), dtype=np.float32), y=np.array([0, 1]))
    with pytest.raises(ValueError, match="lacks EpochDataset fields"):
        load_epoch_dataset(path)


def test_epoch_dataset_rejects_nonphysical_time_axis() -> None:
    dataset = epoch_dataset()
    dataset.preprocessing = replace(dataset.preprocessing, n_times=20)
    with pytest.raises(ValueError, match="Physical time axis"):
        dataset.validate()


def test_preprocessing_trial_reference_accepts_stimulus_locked_window() -> None:
    profile = PreprocessingSpec(
        name="stimulus_locked_trial_reference",
        sfreq=256.0,
        l_freq=None,
        tmin_ms=0.0,
        tmax_ms=1000.0,
        n_times=256,
        baseline_mode="trial_reference",
        trial_reference_window_ms=(0.0, 50.0),
        trial_reference_center="median",
        trial_reference_scale="none",
        reject_threshold_v=None,
    )
    profile.validate()
    assert profile.trial_reference_window_ms == (0.0, 50.0)


def test_preprocessing_rejects_dimensionless_trial_scaling_before_physical_qc() -> None:
    with pytest.raises(ValueError, match="destroys the volts contract"):
        PreprocessingSpec(baseline_mode="trial").validate()
    with pytest.raises(ValueError, match="destroys the volts contract"):
        PreprocessingSpec(
            baseline_mode="trial_reference",
            trial_reference_window_ms=(-200.0, 0.0),
            trial_reference_scale="mad",
        ).validate()


def test_trial_scale_normalization_can_erase_erp_amplitude_information() -> None:
    """Counterexample motivating the volts-preserving ingress restriction."""

    base = np.array([-1.0, 1.0, 2.0, 4.0])
    doubled = 2.0 * base
    normalized_base = (base - base[:2].mean()) / base[:2].std()
    normalized_doubled = (doubled - doubled[:2].mean()) / doubled[:2].std()

    assert np.ptp(doubled) == 2.0 * np.ptp(base)
    np.testing.assert_allclose(normalized_base, normalized_doubled)


def test_preprocessing_trial_reference_rejects_window_outside_epoch() -> None:
    profile = PreprocessingSpec(
        sfreq=250.0,
        tmin_ms=0.0,
        tmax_ms=1000.0,
        n_times=250,
        baseline_mode="trial_reference",
        trial_reference_window_ms=(-1.0, 50.0),
    )
    with pytest.raises(ValueError, match="inside the physical epoch"):
        profile.validate()


def test_select_epoch_channels_is_exact_and_ordered() -> None:
    dataset = epoch_dataset()
    identity_before = materialize_dataset_identity(dataset).payload()
    selected = select_epoch_channels(dataset, ("Pz", "Fz"))
    assert selected.channel_names == ("PZ", "FZ")
    assert np.array_equal(selected.X, dataset.X[:, [2, 0], :])
    assert selected.identity_table is not None
    assert selected.identity_table.payload() == identity_before
    with pytest.raises(ValueError, match="cannot supply channels"):
        select_epoch_channels(dataset, ("Pz", "Oz"))


def test_epoch_dataset_rejects_nonfinite_signal() -> None:
    dataset = epoch_dataset()
    dataset.X[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        dataset.validate()


def test_epoch_dataset_rejects_non_integer_labels() -> None:
    dataset = epoch_dataset()
    dataset.y = dataset.y.astype(np.float32)
    with pytest.raises(ValueError, match="integer dtype"):
        dataset.validate()


def test_epoch_dataset_rejects_integer_signal_and_masks() -> None:
    dataset = epoch_dataset()
    dataset.X = dataset.X.astype(np.int16)
    with pytest.raises(ValueError, match="floating-point dtype"):
        dataset.validate()

    dataset = epoch_dataset()
    dataset.channel_mask = dataset.channel_mask.astype(np.int8)
    with pytest.raises(ValueError, match="channel_mask must have boolean dtype"):
        dataset.validate()


def test_channel_selection_cannot_hide_invalid_source_data() -> None:
    dataset = epoch_dataset()
    dataset.channel_mask[1] = False
    with pytest.raises(ValueError, match="Absent channels"):
        select_epoch_channels(dataset, ("Fz", "Pz"))


def test_preprocessing_rejects_invalid_filter_contract() -> None:
    with pytest.raises(ValueError, match="smaller than h_freq"):
        PreprocessingSpec(l_freq=40.0, h_freq=20.0).validate()
    with pytest.raises(ValueError, match="Nyquist"):
        PreprocessingSpec(h_freq=200.0).validate()
    with pytest.raises(ValueError, match="Nyquist"):
        PreprocessingSpec(h_freq=128.0).validate()
    with pytest.raises(ValueError, match="Nyquist"):
        PreprocessingSpec(l_freq=128.0, h_freq=None).validate()
    with pytest.raises(ValueError, match="must be numeric"):
        PreprocessingSpec(sfreq="256").validate()


def test_default_preprocessing_matches_ms_eegnet_physical_scales() -> None:
    profile = PreprocessingSpec()

    assert profile.name == "p300_ms_eegnet_input_v3"
    assert (profile.sfreq, profile.l_freq, profile.h_freq) == (128.0, 0.1, 30.0)
    assert (profile.tmin_ms, profile.tmax_ms, profile.n_times) == (-200.0, 1200.0, 179)
    assert profile.baseline_mode == "mean_only"
    assert profile.signal_unit == "V"
    assert profile.resample_method == "fft"
    assert profile.resample_npad == "auto"
    assert profile.resample_window == "auto"
    assert profile.resample_pad == "edge"


def test_causal_single_subject_contract_accepts_forward_phase_only() -> None:
    profile = PreprocessingSpec()
    with pytest.raises(ValueError, match="forward"):
        assert_causal_p300_input_contract(profile)

    causal = preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)
    causal.validate()
    assert_causal_p300_input_contract(causal)


def test_epoch_dataset_rejects_mismatched_causal_declaration() -> None:
    dataset = epoch_dataset()
    dataset.preprocessing = PreprocessingSpec(
        name=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.name,
        sfreq=100.0,
        tmin_ms=-200.0,
        tmax_ms=800.0,
        n_times=100,
        filter_phase="forward",
        causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
    )
    with pytest.raises(ValueError, match="online_causal"):
        dataset.validate()

    causal_timeline = replace(dataset.event_timeline, online_causal=True)
    dataset.event_timeline = causal_timeline
    dataset.validate()


def test_mainline_contract_rejects_the_retired_250_hz_cache_geometry() -> None:
    assert_default_p300_input_contract(PreprocessingSpec())
    retired = PreprocessingSpec(
        name="retired_250_hz",
        sfreq=250.0,
        l_freq=0.1,
        h_freq=None,
        tmin_ms=-200.0,
        tmax_ms=800.0,
        n_times=250,
        baseline_mode="none",
    )

    with pytest.raises(ValueError, match="Regenerate the cache"):
        assert_default_p300_input_contract(retired)


def test_concatenation_rejects_unharmonized_source_references() -> None:
    left = epoch_dataset()
    right = epoch_dataset()
    left.provenance["source_reference"] = "right earlobe"
    right.provenance["source_reference"] = "nose"

    with pytest.raises(ValueError, match="common re-reference"):
        concatenate_epoch_datasets((left, right), name="invalid_transfer")


def test_reference_choice_creates_a_common_mode_dataset_confound() -> None:
    """Counterexample: identical scalp potentials differ under two references."""

    scalp = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
    reference_a = np.array([0.5, 1.0, 1.5])
    reference_b = np.array([-1.0, -2.0, -3.0])
    observed_a = scalp - reference_a[None, :]
    observed_b = scalp - reference_b[None, :]

    assert not np.allclose(observed_a, observed_b)
    car_a = observed_a - observed_a.mean(axis=0, keepdims=True)
    car_b = observed_b - observed_b.mean(axis=0, keepdims=True)
    np.testing.assert_allclose(car_a, car_b)


def test_epoch_dataset_rejects_unit_sphere_values_mislabeled_as_metres() -> None:
    dataset = epoch_dataset()
    dataset.channel_positions_m *= 10.0
    with pytest.raises(ValueError, match="registered head-frame coordinates in metres"):
        dataset.validate()


def test_explicit_readonly_same_channel_view_avoids_matrix_copy():
    source = epoch_dataset()
    default = select_epoch_channels(source, source.channel_names)
    shared = select_epoch_channels(source, source.channel_names, copy=False)
    assert not np.shares_memory(default.X, source.X)
    assert np.shares_memory(shared.X, source.X)
    assert not shared.X.flags.writeable and source.X.flags.writeable
    np.testing.assert_array_equal(shared.X, default.X)
    assert shared.lineage.payload() == default.lineage.payload()
