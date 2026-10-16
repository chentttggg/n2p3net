from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from data.domain import namespace_epoch_dataset
from data.epochs import (
    materialize_dataset_identity,
)
from data.identity import DatasetIdentityTable
from experiments.run_within_subject_transfer import (
    _load_trunk,
)
from models.adapters import source_domain_axis_record
from models.n2p3net import N2P3Net
from research.contracts import semantic_sha256
from tests.support.candidate import causal_candidate_dataset
from tests.support.checkpoints import (
    checkpoint_identity_fields,
    dataset_trunk,
    participant_identity_table,
    small_trunk,
)
from transfer.checkpoint import (
    checkpoint_architecture_record,
    checkpoint_classifier_is_trained,
    checkpoint_input_stats,
    checkpoint_scores_to_llr,
    checkpoint_training_contract,
    load_n2p3_trunk_checkpoint,
    predict_n2p3_checkpoint,
    predict_n2p3_checkpoint_with_features,
)


def test_checkpoint_prediction_with_features_matches_unchanged_scores() -> None:
    trunk = small_trunk()
    x = np.random.default_rng(4).normal(size=(2, 3, 128)).astype(np.float32)
    stats = (
        np.zeros((1, 3, 1), dtype=np.float32),
        np.ones((1, 3, 1), dtype=np.float32),
    )
    scores, features = predict_n2p3_checkpoint_with_features(
        trunk,
        x,
        input_stats=stats,
        device=torch.device("cpu"),
    )
    expected = predict_n2p3_checkpoint(
        trunk,
        x,
        input_stats=stats,
        device=torch.device("cpu"),
    )
    assert features is not None and features.ndim == 2
    assert features.shape[0] == len(x)
    assert np.allclose(scores, expected)
    assert np.isfinite(features).all()


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


def test_current_checkpoint_helpers_reject_implicit_defaults() -> None:
    with pytest.raises(ValueError, match="explicit input_mean"):
        checkpoint_input_stats({}, 3)
    with pytest.raises(ValueError, match="explicit boolean"):
        checkpoint_classifier_is_trained({"config": {"training": "supervised"}})
    architecture = small_trunk().architecture_record()
    with pytest.raises(ValueError, match="required input geometry"):
        checkpoint_architecture_record({"architecture": architecture})
    with pytest.raises(ValueError, match="requires a source_calibration mapping"):
        checkpoint_scores_to_llr({}, np.asarray([0.0, 1.0]))


def test_pretrained_trunk_rejects_target_subject_overlap(tmp_path) -> None:
    dataset = causal_candidate_dataset()
    identity = materialize_dataset_identity(dataset)
    checkpoint = tmp_path / "pretrained.pt"
    training_ledger = identity.subset(["g1"])
    payload = {
        "trunk_state_dict": dataset_trunk(dataset).state_dict(),
        **checkpoint_identity_fields(training_ledger, dataset),
        "source_dataset_name": dataset.name,
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

    payload.update(checkpoint_identity_fields(participant_identity_table("synthetic", "other"), dataset))
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
    dataset = causal_candidate_dataset()
    materialize_dataset_identity(dataset)
    trunk = dataset_trunk(dataset)
    architecture = trunk.architecture_record()
    architecture.pop("st_temporal_kernel_samples")
    payload = {
        "trunk_state_dict": trunk.state_dict(),
        **checkpoint_identity_fields(participant_identity_table("synthetic", "other"), dataset),
        "architecture": architecture,
    }
    contract = checkpoint_training_contract(payload)
    contract = replace(contract, architecture=architecture)
    payload["training_contract"] = contract.record()
    payload["training_contract_digest"] = contract.digest()

    with pytest.raises(ValueError, match="lacks current fields"):
        load_n2p3_trunk_checkpoint(payload, dataset, target_subject="g1")


def test_checkpoint_rejects_splitbn_architecture_with_pooled_bn_weights() -> None:
    dataset = causal_candidate_dataset()
    materialize_dataset_identity(dataset)
    pooled = dataset_trunk(dataset)
    split = N2P3Net(
        n_channels=dataset.n_channels,
        n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq,
        tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
        pooling_mode="ms_flatten",
        splitbn_domain_count=2,
        splitbn_target_buffer_source="pooled_source",
    )
    axis_record = source_domain_axis_record(("dataset-a", "dataset-b"))
    axis_digest = semantic_sha256(axis_record)
    architecture = {
        **split.architecture_record(),
        "source_domain_axis_digest": axis_digest,
    }
    payload = {
        "trunk_state_dict": pooled.state_dict(),
        **checkpoint_identity_fields(participant_identity_table("independent-study", "other"), dataset),
        "architecture": architecture,
        "source_domain_axis": axis_record,
        "source_domain_axis_digest": axis_digest,
    }
    contract = checkpoint_training_contract(payload)
    contract = replace(
        contract,
        architecture=architecture,
        objective={**contract.objective, "source_domain_axis": axis_record},
    )
    payload["training_contract"] = contract.record()
    payload["training_contract_digest"] = contract.digest()

    with pytest.raises(ValueError, match="weights do not match its declared architecture"):
        load_n2p3_trunk_checkpoint(payload, dataset, target_subject="g1")


def test_checkpoint_rejects_missing_current_schema() -> None:
    dataset = causal_candidate_dataset()
    materialize_dataset_identity(dataset)
    payload = {
        "trunk_state_dict": dataset_trunk(dataset).state_dict(),
        "input_channel_names": list(dataset.channel_names),
        "input_preprocessing": asdict(dataset.preprocessing),
        "input_source_reference": dataset.provenance["source_reference"],
    }

    with pytest.raises(ValueError, match="schema is unsupported"):
        load_n2p3_trunk_checkpoint(payload, dataset, target_subject="g1")


def test_same_local_number_from_unrelated_source_is_not_false_overlap() -> None:
    dataset = causal_candidate_dataset()
    materialize_dataset_identity(dataset)
    unrelated = participant_identity_table("independent-study", "g1")
    payload = {
        "trunk_state_dict": dataset_trunk(dataset).state_dict(),
        **checkpoint_identity_fields(unrelated, dataset),
        "config": {"pooling_mode": "ms_flatten", "temporal_kernel_size": 35},
    }

    loaded, _ = load_n2p3_trunk_checkpoint(
        payload,
        dataset,
        target_subject="g1",
    )

    assert isinstance(loaded, N2P3Net)


def test_namespaced_derived_target_still_overlaps_its_source_participant() -> None:
    source = causal_candidate_dataset()
    source_identity = materialize_dataset_identity(source)
    target = namespace_epoch_dataset(source, "DERIVED")
    training = source_identity.subset(["g1"])
    payload = {
        "trunk_state_dict": dataset_trunk(source).state_dict(),
        **checkpoint_identity_fields(training, target),
        "architecture": dataset_trunk(source).architecture_record(),
        "config": {"pooling_mode": "ms_flatten", "temporal_kernel_size": 35},
    }

    with pytest.raises(ValueError, match="shared source identity"):
        load_n2p3_trunk_checkpoint(
            payload,
            target,
            target_subject="DERIVED::g1",
        )


def test_explicit_global_identity_rejects_cross_source_same_person() -> None:
    dataset = causal_candidate_dataset()
    source_table = materialize_dataset_identity(dataset)
    records = {record.local_subject_id: record for record in source_table.records}
    records["g1"] = replace(
        records["g1"], global_person_keys=("registry:person-1",), identity_status="global_verified"
    )
    dataset.identity_table = DatasetIdentityTable(records=tuple(records.values()))
    training = participant_identity_table(
        "independent-study",
        "different-local-id",
        global_key="registry:person-1",
    )
    payload = {
        "trunk_state_dict": dataset_trunk(dataset).state_dict(),
        **checkpoint_identity_fields(training, dataset),
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
    dataset = causal_candidate_dataset()
    materialize_dataset_identity(dataset)
    training = participant_identity_table("independent-study", "other")
    payload = {
        "trunk_state_dict": dataset_trunk(dataset).state_dict(),
        **checkpoint_identity_fields(training, dataset),
    }
    payload["training_identity_ledger_digest"] = "0" * 64

    with pytest.raises(ValueError, match="digest disagrees"):
        load_n2p3_trunk_checkpoint(payload, dataset, target_subject="g1")
