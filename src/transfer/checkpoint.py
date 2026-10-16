"""Auditable loading and validation for N2P3 transfer checkpoints.

Transfer checkpoints are executable model contracts, not just collections of
weights.  This module keeps architecture reconstruction, target-subject
exclusion, physical-shape checks, and input-statistics handling in one place so
the GTN and BI runners cannot silently disagree about what was pretrained.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from data.epochs import (
    PreprocessingSpec,
    preprocessing_execution_binding,
    preprocessing_execution_identity,
)
from data.exclusions import assert_identity_participants_allowed
from data.identity import (
    DatasetIdentityTable,
    IdentityExclusionPolicy,
    assert_target_identity_excluded,
    source_dataset_ids_from_identity,
)
from models.adapters import (
    FEATURE_TRANSFER_SCHEMA,
    SOURCE_RESIDUAL_LEDGER_SCHEMA,
    SOURCE_RESIDUAL_R_MAX,
    FeatureDomainTransfer,
    FeatureTransferConfig,
    feature_transfer_architecture_record,
    feature_transfer_config_from_record,
    source_domain_axis_from_record,
)
from models.n2p3net import (
    POOLING_MODES,
    N2P3ArchitectureConfig,
    N2P3Net,
)
from models.splitbn import SplitBNConfig, splitbn_config_from_record
from research.contracts import TrainingRunContract, semantic_sha256
from transfer.decision_loss import DecisionLossConfig

CHECKPOINT_SCHEMA = "n2p3_transfer_checkpoint/2"


def load_checkpoint_payload(path: str | Path) -> dict[str, object]:
    """Load a checkpoint payload and require the mapping format used by runners."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a mapping checkpoint payload.")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema is unsupported; regenerate the checkpoint.")
    ledger = payload.get("training_identity_ledger")
    if isinstance(ledger, Mapping):
        assert_identity_participants_allowed(DatasetIdentityTable.from_payload(ledger))
    return payload


def checkpoint_training_contract(
    payload: Mapping[str, object],
) -> TrainingRunContract:
    """Validate and return the semantic training contract bound to a checkpoint."""

    record = payload.get("training_contract")
    if not isinstance(record, Mapping):
        raise ValueError("checkpoint lacks a training_contract mapping.")
    try:
        contract = TrainingRunContract(**dict(record))
    except (TypeError, ValueError) as error:
        raise ValueError(f"checkpoint training_contract is invalid: {error}") from error
    if payload.get("training_contract_digest") != contract.digest():
        raise ValueError("checkpoint training_contract_digest disagrees with its record.")
    from transfer.source_objective import validate_source_objective_contract

    validate_source_objective_contract(dict(contract.optimizer), dict(contract.objective))
    return contract


def checkpoint_training_identity_ledger(
    payload: Mapping[str, object],
) -> DatasetIdentityTable:
    """Validate and return the participant ledger bound to a checkpoint."""

    training_contract = checkpoint_training_contract(payload)
    ledger_payload = payload.get("training_identity_ledger")
    if not isinstance(ledger_payload, Mapping):
        raise ValueError("pretraining checkpoint lacks a structured training_identity_ledger.")
    ledger = DatasetIdentityTable.from_payload(ledger_payload)
    assert_identity_participants_allowed(ledger)
    if payload.get("training_identity_ledger_digest") != ledger.digest():
        raise ValueError("checkpoint training identity ledger digest disagrees with its records.")
    if set(training_contract.training_participant_keys) != set(ledger.authority_keys()):
        raise ValueError("training_contract participant keys disagree with the identity ledger.")
    membership = payload.get("source_study_membership")
    declared_membership = training_contract.objective.get("source_study_membership")
    if membership is not None or declared_membership is not None:
        if membership is None or declared_membership is None or (
            semantic_sha256(membership) != semantic_sha256(declared_membership)
        ):
            raise ValueError("checkpoint study membership disagrees with its bound training contract")
        from data.study import study_ids_for_identity

        study_ids_for_identity(ledger, membership)
    return ledger


def assert_checkpoint_library_exclusion(
    payload: Mapping[str, object], target_identity: DatasetIdentityTable, *,
    excluded_source_datasets: tuple[str, ...] = (),
    source_membership: Mapping[str, object] | None = None,
    target_membership: Mapping[str, object] | None = None,
) -> None:
    """Reject the entire target library, including people absent from its cache."""
    training = checkpoint_training_identity_ledger(payload)
    if source_membership is None and target_membership is None:
        actual_source = set(source_dataset_ids_from_identity(training))
        target_studies = set(source_dataset_ids_from_identity(target_identity))
    elif source_membership is None or target_membership is None:
        raise ValueError("whole-study exclusion requires both source and target membership proofs")
    else:
        from data.study import study_ids_for_identity

        actual_source = set(study_ids_for_identity(training, source_membership))
        target_studies = set(study_ids_for_identity(target_identity, target_membership))
    forbidden = target_studies | set(excluded_source_datasets)
    overlap = actual_source & forbidden
    if overlap:
        raise ValueError(f"checkpoint source contains excluded target libraries: {sorted(overlap)}")


def assert_checkpoint_target_identity_excluded(
    payload: Mapping[str, object],
    dataset: object,
    *,
    target_subject: str,
    identity_exclusion_policy: IdentityExclusionPolicy = "source_or_global",
) -> None:
    """Require one requested target identity to be absent from checkpoint training."""

    target_identity_table = getattr(dataset, "identity_table", None)
    if not isinstance(target_identity_table, DatasetIdentityTable):
        raise ValueError("target dataset lacks a structured identity table; regenerate its cache.")
    target_identity = target_identity_table.record_for(str(target_subject))
    assert_target_identity_excluded(
        checkpoint_training_identity_ledger(payload),
        target_identity,
        policy=identity_exclusion_policy,
    )


def _architecture_from_payload(
    payload: Mapping[str, object],
) -> tuple[
    N2P3ArchitectureConfig,
    str,
    SplitBNConfig | None,
    FeatureTransferConfig | None,
    dict[str, object] | None,
    dict[str, object],
]:
    """Resolve the complete architecture record required by the current schema."""

    record = payload.get("architecture")
    if not isinstance(record, Mapping):
        raise ValueError("checkpoint architecture must be a mapping.")
    splitbn_config = splitbn_config_from_record(record.get("normalization"))
    pooling = record.get("pooling_mode")
    if not isinstance(pooling, str) or pooling not in POOLING_MODES:
        raise ValueError(f"checkpoint pooling_mode must be one of {sorted(POOLING_MODES)}.")

    values: dict[str, object] = {}
    aliases = {
        "temporal_filters": "st_temporal_filters",
        "temporal_kernel_size": "st_temporal_kernel_samples",
        "st_temporal_dilation": "st_temporal_dilation",
        "spatial_depth_multiplier": "spatial_depth_multiplier",
        "st_pool_size": "st_pool_size",
        "mst_kernel_sizes": "mst_kernel_samples",
        "mst_dilations": "mst_dilations",
        "mst_features_per_scale": "mst_features_per_scale",
        "mst_pool_size": "mst_pool_size",
        "dropout": "dropout",
        "spatial_max_norm": "spatial_max_norm",
    }
    if pooling == "quadratic_full_unfold":
        aliases["interaction_rank"] = "interaction_rank"
    if pooling == "mlp_full_unfold":
        aliases["mlp_hidden_features"] = "mlp_hidden_features"
    missing = sorted(source for source in aliases.values() if source not in record)
    if missing:
        raise ValueError(f"checkpoint architecture lacks current fields {missing}.")
    for destination, source in aliases.items():
        values[destination] = record[source]
    if "channel_masking" in record:
        if record["channel_masking"] != "observed_temporal_bn/1":
            raise ValueError("checkpoint channel masking is unsupported.")
        values["observed_channel_norm"] = True
    if "feature_residual" in record:
        values["feature_residual"] = record["feature_residual"]

    if "mst_kernel_sizes" in values:
        values["mst_kernel_sizes"] = tuple(int(item) for item in values["mst_kernel_sizes"])
    if "mst_dilations" in values:
        values["mst_dilations"] = tuple(int(item) for item in values["mst_dilations"])
    try:
        architecture = N2P3ArchitectureConfig(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint architecture is invalid: {exc}") from exc

    pooling_kwargs: dict[str, object] = {}
    if pooling == "latency_marginal_contrast":
        required_pooling = (
            "evidence_window_ms",
            "reference_window_ms",
            "latency_offsets_ms",
            "latency_temperature",
        )
        missing_pooling = [name for name in required_pooling if name not in record]
        if missing_pooling:
            raise ValueError(
                "checkpoint latency pooling lacks current fields "
                f"{missing_pooling}."
            )
        try:
            pooling_kwargs = {
                "evidence_window_ms": tuple(
                    float(value) for value in record["evidence_window_ms"]
                ),
                "reference_window_ms": tuple(
                    float(value) for value in record["reference_window_ms"]
                ),
                "latency_offsets_ms": tuple(
                    float(value) for value in record["latency_offsets_ms"]
                ),
                "latency_temperature": float(record["latency_temperature"]),
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"checkpoint latency pooling contract is invalid: {exc}"
            ) from exc

    if "domain_adapters" in record:
        raise ValueError(
            "checkpoint uses the retired source-domain adapter bank; regenerate it "
            "with the feature_transfer architecture."
        )
    transfer_record = record.get("feature_transfer")
    transfer_config = None
    if transfer_record is not None:
        if not isinstance(transfer_record, Mapping):
            raise ValueError("checkpoint feature_transfer must be a mapping.")
        if transfer_record.get("schema") != FEATURE_TRANSFER_SCHEMA:
            raise ValueError("checkpoint feature_transfer schema is unsupported.")
        try:
            transfer_config = feature_transfer_config_from_record(transfer_record)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"checkpoint feature_transfer config is invalid: {exc}") from exc
        feature_channels = (
            len(architecture.mst_kernel_sizes) * architecture.mst_features_per_scale
        )
        if dict(transfer_record) != feature_transfer_architecture_record(
            transfer_config,
            feature_channels=feature_channels,
        ):
            raise ValueError(
                "checkpoint feature_transfer wrapper contract is incomplete or inconsistent."
            )

    axis_record = _model_source_domain_axis_from_payload(
        payload,
        architecture_record=record,
        splitbn_config=splitbn_config,
        transfer_config=transfer_config,
    )
    return (
        architecture,
        pooling,
        splitbn_config,
        transfer_config,
        axis_record,
        pooling_kwargs,
    )


def _model_source_domain_axis_from_payload(
    payload: Mapping[str, object],
    *,
    architecture_record: Mapping[str, object],
    splitbn_config: SplitBNConfig | None,
    transfer_config: FeatureTransferConfig | None,
) -> dict[str, object] | None:
    indexed_counts = tuple(
        count
        for count in (
            None if splitbn_config is None else splitbn_config.domain_count,
            None if transfer_config is None else transfer_config.source_domain_count,
            None if transfer_config is None else transfer_config.private_domain_count,
        )
        if count is not None and count > 0
    )
    axis_payload = payload.get("source_domain_axis")
    payload_digest = payload.get("source_domain_axis_digest")
    architecture_digest = architecture_record.get("source_domain_axis_digest")
    if not indexed_counts:
        if (
            axis_payload is not None
            or payload_digest is not None
            or architecture_digest is not None
        ):
            raise ValueError(
                "checkpoint has a model source-domain axis without an indexed mechanism."
            )
        return None
    if not isinstance(axis_payload, Mapping):
        raise ValueError("checkpoint indexed model lacks a source_domain_axis mapping.")
    try:
        vocabulary, weights_by_domain = source_domain_axis_from_record(axis_payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint source_domain_axis is invalid: {exc}") from exc
    axis_record = dict(axis_payload)
    digest = semantic_sha256(axis_record)
    if payload_digest != digest or architecture_digest != digest:
        raise ValueError(
            "checkpoint source_domain_axis digest is absent or disagrees with its records."
        )
    if any(count != len(vocabulary) for count in indexed_counts):
        raise ValueError(
            "checkpoint indexed domain counts disagree with source_domain_axis."
        )
    expected_weights = None if splitbn_config is None else splitbn_config.source_weights
    if expected_weights is None:
        if weights_by_domain is not None:
            raise ValueError(
                "checkpoint source_domain_axis contains weights unused by its SplitBN contract."
            )
    else:
        if weights_by_domain is None:
            raise ValueError(
                "checkpoint source_weighted SplitBN lacks domain-keyed source weights."
            )
        positional_weights = tuple(weights_by_domain[domain] for domain in vocabulary)
        if positional_weights != expected_weights:
            raise ValueError(
                "checkpoint keyed source weights disagree with SplitBN buffer order."
            )
    return axis_record


def _validate_metadata(
    payload: Mapping[str, object],
    dataset: object,
    *,
    target_subject: str | None,
    identity_exclusion_policy: IdentityExclusionPolicy,
) -> None:
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema is unsupported; regenerate the checkpoint.")
    state = payload.get("trunk_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint lacks a non-empty trunk_state_dict mapping.")

    training_contract = checkpoint_training_contract(payload)
    if payload.get("source_cache_sha256") != training_contract.source_cache_sha256:
        raise ValueError("checkpoint source_cache_sha256 disagrees with its training contract.")
    contract_domain_axis = training_contract.objective.get("source_domain_axis")
    payload_domain_axis = payload.get("source_domain_axis")
    if semantic_sha256(contract_domain_axis) != semantic_sha256(payload_domain_axis):
        raise ValueError(
            "checkpoint source_domain_axis disagrees with its training contract."
        )
    contract_residual = training_contract.objective.get(
        "source_residual_diagnostics"
    )
    if contract_residual is not None:
        if not isinstance(contract_residual, Mapping):
            raise ValueError(
                "training contract source_residual_diagnostics must be a mapping."
            )
        durable_residual = {
            "selection": payload.get("source_residual_selection"),
            "refit": payload.get("source_residual_refit"),
        }
        if semantic_sha256(contract_residual) != semantic_sha256(durable_residual):
            raise ValueError(
                "checkpoint source residual ledger disagrees with its training contract."
            )
        architecture_record = payload.get("architecture")
        transfer_record = (
            architecture_record.get("feature_transfer")
            if isinstance(architecture_record, Mapping)
            else None
        )
        source_residual_expected = False
        if isinstance(transfer_record, Mapping):
            try:
                source_residual_expected = (
                    feature_transfer_config_from_record(transfer_record).source_domain_count
                    >= 2
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"checkpoint feature_transfer config is invalid: {error}"
                ) from error
        for phase, ledger in durable_residual.items():
            if ledger is None:
                if source_residual_expected:
                    raise ValueError(
                        f"checkpoint {phase} source residual ledger is missing."
                    )
                continue
            if not isinstance(ledger, Mapping):
                raise ValueError(
                    f"checkpoint {phase} source residual ledger must be a mapping."
                )
            if ledger.get("schema") != SOURCE_RESIDUAL_LEDGER_SCHEMA:
                raise ValueError(
                    f"checkpoint {phase} source residual ledger has an unsupported schema."
                )
            active = ledger.get("active")
            if not isinstance(active, bool):
                raise ValueError(
                    f"checkpoint {phase} source residual ledger lacks an active flag."
                )
            if source_residual_expected and active is not True:
                raise ValueError(
                    f"checkpoint {phase} source residual ledger is inactive."
                )
            if active and not source_residual_expected:
                raise ValueError(
                    f"checkpoint {phase} source residual ledger is active but architecture disables it."
                )
            if active:
                maximum = ledger.get("run_max_ratio")
                if (
                    isinstance(maximum, (bool, np.bool_))
                    or not isinstance(maximum, (int, float, np.integer, np.floating))
                    or not np.isfinite(float(maximum))
                    or float(maximum) > SOURCE_RESIDUAL_R_MAX
                ):
                    raise ValueError(
                        f"checkpoint {phase} source residual maximum ratio is invalid."
                    )
                if ledger.get("status") != "valid" or ledger.get("valid") is not True:
                    raise ValueError(
                        f"checkpoint {phase} source residual ledger is not a valid completed fit."
                    )
            elif ledger.get("status") != "disabled" or ledger.get("valid") is not True:
                raise ValueError(
                    f"checkpoint {phase} disabled source residual ledger is inconsistent."
                )
    elif (
        "source_residual_selection" in payload
        or "source_residual_refit" in payload
    ):
        raise ValueError(
            "checkpoint source residual ledger is absent from its training contract."
        )
    contract_decision_loss = training_contract.objective.get("decision_loss")
    payload_decision_loss = payload.get("decision_loss")
    if semantic_sha256(contract_decision_loss) != semantic_sha256(
        payload_decision_loss
    ):
        raise ValueError("checkpoint decision loss disagrees with its training contract.")
    if payload_decision_loss is not None:
        try:
            decision_config = DecisionLossConfig.from_record(payload_decision_loss)
        except (TypeError, ValueError) as error:
            raise ValueError(f"checkpoint decision loss is invalid: {error}") from error
        if decision_config.record() != dict(payload_decision_loss):
            raise ValueError("checkpoint decision loss record is non-canonical.")
        decision_metadata = payload.get("decision_metadata")
        if not isinstance(decision_metadata, Mapping):
            raise ValueError("checkpoint decision loss lacks direct decision metadata.")
        if payload.get("decision_metadata_digest") != semantic_sha256(
            decision_metadata
        ):
            raise ValueError("checkpoint direct decision metadata digest disagrees.")
        if semantic_sha256(
            training_contract.objective.get("decision_metadata")
        ) != semantic_sha256(decision_metadata):
            raise ValueError(
                "checkpoint direct decision metadata disagrees with its training contract."
            )
        durable_decision_diagnostics = {
            "selection": payload.get("decision_loss_selection"),
            "refit": payload.get("decision_loss_refit"),
        }
        if semantic_sha256(
            training_contract.objective.get("decision_loss_diagnostics")
        ) != semantic_sha256(durable_decision_diagnostics):
            raise ValueError(
                "checkpoint decision loss ledger disagrees with its training contract."
            )
        for phase, ledger in durable_decision_diagnostics.items():
            if not isinstance(ledger, Mapping):
                raise ValueError(f"checkpoint {phase} decision ledger must be a mapping.")
            if ledger.get("status") != "valid" or ledger.get("valid") is not True:
                raise ValueError(
                    f"checkpoint {phase} decision ledger is not a valid completed fit."
                )
    elif any(
        payload.get(name) is not None
        for name in (
            "decision_metadata",
            "decision_metadata_digest",
            "decision_loss_selection",
            "decision_loss_refit",
        )
    ):
        raise ValueError("checkpoint carries decision metadata without a decision loss contract.")
    training_identity_ledger = checkpoint_training_identity_ledger(payload)

    architecture_record = payload.get("architecture")
    if not isinstance(architecture_record, Mapping) or semantic_sha256(
        architecture_record
    ) != semantic_sha256(training_contract.architecture):
        raise ValueError("checkpoint architecture disagrees with its training contract.")
    contract_preprocessing = training_contract.preprocessing
    execution_binding = preprocessing_execution_binding(
        payload.get("input_preprocessing_execution"), PreprocessingSpec(**payload.get("input_preprocessing", {})))
    input_signature = {
        **execution_binding,
        "epoch": payload.get("input_preprocessing"),
        "channel_names": payload.get("input_channel_names"),
        "source_reference": payload.get("input_source_reference"),
    }
    if semantic_sha256(input_signature) != semantic_sha256(contract_preprocessing):
        raise ValueError("checkpoint input signature disagrees with its training contract.")
    expected_execution = execution_binding.get("execution_identity")
    if expected_execution is not None and preprocessing_execution_identity(dataset) != expected_execution:
        raise ValueError("checkpoint requires compatible preprocessing execution evidence")
    checkpoint_classifier_is_trained(payload)
    input_mean, input_std = checkpoint_input_stats(payload, int(dataset.n_channels))
    optimizer_config = training_contract.optimizer.get("config", {})
    if isinstance(optimizer_config, Mapping) and optimizer_config.get("standardize_input") is False:
        if np.any(input_mean != 0) or np.any(input_std != 1):
            raise ValueError("checkpoint with disabled standardization requires an identity input transform")

    if target_subject is not None:
        target = str(target_subject)
        target_identity_table = getattr(dataset, "identity_table", None)
        if not isinstance(target_identity_table, DatasetIdentityTable):
            raise ValueError(
                "target dataset lacks a structured identity table; regenerate its cache."
            )
        target_identity = target_identity_table.record_for(target)
        assert_target_identity_excluded(
            training_identity_ledger,
            target_identity,
            policy=identity_exclusion_policy,
        )

    expected = {
        "n_channels": getattr(dataset, "n_channels", None),
        "n_times": getattr(dataset, "n_times", None),
    }
    for key, value in expected.items():
        declared = payload.get(key)
        if declared is None:
            raise ValueError(f"checkpoint lacks required {key}.")
        if int(declared) != int(value):
            raise ValueError(
                f"checkpoint {key}={declared!r} does not match target dataset {value!r}."
            )
    declared_rate = payload.get("input_sample_rate_hz")
    if declared_rate is None:
        raise ValueError("checkpoint lacks required input_sample_rate_hz.")
    if not np.isclose(
        float(declared_rate), float(dataset.preprocessing.sfreq), rtol=0.0, atol=1e-9
    ):
        raise ValueError("checkpoint input_sample_rate_hz does not match the target dataset.")
    declared_tmin = payload.get("input_tmin_s")
    if declared_tmin is None:
        raise ValueError("checkpoint lacks required input_tmin_s.")
    if not np.isclose(
        float(declared_tmin), float(dataset.preprocessing.tmin_ms) / 1000.0, rtol=0.0, atol=1e-9
    ):
        raise ValueError("checkpoint input_tmin_s does not match the target dataset.")
    declared_channels = payload.get("input_channel_names")
    declared_preprocessing = payload.get("input_preprocessing")
    declared_reference = payload.get("input_source_reference")
    if declared_channels is None or declared_preprocessing is None:
        raise ValueError(
            "checkpoint lacks the ordered channel and preprocessing signature; regenerate it."
        )
    if not isinstance(declared_channels, list) or not all(
        isinstance(value, str) for value in declared_channels
    ):
        raise ValueError("checkpoint input_channel_names must be a list of strings.")
    if not isinstance(declared_preprocessing, Mapping):
        raise ValueError("checkpoint input_preprocessing must be a mapping.")
    target_channels = list(getattr(dataset, "channel_names", ()))
    target_preprocessing = dataset.preprocessing.record()
    required_fields = set(target_preprocessing) - {"notch_freqs_hz", "notch_quality_factor"}
    if not required_fields <= set(declared_preprocessing):
        raise ValueError("checkpoint preprocessing lacks required physical fields")
    if declared_preprocessing.get("notch_freqs_hz") and "notch_quality_factor" not in declared_preprocessing:
        raise ValueError("checkpoint with notch filtering must declare notch_quality_factor")
    target_reference = getattr(dataset, "provenance", {}).get("source_reference")
    input_mismatches = []
    if declared_channels != target_channels:
        input_mismatches.append("channel_names/order")
    declared_spec = PreprocessingSpec(**declared_preprocessing)
    declared_spec.validate()
    if declared_spec.record() != target_preprocessing:
        input_mismatches.append("preprocessing")
    if declared_reference != target_reference:
        input_mismatches.append("source_reference")
    if input_mismatches:
        raise ValueError(
            "checkpoint input domain differs from the target in "
            + ", ".join(input_mismatches)
            + "; use an explicit stem/re-reference/domain-shift adapter."
        )


def checkpoint_input_stats(
    payload: Mapping[str, object],
    n_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return required channel statistics in ``(1,C,1)`` form."""

    mean_value = payload.get("input_mean")
    std_value = payload.get("input_std")
    if mean_value is None or std_value is None:
        raise ValueError("checkpoint requires explicit input_mean and input_std.")

    def normalize(value: object, name: str, *, positive: bool = False) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 3 and array.shape == (1, n_channels, 1):
            array = array.reshape(n_channels)
        elif array.ndim != 1 or array.shape != (n_channels,):
            raise ValueError(
                f"checkpoint {name} must have shape ({n_channels},) or (1,{n_channels},1)."
            )
        if not np.isfinite(array).all() or (positive and np.any(array <= 0.0)):
            raise ValueError(f"checkpoint {name} contains invalid values.")
        return array.reshape(1, n_channels, 1)

    return normalize(mean_value, "input_mean"), normalize(std_value, "input_std", positive=True)


def checkpoint_classifier_is_trained(payload: Mapping[str, object]) -> bool:
    """Whether the checkpoint classifier received supervised target labels."""

    declared = payload.get("classifier_trained")
    if not isinstance(declared, (bool, np.bool_)):
        raise ValueError("checkpoint classifier_trained must be an explicit boolean.")
    return bool(declared)


def checkpoint_scores_to_llr(
    payload: Mapping[str, object], logits: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply a positive-slope source calibration without changing score order.

    A weighted binary cross-entropy classifier estimates a posterior shifted by
    ``log(pos_weight)``.  Removing that shift and the source class prior yields
    an LLR.  An optional source-validation temperature is constrained positive,
    so equal-count candidate argmax decisions are exactly invariant.
    """

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("checkpoint logits must be a finite one-dimensional array.")
    obsolete = [key for key in ("training_pos_weight", "training_prior") if key in payload]
    if obsolete:
        raise ValueError(
            f"checkpoint uses obsolete top-level calibration fields {obsolete}; "
            "store calibration only in source_calibration."
        )
    calibration = payload.get("source_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("checkpoint requires a source_calibration mapping.")
    required = ("pos_weight", "train_prior", "temperature", "source")
    missing = [key for key in required if key not in calibration]
    if missing:
        raise ValueError(f"checkpoint source_calibration lacks required keys {missing}.")
    numeric_values = [calibration[key] for key in required[:3]]
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        for value in numeric_values
    ):
        raise ValueError("checkpoint source_calibration numeric fields are invalid.")
    pos_weight, train_prior, temperature = map(float, numeric_values)
    source = calibration["source"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError("checkpoint source_calibration source must be a non-empty string.")
    if not np.isfinite([pos_weight, train_prior, temperature]).all():
        raise ValueError("checkpoint calibration contains NaN/inf.")
    if pos_weight <= 0.0 or not 0.0 < train_prior < 1.0 or temperature <= 0.0:
        raise ValueError("checkpoint calibration has invalid weight/prior/temperature.")
    offset = float(np.log(pos_weight) + np.log(train_prior / (1.0 - train_prior)))
    return (values - offset) / temperature, {
        "mode": "weighted_ce_llr",
        "source": source,
        "pos_weight": pos_weight,
        "train_prior": train_prior,
        "temperature": temperature,
        "offset": offset,
        "order_preserving": True,
    }


def _checkpoint_prediction_input(
    trunk: N2P3Net | FeatureDomainTransfer,
    X: np.ndarray,
    *,
    input_stats: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    trial_channel_mask: np.ndarray | None,
    eval_buffer_source: str | None,
    eval_buffer_index: int | None,
) -> tuple[np.ndarray, str | None, np.ndarray]:
    """Validate one unchanged-checkpoint suffix and return standardized input."""
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (trunk.n_channels, trunk.n_times):
        raise ValueError(f"checkpoint input must be (N,{trunk.n_channels},{trunk.n_times}).")
    if len(values) == 0:
        raise ValueError("checkpoint input must be finite and non-empty.")
    if trial_channel_mask is None:
        mask = np.ones(values.shape[:2], dtype=bool)
    else:
        mask = np.asarray(trial_channel_mask)
        if mask.dtype != np.dtype(bool) or mask.shape != values.shape[:2]:
            raise ValueError("trial_channel_mask must be boolean and match X.shape[:2].")
        if not bool(mask.any(axis=1).all()):
            raise ValueError("Every checkpoint trial must retain one observed channel.")
    if not np.isfinite(values[mask]).all():
        raise ValueError("checkpoint observed input must be finite and non-empty.")
    mean, std = input_stats
    standardized = ((values - mean) / std).astype(np.float32, copy=False)
    np.copyto(standardized, 0.0, where=~mask[:, :, None])
    core = trunk.core if isinstance(trunk, FeatureDomainTransfer) else trunk
    splitbn_config = core.splitbn_config
    if splitbn_config is not None:
        declared_source = splitbn_config.target_buffer_source
        if eval_buffer_source is not None and eval_buffer_source != declared_source:
            raise ValueError(
                "checkpoint target prediction cannot override its declared SplitBN buffer."
            )
        if eval_buffer_index is not None:
            raise ValueError(
                "checkpoint target prediction cannot select a source-domain buffer index."
            )
        eval_buffer_source = declared_source
    return standardized, eval_buffer_source, mask


def predict_n2p3_checkpoint_with_features(
    trunk: N2P3Net | FeatureDomainTransfer,
    X: np.ndarray,
    *,
    input_stats: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    trial_channel_mask: np.ndarray | None = None,
    eval_buffer_source: str | None = None,
    eval_buffer_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return unchanged-checkpoint scores and the pooled trunk features.

    The features are returned only for plain ``N2P3Net`` trunks whose
    classifier consumes the same adapted map that ``forward_with_features``
    exposes; wrapped feature-transfer trunks keep their existing consensus
    path and report ``None`` features.
    """

    standardized, resolved_buffer_source, mask = _checkpoint_prediction_input(
        trunk,
        X,
        input_stats=input_stats,
        device=device,
        trial_channel_mask=trial_channel_mask,
        eval_buffer_source=eval_buffer_source,
        eval_buffer_index=eval_buffer_index,
    )
    trunk = trunk.to(device)
    trunk.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(standardized)).to(device)
        channel_mask = torch.tensor(mask, device=device)
        if isinstance(trunk, N2P3Net):
            output, features = trunk.forward_with_features(
                tensor,
                channel_mask=channel_mask,
                eval_buffer_source=resolved_buffer_source,
                eval_buffer_index=eval_buffer_index,
            )
            features_value = (
                trunk.pool_features(features).detach().cpu().numpy().astype(np.float32)
            )
        else:
            output = trunk(
                tensor,
                channel_mask=channel_mask,
                eval_buffer_source=resolved_buffer_source,
                eval_buffer_index=eval_buffer_index,
            )
            features_value = None
    scores = (output[:, 1] - output[:, 0]).detach().cpu().numpy().astype(np.float64)
    return scores, features_value


def predict_n2p3_checkpoint(
    trunk: N2P3Net | FeatureDomainTransfer,
    X: np.ndarray,
    *,
    input_stats: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    trial_channel_mask: np.ndarray | None = None,
    eval_buffer_source: str | None = None,
    eval_buffer_index: int | None = None,
) -> np.ndarray:
    """Run a small target suffix through a pretrained classifier unchanged."""

    scores, _ = predict_n2p3_checkpoint_with_features(
        trunk,
        X,
        input_stats=input_stats,
        device=device,
        trial_channel_mask=trial_channel_mask,
        eval_buffer_source=eval_buffer_source,
        eval_buffer_index=eval_buffer_index,
    )
    return scores


def load_n2p3_trunk_checkpoint(
    path: str | Path | Mapping[str, object],
    dataset: object,
    *,
    target_subject: str | None = None,
    identity_exclusion_policy: IdentityExclusionPolicy = "source_or_global",
) -> tuple[N2P3Net | FeatureDomainTransfer, dict[str, object]]:
    """Load one target-excluded trunk using its declared architecture contract."""

    payload = dict(path) if isinstance(path, Mapping) else load_checkpoint_payload(path)
    _validate_metadata(
        payload,
        dataset,
        target_subject=target_subject,
        identity_exclusion_policy=identity_exclusion_policy,
    )
    (
        architecture,
        pooling,
        splitbn_config,
        transfer_config,
        axis_record,
        pooling_kwargs,
    ) = _architecture_from_payload(payload)
    core = N2P3Net(
        int(dataset.n_channels),
        n_times=int(dataset.n_times),
        sfreq=float(dataset.preprocessing.sfreq),
        tmin_s=float(dataset.preprocessing.tmin_ms) / 1000.0,
        pooling_mode=pooling,
        **architecture.model_kwargs(),
        **({} if splitbn_config is None else splitbn_config.model_kwargs()),
        **pooling_kwargs,
    )
    trunk: N2P3Net | FeatureDomainTransfer = core
    if transfer_config is not None:
        trunk = FeatureDomainTransfer(core, config=transfer_config)
    reconstructed_architecture = core.architecture_record()
    if transfer_config is not None:
        reconstructed_architecture["feature_transfer"] = (
            feature_transfer_architecture_record(
                transfer_config,
                feature_channels=(
                    len(architecture.mst_kernel_sizes)
                    * architecture.mst_features_per_scale
                ),
            )
        )
    if axis_record is not None:
        reconstructed_architecture["source_domain_axis_digest"] = semantic_sha256(
            axis_record
        )
    if semantic_sha256(reconstructed_architecture) != semantic_sha256(
        payload["architecture"]
    ):
        raise ValueError(
            "checkpoint architecture record does not exactly match the reconstructed model."
        )
    state = payload["trunk_state_dict"]
    try:
        missing, unexpected = trunk.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise ValueError(
            f"checkpoint weights do not match its declared architecture: {exc}"
        ) from exc
    if missing or unexpected:
        raise ValueError(
            f"checkpoint does not match the target trunk: missing={missing} unexpected={unexpected}."
        )
    if transfer_config is not None and not isinstance(trunk, FeatureDomainTransfer):
        raise RuntimeError("feature_transfer checkpoint did not reconstruct its wrapper.")
    return trunk, payload


class PreparedTransferCheckpoint:
    """Validate one immutable checkpoint/dataset pair, then check each target cheaply."""

    def __init__(self, path, dataset):
        if not isinstance(getattr(dataset, "identity_table", None), DatasetIdentityTable):
            raise ValueError("prepared checkpoint requires a structured target identity table")
        self._trunk, payload = load_n2p3_trunk_checkpoint(path, dataset)
        self._training_identity = DatasetIdentityTable.from_payload(payload["training_identity_ledger"])
        self._target_identity = dataset.identity_table
        self.source_seed = int(payload["training_contract"]["seed"])
        self.training_contract_digest = str(payload["training_contract_digest"])
        self.input_statistics = checkpoint_input_stats(payload, dataset.n_channels)
        self.payload = payload

    def assert_target_excluded(self, subject):
        assert_target_identity_excluded(self._training_identity,
            self._target_identity.record_for(str(subject)), policy="source_or_global")

    def trunk_for_target(self, subject):
        self.assert_target_excluded(subject)
        return copy.deepcopy(self._trunk)

    def model_origin(self, checkpoint_sha256):
        return {"kind": "checkpoint", "checkpoint_sha256": checkpoint_sha256,
                "training_contract_digest": self.training_contract_digest}


def checkpoint_architecture_record(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a normalized architecture summary for run ledgers and diagnostics."""

    (
        architecture,
        pooling,
        splitbn_config,
        transfer_config,
        axis_record,
        pooling_kwargs,
    ) = _architecture_from_payload(payload)
    required = ("n_channels", "input_tmin_s", "input_sample_rate_hz", "n_times")
    missing = [field for field in required if payload.get(field) is None]
    if missing:
        raise ValueError(f"checkpoint lacks required input geometry {missing}.")
    reconstructed = N2P3Net(
        int(payload["n_channels"]),
        pooling_mode=pooling,
        tmin_s=float(payload["input_tmin_s"]),
        sfreq=float(payload["input_sample_rate_hz"]),
        n_times=int(payload["n_times"]),
        **architecture.model_kwargs(),
        **({} if splitbn_config is None else splitbn_config.model_kwargs()),
        **pooling_kwargs,
    )
    record = reconstructed.architecture_record()
    if transfer_config is not None:
        record["feature_transfer"] = feature_transfer_architecture_record(
            transfer_config,
            feature_channels=(
                len(architecture.mst_kernel_sizes) * architecture.mst_features_per_scale
            ),
        )
    if axis_record is not None:
        record["source_domain_axis_digest"] = semantic_sha256(axis_record)
    declared = payload.get("architecture")
    if semantic_sha256(record) != semantic_sha256(declared):
        raise ValueError(
            "checkpoint architecture record does not exactly match the reconstructed model."
        )
    return record


__all__ = [
    "CHECKPOINT_SCHEMA",
    "PreparedTransferCheckpoint",
    "assert_checkpoint_target_identity_excluded",
    "checkpoint_architecture_record",
    "checkpoint_classifier_is_trained",
    "checkpoint_input_stats",
    "checkpoint_training_contract",
    "checkpoint_training_identity_ledger",
    "checkpoint_scores_to_llr",
    "load_checkpoint_payload",
    "load_n2p3_trunk_checkpoint",
    "predict_n2p3_checkpoint",
]
