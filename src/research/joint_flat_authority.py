"""Original raw-source ingress for the joint flat / same-record Gamma program.

Numerical role strings issue no read/label authority. This module consumes the
original MAT compiler's admitted C, reconstructs its samples/events and labels,
and calls the existing noise and response owners for parameter estimation.
The complete C and its future continuation share ONE recording scale and one
flat time chart. Source-only and C-plugin parameter programs have different
factories and independently attributable training ledgers.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from research.channel_noise_mechanism import (
    FittedChannelNoise,
    FrozenERPResponse,
    NoiseGeometry,
    ResidualRecord,
)
from research.joint_flat_response import compile_owned_flat_decision, prepare_flat_model
from research.joint_response_sampling import run_gibbs
from research.joint_response_scale import FixedHyperparameters, GlobalGaussianPrior
from transfer.continuous_decision import ContinuousDecisionModel, _event_design
from transfer.ensemble_views import CalibrationPrefixCompiler, calibration_prefix_identity
from transfer.noise_model import array_digest, freeze_metadata, frozen_array, metadata_digest


def _original_c(compiler, prefix):
    if type(compiler) is not CalibrationPrefixCompiler:
        raise TypeError("actual original MAT decoder is required; prepared replay carriers issue no C labels")
    if (hashlib.sha256(compiler._history_source_payload).hexdigest() != compiler._record.source_sha256
            or compiler._history_source_identity != array_digest(compiler._record.eeg_uv,
                compiler._record.flash_sample, compiler._record.candidate_id,
                compiler._record.selection_index, compiler._record.repetition_index)):
        raise ValueError("original source decoder operands changed")
    prefix = compiler.assert_original_prefix(prefix)
    if prefix.raw is None:
        raise ValueError("this C-plugin requires actual authorized C observations")
    rows = compiler._record.flash_sample < prefix.target_contract["raw_scope"][1]
    if (np.count_nonzero(rows) != prefix.budget
            or not np.array_equal(compiler._record.flash_sample[rows], prefix.raw.onsets)
            or not np.array_equal(compiler._record.raw_is_target[rows], prefix.raw.target_mask)
            or not np.array_equal(prefix.raw.samples,
                np.asarray(compiler._record.eeg_uv[:len(prefix.raw.samples)], dtype=float)*1e-6)):
        raise ValueError("C labels/events/volts differ from original source authority")
    return prefix


def _geometry(prefix):
    target = prefix.target_contract
    return NoiseGeometry(tuple(target["channel_names"]), target["sample_rate_hz"], "V",
                         target["reference"], "native_unfiltered_volts/1", len(target["channel_names"]))


def physical_time_nuisance(samples, channels, sfreq):
    """Channel-local intercept and slope per SECOND, unchanged on continuation."""
    time = np.arange(samples, dtype=float)/sfreq
    common = np.column_stack((np.ones(samples), time))
    return np.einsum("nj,cd->ncjd", common, np.eye(channels)).reshape(samples, channels, 2*channels)


def physical_event_image(onsets, labels, samples, basis, channels):
    design = _event_design(onsets, labels, samples, basis)
    return np.einsum("nk,cd->nckd", design, np.eye(channels)).reshape(samples, channels, -1)


def _external_prior(plan_bytes, geometry):
    if not isinstance(plan_bytes, bytes):
        raise TypeError("the original predeclared external numeric prior plan bytes are required")
    plan = json.loads(plan_bytes)
    if (set(plan) != {"schema", "freeze_role", "geometry", "basis", "mean", "covariance_root"}
            or plan["schema"] != "n2p3_joint_flat_external_prior/1"
            or plan["freeze_role"] != "predeclared_external_numeric_parameters"
            or plan["geometry"] != geometry.record()):
        raise ValueError("external prior plan differs from the actual physical geometry")
    basis = np.asarray(plan["basis"], dtype=float)
    prior = GlobalGaussianPrior(plan["mean"], plan["covariance_root"], "external")
    if (basis.ndim != 2 or not min(basis.shape) or not np.isfinite(basis).all()
            or len(prior.mean) != 2*basis.shape[1]*len(geometry.channels)):
        raise ValueError("external beta prior must use actual common/increment/channel coordinates")
    return prior, frozen_array(basis), {"program": "predeclared_external_numeric_prior",
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "actual_plan": plan}


@dataclass(frozen=True, init=False)
class JointFlatParameters:
    prior: GlobalGaussianPrior
    hyperparameters: FixedHyperparameters
    basis: np.ndarray
    noise: FittedChannelNoise
    binding: dict

    @property
    def digest(self):
        return metadata_digest(self.binding)


def _parameters(prior, hyperparameters, basis, noise, binding):
    result = object.__new__(JointFlatParameters)
    for key, value in {"prior": prior, "hyperparameters": hyperparameters, "basis": frozen_array(basis),
                       "noise": noise, "binding": freeze_metadata(binding)}.items():
        object.__setattr__(result, key, value)
    return result


def build_c_plugin_parameters(compiler, prefix, *, external_prior_plan, degrees_of_freedom,
        relative_noise_shape, noise_family, noise_order, conditioning_samples,
        minimum_samples_per_parameter=20):
    """Freeze actual C residual noise with the CENTRAL existing noise fitter.

    Residual mean is the declared external prior mean. It can contain response
    mismatch and drift; this method is explicitly a C-plugin K0 program.
    Neither this call nor Gibbs refits nu, shape or the external beta prior.
    """
    prefix = _original_c(compiler, prefix)
    geometry = _geometry(prefix)
    prior, basis, prior_binding = _external_prior(external_prior_plan, geometry)
    design = _event_design(prefix.raw.onsets, prefix.raw.target_mask, len(prefix.raw.samples), basis)
    mean = prior.mean.reshape(2*basis.shape[1], len(geometry.channels))
    residual = ResidualRecord.from_arrays(samples=prefix.raw.samples, design=design,
        kernel_mean=mean, nuisance=np.zeros_like(prefix.raw.samples), record_id=prefix.raw.record_id,
        participant_key=prefix.target_contract["participant_key"],
        recording_id=prefix.target_contract["source_sha256"], sample_start=0,
        geometry=geometry, role="target_calibration",
        source_model_sha256=prior_binding["plan_sha256"],
        source_input_sha256=prefix.target_contract["source_sha256"])
    noise = FittedChannelNoise.fit((residual,), family=noise_family, order=noise_order,
        conditioning_samples=conditioning_samples, minimum_samples_per_parameter=minimum_samples_per_parameter)
    hyper = FixedHyperparameters(degrees_of_freedom, relative_noise_shape, "frozen_C_plugin")
    binding = {"schema": "n2p3_joint_flat_parameter_authority/1", "program": "C_residual_noise_plugin_external_beta",
        "prior": prior_binding, "noise": noise.record(), "K0_definition": "frozen_C_residual_noise_shape",
        "C_prefix_digest": calibration_prefix_identity(prefix), "target_contract": prefix.target_contract,
        "label_use": prefix.label_use["raw"], "residual": residual.ledger(),
        "beta_prior_sha256": array_digest(prior.mean, prior.root), "basis_sha256": array_digest(basis),
        "Gamma_recording_law": "one_scale_for_entire_same_original_record_C_and_future",
        "relative_noise_shape": relative_noise_shape, "degrees_of_freedom": str(degrees_of_freedom),
        "nuisance_measure": "physical_intercept_V_and_slope_V_per_second_each_channel",
        "future_parameter_fits": 0, "original_compiler_source_sha256": compiler.history_owner_source}
    return _parameters(prior, hyper, basis, noise, binding)


def _source_cohort(source_cohort, target_participant_key):
    cohort = tuple((compiler, _original_c(compiler, prefix)) for compiler, prefix in source_cohort)
    if not cohort or not isinstance(target_participant_key, str) or not target_participant_key:
        raise ValueError("actual nonempty source cohort and explicit excluded target are required")
    geometry = _geometry(cohort[0][1])
    participants = [prefix.target_contract["participant_key"] for _, prefix in cohort]
    originals = [prefix.target_contract["source_sha256"] for _, prefix in cohort]
    if len(set(originals)) != len(originals) or target_participant_key in participants:
        raise ValueError("source cohort repeats overlapping original recordings or consumes the excluded target")
    if any(_geometry(prefix) != geometry for _, prefix in cohort):
        raise ValueError("source and target physical geometry requires an explicit transport owner")
    return cohort, geometry


def build_source_parameters(source_cohort, *, target_participant_key, degrees_of_freedom,
        relative_noise_shape, basis, ridge, noise_order, conditioning_samples, source_model_path):
    """Actual verified source records -> original response/noise owners -> prior.

    source_cohort supplies original (compiler,prefix) pairs, with explicit target
    exclusion. Native geometry must agree: this slice makes no spatial transport
    claim. It grants no target C statistics or labels to a no-label program.
    """
    cohort, geometry = _source_cohort(source_cohort, target_participant_key)
    basis = np.asarray(basis, dtype=float)
    records = tuple(prefix.raw for _, prefix in cohort)
    model = ContinuousDecisionModel.fit(records, sfreq=geometry.sfreq, channels=geometry.channels,
        candidate_ids=cohort[0][1].target_contract["candidate_vocabulary"], basis=basis, ridge=ridge,
        fit_scope="source_training", noise_order=noise_order, conditioning_samples=conditioning_samples,
        operator_identity=geometry.operator_identity)
    model.save(source_model_path)
    return source_parameters_from_saved_model(cohort, target_participant_key=target_participant_key,
        degrees_of_freedom=degrees_of_freedom, relative_noise_shape=relative_noise_shape,
        source_model_path=source_model_path)


def _reexport_cached_source(frozen_source, *, original_bytes, model, geometry, training,
                            target_participant_key, degrees_of_freedom, relative_noise_shape):
    """Rebind target exclusion after original-data checks; preserve all source operands."""
    from transfer.joint_flat_programme import FrozenJointFlatSource

    if not isinstance(frozen_source, FrozenJointFlatSource):
        raise TypeError('cached source export requires an admitted FrozenJointFlatSource')
    parameters = frozen_source.parameters
    binding = parameters.binding
    hyper = FixedHyperparameters(degrees_of_freedom, relative_noise_shape, 'externally_fixed')
    if (original_bytes != frozen_source.original_model_content
            or hashlib.sha256(original_bytes).hexdigest() != binding['source_model_sha256']
            or model.digest() != binding['source_model_digest']
            or parameters.noise.geometry != geometry
            or not np.array_equal(parameters.basis, model.basis)
            or array_digest(parameters.prior.mean, parameters.prior.root) != binding['beta_prior_sha256']
            or array_digest(parameters.basis) != binding['basis_sha256']
            or hyper != parameters.hyperparameters):
        raise ValueError('cached source must preserve the original NPZ, geometry and complete frozen parameter law')
    original_training = binding['source_prior_factor_conversion']['source_covariance_producer']['provenance']['records']
    keys = ('record_id', 'participant_key', 'source_sha256', 'read_support',
            'fit_scope', 'input_sha256', 'n_samples')
    def identities(rows):
        return {row['record_id']: metadata_digest({key: row[key] for key in keys}) for row in rows}
    if len(training) != len(original_training) or identities(training) != identities(original_training):
        raise ValueError('cached source cohort differs from original fitted records, identities or physical support')
    rebound = {**binding, 'excluded_target_participant': target_participant_key}
    return _parameters(parameters.prior, parameters.hyperparameters, parameters.basis, parameters.noise, rebound)


def source_parameters_from_saved_model(source_cohort, *, target_participant_key, degrees_of_freedom,
                                        relative_noise_shape, source_model_path, frozen_source=None):
    """Read-only original source model + original EEG authority, without refitting."""
    cohort, geometry = _source_cohort(source_cohort, target_participant_key)
    original_bytes = Path(source_model_path).read_bytes()
    model_file_sha256 = hashlib.sha256(original_bytes).hexdigest()
    model = ContinuousDecisionModel.load(io.BytesIO(original_bytes))
    model_digest = model.digest()
    basis = model.basis
    if (model.provenance["fit_scope"] != "source_training" or model.noise is None
            or model.channels != geometry.channels or model.sfreq != geometry.sfreq
            or model.operator_identity != geometry.operator_identity
            or model.candidate_ids != tuple(cohort[0][1].target_contract["candidate_vocabulary"])):
        raise ValueError("saved source model must retain its actual source-only native program")
    training = []
    fitted_records = {item["record_id"]: item for item in model.provenance["records"]}
    if set(fitted_records) != {prefix.raw.record_id for _, prefix in cohort}:
        raise ValueError("source cohort must recover every original source fit record")
    for compiler, prefix in cohort:
        record = prefix.raw
        original = fitted_records[record.record_id]
        actual_input = array_digest(record.samples, record.onsets, record.target_mask)
        if original["input_sha256"] != actual_input or original["n_samples"] != len(record.samples):
            raise ValueError("saved source model differs from actual original cohort input")
        original_view = {**prefix.target_contract, "record_id": record.record_id,
            "raw_scope": list(prefix.target_contract["raw_scope"]),
            "original_prefix_digest": calibration_prefix_identity(prefix),
            "original_compiler_source_sha256": compiler.history_owner_source,
            "actual_input_sha256": actual_input, "label_use": prefix.label_use["raw"]}
        training.append({"record_id": record.record_id,
            "participant_key": prefix.target_contract["participant_key"],
            "recording_id": prefix.target_contract["source_sha256"],
            "source_sha256": prefix.target_contract["source_sha256"],
            "read_support": list(prefix.target_contract["raw_scope"]),
            "fit_scope": model.provenance["fit_scope"], "input_sha256": original["input_sha256"],
            "n_samples": original["n_samples"], "original_view": original_view})
    if frozen_source is not None:
        return _reexport_cached_source(frozen_source, original_bytes=original_bytes, model=model,
            geometry=geometry, training=training, target_participant_key=target_participant_key,
            degrees_of_freedom=degrees_of_freedom, relative_noise_shape=relative_noise_shape), model
    response = FrozenERPResponse.from_continuous_model(model, geometry=geometry,
        source_model_sha256=model_file_sha256, response_id="joint-flat-source-beta", training_records=training)
    # F.T=Q R => F F.T=R.T R. Keep the original covariance factor's
    # information through square-root QR; never form/re-factor its dense Gram.
    q, upper = np.linalg.qr(response.coefficient_factor.T, mode="reduced")
    signs = np.where(np.diag(upper) < 0, -1., 1.)
    root = upper.T*signs[None, :]
    prior = GlobalGaussianPrior(response.mean.ravel(), root, "source_only")
    residuals = []
    for _, prefix in cohort:
        design = _event_design(prefix.raw.onsets, prefix.raw.target_mask, len(prefix.raw.samples), basis)
        residuals.append(ResidualRecord.from_arrays(samples=prefix.raw.samples, design=design,
            kernel_mean=model.kernel_mean, nuisance=np.zeros_like(prefix.raw.samples),
            record_id=prefix.raw.record_id, participant_key=prefix.target_contract["participant_key"],
            recording_id=prefix.target_contract["source_sha256"], sample_start=0,
            geometry=geometry, role="source_training", source_model_sha256=model_file_sha256,
            source_input_sha256=prefix.target_contract["source_sha256"]))
    noise = FittedChannelNoise.from_temporal_noise(model.noise, model.innovation_covariance,
        geometry=geometry, conditioning_samples=model.conditioning_samples,
        source_model_sha256=model_file_sha256, training_records=residuals)
    hyper = FixedHyperparameters(degrees_of_freedom, relative_noise_shape, "externally_fixed")
    binding = {"schema": "n2p3_joint_flat_parameter_authority/1", "program": "actual_source_response_and_noise",
        "excluded_target_participant": target_participant_key, "source_model": model._metadata(),
        "source_model_digest": model_digest, "source_model_sha256": model_file_sha256, "noise": noise.record(),
        "source_prefixes": [{"prefix_digest": calibration_prefix_identity(prefix),
            "target_contract": prefix.target_contract, "label_use": prefix.label_use["raw"],
            "compiler_source_sha256": compiler.history_owner_source} for compiler, prefix in cohort],
        "beta_prior_sha256": array_digest(prior.mean, prior.root), "basis_sha256": array_digest(basis),
        "source_prior_factor_conversion": {"program": "original_factor_transpose_QR_positive_lower_chart",
            "original_factor_sha256": array_digest(response.coefficient_factor),
            "chart_root_sha256": array_digest(root), "orthogonal_sha256": array_digest(q),
            "column_signs": signs.tolist(),
            "actual_reconstruction_norm": float(np.linalg.norm(response.coefficient_factor.T-q@upper)),
            "dense_Gram_formed": False, "source_covariance_producer": response.record()},
        "K0_definition": "original_saved_source_temporal_noise_and_innovation_covariance",
        "diagnostic_final_residual_records": [record.ledger() for record in residuals],
        "noise_parameter_fits_after_original_source_model": 0,
        "noise_original_fit_inputs": model.noise.provenance,
        "Gamma_recording_law": "one_scale_for_entire_original_record",
        "relative_noise_shape": relative_noise_shape, "degrees_of_freedom": str(degrees_of_freedom),
        "target_parameter_fits": 0, "target_labels_read": 0}
    return _parameters(prior, hyper, basis, noise, binding), model


def _original_record_source_cohort(source_cohort, target_participant_key):
    """Named original multi-record source issuer; the MAT gate stays intact."""
    from transfer.ensemble_views import RecordCalibrationPrefixCompiler

    cohort, records, authorities = [], [], []
    original_inputs = set()
    for compiler, declared_prefix in source_cohort:
        if type(compiler) is not RecordCalibrationPrefixCompiler:
            raise TypeError('record source parameters require the original registered recording compiler')
        prefix = compiler.assert_original_prefix(declared_prefix)
        if prefix.target_contract['participant_key'] == target_participant_key:
            raise ValueError('actual source programme includes the excluded target participant')
        for record in prefix.calibration_records:
            scope = prefix.target_contract['record_scopes'][record.record_id]
            matches = [r for r in compiler.sources if r.record_id == record.record_id
                       and r.source_sha256 == scope['source_sha256']]
            if len(matches) != 1 or tuple(scope['read_sample_support']) != (0, len(record.samples)):
                raise ValueError('source fit has no unique original acquisition and admitted native C scope')
            source = matches[0]
            count = len(record.onsets)
            if (source.source_sha256 in original_inputs
                    or not np.array_equal(record.samples, source.read_native(0, len(record.samples)))
                    or not np.array_equal(record.onsets, source.native_onsets[:count])
                    or not np.array_equal(record.target_mask, source.authorized_binary_labels[:count])
                    or source.participant_key != prefix.target_contract['participant_key']):
                raise ValueError('source original bytes, clocks, C labels, participant or overlap changed')
            original_inputs.add(source.source_sha256)
            original_view = {**prefix.target_contract, 'record_id': record.record_id,
                'source_sha256': source.source_sha256, 'raw_scope': scope['read_sample_support'],
                'actual_record_scope': scope, 'original_prefix_digest': calibration_prefix_identity(prefix),
                'original_compiler_source_sha256': compiler.history_owner_source,
                'original_factory_source_sha256': compiler.adapter.original_factory_source_sha256,
                'actual_input_sha256': array_digest(record.samples, record.onsets, record.target_mask),
                'label_use': prefix.label_use['raw']}
            records.append(record)
            authorities.append(original_view)
        cohort.append((compiler, prefix))
    if not records or not target_participant_key:
        raise ValueError('actual record source cohort and excluded target must be nonempty')
    geometry = _geometry(cohort[0][1])
    if any(_geometry(prefix) != geometry for _, prefix in cohort):
        raise ValueError('source records require one actual native physical geometry or explicit transport')
    if len({record.record_id for record in records}) != len(records):
        raise ValueError('source record identifiers must retain their original unique acquisition keys')
    return tuple(cohort), tuple(records), tuple(authorities), geometry


def build_source_record_parameters(source_cohort, *, target_participant_key, degrees_of_freedom,
        relative_noise_shape, basis, ridge, noise_order, conditioning_samples, source_model_path):
    """Original source-record C authority -> existing response/noise source fit."""
    cohort, records, _views, geometry = _original_record_source_cohort(source_cohort, target_participant_key)
    model = ContinuousDecisionModel.fit(records, sfreq=geometry.sfreq, channels=geometry.channels,
        candidate_ids=cohort[0][1].target_contract['candidate_vocabulary'], basis=np.asarray(basis), ridge=ridge,
        fit_scope='source_training', noise_order=noise_order, conditioning_samples=conditioning_samples,
        operator_identity=geometry.operator_identity)
    model.save(source_model_path)
    return source_record_parameters_from_saved_model(cohort, target_participant_key=target_participant_key,
        degrees_of_freedom=degrees_of_freedom, relative_noise_shape=relative_noise_shape,
        source_model_path=source_model_path)


def source_record_parameters_from_saved_model(source_cohort, *, target_participant_key, degrees_of_freedom,
        relative_noise_shape, source_model_path, frozen_source=None):
    """Read-only original multi-record source export, with real cohort checks."""
    cohort, records, views, geometry = _original_record_source_cohort(source_cohort, target_participant_key)
    original_bytes = Path(source_model_path).read_bytes()
    file_sha = hashlib.sha256(original_bytes).hexdigest()
    model = ContinuousDecisionModel.load(io.BytesIO(original_bytes))
    if (model.provenance['fit_scope'] != 'source_training' or model.posterior_state is None or model.noise is None
            or model.channels != geometry.channels or model.sfreq != geometry.sfreq
            or model.operator_identity != geometry.operator_identity):
        raise ValueError('source model differs from the actual original native source programme')
    fitted = {row['record_id']: row for row in model.provenance['records']}
    if set(fitted) != {record.record_id for record in records}:
        raise ValueError('source export must recover every actual original fitted recording')
    training, residuals = [], []
    for record, view in zip(records, views, strict=True):
        original = fitted[record.record_id]
        if (original['input_sha256'] != view['actual_input_sha256']
                or original['n_samples'] != len(record.samples)):
            raise ValueError('saved source producer differs from actual source C arrays or clock')
        training.append({'record_id': record.record_id, 'participant_key': view['participant_key'],
            'recording_id': view['source_sha256'], 'read_support': list(view['raw_scope']),
            'source_sha256': view['source_sha256'], 'source_input_sha256': view['source_sha256'],
            'fit_scope': 'source_training', 'input_sha256': original['input_sha256'],
            'n_samples': original['n_samples'], 'original_view': view})
    if frozen_source is not None:
        return _reexport_cached_source(frozen_source, original_bytes=original_bytes, model=model,
            geometry=geometry, training=training, target_participant_key=target_participant_key,
            degrees_of_freedom=degrees_of_freedom, relative_noise_shape=relative_noise_shape), model
    for record, view in zip(records, views, strict=True):
        design = _event_design(record.onsets, record.target_mask, len(record.samples), model.basis)
        residuals.append(ResidualRecord.from_arrays(samples=record.samples, design=design,
            kernel_mean=model.kernel_mean, nuisance=np.zeros_like(record.samples), record_id=record.record_id,
            participant_key=view['participant_key'], recording_id=view['source_sha256'], sample_start=0,
            geometry=geometry, role='source_training', source_model_sha256=file_sha,
            source_input_sha256=view['source_sha256']))
    response = FrozenERPResponse.from_continuous_model(model, geometry=geometry,
        source_model_sha256=file_sha, response_id='joint-flat-record-source-beta', training_records=training)
    q, upper = np.linalg.qr(response.coefficient_factor.T, mode='reduced')
    signs = np.where(np.diag(upper) < 0, -1., 1.)
    root = upper.T*signs[None, :]
    prior = GlobalGaussianPrior(response.mean.ravel(), root, 'source_only')
    noise = FittedChannelNoise.from_temporal_noise(model.noise, model.innovation_covariance,
        geometry=geometry, conditioning_samples=model.conditioning_samples,
        source_model_sha256=file_sha, training_records=residuals)
    hyper = FixedHyperparameters(degrees_of_freedom, relative_noise_shape, 'externally_fixed')
    binding = {'schema': 'n2p3_joint_flat_parameter_authority/1', 'program': 'actual_source_response_and_noise',
        'excluded_target_participant': target_participant_key, 'source_model': model._metadata(),
        'source_model_digest': model.digest(), 'source_model_sha256': file_sha, 'noise': noise.record(),
        'source_prefixes': [{'prefix_digest': calibration_prefix_identity(prefix),
            'target_contract': prefix.target_contract, 'label_use': prefix.label_use['raw'],
            'compiler_source_sha256': compiler.history_owner_source,
            'original_factory_source_sha256': compiler.adapter.original_factory_source_sha256} for compiler, prefix in cohort],
        'actual_source_record_views': list(views),
        'beta_prior_sha256': array_digest(prior.mean, prior.root), 'basis_sha256': array_digest(model.basis),
        'source_prior_factor_conversion': {'program': 'original_factor_transpose_QR_positive_lower_chart',
            'original_factor_sha256': array_digest(response.coefficient_factor), 'chart_root_sha256': array_digest(root),
            'orthogonal_sha256': array_digest(q), 'column_signs': signs.tolist(),
            'actual_reconstruction_norm': float(np.linalg.norm(response.coefficient_factor.T-q@upper)),
            'dense_Gram_formed': False, 'source_covariance_producer': response.record()},
        'K0_definition': 'original_saved_source_temporal_noise_and_innovation_covariance',
        'diagnostic_final_residual_records': [record.ledger() for record in residuals],
        'noise_parameter_fits_after_original_source_model': 0, 'noise_original_fit_inputs': model.noise.provenance,
        'Gamma_recording_law': 'one_scale_for_entire_original_record',
        'relative_noise_shape': relative_noise_shape, 'degrees_of_freedom': str(degrees_of_freedom),
        'target_parameter_fits': 0, 'target_labels_read': 0}
    return _parameters(prior, hyper, model.basis, noise, binding), model


@dataclass(frozen=True, init=False)
class AuthorizedJointFlatC:
    parameters: JointFlatParameters
    compiled: object
    binding: dict


def compile_original_c(parameters, compiler, prefix):
    prefix = _original_c(compiler, prefix)
    if (not isinstance(parameters, JointFlatParameters) or parameters.noise.geometry != _geometry(prefix)
            or parameters.binding["program"] == "C_residual_noise_plugin_external_beta"
            and parameters.binding["C_prefix_digest"] != calibration_prefix_identity(prefix)):
        raise ValueError("C authority and parameter producer differ")
    if (parameters.binding["program"] == "actual_source_response_and_noise"
            and parameters.binding["excluded_target_participant"] != prefix.target_contract["participant_key"]):
        raise ValueError("source prior/noise must exclude this actual target participant")
    if (parameters.binding["program"] == "actual_source_response_and_noise"
            and prefix.target_contract["source_sha256"] in {
                item["target_contract"]["source_sha256"] for item in parameters.binding["source_prefixes"]}):
        raise ValueError("source input bytes cannot re-enter as target under another participant alias")
    raw, channels = prefix.raw, len(parameters.noise.geometry.channels)
    design = physical_event_image(raw.onsets, raw.target_mask, len(raw.samples), parameters.basis, channels)
    nuisance = physical_time_nuisance(len(raw.samples), channels, parameters.noise.geometry.sfreq)
    compiled = compile_owned_flat_decision(parameters.prior, parameters.hyperparameters, parameters.noise,
        decision_id=raw.record_id, role="target_calibration", native_samples=len(raw.samples),
        raw_batches=((0, raw.samples, design, nuisance),))
    result = object.__new__(AuthorizedJointFlatC)
    for key, value in {"parameters": parameters, "compiled": compiled,
        "binding": freeze_metadata({"schema": "n2p3_joint_flat_actual_C/1",
            "parameter_authority": parameters.digest, "C_prefix_digest": calibration_prefix_identity(prefix),
            "source_sha256": prefix.target_contract["source_sha256"], "record_id": raw.record_id,
            "read_support": prefix.target_contract["raw_scope"], "label_use": prefix.label_use["raw"],
            "original_Y_D_Z_sha256": compiled.raw_input_identity,
            "raw_scalar_rows": compiled.observed_rows, "flat_rank": compiled.nuisance_rank,
            "effective_rows": compiled.effective_rows, "actual_Gamma_variables": 1})}.items():
        object.__setattr__(result, key, value)
    return result


def sample_original_c(authorized, *, burnin, draws, seeds, initial_charts):
    if not isinstance(authorized, AuthorizedJointFlatC):
        raise TypeError("actual original C issuer is required")
    parameters = authorized.parameters
    model = prepare_flat_model(parameters.prior, (authorized.compiled,), parameters.hyperparameters)
    return run_gibbs(model, burnin=burnin, draws=draws, seeds=seeds, initial_charts=initial_charts)


def compile_original_continuation(authorized, compiler, prefix, view, *, candidate_id,
                                  conditioning_stop=None):
    """Original C -> additional observed prefix -> tail, all on the SAME lambda.

    All already known C event labels are preserved. A partly calibrated earlier
    selection may continue under its uniquely identified C target. Intervening
    selections with multiple still possible targets need a joint target law.
    Scores are conditional on the declared current q; the candidate caller
    chooses a fixed-current-prior tail law or a whole-observed law explicitly.
    """
    from transfer.ensemble_views import PredictionDecisionView, _bnci_prediction_view

    prefix = _original_c(compiler, prefix)
    if (not isinstance(authorized, AuthorizedJointFlatC)
            or authorized.binding["C_prefix_digest"] != calibration_prefix_identity(prefix)
            or not isinstance(view, PredictionDecisionView) or view.samples is None):
        raise ValueError("the actual issuer C and original observed continuation are required")
    raw, parameters = compiler._record, authorized.parameters
    rows = np.flatnonzero(raw.selection_id == view.decision_id)
    if not len(rows):
        raise ValueError("current selection is absent from original event authority")
    original = _bnci_prediction_view(raw, rows, *view.raw_scope, available_sample=view.available_sample)
    if (original.physical_record() != view.physical_record()
            or not np.array_equal(original.samples, view.samples)
            or original.decision.record() != view.decision.record()
            or not np.array_equal(original.onsets, view.onsets)
            or view.participant_key != prefix.target_contract["participant_key"]):
        raise ValueError("continuation samples/membership/cutoff differ from original decoder")
    if (not view.decision.schedule_complete or np.any(view.decision.evidence_indices < 0)
            or view.available_sample != view.raw_scope[1] or view.raw_scope[0] < 0):
        raise ValueError("one actual complete current membership and observation cutoff are required")
    stop, cstop = view.raw_scope[1], len(prefix.raw.samples)
    pstop = cstop if conditioning_stop is None else conditioning_stop
    if type(pstop) is not int or not cstop <= pstop < stop:
        raise ValueError("prediction prefix must extend actual C and precede the observed tail")
    occurred = np.flatnonzero(raw.flash_sample < stop)
    unknown = occurred[occurred >= prefix.budget]
    vocabulary = view.decision.candidate_vocabulary
    if candidate_id not in vocabulary:
        raise ValueError("current q must remain in the original full vocabulary")
    index = vocabulary.index(candidate_id)
    known = rows[rows < prefix.budget]
    current_labels = view.decision.membership[:, index]
    if len(known) and not np.array_equal(current_labels[:len(known)], prefix.raw.target_mask[known]):
        raise ValueError("current q is incompatible with actual authorized partial-C labels")
    # No held target labels are accessed: future binary labels come solely
    # from this candidate's public physical membership.
    labels = np.zeros(len(occurred))
    labels[:prefix.budget] = prefix.raw.target_mask
    current_unknown = unknown[raw.selection_id[unknown] == view.decision_id]
    positions = np.searchsorted(rows, current_unknown)
    labels[current_unknown] = current_labels[positions]
    known_continuations = []
    for selection in dict.fromkeys(raw.selection_id[unknown]):
        if selection == view.decision_id:
            continue
        from data.bnci2014_008_candidate import BNCI2014_008_GRID_SHAPE
        from data.candidate_task import MembershipDecision

        selection_rows = occurred[raw.selection_id[occurred] == selection]
        membership = MembershipDecision.row_column(raw.candidate_id[selection_rows],
            n_rows=BNCI2014_008_GRID_SHAPE[0], n_columns=BNCI2014_008_GRID_SHAPE[1],
            event_ids=tuple(f"{view.source_dataset_id}:{raw.subject_id}:{int(row)}" for row in selection_rows),
            repetition_indices=raw.repetition_index[selection_rows], repetitions=view.decision.repetitions)
        known_rows = selection_rows[selection_rows < prefix.budget]
        possible = np.all(membership.membership[:len(known_rows)] == prefix.raw.target_mask[known_rows, None], axis=0)
        if np.count_nonzero(possible) != 1:
            raise ValueError("intervening unknown selections require a complete declared joint target law")
        selected = int(np.flatnonzero(possible)[0])
        remaining_rows = selection_rows[selection_rows >= prefix.budget]
        labels[remaining_rows] = membership.membership[len(known_rows):, selected]
        known_continuations.append({"selection_id": str(selection),
            "authorized_C_labels": len(known_rows), "possible_targets_after_C": 1,
            "target_from_C_only": membership.candidate_vocabulary[selected],
            "authorized_C_constraint_sha256": array_digest(prefix.raw.target_mask[known_rows]),
            "already_occurred_public_membership_sha256": array_digest(membership.membership),
            "new_labels_read": 0})
    channels = len(parameters.noise.geometry.channels)
    samples = np.asarray(raw.eeg_uv[:stop], dtype=float)*1e-6
    design = physical_event_image(raw.flash_sample[occurred], labels, stop, parameters.basis, channels)
    nuisance = physical_time_nuisance(stop, channels, parameters.noise.geometry.sfreq)
    earlier = () if pstop == cstop else (cstop,)
    full = compile_owned_flat_decision(parameters.prior, parameters.hyperparameters, parameters.noise,
        decision_id=prefix.raw.record_id, role="read_only_inference", native_samples=stop,
        raw_batches=((0, samples, design, nuisance),), conditioning_samples=pstop,
        earlier_conditioning_samples=earlier)
    receipt = freeze_metadata({"schema": "n2p3_joint_flat_same_record_continuation/1",
        "actual_C": authorized.binding, "source_sha256": raw.source_sha256,
        "current_selection": view.decision_id, "candidate_vocabulary": vocabulary,
        "current_q": candidate_id, "full_read_support": [0, stop],
        "calibration_support": [0, cstop], "conditioning_support": [0, pstop],
        "scored_tail_support": [pstop, stop], "truth_consumed": False,
        "Gamma_axes": [prefix.raw.record_id], "reset_at_selection_boundary": False,
        "known_C_selection_continuations": known_continuations,
        "original_public_events": {"onsets_sha256": array_digest(raw.flash_sample[occurred]),
            "membership_sha256": array_digest(view.decision.membership)},
        "physical_da_chart": "same_intercept_V_slope_V_per_second_entire_record",
        "prior_candidate_role": "declared_current_q_conditional_not_automatic_q_posterior"})
    return full, receipt


def compile_original_no_label(parameters, compiler, zero_prefix, view, *, candidate_id,
                               conditioning_stop):
    """Source-only parameters and actual K=0 observations, without target fitting.

    This admits the original first selection only. Its observed prefix updates
    flat drift, response and Gamma by predictive conditioning, while source
    parameters remain frozen. Other observed unknown selections require their
    declared joint target law and are not guessed independent.
    """
    from transfer.ensemble_views import PredictionDecisionView, _bnci_prediction_view

    if type(compiler) is not CalibrationPrefixCompiler:
        raise TypeError("original MAT source owner is required")
    zero_prefix = compiler.assert_original_prefix(zero_prefix)
    if (not isinstance(parameters, JointFlatParameters)
            or parameters.binding["program"] != "actual_source_response_and_noise"
            or zero_prefix.budget != 0 or zero_prefix.raw is not None
            or parameters.binding["excluded_target_participant"] != zero_prefix.target_contract["participant_key"]
            or parameters.noise.geometry != _geometry(zero_prefix)
            or not isinstance(view, PredictionDecisionView) or view.samples is None):
        raise ValueError("source-only prior/noise, actual excluded target K0 and observations are required")
    if zero_prefix.target_contract["source_sha256"] in {
            item["target_contract"]["source_sha256"] for item in parameters.binding["source_prefixes"]}:
        raise ValueError("source input bytes cannot re-enter as target under another participant alias")
    if (hashlib.sha256(compiler._history_source_payload).hexdigest() != compiler._record.source_sha256
            or compiler._history_source_identity != array_digest(compiler._record.eeg_uv,
                compiler._record.flash_sample, compiler._record.candidate_id,
                compiler._record.selection_index, compiler._record.repetition_index)):
        raise ValueError("original decoder changed after source admission")
    raw = compiler._record
    rows = np.flatnonzero(raw.selection_id == view.decision_id)
    original = _bnci_prediction_view(raw, rows, *view.raw_scope, available_sample=view.available_sample)
    if (original.physical_record() != view.physical_record()
            or not np.array_equal(original.samples, view.samples)
            or original.decision.record() != view.decision.record()
            or not np.array_equal(original.onsets, view.onsets)
            or view.participant_key != zero_prefix.target_contract["participant_key"]
            or not view.decision.schedule_complete or np.any(view.decision.evidence_indices < 0)
            or view.available_sample != view.raw_scope[1]):
        raise ValueError("no-label current observation differs from original source membership/cutoff")
    stop = view.raw_scope[1]
    if type(conditioning_stop) is not int or not 0 < conditioning_stop < stop:
        raise ValueError("one actual observed prediction prefix and tail are required")
    occurred = np.flatnonzero(raw.flash_sample < stop)
    if np.any(raw.selection_id[occurred] != view.decision_id):
        raise ValueError("no-label route requires the declared first-selection joint target law")
    vocabulary = view.decision.candidate_vocabulary
    if candidate_id not in vocabulary:
        raise ValueError("current q must preserve the original requested vocabulary")
    labels = view.decision.membership[:, vocabulary.index(candidate_id)][np.searchsorted(rows, occurred)]
    channels = len(parameters.noise.geometry.channels)
    samples = np.asarray(raw.eeg_uv[:stop], dtype=float)*1e-6
    design = physical_event_image(raw.flash_sample[occurred], labels, stop, parameters.basis, channels)
    nuisance = physical_time_nuisance(stop, channels, parameters.noise.geometry.sfreq)
    full = compile_owned_flat_decision(parameters.prior, parameters.hyperparameters, parameters.noise,
        decision_id=f"{view.participant_key}:original-record:{view.source_sha256}", role="read_only_inference",
        native_samples=stop, raw_batches=((0, samples, design, nuisance),),
        conditioning_samples=conditioning_stop)
    return full, freeze_metadata({"schema": "n2p3_joint_flat_actual_no_label/1",
        "parameter_authority": parameters.digest, "source_sha256": view.source_sha256,
        "target_participant": view.participant_key, "target_labels_read": 0,
        "target_parameter_fits": 0, "calibration_budget": 0,
        "read_support": [0, stop], "conditioning_support": [0, conditioning_stop],
        "tail_support": [conditioning_stop, stop], "current_q": candidate_id,
        "candidate_vocabulary": vocabulary, "actual_Gamma_variables": 1,
        "truth_consumed": False, "read_program": "same_original_record_first_selection"})
