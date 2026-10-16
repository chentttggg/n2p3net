"""Typed source reference and one caller for the finite response programme.

Scientific shapes, normalized state masses and Gaussian algebra belong to their
existing owners. This module binds an explicitly declared continuation to the
actual source, then calls the original BNCI C/history issuer.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from transfer.joint_flat_programme import FrozenJointFlatSource
from transfer.noise_model import array_digest

SOURCE_RESPONSE_FAMILY = 'source_response'
SOURCE_ROLE = 'original_source_response_and_noise_parameters'
SOURCE_REF_SCHEMA = 'n2p3_source_response_reference/1'
_ADAPTER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class SourceResponseSourceRef:
    source_exclusion_scope: str
    continuation_role: str
    terminal_policy: str
    terminal_values: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.source_exclusion_scope not in {'target_participant', 'whole_target_dataset'}:
            raise ValueError('source response requires its declared source exclusion scope')
        if self.continuation_role == 'original_saved_continuous_function':
            if self.terminal_policy != 'original_saved' or self.terminal_values is not None:
                raise ValueError('historical response continuation must use the actual saved function')
        elif self.continuation_role == 'new_declared_continuation_of_source_native_arrays':
            if self.terminal_policy not in {'zero_native_basis_vector', 'explicit_native_basis_vector'}:
                raise ValueError('new source-native function needs an explicit terminal policy')
            if self.terminal_policy == 'zero_native_basis_vector':
                if self.terminal_values is not None:
                    raise ValueError('zero terminal policy cannot also declare separate terminal values')
            else:
                if self.terminal_values is None or not len(self.terminal_values):
                    raise ValueError('explicit terminal policy requires its complete native basis vector')
                values = tuple(float(value) for value in self.terminal_values)
                if not np.isfinite(values).all():
                    raise ValueError('source response terminal values must be finite')
                object.__setattr__(self, 'terminal_values', values)
        else:
            raise ValueError('source response continuation role must be explicitly declared')

    def record(self):
        return {'schema': SOURCE_REF_SCHEMA, 'source_exclusion_scope': self.source_exclusion_scope,
            'continuation_role': self.continuation_role, 'terminal_policy': self.terminal_policy,
            'terminal_values': None if self.terminal_values is None else list(self.terminal_values)}

    @classmethod
    def from_record(cls, record):
        if set(record) != {'schema', 'source_exclusion_scope', 'continuation_role', 'terminal_policy', 'terminal_values'}:
            raise ValueError('source response reference changed its complete typed declaration')
        if record['schema'] != SOURCE_REF_SCHEMA:
            raise ValueError('unsupported source response reference schema')
        return cls(record['source_exclusion_scope'], record['continuation_role'],
            record['terminal_policy'], record['terminal_values'])

    def parameters(self, source, prefix):
        from research.source_response_states import SourceInformedResponseParameters
        from transfer.continuous_decision import ContinuousDecisionModel
        from transfer.native_response_dictionary import NativeResponseDictionary

        if not isinstance(source, FrozenJointFlatSource):
            raise TypeError('source response reference requires the actual frozen original parameters')
        self.check_source(source,prefix.target_contract)
        if self.continuation_role == 'original_saved_continuous_function':
            dictionary = ContinuousDecisionModel.load(io.BytesIO(source.original_model_content)).dictionary
            if dictionary is None:
                raise ValueError('original source contains no saved continuous response function')
        else:
            basis = source.parameters.basis
            terminal = (np.zeros(basis.shape[1]) if self.terminal_policy == 'zero_native_basis_vector'
                        else np.asarray(self.terminal_values))
            dictionary = NativeResponseDictionary.from_source_matrix(basis,
                sample_rate_hz=source.parameters.noise.geometry.sfreq,
                source_basis_sha256=array_digest(basis), terminal_values=terminal)
        return SourceInformedResponseParameters.from_source(source, target_authority=prefix,
            continuation=dictionary, continuation_role=self.continuation_role)

    def check_source(self, source, target_contract):
        source.assert_target(target_contract)
        if self.source_exclusion_scope == 'whole_target_dataset' and any(
                row['target_contract']['source_dataset_id'] == target_contract['source_dataset_id']
                for row in source.parameters.binding['source_prefixes']):
            raise ValueError('source response whole-study exclusion includes the actual target study')

    def check_continuation(self, source, dictionary):
        """Check the saved function's declared relation without recompiling it."""
        from transfer.continuous_decision import ContinuousDecisionModel
        from transfer.native_response_dictionary import NativeResponseDictionary
        from transfer.response_dictionary import ResponseDictionary

        if self.continuation_role == 'original_saved_continuous_function':
            original = ContinuousDecisionModel.load(io.BytesIO(source.original_model_content)).dictionary
            if (not isinstance(dictionary, ResponseDictionary) or original is None
                    or original.digest() != dictionary.digest()):
                raise ValueError('saved historical response function differs from its actual original source')
        else:
            if not isinstance(dictionary, NativeResponseDictionary):
                raise ValueError('saved source-native response requires its declared function owner')
            dictionary.digest()
            terminal = (np.zeros(source.parameters.basis.shape[1]) if self.terminal_policy == 'zero_native_basis_vector'
                        else np.asarray(self.terminal_values))
            if (array_digest(dictionary.matrix) != array_digest(source.parameters.basis)
                    or dictionary.sample_rate_hz != source.parameters.noise.geometry.sfreq
                    or array_digest(dictionary.terminal_values) != array_digest(terminal)):
                raise ValueError('saved native function changed its original basis, clock or declared terminal')


def fit_source_response(source, prefix, compiler, programme, source_ref):
    from research.channel_noise_mechanism import DiffuseProjectionResolutionError
    from research.execution import ExpectedSubjectError, SubjectFailureCode
    from research.finite_response_law import FiniteResponseResolutionError
    from research.history_authority import HistoryUnavailableError
    from research.joint_response_scale import NumericalResolutionError
    from research.response_history_identification import HistoryNuisanceResolutionError
    from research.source_response_states import fit_source_informed_response

    try:
        return fit_source_informed_response(prefix, compiler=compiler,
            parameters=source_ref.parameters(source, prefix), program=programme)
    except (HistoryUnavailableError, FiniteResponseResolutionError, DiffuseProjectionResolutionError,
            NumericalResolutionError, HistoryNuisanceResolutionError, np.linalg.LinAlgError) as error:
        failure = error.record() if hasattr(error, 'record') else {'type': type(error).__name__, 'reason': str(error)}
        raise ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE, stage='source_response_C',
            detail=json.dumps(failure, sort_keys=True)) from error


def score_source_response(model, compiler, prefix, view, *, permission_recorder=None):
    from research.response_history_identification import SparseConditionalHistoryEvidence
    from research.source_response_states import WholeResponseEvidence

    cutoff = int(view.raw_scope[0])+int(view.onsets[0])+model.program.conditioning_offset_samples
    capability = model.history_scope(compiler, view, prefix=prefix, conditioning_stop=cutoff)
    if permission_recorder is not None:
        permission_recorder(capability._original_scope)
    evidence, binding = model.score_history(view, capability)
    if type(evidence) is WholeResponseEvidence:
        resolution = evidence.resolution
        kind = 'whole_response'
    elif type(evidence) is SparseConditionalHistoryEvidence:
        resolution = evidence.mixture_resolution
        kind = 'finite_conditional_mixture'
    else:
        raise TypeError('source response returned an unsupported mathematical evidence owner')
    record = {field.name: (getattr(evidence,field.name).tolist()
        if isinstance(getattr(evidence,field.name),np.ndarray) else getattr(evidence,field.name))
        for field in fields(evidence)}
    prediction = {'schema': 'n2p3_source_response_prediction_artifact/1',
        'core_prediction': binding, 'evidence_kind': kind, 'evidence': record, 'resolution': resolution,
        'adapter_source_sha256': _ADAPTER_SHA256}
    return evidence, prediction, capability


def check_source_response_fit(model, prefix, source_digest, programme, source_ref):
    """Join the saved mathematical producer to actual source and C authority."""
    from transfer.ensemble_views import calibration_prefix_identity

    if (model.parameters.source.digest != source_digest or model.program.digest() != programme.digest()
            or model.binding['prefix_identity'] != calibration_prefix_identity(prefix)
            or model.parameters.binding['continuation_role'] != source_ref.continuation_role):
        raise ValueError('source response producer differs from actual source, C prefix or programme')
    expected = None if prefix.raw is None else array_digest(prefix.raw.samples, prefix.raw.onsets, prefix.raw.target_mask)
    if model.binding['C_input_sha256'] != expected:
        raise ValueError('source response C posterior differs from actual authorized arrays and labels')
    source_ref.check_continuation(model.parameters.source, model.parameters.dictionary)
    source_ref.check_source(model.parameters.source,prefix.target_contract)
