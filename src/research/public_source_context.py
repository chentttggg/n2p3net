"""One typed public source view over the original source-admission owners.

The context creates no C grants or fitted parameters. Population and the named
single-person development source retain their distinct original live issuers.
"""
from __future__ import annotations

from dataclasses import dataclass
from weakref import WeakKeyDictionary, ref

import numpy as np

from research.frozen_metadata import freeze_metadata
from research.gaussian_local_factor import SharedResponseChart
from research.public_native_source_admission import (
    PublicNativeSourceAdmission,
    require_public_native_source,
)
from transfer.noise_model import array_digest, metadata_digest

_CONTEXTS, _LIVE = WeakKeyDictionary(), WeakKeyDictionary()


def _source_values(admission):
    if type(admission) is PublicNativeSourceAdmission:
        require_public_native_source(admission)
        model, parameters = admission.model, admission.model.source.parameters
        geometry, basis = model.target_observation_geometry, parameters.basis
        graph = model.graph()
        p, c = 2*basis.shape[1], len(geometry.channels)
        source = SharedResponseChart(graph.source_mean.reshape(p, c), graph.source_factor.reshape(p, c, -1),
            tuple(model.engine.coordinate_ids), geometry.channels, {'source_graph': graph.digest(),
            'source_codec_sha256': model.source.digest, 'source_parameter_authority': parameters.digest,
            'source_role': admission.record['source_role'], 'source_prior_count': 1})
        return source, basis, model.continuation, geometry, None, parameters.noise
    from research.public_population_source import (
        PublicPopulationSourceAdmission,
        require_public_population_source,
    )

    if type(admission) is not PublicPopulationSourceAdmission:
        raise TypeError('public source context requires an explicit original native or population source admission')
    require_public_population_source(admission)
    return (admission.source_chart, admission.basis, admission.dictionary, admission.geometry,
        admission.within_reference, admission.noise_reference)


@dataclass(frozen=True, eq=False)
class PublicSourceContext:
    admission: object
    source_chart: SharedResponseChart
    basis: np.ndarray
    dictionary: object
    geometry: object
    within_reference: object
    noise_reference: object
    compiler: object
    prefixes: tuple
    calibration_inputs: tuple
    record: dict

    @property
    def identity(self):
        return metadata_digest({'record': self.record, 'source': self.source_chart.identity,
            'basis': array_digest(self.basis), 'dictionary': self.dictionary.digest(),
            'geometry': self.geometry.record(),
            'within': None if self.within_reference is None else self.within_reference.identity,
            'noise_reference': self.noise_reference._runtime_state_digest()})


def public_source_context(admission):
    existing = _CONTEXTS.get(admission)
    if existing is not None and existing() is not None:
        return require_public_source_context(existing())
    source, basis, dictionary, geometry, within, noise = _source_values(admission)
    noise.assert_integrity()
    record = freeze_metadata({'schema': 'typed_original_public_source_context/1',
        'admission_type': type(admission).__name__, 'source_admission': metadata_digest(admission.record),
        'source_role': admission.record['source_role'], 'complete_shared_factor': array_digest(source.mean, source.factor),
        'source_prior_count': 1, 'C_admission_owner': 'original_public_record_physical_bridge',
        'additional_source_fits_or_C_grants': 0})
    value = PublicSourceContext(admission, source, basis, dictionary, geometry, within, noise,
        admission.compiler, admission.prefixes, admission.calibration_inputs, record)
    _CONTEXTS[admission] = ref(value)
    _LIVE[value] = value.identity
    return value


def require_public_source_context(value):
    if type(value) is not PublicSourceContext or value not in _LIVE:
        raise TypeError('public source use requires its original typed source context')
    source, basis, dictionary, geometry, within, noise = _source_values(value.admission)
    expected = value.admission
    if (value.compiler is not expected.compiler or value.prefixes is not expected.prefixes
            or value.calibration_inputs is not expected.calibration_inputs
            or value.source_chart.identity != source.identity or array_digest(value.basis) != array_digest(basis)
            or value.dictionary is not dictionary or value.geometry != geometry or value.within_reference is not within
            or value.noise_reference is not noise or value.identity != _LIVE[value]):
        raise ValueError('the original complete source or its live public C/clock context changed')
    value.noise_reference.assert_integrity()
    return value
