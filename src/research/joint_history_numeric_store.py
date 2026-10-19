"""Shared immutable proposal arrays; numerical methods remain original owners."""
from __future__ import annotations

import hashlib
import io
import uuid
import zipfile
from pathlib import Path

import numpy as np

from research.collapsed_continuous_inference import GaussianMixtureProposal, GaussianProposal
from research.frozen_metadata import freeze_metadata
from research.structured_continuous_proposal import proposal_from_record
from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class SharedHistoryNumericStore:
    """Content-addressed exact array bytes with atomic publication."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._references = {}

    def put(self, array):
        value = np.asarray(array, dtype=np.float64)
        identity = array_digest(value)
        if identity in self._references:
            return self._references[identity]
        native = io.BytesIO()
        np.save(native, value, allow_pickle=False)
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            entry = zipfile.ZipInfo('value.npy', (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, native.getvalue())
        content = payload.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        target = self.directory/(digest+'.npz')
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError('an existing shared numeric array changed its immutable bytes')
        else:
            staging = self.directory/(digest+'.'+uuid.uuid4().hex+'.staging')
            staging.write_bytes(content)
            # Concurrent writers of this SHA publish identical complete bytes.
            staging.replace(target)
        reference = freeze_metadata({'sha256': digest, 'numeric_identity': identity,
            'shape': value.shape, 'bytes': len(content)})
        self._references[identity] = reference
        return reference

    def get(self, reference):
        path = self.directory/(reference['sha256']+'.npz')
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != reference['sha256'] or len(content) != reference['bytes']:
            raise ValueError('shared proposal array differs from its original external dependency')
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            value = archive['value']
        if tuple(value.shape) != tuple(reference['shape']) or array_digest(value) != reference['numeric_identity']:
            raise ValueError('shared proposal array changed its complete numerical coordinates')
        return value


def _store_proposal(proposal, store):
    if type(proposal) is GaussianProposal:
        return {'kind': 'original_GaussianProposal', 'mean': store.put(proposal.mean), 'root': store.put(proposal.root)}
    if type(proposal) is GaussianMixtureProposal:
        return {'kind': 'original_GaussianMixtureProposal', 'log_weights': store.put(proposal.log_weights),
            'components': [_store_proposal(component, store) for component in proposal.components]}
    raise TypeError('this shared codec consumes original Gaussian proposal owners explicitly')


def _load_proposal(record, store):
    if record['kind'] == 'original_GaussianProposal':
        return GaussianProposal(store.get(record['mean']), store.get(record['root']))
    if record['kind'] == 'original_GaussianMixtureProposal':
        return GaussianMixtureProposal(tuple(_load_proposal(row, store) for row in record['components']),
            store.get(record['log_weights']))
    raise ValueError('unknown shared original-proposal numerical kind')


class ReferencedHistoryProposal:
    """Serialization adapter; all sampling/density/move calls retain original code."""

    def __init__(self, proposal, store):
        self.original = proposal
        self._record = freeze_metadata({'kind': 'joint_history_shared_original_proposal/1',
            'store': str(store.directory), 'proposal': _store_proposal(proposal, store),
            'producer_sha256': IMPLEMENTATION_SHA256,
            'sampling_density_and_reversible_move_owner': 'original_GaussianProposal_and_GaussianMixtureProposal'})

    def draw(self, rng, count):
        return self.original.draw(rng, count)

    def log_density(self, points):
        return self.original.log_density(points)

    def reversible_move(self, rng, points, scale):
        return self.original.reversible_move(rng, points, scale)

    def record(self):
        return self._record


def _proposal_fingerprint(proposal):
    if type(proposal) is GaussianProposal:
        return metadata_digest({'kind': 'GaussianProposal', 'complete_arrays': array_digest(proposal.mean, proposal.root)})
    if type(proposal) is GaussianMixtureProposal:
        return metadata_digest({'kind': 'GaussianMixtureProposal', 'log_weights': array_digest(proposal.log_weights),
            'components': [_proposal_fingerprint(value) for value in proposal.components]})
    raise TypeError('proposal recipe fingerprints the original Gaussian numerical owners explicitly')


class RecipeHistoryProposal(ReferencedHistoryProposal):
    """Store the generation recipe and original C state, never candidate roots."""

    def __init__(self, proposal, target, C_posterior, store, scales, *, target_dependency=None):
        from research.joint_history_proposal import IMPLEMENTATION_SHA256 as RECIPE_OWNER_SHA256

        self.original = proposal
        self._record = freeze_metadata({'kind': 'joint_history_complete_reference_proposal_recipe/1',
            'store': str(store.directory), 'target_identity': target.identity,
            'target_dependency': target_dependency,
            'C_posterior': {'upper': store.put(C_posterior.upper), 'rhs': store.put(C_posterior.rhs),
                'log_density': C_posterior.log_density, 'shared_dimension': C_posterior.shared_dimension,
                'drift_coordinate_ids': C_posterior.drift_coordinate_ids, 'binding': C_posterior.binding},
            'original_proposal_fingerprint': _proposal_fingerprint(proposal),
            'component_scales_sha256': array_digest(scales), 'recipe_owner_sha256': RECIPE_OWNER_SHA256,
            'producer_sha256': IMPLEMENTATION_SHA256,
            'replay_operations': 'native_statistic_component_recompile_and_Gaussian_QR_only_zero_new_fits',
            'candidate_mean_or_root_arrays_stored': False,
            'sampling_density_and_reversible_move_owner': 'original_GaussianProposal_and_GaussianMixtureProposal'})


def history_proposal_from_record(record, *, target=None):
    if record['kind'] == 'joint_history_complete_reference_proposal_recipe/1':
        from research.joint_history_proposal import IMPLEMENTATION_SHA256 as RECIPE_OWNER_SHA256
        from research.joint_history_proposal import complete_history_proposal
        from research.joint_response_noise_calibration import JointMarginal

        if (record['producer_sha256'] != IMPLEMENTATION_SHA256 or record['recipe_owner_sha256'] != RECIPE_OWNER_SHA256
                or target is None or target.identity != record['target_identity']):
            raise ValueError('proposal recipe requires its exact actual numeric target and generation source')
        store, row = SharedHistoryNumericStore(record['store']), record['C_posterior']
        posterior = JointMarginal(row['log_density'], store.get(row['upper']), store.get(row['rhs']),
            row['shared_dimension'], tuple(row['drift_coordinate_ids']), row['binding'])
        proposal, scales, _ = complete_history_proposal(target, posterior)
        if (_proposal_fingerprint(proposal) != record['original_proposal_fingerprint']
                or array_digest(scales) != record['component_scales_sha256']):
            raise ValueError('proposal recipe changed an original mean, root, weight, scale or physical measure')
        return proposal
    if record['kind'] != 'joint_history_shared_original_proposal/1':
        return proposal_from_record(record)
    if record['producer_sha256'] != IMPLEMENTATION_SHA256:
        raise ValueError('shared proposal replay requires its original numeric codec source')
    return _load_proposal(record['proposal'], SharedHistoryNumericStore(record['store']))
