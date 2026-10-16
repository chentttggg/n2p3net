"""Actual mixed NN/joint packages share original NN bytes across prefix doses."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np

from research.joint_flat_authority import build_source_parameters
from research.joint_flat_replay import (
    NN_SOURCE_REFERENCE_SCHEMA,
    JointFlatPackageAdmission,
    JointFlatReplayRecorder,
    persist_joint_flat_prefix,
)
from research.source_metadata_store import (
    SourceMetadataStore,
    export_source_package_closure,
    source_metadata_scope,
)
from tests.research.test_joint_flat_response import compiler_fixture
from tests.research.test_source_storage_prefix import _case
from train.runtime import cpu_thread_budget
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import (
    CalibrationExpertProgram,
    FrozenSourceCheckpoint,
    fit_calibration_prefix_programs,
)
from transfer.joint_flat_programme import (
    FrozenJointFlatSource,
    JointFlatProgram,
    JointFlatSamplingProgram,
)


def test_mixed_zero_positive_packages_share_one_nn_blob_and_export_without_original_store(tmp_path):
    with cpu_thread_budget(1):
        views, nn_path, nn_spec, _ = _case(tmp_path)
        nn_bytes = nn_path.read_bytes()
        source_issuer = compiler_fixture('A02.mat', 840)
        source_prefix = source_issuer.compile(120)
        joint_npz = tmp_path/'joint-original.npz'
        parameters, _ = build_source_parameters(((source_issuer, source_prefix),),
            target_participant_key=views.calibration[0].target_contract['participant_key'],
            degrees_of_freedom=4., relative_noise_shape=1., basis=np.ones((8, 1)), ridge=1.,
            noise_order=0, conditioning_samples=4, source_model_path=joint_npz)
        joint_source = FrozenJointFlatSource.from_parameters(parameters, original_model_content=joint_npz.read_bytes())
        sampler = JointFlatSamplingProgram(4., 8, 8, (2, 3), 1, 2, 1., 1e9, 0., 0., 1e3)
        joint_spec = CalibrationExpertProgram('joint', 'joint_flat',
            joint_flat=JointFlatProgram(sampler, 32, 'whole_observed_current_q'))
        scratch = replace(nn_spec, expert_id='scratch', recipe=replace(nn_spec.recipe,
            name='scratch', initialization='scratch', input_statistics='target_prefix'))
        nn_source = FrozenSourceCheckpoint(nn_bytes)
        run = tmp_path/'run'
        run.mkdir()
        store = SourceMetadataStore(run/'source_store')
        packages = {}
        for budget, specs in ((0, (joint_spec, nn_spec, scratch)), (200, (joint_spec, nn_spec))):
            prefix = views.calibration[budget]
            programme = EnsembleProgram.equal_weights('mixed_storage', tuple(spec.expert_id for spec in specs))
            sources = {spec.expert_id: joint_source if spec.family == 'joint_flat' else nn_source for spec in specs}
            context = {'source_seed': 0, 'target_seed': 0, 'requested_Ks': [budget],
                'requested_to_actual_K': {str(budget): budget}, 'test_scope': views.ledger['test_scope'],
                'requested_decision_ids': [], 'evidence_scope': 'software_storage_only'}
            recorder = JointFlatReplayRecorder()
            programs = fit_calibration_prefix_programs(prefix, specs, programme, strategies=('fixed',),
                source_checkpoints=sources, prefix_compiler=views.prefix_compiler,
                inner_replay_recorder=recorder, inner_replay_context=context)
            assert programs['fixed'].expert_fit_available['joint']
            assert programs['fixed'].expert_fit_available[nn_spec.expert_id]
            if budget == 200:
                assert programs['fixed']._experts[1].predictor.source_contract_reference is None
            package = run/f'K{budget}'
            with source_metadata_scope(store):
                persist_joint_flat_prefix(prefix, programs, package, recorder=recorder, context=context)
            packages[budget] = package
            manifest = json.loads((package/'prefix_inner_replay.json').read_bytes())['record']
            for name, row in manifest['original_source_files'].items():
                if name == 'joint':
                    assert row['path'].endswith('.json')
                else:
                    assert row['schema'] == NN_SOURCE_REFERENCE_SCHEMA and 'path' not in row
                    assert row['reference']['checkpoint_sha256'] == hashlib.sha256(nn_bytes).hexdigest()
            admitted = JointFlatPackageAdmission(package)
            assert admitted._metadata_admission.source_admissions == 1
            assert admitted.sources[nn_spec.expert_id].content == nn_bytes
            if budget == 0:
                assert manifest['original_source_files']['scratch']['role'] == 'architecture_only'
                assert admitted.sources['scratch'].content == nn_bytes
        assert len([path for path in run.rglob('*.pt') if path.read_bytes() == nn_bytes]) == 1
        # Positive /2 has no model-owned source reference: its typed transport
        # must independently supply the complete export dependency closure.
        entry = export_source_package_closure(packages[200], tmp_path/'isolated')
        assert len([path for path in (tmp_path/'isolated').rglob('*.pt') if path.read_bytes() == nn_bytes]) == 1
        offline = run/'source_store.offline'
        assert store.directory.parent == offline.parent.resolve()
        store.directory.rename(offline)
        restored = JointFlatPackageAdmission(entry)
        assert restored.sources[nn_spec.expert_id].content == nn_bytes
        assert restored._metadata_admission.source_admissions == 1
        assert restored.sources['joint'].original_model_content == joint_source.original_model_content
