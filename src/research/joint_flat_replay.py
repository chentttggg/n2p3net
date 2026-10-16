"""Same-run original history posterior capture and model-only prefix replay.

The numerical package stores the original draws and complete q components.
Original sample permission remains in the existing input capsule, whose
prepared compiler restores only original issuer histories. Replay runs no
sampler or fitter and the public parent reconstructs the original decoder.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from models.candidate_evidence import CandidateEvidence
from research.contracts import semantic_sha256
from research.execution import ExpectedSubjectError
from research.nested_selection_replay import InnerReplayRecorder
from transfer.ensemble import EnsembleScorer
from transfer.ensemble_fit import FittedEnsemble
from transfer.joint_flat_model import (
    JointFlatPredictionCapture,
    _restore_sampling,
    _sampling_record,
)
from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation
from transfer.noise_model import array_digest

SCHEMA = 'n2p3_joint_flat_prefix_inner_replay/1'
NN_SOURCE_REFERENCE_SCHEMA = 'n2p3_joint_source_checkpoint_reference/1'


def _nn_source_reference(prefix, spec, source, preparation):
    from data.identity import source_dataset_ids_from_identity
    from data.lineage import DataLineage
    from data.study import _from_metadata

    target = prefix.trial_target(spec.profile)
    subject = target.identity_table.local_subject_ids[0]
    prepared = preparation.prepare(source.content, target, subject)
    lineage = DataLineage.source(parameters={'source_dataset_ids': list(source_dataset_ids_from_identity(target.identity_table)),
        'target_contract': prefix.target_contract, 'source_provenance': target.provenance})
    membership = _from_metadata(target.identity_table, lineage, target.provenance)
    prepared.assert_library_exclusion(target_membership=membership)
    context = prepared.source_storage_context(subject)
    if context is None or context[0] != source.content:
        raise ValueError('joint NN byte transport requires its original admitted checkpoint and current target binding')
    return context[1]


def _joint_source_dependencies(directory, packages, source_files, *, store=None):
    from research.source_metadata_store import (
        SourceMetadataStore,
        collect_source_dependencies,
        source_dependency_descriptor,
    )

    numeric = collect_source_dependencies(directory, packages)
    references = [] if numeric is None else list(numeric['references'])
    if numeric is not None:
        actual_store = SourceMetadataStore(Path(directory)/numeric['directory'])
        if store is not None and store.directory != actual_store.directory:
            raise ValueError('joint numerical and transport dependencies use different actual source stores')
        store = actual_store
    for row in source_files.values():
        if row.get('schema') == NN_SOURCE_REFERENCE_SCHEMA:
            if set(row) != {'schema', 'family', 'role', 'reference'} or row['family'] != 'trial':
                raise ValueError('joint source transport requires its complete explicit NN reference schema')
            references.append(row['reference'])
        elif set(row) != {'path', 'sha256', 'bytes'}:
            raise ValueError('joint source transport has an unsupported source artifact schema')
    if not references:
        return None
    if store is None:
        raise ValueError('joint NN source references require their actual shared source store locator')
    return source_dependency_descriptor(directory, store, references)


def _check_nn_source_target(reference, spec, target_contract):
    from transfer.ensemble_views import _trial_target_from_contract

    target = _trial_target_from_contract(target_contract, spec.profile)
    expected = {'subject': str(target.identity_table.local_subject_ids[0]), 'identity': target.identity_table.payload(),
        'preprocessing': target.preprocessing.record(), 'channel_names': list(target.channel_names),
        'provenance': target.provenance}
    binding = reference['target_binding']
    if any(semantic_sha256(binding[key]) != semantic_sha256(value) for key, value in expected.items()):
        raise ValueError('joint NN source transport differs from its actual target and expert input programme')


def _authority_record(prefix, specs, program, sources, context):
    from research.prefix_inner_replay import (
        _check_aliases,
        _check_seeds,
        _source_roles,
        prefix_input_identity,
        prefix_native_fit_inputs,
    )

    aliases, mapping = _check_aliases(prefix, context)
    identity = prefix_input_identity(prefix)
    return {'participant_key': prefix.target_contract['participant_key'], 'actual_K': prefix.budget,
        'prefix_input': identity, 'prefix_identity': identity,
        'prefix_sha256': semantic_sha256(prefix.record()), 'prefix_record_sha256': semantic_sha256(prefix.record()),
        'native_fit_inputs': prefix_native_fit_inputs(prefix, specs),
        'expert_programs': [spec.record() for spec in specs], 'ensemble_program': program.record(),
        'source_checkpoint_digests': {name: source.digest for name, source in sources.items()},
        'source_checkpoint_roles': _source_roles(specs, sources),
        'actual_source_seeds': _check_seeds(specs, sources, context),
        'source_seed': context.get('source_seed'), 'target_seed': context.get('target_seed'),
        'requested_Ks': aliases, 'requested_to_actual_K': mapping,
        'requested_denominator': {'requested_Ks': aliases, 'actual_K': prefix.budget,
            'source_seed': context.get('source_seed'), 'target_seed': context.get('target_seed')},
        'fit_context': context,
        'source_exclusion_scope': {spec.expert_id: spec.joint_flat.source_exclusion_scope
                                   for spec in specs if spec.family == 'joint_flat'}}


def _score_programs(programs, view, *, prefix, specs, **history):
    """Keep an unavailable probability learner in the same requested roster."""
    from research.prefix_inner_replay import unavailable_prefix_expert_scores

    rows = FittedEnsemble.score_programs(programs, view, **history)
    return {name: (unavailable_prefix_expert_scores(prefix.record(),
        [spec.record() for spec in specs], name,
        evidence.record(subject=view.participant_key), view) if isinstance(evidence, ExpectedSubjectError) else evidence)
        for name, evidence in rows.items()}


def _replay_expert_scores(model, view, *, compiler, prefix, captures, saved, permissions, refusals=None):
    """Recompute captured scores; preserve original unavailable attempts without RNG."""
    result = {}
    for expert in model._experts:
        name = expert.program.expert_id
        original = CandidateEvidence.from_record(saved[name])
        if expert.program.family == 'joint_flat' and name not in captures:
            if original.probability_available:
                raise ValueError('available joint prediction has no original captured P measure')
            if name in permissions:
                permission = permissions[name]
                stop = permission['binding']['conditioning_sample_support'][1]
                capability = compiler.record_read_scope(view, prefix=prefix, conditioning_stop=stop)
                if capability.digest != permission['sha256']:
                    raise ValueError('preserved unavailable history changed its original issued permission')
                expert.predictor.validate_history_permission(view, capability)
            elif refusals is not None and name in refusals:
                from research.history_authority import HistoryUnavailableError
                from transfer.ensemble_views import _lookup_record_history_refusal

                expected = refusals[name]
                request = expected['binding']['request']
                actual_stop = int(view.onsets[0])+view.raw_scope[0]+expert.program.joint_flat.conditioning_samples
                if (request['conditioning_stop'] != actual_stop or request['available_stop'] != view.available_sample
                        or request['program'] != 'record_prequential_public_history/1'):
                    raise ValueError('joint refusal belongs to another actual model history request')
                try:
                    compiler.record_read_scope(view, prefix=prefix, conditioning_stop=request['conditioning_stop'],
                        available_stop=request['available_stop'], program=request['program'])
                except HistoryUnavailableError as error:
                    actual = _lookup_record_history_refusal(compiler, view, prefix=prefix,
                        conditioning_stop=request['conditioning_stop'], available_stop=request['available_stop'],
                        program=request['program'])
                    if (actual.digest != expected['sha256']
                            or semantic_sha256(actual.binding) != semantic_sha256(expected['binding'])
                            or semantic_sha256(error.record()) != semantic_sha256(actual.binding['error'])
                            or original.failure_reason != json.dumps(error.record(), sort_keys=True, allow_nan=False)
                            or original.failure_kind != error.failure_kind):
                        raise ValueError('joint exact-request history refusal changed during read-only replay') from error
                else:
                    raise ValueError('original refused history request unexpectedly obtained a reading capability')
            elif expert.predictor is not None:
                allowed = {None, 'previous_unknown_decision_overlaps_finite_response_support',
                    'next_unknown_decision_overlaps_finite_response_support',
                    'previous_unknown_selection_overlaps_response', 'next_unknown_selection_overlaps_response'}
                if view.unavailable_reason not in allowed:
                    if original.failure_reason != view.unavailable_reason:
                        raise ValueError('original embargo failure differs from its actual view')
                elif view.decision.schedule_complete:
                    from research.history_authority import HistoryUnavailableError

                    stop = int(view.onsets[0])+view.raw_scope[0]+expert.program.joint_flat.conditioning_samples
                    try:
                        compiler.record_read_scope(view, prefix=prefix, conditioning_stop=stop)
                    except HistoryUnavailableError as error:
                        raise ValueError('original history issuer failure lacks its exact-request refusal witness') from error
                    else:
                        raise ValueError('joint failure lacks its original issued history permission')
            evidence = expert.unavailable(view, original.failure_reason, original.failure_kind)
        else:
            evidence = expert.score(view, history_context=(compiler, prefix), captured_history=captures.get(name))
        if (expert.program.family == 'joint_flat' and name not in captures
                and semantic_sha256(evidence.record()) != semantic_sha256(original.record())):
            raise ValueError('original captured joint evidence or preserved unavailable identity changed')
        result[name] = evidence
    return result


def _permission_records(permissions):
    return {name: {'sha256': cap.digest, 'binding': cap.binding} for name, cap in permissions.items()}


def _check_joint_authority(record, loaded):
    from research.prefix_inner_replay import _check_final_authority, _prefix_record

    authority = _check_final_authority(record, loaded)
    identity = record['prefix_input']
    if (semantic_sha256(record['prefix_identity']) != semantic_sha256(identity)
            or record['prefix_sha256'] != record['prefix_record_sha256']):
        raise ValueError('joint top prefix identity differs from its native fitted authority')
    prefix_content = {'prefix': _prefix_record(identity),
        'raw': None if identity['raw'] is None else identity['raw']['fit_input_digest'],
        'trial': {name: row['numerical_input_digest'] for name, row in identity['trial_datasets'].items()}}
    if identity.get('native_records'):
        prefix_content['native_records'] = [{'record_id': row['record_id'], 'input_sha256': row['fit_input_digest']}
                                             for row in identity['native_records']]
    scopes = {spec['expert_id']: spec['joint_flat']['source_exclusion_scope']
              for spec in record['expert_programs'] if spec['family'] == 'joint_flat'}
    if scopes != record['source_exclusion_scope']:
        raise ValueError('joint source exclusion declaration differs from its actual expert programme')
    for model in loaded.values():
        for expert in model._experts:
            if expert.program.family != 'joint_flat' or expert.predictor is None:
                continue
            predictor, target = expert.predictor, identity['target_contract']
            provenance = predictor.provenance
            if (provenance['original_C_prefix_identity'] != semantic_sha256(prefix_content)
                    or semantic_sha256(provenance['target_contract']) != semantic_sha256(target)
                    or provenance['requested_labels'] != record['actual_K']
                    or provenance['source_checkpoint_sha256'] != record['source_checkpoint_digests'][expert.program.expert_id]
                    or tuple(predictor.calibration_record_ids) != tuple(row['C_record_id'] for row in predictor.calibration_inputs)):
                raise ValueError('joint actual C posterior differs from original source, dose, record order or physical prefix')
            predictor.source.assert_target(target)
            if scopes[expert.program.expert_id] == 'whole_target_dataset' and any(
                    row['target_contract']['source_dataset_id'] == target['source_dataset_id']
                    for row in predictor.source.parameters.binding['source_prefixes']):
                raise ValueError('joint whole-study source exclusion includes the actual target study')
    return authority


def _artifact(path, directory):
    path = Path(path)
    return {'path': path.relative_to(directory).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def _save_capture(path, capture):
    state = {'schema': 'n2p3_joint_flat_actual_prefix_measure/1', 'sampling': _sampling_record(capture.samples),
        'binding': capture.binding, 'diagnostics': capture.diagnostics, 'capture_sha256': capture.digest,
        'components_sha256': array_digest(capture.logweights, capture.logdensity)}
    np.savez_compressed(path, metadata_json=np.frombuffer(json.dumps(state, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf8'), dtype=np.uint8),
        coefficients=capture.samples.coefficients, precisions=capture.samples.precisions,
        targets=capture.samples.targets, logweights=capture.logweights, logdensity=capture.logdensity)


def _load_capture(path):
    with np.load(path, allow_pickle=False) as archive:
        state = json.loads(archive['metadata_json'].tobytes())
        if state['schema'] != 'n2p3_joint_flat_actual_prefix_measure/1':
            raise ValueError('captured original P measure has an unsupported numerical schema')
        samples = _restore_sampling(state['sampling'], archive['coefficients'], archive['precisions'], archive['targets'])
        result = JointFlatPredictionCapture(samples, archive['logweights'], archive['logdensity'],
                                           state['binding'], state['diagnostics'])
    if (result.digest != state['capture_sha256']
            or array_digest(result.logweights, result.logdensity) != state['components_sha256']):
        raise ValueError('original P sampling/component arrays changed')
    return result


class JointFlatReplayRecorder(InnerReplayRecorder):
    """Original fit call sink; no reconstruction of a numerical producer."""
    def __init__(self):
        super().__init__()
        self.joint_oof = []
        self.joint_future = []

    def record_joint_forward_oof(self, *, earlier, held, fitted, row, context, actual_captures, issued_capabilities,
                                 issued_refusals=None):
        record = {'prefix': earlier, 'held': held, 'model': fitted, 'row': row,
            'context': context, 'captures': dict(actual_captures), 'issued_capabilities': dict(issued_capabilities)}
        if issued_refusals:
            record['issued_refusals'] = dict(issued_refusals)
        self.joint_oof.append(record)

    def score_future(self, programs, view, *, compiler, prefix):
        captures, issued, refusals = {}, {}, {}
        def remember(name, capability, capture):
            captures[name] = (capability, capture)
        def remember_permission(name, capability):
            issued[name] = capability
        def remember_refusal(name, refusal):
            refusals[name] = refusal
        if any(row['view'].record_id == view.record_id for row in self.joint_future):
            raise ValueError('joint future recorder repeats an original requested decision')
        specs = self.prefix_fit_declaration['expert_programs']
        scored = _score_programs(programs, view, prefix=prefix, specs=specs,
            history_context=(compiler, prefix), history_recorder=remember,
            history_capability_recorder=remember_permission, history_refusal_recorder=remember_refusal)
        for strategy, evidence in scored.items():
            self.record_prefix_future_score(strategy, view=view, evidence=evidence)
        record = {'view': view, 'captures': captures, 'issued_capabilities': issued,
            'scores': {name: {expert: item.record() for expert, item in rows.items()} for name, rows in scored.items()}}
        if refusals:
            record['issued_refusals'] = refusals
        self.joint_future.append(record)
        return scored

    @property
    def original_history_capabilities(self):
        result = {}
        for row in (*self.joint_oof, *self.joint_future):
            for cap in row['issued_capabilities'].values():
                result[cap.digest] = cap
        return tuple(result.values())

    @property
    def original_history_refusals(self):
        result = {}
        for row in (*self.joint_oof, *self.joint_future):
            for refusal in row.get('issued_refusals', {}).values():
                result[refusal.digest] = refusal
        return tuple(result.values())


def persist_joint_flat_prefix(prefix, programs, directory, *, recorder, context=None, device='cpu', **_declarations):
    from research.prefix_inner_replay import _fit_record, prefix_input_identity

    if not isinstance(recorder, JointFlatReplayRecorder) or recorder.prefix_fit_declaration['prefix'] is not prefix:
        raise TypeError('joint prefix persistence needs the original same-run recorder and C prefix')
    if context is not None and semantic_sha256(context) != semantic_sha256(recorder.prefix_fit_declaration['context']):
        raise ValueError('joint prefix package changes the original execution context')
    if (set(programs) != set(recorder.prefix_programs)
            or any(programs[name] is not recorder.prefix_programs[name] for name in programs)):
        raise ValueError('joint prefix persistence changed the actual shared final probability programmes')
    declaration = recorder.prefix_fit_declaration
    specs, program, sources = declaration['expert_programs'], declaration['program'], declaration['source_checkpoints']
    context = declaration['context']
    if semantic_sha256(prefix_input_identity(prefix)) != semantic_sha256(declaration['input_identity']):
        raise ValueError('joint original C samples, labels or scope changed after the actual fit')
    for key, expected in (('expert_programs', [spec.record() for spec in specs]), ('program', program.record()),
                          ('source_checkpoints', {name: source.digest for name, source in sources.items()})):
        supplied = _declarations.get(key)
        if supplied is not None:
            actual = ([spec.record() for spec in supplied] if key == 'expert_programs' else
                      supplied.record() if key == 'program' else {name: source.digest for name, source in supplied.items()})
            if semantic_sha256(actual) != semantic_sha256(expected):
                raise ValueError('joint persistence changes the original '+key+' declaration')
    from research.source_metadata_store import current_source_metadata_store
    from transfer.checkpoint import ScopedSourceCheckpointPreparation

    store = current_source_metadata_store()
    by_id = {spec.expert_id: spec for spec in specs}
    if any(by_id[name].family == 'trial' for name in sources) and store is None:
        raise ValueError('mixed joint NN persistence requires an explicit shared source metadata scope')
    preparation = _declarations.get('_source_preparation') or ScopedSourceCheckpointPreparation()
    directory = Path(directory)
    FittedEnsemble.save_programs(directory, programs, participant_key=prefix.target_contract['participant_key'])
    (directory/'captured_measures').mkdir()
    (directory/'inner_models').mkdir()
    (directory/'original_sources').mkdir()
    source_files = {}
    for name, source in sources.items():
        spec = by_id[name]
        if spec.family == 'joint_flat':
            file = directory/'original_sources'/f'{source.digest}.json'
            if not file.exists():
                file.write_bytes(source.content)
            source_files[name] = _artifact(file, directory)
        else:
            reference = _nn_source_reference(prefix, spec, source, preparation)
            if store.publish(source.content) != source.digest:
                raise ValueError('joint NN source publisher changed its original checkpoint bytes')
            source_files[name] = {'schema': NN_SOURCE_REFERENCE_SCHEMA, 'family': spec.family,
                'role': spec.source_checkpoint_role, 'reference': reference}
    oof, future = [], []
    for ordinal, original in enumerate(recorder.joint_oof):
        path = directory/'inner_models'/f'{ordinal:03d}'
        inner = original['model']
        if inner is not None:
            inner.save(path)
        captured = {}
        for name, (capability, capture) in original['captures'].items():
            file = directory/'captured_measures'/f'OOF_{ordinal:03d}_{len(captured):03d}.npz'
            _save_capture(file, capture)
            captured[name] = {'capture': _artifact(file, directory), 'history_capability_sha256': capability.digest,
                              'capture_sha256': capture.digest}
        inner_specs = tuple(spec for spec in specs if spec.expert_id in original['row']['expert_fit_digests'])
        inner_program = (inner.scorer.program if inner is not None else
            type(program).equal_weights(program.name+':forward_temperature', tuple(spec.expert_id for spec in inner_specs)))
        inner_sources = {name: source for name, source in sources.items() if name in inner_program.expert_ids}
        inner_context = {**context, 'requested_Ks': [original['prefix'].budget],
                         'requested_to_actual_K': {str(original['prefix'].budget): original['prefix'].budget}}
        oof.append({'earlier_actual_K': original['prefix'].budget,
            'earlier_prefix_sha256': semantic_sha256(original['prefix'].record()),
            'authority': _authority_record(original['prefix'], inner_specs, inner_program, inner_sources, inner_context),
            'issued_history_permissions': _permission_records(original['issued_capabilities']),
            'held_view': original['held'].view.record(),
            'model_directory': None if inner is None else path.relative_to(directory).as_posix(),
            'row': original['row'], 'captures': captured,
            **({'issued_history_refusals': _permission_records(original['issued_refusals'])}
               if original.get('issued_refusals') else {})})
    for ordinal, original in enumerate(recorder.joint_future):
        captured = {}
        for name, (capability, capture) in original['captures'].items():
            file = directory/'captured_measures'/f'future_{ordinal:03d}_{len(captured):03d}.npz'
            _save_capture(file, capture)
            captured[name] = {'capture': _artifact(file, directory), 'history_capability_sha256': capability.digest,
                              'capture_sha256': capture.digest}
        future.append({'view': original['view'].record(), 'scores': original['scores'], 'captures': captured,
                       'issued_history_permissions': _permission_records(original['issued_capabilities']),
                       **({'issued_history_refusals': _permission_records(original['issued_refusals'])}
                          if original.get('issued_refusals') else {})})
    record = {**_authority_record(prefix, specs, program, sources, context), 'schema': SCHEMA,
        'device': str(device),
        'actual_K': prefix.budget, 'prefix_sha256': semantic_sha256(prefix.record()),
        'prefix_identity': recorder.prefix_fit_declaration['input_identity'],
        'programs': {name: _fit_record(value, participant_key=prefix.target_contract['participant_key'])
                     for name, value in programs.items()},
        'strategy_order': list(programs),
        'future_score_rows': recorder.prefix_future_scores,
        'original_source_files': source_files,
        'inner_producers': [{'fit_key': semantic_sha256(row['authority']),
            'oof_rows': [row['row']] if row['row']['status'] == 'scored' else [],
            'oof_score_failures': [] if row['row']['status'] == 'scored' else [row['row']]}
            for row in oof],
        'OOF': oof, 'future': future, 'selection_performed': False, 'truth_free_future_replay': True,
        'posterior_replay': 'captured_original_P_states_no_random_sampler_or_parameter_fit',
        'scientific_endpoint': 'original_independent_future_full_selection',
        'source_exclusion_scope': {spec.expert_id: spec.joint_flat.source_exclusion_scope
            for spec in specs if spec.family == 'joint_flat'}}
    source_dependencies = _joint_source_dependencies(directory,
        [directory, *(directory/row['model_directory'] for row in oof if row['model_directory'] is not None)],
        source_files, store=store)
    if source_dependencies is not None:
        record['source_metadata'] = source_dependencies
    record['members'] = [_artifact(file, directory) for file in sorted(directory.rglob('*')) if file.is_file()]
    _check_joint_authority(record, {name: value for name, value in programs.items() if isinstance(value, FittedEnsemble)})
    payload = {'record': record, 'record_sha256': semantic_sha256(record)}
    (directory/'prefix_inner_replay.json').write_text(json.dumps(payload, sort_keys=True, indent=2,
        allow_nan=False), encoding='utf8')
    recorder.prefix_receipt_sha256 = payload['record_sha256']
    return directory


def _check_member_references(directory, record):
    """Every consumed source, model and capture belongs to the verified member set."""
    members = {row['path']: row for row in record['members']}
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob('*')
              if path.is_file() and path != directory/'prefix_inner_replay.json'}
    if actual != set(members):
        raise ValueError('joint package files differ from its complete saved member roster')
    references = [row for row in record['original_source_files'].values()
                  if row.get('schema') != NN_SOURCE_REFERENCE_SCHEMA]
    references.extend(entry['capture'] for row in (*record['OOF'], *record['future'])
                      for entry in row['captures'].values())
    if any(members.get(row['path']) != row for row in references):
        raise ValueError('joint source or captured measure reference is outside the verified package members')
    for row in record['OOF']:
        relative = row['model_directory']
        if relative is None:
            continue
        path = directory/relative
        if (path.is_symlink() or directory.resolve() not in path.resolve().parents
                or (Path(relative)/'ensemble.json').as_posix() not in members):
            raise ValueError('joint forward model directory is outside the verified package members')


class JointFlatPackageAdmission:
    """Original package bytes, and only input-owner restored histories."""
    def __init__(self, directory, *, expected_record_sha256=None, metadata_store=None, _metadata_admission=None,
                 _joint_source_preparation=None):
        from research.prefix_inner_replay import _check_program_declarations
        from research.source_metadata_store import (
            SourceMetadataStore,
            _SourceMetadataAdmission,
            admit_source_dependencies,
        )

        self.directory = Path(directory)
        if self.directory.is_symlink() or (self.directory/'prefix_inner_replay.json').is_symlink():
            raise ValueError('joint original numerical package path is a symlink')
        payload = json.loads((self.directory/'prefix_inner_replay.json').read_bytes())
        self.record = payload['record']
        if (self.record['schema'] != SCHEMA or semantic_sha256(self.record) != payload['record_sha256']
                or expected_record_sha256 is not None and expected_record_sha256 != payload['record_sha256']
                or self.record['truth_free_future_replay'] is not True or self.record['selection_performed'] is not False):
            raise ValueError('joint original prefix package receipt changed')
        member_paths = [member['path'] for member in self.record['members']]
        if len(member_paths) != len(set(member_paths)):
            raise ValueError('joint package member roster repeats an artifact')
        for member in self.record['members']:
            path = self.directory/member['path']
            if (path.is_symlink() or self.directory.resolve() not in path.resolve().parents
                    or _artifact(path, self.directory) != member):
                raise ValueError('joint saved source/model/captured measure package member changed')
        _check_member_references(self.directory, self.record)
        self._metadata_store = metadata_store
        descriptor = self.record.get('source_metadata')
        transport_store = (metadata_store if metadata_store is not None else
            None if descriptor is None else SourceMetadataStore(self.directory/descriptor['directory']))
        actual_dependencies = _joint_source_dependencies(self.directory, [self.directory, *(
            self.directory/row['model_directory'] for row in self.record['OOF'] if row['model_directory'] is not None)],
            self.record['original_source_files'], store=transport_store)
        if semantic_sha256(descriptor) != semantic_sha256(actual_dependencies):
            raise ValueError('joint package changed its complete final and OOF source dependency closure')
        self._metadata_admission = (_metadata_admission or _SourceMetadataAdmission(store=metadata_store)
            if descriptor is None else admit_source_dependencies(descriptor, self.directory,
                store=metadata_store, admission=_metadata_admission))
        self._source_preparation = (_joint_source_preparation if _joint_source_preparation is not None
                                    else FrozenJointFlatSourcePreparation())
        actual_programs, loaded = FittedEnsemble._admit_programs(self.directory,
            metadata_store=metadata_store, _metadata_admission=self._metadata_admission,
            _joint_source_preparation=self._source_preparation)
        _check_program_declarations(self.record, loaded, actual_programs['programs'])
        self.authority = _check_joint_authority(self.record, loaded)
        self.programs = {name: (loaded[name] if row['status'] == 'completed' else ExpectedSubjectError(
            row['failure']['error_code'], stage=row['failure']['stage'], detail=row['failure']['detail']))
            for name, row in self.record['programs'].items()}
        from transfer.ensemble_fit import CalibrationExpertProgram, FrozenSourceCheckpoint

        specs = {row['expert_id']: CalibrationExpertProgram.from_record(row) for row in self.record['expert_programs']}
        self.sources = {}
        for name, row in self.record['original_source_files'].items():
            spec = specs[name]
            if row.get('schema') == NN_SOURCE_REFERENCE_SCHEMA:
                if spec.family != row['family'] or spec.source_checkpoint_role != row['role']:
                    raise ValueError('joint NN source transport changed its actual expert family or checkpoint role')
                reference = row['reference']
                _check_nn_source_target(reference, spec, self.record['prefix_input']['target_contract'])
                self._metadata_admission.resolve(reference)
                content = self._metadata_admission.content(reference)
            else:
                content = (self.directory/row['path']).read_bytes()
            digest = self.record['source_checkpoint_digests'][name]
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError('joint original source file differs from its original declared checkpoint')
            self.sources[name] = (self._source_preparation.prepare(content, expected_sha256=digest)
                                  if spec.family == 'joint_flat' else FrozenSourceCheckpoint(content))
        if set(self.sources) != set(self.record['source_checkpoint_digests']):
            raise ValueError('joint package lost its complete original source roster')
        target = self.record['prefix_input']['target_contract']
        for spec in self.record['expert_programs']:
            if spec['family'] != 'joint_flat':
                continue
            source = self.sources[spec['expert_id']]
            source.assert_target(target)
            if spec['joint_flat']['source_exclusion_scope'] == 'whole_target_dataset' and any(
                    row['target_contract']['source_dataset_id'] == target['source_dataset_id']
                    for row in source.parameters.binding['source_prefixes']):
                raise ValueError('joint whole-study source exclusion includes the actual target study')
        self._check_saved_rosters()
        for model in loaded.values():
            if model.oof_probability_record is not None:
                original = model.oof_probability_record
                if (original['schema'] != 'n2p3_joint_flat_forward_probability_OOF/1'
                        or semantic_sha256(original['rows']) != semantic_sha256([row['row'] for row in self.record['OOF']])
                        or semantic_sha256(original['OOF_expert_programs']) != semantic_sha256(
                            self.record['OOF'][0]['authority']['expert_programs'])
                        or semantic_sha256(original['temperature_calibration']['OOF_groups']) != semantic_sha256(
                            [{key: value for key, value in row['row'].items() if key != 'expert_evidence'}
                             for row in self.record['OOF']])):
                    raise ValueError('joint temperature evidence differs from its actual captured forward numerical producers')
        for row in self.record['OOF']:
            inner_sources = row['authority']['source_checkpoint_digests']
            if (not set(inner_sources) <= set(self.record['source_checkpoint_digests'])
                    or any(digest != self.record['source_checkpoint_digests'][name] for name, digest in inner_sources.items())):
                raise ValueError('forward joint model changed the original declared source checkpoint subset')
            requested = {spec['expert_id']: spec for spec in self.record['expert_programs']}
            if any(semantic_sha256(spec) != semantic_sha256(requested[spec['expert_id']])
                   for spec in row['authority']['expert_programs']):
                raise ValueError('forward joint numerical producer changed its actual expert programme subset')

    def _check_saved_rosters(self):

        record = self.record
        oof_ids = [row['held_view']['record_id'] for row in record['OOF']]
        groups = [row['record_id'] for row in record['prefix_input']['complete_groups']]
        needs_oof = any(name != 'fixed' and (row['status'] == 'completed'
            or row['failure']['error_code'] != 'insufficient_calibration_groups')
            for name, row in record['programs'].items())
        if (len(set(oof_ids)) != len(oof_ids) or (needs_oof and oof_ids != groups)
                or not needs_oof and oof_ids):
            raise ValueError('joint original forward OOF lost its complete ordered held roster')
        future_ids = [row['view']['record_id'] for row in record['future']]
        declared = record['fit_context'].get('requested_decision_ids',
            record['fit_context'].get('independent_future_requested', future_ids))
        if len(set(future_ids)) != len(future_ids) or future_ids != list(declared):
            raise ValueError('joint original future roster changed order, duplicates or requested decisions')
        rows = record['future_score_rows']
        strategies = record['strategy_order']
        if len(strategies) != len(set(strategies)) or set(strategies) != set(record['programs']):
            raise ValueError('joint original strategy order differs from its declared probability programmes')
        expected = [(strategy, row['view']['record_id']) for row in record['future'] for strategy in strategies]
        if [(row['strategy'], row['input']['view']['record_id']) for row in rows] != expected:
            raise ValueError('joint common graph score rows differ from its ordered requested future roster')
        for original, future in zip(rows, (row for row in record['future'] for _ in record['programs']), strict=True):
            if (semantic_sha256(original['expert_evidence']) != semantic_sha256(future['scores'][original['strategy']])
                    or original['input']['view_record_sha256'] != semantic_sha256(future['view'])):
                raise ValueError('joint graph score rows differ from original captured expert evidence')
        for row in record['OOF']:
            evidence = row['row']['expert_evidence']
            available = all(CandidateEvidence.from_record(value).probability_available for value in evidence.values())
            if row['row']['status'] != ('scored' if available else 'unavailable'):
                raise ValueError('joint original OOF numerical availability status changed')
            actual_captures = {name: {'capability_sha256': entry['history_capability_sha256'],
                'captured_P_measure_sha256': entry['capture_sha256']} for name, entry in row['captures'].items()}
            if (semantic_sha256(actual_captures) != semantic_sha256(row['row']['original_prediction_captures'])
                    or row['earlier_actual_K'] != row['authority']['actual_K']
                    or row['row']['earlier_actual_K'] != row['earlier_actual_K']
                    or row['row']['held_record_id'] != row['held_view']['record_id']):
                raise ValueError('joint forward OOF changed its actual numerical capture, earlier dose or held identity')
            if semantic_sha256(row.get('issued_history_refusals', {})) != semantic_sha256(
                    row['row'].get('original_history_refusals', {})):
                raise ValueError('joint forward OOF lost its original exact-request refusal witness')
        for row in (*record['OOF'], *record['future']):
            view = row.get('held_view', row.get('view'))
            for name, refusal in row.get('issued_history_refusals', {}).items():
                binding = refusal['binding']
                if (name in row['captures'] or name in row['issued_history_permissions']
                        or semantic_sha256(binding) != refusal['sha256']
                        or semantic_sha256(binding['request']['current_view']) != semantic_sha256(view)
                        or binding['raw_read_permission_granted'] is not False or binding['future_labels_granted'] is not False):
                    raise ValueError('joint exact-request refusal changed its original view or grants reading permissions')
        expected_producers = [{'fit_key': semantic_sha256(row['authority']),
            'oof_rows': [row['row']] if row['row']['status'] == 'scored' else [],
            'oof_score_failures': [] if row['row']['status'] == 'scored' else [row['row']]}
            for row in record['OOF']]
        if semantic_sha256(record['inner_producers']) != semantic_sha256(expected_producers):
            raise ValueError('joint graph inner producers differ from the actual ordered OOF ledger')

    def _captured(self, entries, compiler, prefix, view):
        result = {}
        for name, entry in entries.items():
            capture = _load_capture(self.directory/entry['capture']['path'])
            cutoff = capture.binding['input']['history_capability']['conditioning_sample_support'][1]
            capability = compiler.record_read_scope(view, prefix=prefix, conditioning_stop=cutoff)
            if capability.digest != entry['history_capability_sha256'] or capture.digest != entry['capture_sha256']:
                raise ValueError('captured measure differs from the original issuer history and fitted prefix')
            result[name] = capture
        return result

    def replay(self, *, prefix, future_views, prefix_compiler, source_checkpoints=None, context=None, **_options):
        from research.prefix_inner_replay import (
            future_input_identity,
            prefix_input_identity,
            unavailable_prefix_expert_scores,
        )
        from transfer.ensemble import EnsembleProgram
        from transfer.ensemble_fit import CalibrationExpertProgram, _prefix_sources

        if prefix is None or prefix_compiler is None or future_views is None:
            raise ValueError('joint numerical history replay requires original prefix, compiler and ordered future views')
        prefix_compiler.assert_original_prefix(prefix)
        if (semantic_sha256(prefix_input_identity(prefix)) != semantic_sha256(self.record['prefix_input'])
                or prefix.budget != self.record['actual_K']
                or context is not None and semantic_sha256(context) != semantic_sha256(self.record['fit_context'])):
            raise ValueError('joint replay original C prefix, dose or context changed')
        specs = tuple(CalibrationExpertProgram.from_record(row) for row in self.record['expert_programs'])
        if source_checkpoints is not None:
            sources = _prefix_sources(source_checkpoints, specs)
            if {name: source.digest for name, source in sources.items()} != self.record['source_checkpoint_digests']:
                raise ValueError('joint original source cohort/parameter checkpoint changed')
        checks, preserved = 0, []
        groups = {group.view.record_id: group for group in prefix.complete_groups}
        for row in self.record['OOF']:
            held = groups[row['held_view']['record_id']]
            earlier = prefix_compiler.before_group(prefix, held)
            if (earlier.budget != row['earlier_actual_K']
                    or semantic_sha256(earlier.record()) != row['earlier_prefix_sha256']
                    or semantic_sha256(held.view.record()) != semantic_sha256(row['held_view'])):
                raise ValueError('forward OOF changed its actual earlier C or original held group')
            if semantic_sha256(prefix_input_identity(earlier)) != semantic_sha256(row['authority']['prefix_input']):
                raise ValueError('forward joint OOF original numerical C arrays or labels changed')
            if row['model_directory'] is None:
                if row['captures'] or row['issued_history_permissions'] or row['row']['fit_failure'] is None:
                    raise ValueError('failed forward fit has fabricated numerical state or lost its failure')
                scored = unavailable_prefix_expert_scores(earlier.record(), row['authority']['expert_programs'],
                    'fixed', row['row']['fit_failure'], held.view)
            else:
                path = self.directory/row['model_directory']
                inner = FittedEnsemble._load_content(path, (path/'ensemble.json').read_bytes(),
                    metadata_store=self._metadata_store, _metadata_admission=self._metadata_admission,
                    _joint_source_preparation=self._source_preparation)
                _check_joint_authority(row['authority'], {'fixed': inner})
                if (row['row']['expert_fit_digests'] != {expert.program.expert_id: expert.fit_digest for expert in inner._experts}
                        or row['row']['training_receipts_sha256'] != semantic_sha256(inner.fit_receipts)):
                    raise ValueError('forward joint captured row differs from its actual original fitted producer')
                captured = self._captured(row['captures'], prefix_compiler, earlier, held.view)
                raw = _replay_expert_scores(inner, held.view, compiler=prefix_compiler, prefix=earlier,
                    captures=captured, saved=row['row']['expert_evidence'], permissions=row['issued_history_permissions'],
                    refusals=row.get('issued_history_refusals'))
                scored = inner._apply_probability_program(raw)
            if semantic_sha256({name: item.record() for name, item in scored.items()}) != semantic_sha256(row['row']['expert_evidence']):
                raise ValueError('captured original forward OOF probabilities changed on model-only replay')
            if row['row']['status'] == 'scored':
                checks += 1
            else:
                preserved.append({'fit_key': semantic_sha256(row['authority']), **row['row']})
        future_views = tuple(future_views)
        if ([view.record_id for view in future_views] != [row['view']['record_id'] for row in self.record['future']]
                or len({view.record_id for view in future_views}) != len(future_views)):
            raise ValueError('joint replay must preserve the complete original requested future roster')
        scores = []
        first = next((model for model in self.programs.values() if isinstance(model, FittedEnsemble)), None)
        for row, view in zip(self.record['future'], future_views, strict=True):
            if semantic_sha256(view.record()) != semantic_sha256(row['view']):
                raise ValueError('future physical input differs from the original captured request')
            common = [entry for entry in self.record['future_score_rows'] if entry['input']['view']['record_id'] == view.record_id]
            if any(semantic_sha256(entry['input']) != semantic_sha256(future_input_identity(view)) for entry in common):
                raise ValueError('joint graph future input differs from freshly compiled original observations')
            captured = self._captured(row['captures'], prefix_compiler, prefix, view)
            if first is None:
                if captured or row['issued_history_permissions']:
                    raise ValueError('all-unavailable joint programmes have fabricated numerical P state')
                scored = _score_programs(self.programs, view, prefix=prefix, specs=specs)
            else:
                saved = row['scores'][first.probability_strategy]
                raw = _replay_expert_scores(first, view, compiler=prefix_compiler, prefix=prefix,
                    captures=captured, saved=saved, permissions=row['issued_history_permissions'],
                    refusals=row.get('issued_history_refusals'))
                scored = {name: (unavailable_prefix_expert_scores(prefix.record(), [spec.record() for spec in specs],
                    name, model.record(subject=view.participant_key), view) if isinstance(model, ExpectedSubjectError)
                    else model._apply_probability_program(raw)) for name, model in self.programs.items()}
            records = {strategy: {name: item.record() for name, item in evidence.items()} for strategy, evidence in scored.items()}
            if semantic_sha256(records) != semantic_sha256(row['scores']):
                raise ValueError('captured original future expert/ensemble probability state changed')
            program = EnsembleProgram.from_record(self.record['ensemble_program'])
            scores.append({'decision_id': view.decision_id, 'strategies': {
                name: (model.scorer if isinstance(model, FittedEnsemble) else EnsembleScorer(program)).score(scored[name]).record()
                for name, model in self.programs.items()}})
            checks += len(scored)
        return {'status': 'replayed', 'actual_K': prefix.budget, 'requested_Ks': self.record['requested_Ks'],
            'producer_count': len(self.record['inner_producers']), 'exact_score_checks': checks,
            'preserved_oof_score_failures': preserved, 'scores': scores, 'truth_free': True}


def validate_joint_flat_public_parent(directory, *, original_views, source_cohort):
    """Independent original source/C/future authority, then captured-only replay."""
    from research.joint_flat_authority import _original_record_source_cohort
    from transfer.continuous_decision import ContinuousDecisionModel
    from transfer.ensemble_views import RecordCalibrationPrefixCompiler

    if type(original_views.prefix_compiler) is not RecordCalibrationPrefixCompiler:
        raise TypeError('joint public parent requires the original registered dataset decoder and prefix compiler')
    admission = JointFlatPackageAdmission(directory)
    prefix = original_views.prefix_compiler.compile(admission.record['actual_K'])
    if prefix.target_contract['participant_key'] != admission.record['participant_key']:
        raise ValueError('joint public parent changed the actual target participant')
    cohort, raw_sources, source_views, geometry = _original_record_source_cohort(source_cohort,
        prefix.target_contract['participant_key'])
    source_artifacts = admission.sources
    for name, source in source_artifacts.items():
        if name not in admission.record['source_exclusion_scope']:
            continue
        if (source.parameters.noise.geometry != geometry
                or source.parameters.binding['excluded_target_participant'] != prefix.target_contract['participant_key']):
            raise ValueError('joint public source producer scope or original native geometry changed')
        source.assert_target(prefix.target_contract)
        if admission.record['source_exclusion_scope'][name] == 'whole_target_dataset' and any(
                p.target_contract['source_dataset_id'] == prefix.target_contract['source_dataset_id'] for _, p in cohort):
            raise ValueError('public whole-study-excluded programme consumed the original target study')
        original = ContinuousDecisionModel.load(io.BytesIO(source.original_model_content))
        fitted = {row['record_id']: row for row in original.provenance['records']}
        if list(fitted) != [raw.record_id for raw in raw_sources]:
            raise ValueError('public source gate did not recover the actual fitted source recording order')
        for raw, view in zip(raw_sources, source_views, strict=True):
            if (fitted[raw.record_id]['input_sha256'] != view['actual_input_sha256']
                    or fitted[raw.record_id]['n_samples'] != len(raw.samples)):
                raise ValueError('public source gate changed the original actual source fit operands')
        actual_prefixes = source.parameters.binding['source_prefixes']
        if (len(actual_prefixes) != len(cohort) or any(
                semantic_sha256(actual['target_contract']) != semantic_sha256(p.target_contract)
                or semantic_sha256(actual['label_use']) != semantic_sha256(p.label_use['raw'])
                for actual, (_, p) in zip(actual_prefixes, cohort, strict=True))):
            raise ValueError('public source gate changed the original label dose or dataset source authority')
    replay = admission.replay(prefix=prefix, future_views=original_views.decisions,
        prefix_compiler=original_views.prefix_compiler, source_checkpoints=source_artifacts)
    if set(original_views.scoring_truth) != {view.decision_id for view in original_views.decisions}:
        raise ValueError('original parent scoring truth does not cover the complete requested future roster')
    from research.candidate_metrics import evaluate_candidate_evidence
    from research.contracts import DecisionPlanContract, DecisionPlanEntry
    from transfer.outcomes import CandidateCoverage, DecisionKey

    views = tuple(original_views.decisions)
    plan = DecisionPlanContract(original_views.ledger['source_sha256'], semantic_sha256([v.record() for v in views]),
        (prefix.target_contract['participant_key'],), tuple(DecisionPlanEntry(v.participant_key,
            v.decision_id, original_views.scoring_truth[v.decision_id]) for v in views))
    horizons = {DecisionKey(v.participant_key, v.decision_id): v.decision.repetitions for v in views}
    coverage = {DecisionKey(v.participant_key, v.decision_id): CandidateCoverage.from_mappings(
        dict(zip(v.decision.candidate_vocabulary, map(int, v.decision.membership.sum(axis=0)), strict=True)), {})
        for v in views}
    metrics = {name: evaluate_candidate_evidence({0: [CandidateEvidence.from_record(row['strategies'][name])
        for row in replay['scores']]}, decision_plan=plan, seeds=(0,),
        candidate_vocabulary=prefix.target_contract['candidate_vocabulary'], requested_evidence_levels=horizons,
        requested_coverage=coverage).record() for name in admission.programs}
    # Truth stays with this parent and is never passed to a prediction reader.
    return {'schema': 'n2p3_joint_flat_original_public_parent/1', 'participant_key': admission.record['participant_key'],
        'requested_decisions': len(original_views.decisions), 'replay': replay,
        'ordered_requested_record_ids': [view.record_id for view in views], 'metrics': metrics,
        'source_cohort_participants': [p.target_contract['participant_key'] for _, p in cohort],
        'source_exclusion_scope': admission.record['source_exclusion_scope'],
        'scoring_truth_sha256': semantic_sha256(original_views.scoring_truth),
        'source_parameter_refits': 0, 'new_Gibbs_calls': 0, 'truth_consumed_by_prediction': False}
