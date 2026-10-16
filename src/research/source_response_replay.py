"""Original finite-response C state and deterministic history replay.

This owner stores model bytes, actual native permissions and full prediction
records. It reuses prefix scope/graph/input owners and performs no C/source fit.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from models.candidate_evidence import CandidateEvidence
from research.contracts import semantic_sha256
from research.execution import ExpectedSubjectError
from research.nested_selection_replay import InnerReplayRecorder
from transfer.ensemble import EnsembleScorer
from transfer.ensemble_fit import FittedEnsemble
from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation

SCHEMA = 'n2p3_source_response_prefix_replay/1'


def _artifact(path, directory):
    data = path.read_bytes()
    return {'path': path.relative_to(directory).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


class SourceResponseReplayRecorder(InnerReplayRecorder):
    def __init__(self):
        super().__init__()
        self.predictions = []

    def score_future(self, programs, view, *, compiler, prefix):
        from research.prefix_inner_replay import unavailable_prefix_expert_scores

        if any(row['view'].record_id == view.record_id for row in self.predictions):
            raise ValueError('source response recorder repeats an original requested decision')
        permissions, predictions, refusals = {}, {}, {}
        def remember_permission(name, capability):
            permissions[name] = capability
        def remember_prediction(name, capability, binding):
            predictions[name] = binding
        def remember_refusal(name, refusal):
            refusals[name] = refusal
        specs = self.prefix_fit_declaration['expert_programs']
        scored = FittedEnsemble.score_programs(programs, view, history_context=(compiler, prefix),
            history_capability_recorder=remember_permission, history_recorder=remember_prediction,
            history_refusal_recorder=remember_refusal)
        scored = {name: unavailable_prefix_expert_scores(prefix.record(), [spec.record() for spec in specs], name,
            value.record(subject=view.participant_key), view) if isinstance(value, ExpectedSubjectError) else value
            for name, value in scored.items()}
        for strategy, evidence in scored.items():
            self.record_prefix_future_score(strategy, view=view, evidence=evidence)
        self.predictions.append({'view': view, 'permissions': permissions, 'predictions': predictions,
            'refusals': refusals,
            'scores': {name: {expert: item.record() for expert, item in rows.items()} for name, rows in scored.items()}})
        return scored

    @property
    def original_history_capabilities(self):
        return tuple({cap.digest: cap for row in self.predictions for cap in row['permissions'].values()}.values())

    @property
    def original_history_refusals(self):
        return tuple({refusal.digest: refusal for row in self.predictions
                      for refusal in row['refusals'].values()}.values())


def persist_source_response_prefix(prefix, programs, directory, *, recorder, context=None, device='cpu', **declarations):
    from research.joint_flat_replay import _authority_record
    from research.prefix_inner_replay import (
        _check_final_authority,
        _fit_record,
        prefix_input_identity,
    )

    actual = recorder.prefix_fit_declaration
    if (actual['prefix'] is not prefix or set(programs) != {'fixed'}
            or programs['fixed'] is not recorder.prefix_programs['fixed']
            or context is not None and semantic_sha256(context) != semantic_sha256(actual['context'])
            or semantic_sha256(prefix_input_identity(prefix)) != semantic_sha256(actual['input_identity'])):
        raise ValueError('source response persistence changed its actual same-run prefix or programme')
    specs, program, sources = actual['expert_programs'], actual['program'], actual['source_checkpoints']
    if any(spec.family != 'source_response' for spec in specs):
        raise ValueError('source response first replay programme requires its complete finite-family roster')
    for key, value in (('expert_programs', [spec.record() for spec in specs]), ('program', program.record()),
                        ('source_checkpoints', {name: source.digest for name, source in sources.items()})):
        supplied = declarations.get(key)
        if supplied is not None:
            supplied = ([spec.record() for spec in supplied] if key == 'expert_programs' else
                supplied.record() if key == 'program' else {name: source.digest for name, source in supplied.items()})
            if semantic_sha256(supplied) != semantic_sha256(value):
                raise ValueError('source response persistence changed '+key)
    directory = Path(directory)
    FittedEnsemble.save_programs(directory, programs, participant_key=prefix.target_contract['participant_key'])
    source_files = {}
    (directory/'original_sources').mkdir()
    for name, source in sources.items():
        path = directory/'original_sources'/f'{source.digest}.json'
        if not path.exists():
            path.write_bytes(source.content)
        source_files[name] = _artifact(path, directory)
    record = {**_authority_record(prefix, specs, program, sources, actual['context']), 'schema': SCHEMA,
        'device': str(device), 'actual_K': prefix.budget,
        'prefix_input': prefix_input_identity(prefix), 'strategy_order': list(programs),
        'programs': {name: _fit_record(value, participant_key=prefix.target_contract['participant_key'])
                     for name, value in programs.items()},
        'inner_producers': [], 'original_source_files': source_files, 'future_score_rows': recorder.prefix_future_scores,
        'future': [{'view': row['view'].record(), 'scores': row['scores'], 'predictions': row['predictions'],
            'issued_history_permissions': {name: {'sha256': cap.digest, 'binding': cap.binding}
                                           for name, cap in row['permissions'].items()},
            'issued_history_refusals': {name: {'sha256': refusal.digest, 'binding': refusal.binding}
                                       for name, refusal in row['refusals'].items()}}
                   for row in recorder.predictions],
        'source_exclusion_scope': {spec.expert_id: spec.source_ref.source_exclusion_scope for spec in specs},
        'selection_performed': False, 'truth_free_future_replay': True,
        'posterior_replay': 'saved_original_C_posteriors_deterministic_actual_history'}
    record['members'] = [_artifact(path, directory) for path in sorted(directory.rglob('*')) if path.is_file()]
    _check_final_authority(record, {name: value for name, value in programs.items() if isinstance(value, FittedEnsemble)})
    payload = {'record': record, 'record_sha256': semantic_sha256(record)}
    (directory/'prefix_inner_replay.json').write_text(json.dumps(payload, sort_keys=True, indent=2,
        allow_nan=False), encoding='utf-8')
    recorder.prefix_receipt_sha256 = payload['record_sha256']
    return directory


class SourceResponsePackageAdmission:
    def __init__(self, directory, *, expected_record_sha256=None, _source_preparation=None):
        from research.prefix_inner_replay import _check_final_authority, _check_program_declarations

        self.directory = Path(directory)
        path = self.directory/'prefix_inner_replay.json'
        if self.directory.is_symlink() or path.is_symlink():
            raise ValueError('source response package path is a symlink')
        payload = json.loads(path.read_bytes())
        self.record = payload['record']
        if (self.record['schema'] != SCHEMA or semantic_sha256(self.record) != payload['record_sha256']
                or expected_record_sha256 is not None and payload['record_sha256'] != expected_record_sha256
                or self.record['selection_performed'] is not False or self.record['truth_free_future_replay'] is not True
                or self.record['strategy_order'] != ['fixed'] or self.record['inner_producers']):
            raise ValueError('source response original prefix package receipt changed')
        members = {row['path']: row for row in self.record['members']}
        actual = {path.relative_to(self.directory).as_posix() for path in self.directory.rglob('*')
                  if path.is_file() and path.name != 'prefix_inner_replay.json'}
        if len(members) != len(self.record['members']) or actual != set(members):
            raise ValueError('source response package lost its complete original member roster')
        for relative, expected in members.items():
            path = self.directory/relative
            if (path.is_symlink() or not path.resolve().is_relative_to(self.directory.resolve())
                    or _artifact(path, self.directory) != expected):
                raise ValueError('source response original model or source member changed')
        self._source_preparation = _source_preparation or FrozenJointFlatSourcePreparation()
        manifest, loaded = FittedEnsemble._admit_programs(self.directory,
            _joint_source_preparation=self._source_preparation)
        _check_program_declarations(self.record, loaded, manifest['programs'])
        self.authority = _check_final_authority(self.record, loaded)
        self.programs = {name: loaded[name] if row['status'] == 'completed' else ExpectedSubjectError(
            row['failure']['error_code'], stage=row['failure']['stage'], detail=row['failure']['detail'])
            for name, row in self.record['programs'].items()}
        if any(spec['family'] != 'source_response' for spec in self.record['expert_programs']):
            raise ValueError('source response package changed its typed family roster')
        if self.record['source_exclusion_scope'] != {spec['expert_id']: spec['source_ref']['source_exclusion_scope']
                for spec in self.record['expert_programs']}:
            raise ValueError('source response package changed its declared actual source exclusion scope')
        self.sources = {}
        for name, receipt in self.record['original_source_files'].items():
            if members.get(receipt['path']) != receipt:
                raise ValueError('source response source reference is outside its original package')
            digest = self.record['source_checkpoint_digests'][name]
            self.sources[name] = self._source_preparation.prepare((self.directory/receipt['path']).read_bytes(),
                expected_sha256=digest)
        if set(self.sources) != set(self.record['source_checkpoint_digests']):
            raise ValueError('source response package lost its complete actual source roster')
        ids = [row['view']['record_id'] for row in self.record['future']]
        declared = self.record['fit_context'].get('requested_decision_ids',
            self.record['fit_context'].get('independent_future_requested', ids))
        if ids != list(declared) or len(set(ids)) != len(ids):
            raise ValueError('source response future lost its complete requested ordered roster')
        scores = self.record['future_score_rows']
        if ([(row['strategy'], row['input']['view']['record_id']) for row in scores] !=
                [('fixed', value) for value in ids]):
            raise ValueError('source response graph lost its ordered future score rows')
        for row, score in zip(self.record['future'], scores, strict=True):
            if semantic_sha256(row['scores']['fixed']) != semantic_sha256(score['expert_evidence']):
                raise ValueError('source response saved evidence differs from its graph score')

    def replay(self, *, prefix, future_views, prefix_compiler, source_checkpoints=None, context=None,
               device=None, atol=0., rtol=0.):
        from research.prefix_inner_replay import (
            _validate_future,
            prefix_input_identity,
            unavailable_prefix_expert_scores,
        )
        from transfer.source_response_programme import check_source_response_fit

        if (atol != 0. or rtol != 0.
                or device is not None and str(device) != self.record['device']):
            raise ValueError('source response replay requires its original device and exact comparison')
        prefix_compiler.assert_original_prefix(prefix)
        if (semantic_sha256(prefix_input_identity(prefix)) != semantic_sha256(self.record['prefix_input'])
                or prefix.budget != self.record['actual_K']
                or context is not None and semantic_sha256(context) != semantic_sha256(self.record['fit_context'])):
            raise ValueError('source response replay changed its actual original C inputs or context')
        sources = self.sources if source_checkpoints is None else source_checkpoints
        if {name: source.digest for name, source in sources.items()} != self.record['source_checkpoint_digests']:
            raise ValueError('source response replay changed its actual original source parameters')
        for model in self.programs.values():
            if isinstance(model, ExpectedSubjectError):
                continue
            for expert in model._experts:
                if expert.predictor is not None:
                    check_source_response_fit(expert.predictor, prefix, sources[expert.program.expert_id].digest,
                        expert.program.source_response_program, expert.program.source_ref)
        views = tuple(future_views)
        if [view.record_id for view in views] != [row['view']['record_id'] for row in self.record['future']]:
            raise ValueError('source response replay must retain the complete ordered requested future')
        scores = []
        for view, original in zip(views, self.record['future'], strict=True):
            if semantic_sha256(view.record()) != semantic_sha256(original['view']):
                raise ValueError('source response replay changed the actual physical future view')
            _validate_future(view, self.record)
            permissions, predictions, refusals = {}, {}, {}
            def remember_permission(name, cap, permissions=permissions):
                permissions[name] = {'sha256': cap.digest, 'binding': cap.binding}
            def remember_prediction(name, cap, binding, predictions=predictions):
                predictions[name] = binding
            def remember_refusal(name, refusal, refusals=refusals):
                refusals[name] = {'sha256': refusal.digest, 'binding': refusal.binding}
            scored = FittedEnsemble.score_programs(self.programs, view, history_context=(prefix_compiler, prefix),
                history_capability_recorder=remember_permission, history_recorder=remember_prediction,
                history_refusal_recorder=remember_refusal)
            scored = {name: unavailable_prefix_expert_scores(prefix.record(), self.record['expert_programs'], name,
                value.record(subject=view.participant_key), view) if isinstance(value, ExpectedSubjectError) else value
                for name, value in scored.items()}
            records = {name: {key: value.record() for key, value in rows.items()} for name, rows in scored.items()}
            if (semantic_sha256(records) != semantic_sha256(original['scores'])
                    or semantic_sha256(permissions) != semantic_sha256(original['issued_history_permissions'])
                    or semantic_sha256(refusals) != semantic_sha256(original['issued_history_refusals'])
                    or semantic_sha256(predictions) != semantic_sha256(original['predictions'])):
                raise ValueError('source response replay differs from original actual permission, C state or probability')
            scores.append({'decision_id': view.decision_id, 'strategies': {
                name: (model.scorer if isinstance(model, FittedEnsemble) else
                       EnsembleScorer(self._ensemble_program())).score(scored[name]).record()
                for name, model in self.programs.items()}})
        return {'status': 'replayed', 'actual_K': prefix.budget, 'requested_Ks': self.record['requested_Ks'],
            'producer_count': 0, 'exact_score_checks': len(views)*len(self.programs), 'preserved_oof_score_failures': [],
            'scores': scores, 'truth_free': True}

    def _ensemble_program(self):
        from transfer.ensemble import EnsembleProgram

        return EnsembleProgram.from_record(self.record['ensemble_program'])


def validate_source_response_public_parent(directory, *, original_views, source_cohort):
    from research.joint_source_catalog import _verify_original_cohort
    from transfer.continuous_decision import ContinuousDecisionModel
    from transfer.ensemble_views import CalibrationPrefixCompiler

    compiler = original_views.prefix_compiler
    if type(compiler) is not CalibrationPrefixCompiler:
        raise TypeError('source response first public parent requires the original BNCI single-record MAT issuer')
    admission = SourceResponsePackageAdmission(directory)
    prefix = compiler.compile(admission.record['actual_K'])
    for name, source in admission.sources.items():
        original = ContinuousDecisionModel.load(io.BytesIO(source.original_model_content))
        summary = _verify_original_cohort(source, original, source_cohort, prefix.target_contract['participant_key'])
        source.assert_target(prefix.target_contract)
        if admission.record['source_exclusion_scope'][name] == 'whole_target_dataset' and any(
                row['source_dataset_id'] == prefix.target_contract['source_dataset_id'] for row in summary['source_records']):
            raise ValueError('source response whole-study source contains the actual target study')
    if set(original_views.scoring_truth) != {view.decision_id for view in original_views.decisions}:
        raise ValueError('source response original scoring truth lost requested future decisions')
    replay = admission.replay(prefix=prefix, future_views=original_views.decisions, prefix_compiler=compiler,
        source_checkpoints=admission.sources)
    from research.candidate_metrics import evaluate_candidate_evidence
    from research.contracts import DecisionPlanContract, DecisionPlanEntry
    from transfer.outcomes import CandidateCoverage, DecisionKey

    views = tuple(original_views.decisions)
    plan = DecisionPlanContract(original_views.ledger['source_sha256'], semantic_sha256([v.record() for v in views]),
        (prefix.target_contract['participant_key'],), tuple(DecisionPlanEntry(v.participant_key, v.decision_id,
            original_views.scoring_truth[v.decision_id]) for v in views))
    horizons = {DecisionKey(v.participant_key, v.decision_id): v.decision.repetitions for v in views}
    coverage = {DecisionKey(v.participant_key, v.decision_id): CandidateCoverage.from_mappings(
        dict(zip(v.decision.candidate_vocabulary, map(int, v.decision.membership.sum(axis=0)), strict=True)), {})
        for v in views}
    metrics = {name: evaluate_candidate_evidence({0: [CandidateEvidence.from_record(row['strategies'][name])
        for row in replay['scores']]}, decision_plan=plan, seeds=(0,),
        candidate_vocabulary=prefix.target_contract['candidate_vocabulary'], requested_evidence_levels=horizons,
        requested_coverage=coverage).record() for name in admission.programs}
    return {'schema': 'n2p3_source_response_original_parent/1', 'replay': replay,
        'requested_decisions': len(original_views.decisions),
        'metrics': metrics,
        'ordered_requested_record_ids': [view.record_id for view in original_views.decisions],
        'scoring_truth_sha256': semantic_sha256(original_views.scoring_truth),
        'source_parameter_refits': 0, 'new_C_fits': 0, 'noise_refits': 0,
        'truth_consumed_by_prediction': False,
        'source_cohort_participants': sorted({p.target_contract['participant_key'] for _, p in source_cohort})}
