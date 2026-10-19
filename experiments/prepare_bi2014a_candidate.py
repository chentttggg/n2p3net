"""Prepare original causal BI caches or an explicit independent episode programme."""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from data.bi2014a_candidate import (  # noqa: E402
    BI2014A_CHANNELS,
    BI2014A_SOURCE_BIDIRECTIONAL_PROFILE,
    assert_bi2014a_candidate_input_contract,
    build_bi2014a_prepared_participant,
    build_bi2014a_subject_dataset,
)
from data.cache_output import new_epoch_cache_path  # noqa: E402
from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT  # noqa: E402
from data.epochs import (  # noqa: E402
    PreprocessingSpec,
    concatenate_epoch_datasets,
    parse_preprocessing_spec,
    preprocessing_spec_from_contract,
    save_epoch_dataset,
)
from data.exclusions import select_allowed_subjects  # noqa: E402


def _file_receipt(path):
    path = Path(path)
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'sha256': sha, 'bytes': path.stat().st_size}


def source_profile_from_checkpoint(path, *, expected_sha256=None):
    """Use every actual input field, validating both checkpoint profile owners."""
    from transfer.checkpoint import load_checkpoint_payload

    receipt = _file_receipt(path)
    if expected_sha256 is not None and receipt['sha256'] != expected_sha256:
        raise ValueError('BI source profile checkpoint differs from the declared producer')
    payload = load_checkpoint_payload(path)
    spec = parse_preprocessing_spec(payload['input_preprocessing'])
    declared = payload['training_contract']['preprocessing']
    if parse_preprocessing_spec(declared['epoch']) != spec:
        raise ValueError('BI source checkpoint input and training preprocessing differ')
    channels = tuple(payload['input_channel_names'])
    reference = payload['input_source_reference']
    if (tuple(declared['channel_names']) != channels or declared['source_reference'] != reference
            or payload['n_times'] != spec.n_times or payload['input_sample_rate_hz'] != spec.sfreq
            or payload['n_channels'] != len(channels) or payload['input_tmin_s'] != spec.tmin_ms / 1000):
        raise ValueError('BI source checkpoint geometry or reference owners differ')
    assert_bi2014a_candidate_input_contract(spec)
    receipt.update(preprocessing=spec.record(), input_channel_names=list(channels), input_source_reference=reference)
    return spec, channels, reference, receipt


def prepare_episode_subject(arguments):
    """One resource-owned participant; results return paths, never EEG matrices."""
    from research.execution import atomic_write_json
    from research.source_execution import current_input_admission

    output = Path(arguments['output'])
    output.mkdir(parents=True, exist_ok=False)
    owner = current_input_admission()
    prepared = build_bi2014a_prepared_participant(arguments['subject_dir'],
        preprocessing=PreprocessingSpec(**arguments['preprocessing']),
        calibration_episode_count=arguments['calibration_episode_count'],
        output_channels=tuple(arguments['output_channels']), observed_raw_channels=tuple(arguments['observed_raw_channels']),
        reference_channel=arguments['reference_channel'], maximum_horizon=arguments['maximum_horizon'])
    prepared.calibration.provenance['source_profile_checkpoint'] = arguments['source_profile_checkpoint']
    prepared.future.provenance['source_profile_checkpoint'] = arguments['source_profile_checkpoint']
    c = save_epoch_dataset(output / 'calibration.npz', prepared.calibration, compressed=False, exclusive=True)
    f = save_epoch_dataset(output / 'future.npz', prepared.future, compressed=False, exclusive=True)
    ledger = {**prepared.ledger, 'source_profile_checkpoint': arguments['source_profile_checkpoint'],
        'calibration_cache': _file_receipt(c), 'future_cache': _file_receipt(f),
        'preparation_module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'data_module_sha256': hashlib.sha256((ROOT / 'src/data/bi2014a_candidate.py').read_bytes()).hexdigest()}
    atomic_write_json(output / 'preparation.json', ledger, input_admission=owner)
    return {'subject': prepared.ledger['subject'], 'calibration_labels': prepared.calibration.n_epochs,
        'requested_future_episodes': len(prepared.future_decisions), 'requested_future_flashes': prepared.future.event_timeline.n_events,
        'retained_future_epochs': prepared.future.n_epochs, 'dropped_tail_flashes': prepared.ledger['dropped_tail_flashes'],
        'preparation': _file_receipt(output / 'preparation.json'), 'calibration_cache': ledger['calibration_cache'],
        'future_cache': ledger['future_cache']}


def _episode_programme(args, directories, spec, channels, reference, profile_receipt):
    from research.cloud_execution import CloudExecutor, CloudTask, PhaseResources
    from research.execution import atomic_write_json
    from research.source_execution import (
        HostMemoryAdmission,
        SourceInputAdmission,
        resolve_source_resources,
    )
    from train.runtime import cpu_thread_budget

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    native_reference = reference.removeprefix('measured_channel:')
    if reference != f'measured_channel:{native_reference}' or native_reference not in {name.upper() for name in BI2014A_CHANNELS}:
        raise ValueError('BI episode preparation requires the actual measured source-reference channel')
    observed = tuple(name for name in channels if name in {name.upper() for name in BI2014A_CHANNELS})
    resources = resolve_source_resources({'cpu_budget': args.cpu_budget, 'source_workers': 1,
        'source_worker_memory_gib': 2, 'target_worker_memory_gib': 2, 'host_memory_fraction': .95,
        'shared_staging_memory_gib': 0}, 1, 'cpu')
    workers = resources['worker_cpu_budget'] if args.workers is None else args.workers
    if workers < 1 or workers > resources['worker_cpu_budget']:
        raise ValueError('BI preparation workers exceed the actual shared CPU budget')
    tasks = tuple(CloudTask(f'bi2014a-prepare:{directory.name}',
        'experiments.prepare_bi2014a_candidate:prepare_episode_subject',
        {'subject_dir': str(directory.resolve()), 'output': str((output / directory.name).resolve()),
         'preprocessing': spec.record(), 'output_channels': list(channels), 'observed_raw_channels': list(observed),
         'reference_channel': native_reference, 'source_profile_checkpoint': profile_receipt,
         'calibration_episode_count': args.calibration_episodes, 'maximum_horizon': args.maximum_horizon},
        str(output / directory.name), PhaseResources('preparation', 1, 2 * 2**30, 1, 'cpu')) for directory in directories)
    accepted = []
    with cpu_thread_budget(1), multiprocessing.get_context('spawn').Manager() as manager:
        inputs = SourceInputAdmission(manager, capacity_bytes=resources['host_memory_budget_bytes'], load_workers=workers)
        admission = HostMemoryAdmission(manager, resources, input_admission=inputs)

        def finished(outcome):
            if outcome.status == 'completed':
                for key in ['preparation', 'calibration_cache', 'future_cache']:
                    if _file_receipt(outcome.value[key]['path']) != outcome.value[key]:
                        raise ValueError('BI preparation output changed before parent admission')
                accepted.append(outcome.value)
            atomic_write_json(output / 'progress.json', {'completed': len(accepted), 'requested': len(tasks),
                'last_subject': outcome.task_id, 'last_status': outcome.status}, input_admission=inputs)
            return ()

        run = CloudExecutor(admission, max_workers=workers, diagnostic_directory=output / 'worker_diagnostics',
            identity={'programme': 'BI2014a independent raw episode preparation', 'source_profile_checkpoint': profile_receipt}).run(
                tasks, on_finished=finished,
                on_observation=lambda observation: atomic_write_json(output / 'resources.live.json', observation, input_admission=inputs))
        report = {'schema': 'n2p3_bi2014a_preparation_cohort/1', 'status': run.status,
            'requested_subjects': [directory.name for directory in directories],
            'calibration_episode_count': args.calibration_episodes, 'source_profile_checkpoint': profile_receipt,
            'participants': sorted(accepted, key=lambda value: int(value['subject'])),
            'resources': resources, 'actual_worker_limit': workers,
            'failures': [{'task_id': row.task_id, 'status': row.status, 'failure': row.failure,
                          'supplementary_failures': row.supplementary_failures}
                         for row in run.results.values() if row.status != 'completed'],
            'worker_results': [{'task_id': row.task_id, 'status': row.status, 'worker': row.worker} for row in run.results.values()],
            'host_admission': admission.record(), 'input_admission': inputs.record(),
            'public_training_fits': 0}
        atomic_write_json(output / 'cohort.json', report, input_admission=inputs)
    if report['status'] != 'completed':
        raise RuntimeError('BI preparation did not complete its full declared cohort; original failures preserved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'mne_data/MNE-braininvaders2014a-data/zenodo/3266223')
    parser.add_argument('--output', type=Path, required=True, help='New .npz cache file; with --calibration-episodes, a new participant-output directory. Existing outputs are refused.')
    parser.add_argument('--subjects', default='')
    parser.add_argument('--max-subjects', type=int)
    parser.add_argument('--uncompressed', action='store_true')
    parser.add_argument('--preprocessing-profile', choices=('causal_v3', 'brainsync_bidirectional_v1'), default='causal_v3')
    parser.add_argument('--output-channels')
    parser.add_argument('--observed-raw-channels')
    parser.add_argument('--reference-channel')
    parser.add_argument('--source-reference', default='right earlobe')
    parser.add_argument('--source-checkpoint', type=Path)
    parser.add_argument('--source-checkpoint-sha256')
    parser.add_argument('--calibration-episodes', type=int)
    parser.add_argument('--maximum-horizon', type=int, default=8)
    parser.add_argument('--cpu-budget', type=int, default=25)
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(f'refusing to replace an existing candidate cache or episode directory: {args.output}')
    if args.calibration_episodes is None:
        try:
            args.output = new_epoch_cache_path(args.output)
        except ValueError as error:
            parser.error(str(error))
    directories = sorted((p for p in args.root.glob('subject_*') if p.is_dir()), key=lambda p: int(p.name.split('_')[-1]))
    if args.subjects:
        chosen = []
        for token in args.subjects.split(','):
            if '-' in token:
                first, last = map(int, token.split('-'))
                chosen.extend(range(first, last + 1))
            else:
                chosen.append(int(token))
        allowed = set(select_allowed_subjects('BI2014a', chosen, explicit=True))
    else:
        allowed = set(select_allowed_subjects('BI2014a', [int(p.name.split('_')[-1]) for p in directories], explicit=False))
    directories = [p for p in directories if int(p.name.split('_')[-1]) in allowed]
    if args.max_subjects is not None:
        directories = directories[:args.max_subjects]
    if not directories:
        raise ValueError('No admitted BI2014a participant directories matched')
    if args.source_checkpoint:
        spec, channels, reference, profile_receipt = source_profile_from_checkpoint(args.source_checkpoint,
            expected_sha256=args.source_checkpoint_sha256)
    else:
        spec = preprocessing_spec_from_contract(BI2014A_SOURCE_BIDIRECTIONAL_PROFILE
            if args.preprocessing_profile == 'brainsync_bidirectional_v1' else SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)
        channels = BI2014A_CHANNELS if args.output_channels is None else tuple(args.output_channels.split(','))
        reference = args.source_reference
        profile_receipt = None
    if args.calibration_episodes is not None:
        if profile_receipt is None:
            raise ValueError('Independent BI episode preparation requires its actual source profile checkpoint')
        _episode_programme(args, directories, spec, channels, reference, profile_receipt)
    else:
        observed = BI2014A_CHANNELS if args.observed_raw_channels is None else tuple(args.observed_raw_channels.split(','))
        datasets = [build_bi2014a_subject_dataset(directory, preprocessing=spec, output_channels=channels,
            observed_raw_channels=observed, reference_channel=args.reference_channel, source_reference=args.source_reference)
            for directory in directories]
        combined = concatenate_epoch_datasets(datasets, name='BI2014a-candidate', provenance={
            'source': 'bi2014a_raw_csv', 'source_reference': datasets[0].provenance['source_reference'],
            'source_root': str(args.root.resolve()), 'candidate_task_contract': datasets[0].provenance['candidate_task_contract']})
        save_epoch_dataset(args.output, combined, compressed=not args.uncompressed, exclusive=True)


if __name__ == '__main__':
    main()
