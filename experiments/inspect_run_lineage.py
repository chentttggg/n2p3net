"""Read-only historical lineage lookup; indexing writes only a new derived index.

No weights, EEG, cache arrays, archive members, commands or model code are
loaded. Archive verification streams only small, explicitly declared source
archives. Source availability is separate from scientific result acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA = 'n2p3_historical_run_lineage_retrieval/1'
TIERS = ('original_archive_verified', 'source_only', 'hash_only', 'result_only', 'no_source')
FORBIDDEN_PAYLOADS = {'.pt', '.pth', '.pkl', '.pickle', '.npy', '.npz', '.edf', '.bdf', '.fif', '.set'}


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def small_json(path, maximum=1_500_000):
    path = Path(path)
    if path.suffix != '.json' or path.stat().st_size > maximum:
        return None
    try:
        return json.loads(path.read_bytes())
    except (ValueError, UnicodeError):
        return None


def walk_fields(value, prefix=()):
    if isinstance(value, dict):
        for key, item in value.items():
            path = (*prefix, str(key))
            yield path, item
            yield from walk_fields(item, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_fields(item, (*prefix, str(index)))


def source_archive_reference(manifest_path, *, verify=False, maximum_archive=8_000_000):
    """Verify an original declared physical archive; never restamp its manifest."""
    manifest_path = Path(manifest_path).resolve()
    value = small_json(manifest_path)
    if not isinstance(value, dict) or value.get('schema') != 'n2p3_source_freeze/1':
        return None
    result = {'manifest': str(manifest_path), 'manifest_sha256': file_sha256(manifest_path),
        'archive_sha256': value.get('archive_sha256'), 'commit_role': value.get('source_commit_role', 'undeclared'),
        'verification': 'not_requested'}
    archive = (manifest_path.parent/str(value.get('archive', ''))).resolve()
    result['archive'] = str(archive)
    if not archive.is_relative_to(manifest_path.parent):
        result['verification'] = 'invalid_archive_path'
        return result
    if not archive.is_file():
        result['verification'] = 'archive_missing'
        return result
    if (not isinstance(value.get('files'), dict) or not value['files']
            or any(Path(name).suffix.lower() in FORBIDDEN_PAYLOADS for name in value['files'])):
        result['verification'] = 'not_a_source_only_file_declaration'
        return result
    size = archive.stat().st_size
    result['archive_bytes'] = size
    if size > maximum_archive:
        result['verification'] = 'archive_exceeds_explicit_read_limit'
        return result
    if size != value.get('byte_size'):
        result['verification'] = 'archive_size_mismatch'
        return result
    digest = value.get('archive_sha256')
    if not isinstance(digest, str) or re.fullmatch('[a-f0-9]{64}', digest) is None or len(set(digest)) == 1:
        result['verification'] = 'invalid_or_obvious_placeholder_declaration'
        return result
    if verify:
        result['verification'] = ('physical_archive_sha256_verified' if file_sha256(archive) == digest
            else 'archive_sha256_mismatch')
    return result


def classify_run(run, inventory, declarations, runs_root, *, verify_archives=False):
    base = Path(runs_root)/run
    names = inventory['names']
    manifests = [name for name in names if name.endswith('source.manifest.json')]
    maps = [name for name in names if name.endswith(('source.json', 'execution_source.json'))]
    py_source = [name for name in names if name.endswith('.py') and any(
        marker in name for marker in ('executed_source/', 'checkout/src/', 'execution_snapshots/'))
        and (base/name).is_file()]
    archives = []
    for relative in manifests:
        path = base/relative
        if path.is_file():
            reference = source_archive_reference(path, verify=verify_archives)
            if reference is not None:
                archives.append(reference)
    # Probe original metadata only. Limit and selection are explicit so a
    # missing finding never silently becomes a claim that the field is absent.
    candidates = sorted((name for name in names if name.endswith('.json')
        and 'checkout/' not in name and 'executed_source/' not in name),
        key=lambda name: (len(Path(name).parts), 0 if any(x in Path(name).name for x in ('execution', 'record', 'result', 'completion', 'plan')) else 1, name))
    fields, placeholders, scanned, source_digests = {}, [], 0, set()
    groups = {
        'seed': {'seed', 'source_seed', 'target_seed', 'source_seeds', 'target_seeds', 'inference_seed', 'randomness'},
        'exit_status': {'returncode', 'exit_code', 'status'},
        'denominator': {'requested_cells', 'requested_tasks', 'planned_cells', 'expected_cells', 'failed_cells', 'requested_participant_keys'},
        'data': {'source_cache_sha256', 'target_cache_sha256', 'input_caches', 'raw_hashes', 'source_files', 'input_manifests'},
        'scope': {'evidence_scope', 'synthetic', 'software_correctness'},
    }
    for relative in candidates[:80]:
        path = base/relative
        if not path.is_file():
            continue
        value = small_json(path)
        if value is None:
            continue
        scanned += 1
        for pointer, item in walk_fields(value):
            key = pointer[-1]
            for group, keys in groups.items():
                if key in keys and len(fields.setdefault(group, [])) < 2:
                    fields[group].append({'path': relative, 'pointer': '/'+ '/'.join(pointer)})
            source_context = any('snapshot' in part or part in {'snapshot', 'source'} for part in pointer[:-1])
            source_key = key in {'source_snapshot_sha256', 'source_archive_sha256', 'executed_source_manifest_sha256', 'training_and_replay_source_manifest_sha256'}
            if isinstance(item, str) and re.fullmatch('[a-f0-9]{64}', item) and (source_key or source_context and key in {'archive_sha256', 'manifest_sha256'}):
                if len(set(item)) == 1:
                    if len(placeholders) < 3:
                        placeholders.append({'path': relative, 'pointer': '/'+ '/'.join(pointer), 'digest': item})
                else:
                    source_digests.add(item)
    declared = declarations.get(run, {})
    has_declared_hash = bool(source_digests or declared.get('snapshot_decl'))
    verified = [item for item in archives if item['verification'] == 'physical_archive_sha256_verified']
    result_names = [name for name in names if Path(name).suffix in {'.pt', '.npz'} or name.endswith(('result.json', 'results.json', 'record.json', 'summary.json'))]
    if verified:
        tier = 'original_archive_verified'
    elif py_source:
        tier = 'source_only'
    elif has_declared_hash:
        tier = 'hash_only'
    elif result_names:
        tier = 'result_only'
    else:
        tier = 'no_source'
    gaps = ['source recovery tier does not certify every result producer or scientific acceptance']
    if tier == 'source_only':
        gaps.append('source bytes exist; original archive/manifest execution binding is not established')
    elif tier == 'hash_only':
        gaps.append('original source digest is a retrieval key; local original archive is unresolved by this index')
    elif tier in {'result_only', 'no_source'}:
        gaps.append('no original executed source established; do not reconstruct it from current HEAD')
    if placeholders:
        gaps.append('obvious historical placeholder source digests found; original records preserved unchanged')
    return {'run': run, 'source_recovery_tier': tier, 'run_locator': str(base),
        'original_G8_field_presence_counts': declared,
        'metadata_probe': {'parsed_files': scanned, 'candidate_files': len(candidates), 'limit': 80,
            'maximum_json_bytes': 1_500_000, 'findings_are_presence_only': True, 'pointers': fields},
        'restoration': {'original_source_archives': verified[:2],
            'all_original_source_manifest_locators': manifests,
            'source_map_locators': maps, 'source_byte_examples': py_source[:2],
            'unresolved_source_digests': sorted(source_digests)[:3], 'result_metadata_examples': result_names[:2]},
        'historical_placeholder_examples': placeholders, 'gaps': gaps,
        'nested_archive_checks': {'declared_manifests': len(archives),
            'status_counts': dict(Counter(item['verification'] for item in archives))}}


def build_index(repository, *, verify_archives=False):
    repository = Path(repository).resolve()
    inventory_path = repository/'tmp/audit/scratch/g8_inventory.json'
    declarations_path = repository/'tmp/audit/scratch/g8_decl.json'
    inventory = json.loads(inventory_path.read_bytes())
    declarations = json.loads(declarations_path.read_bytes())
    records = [classify_run(name, value, declarations, repository/'experiments/runs',
        verify_archives=verify_archives) for name, value in sorted(inventory.items())]
    return {'schema': SCHEMA, 'purpose': 'read_only_retrieval_not_run_admission_or_scientific_acceptance',
        'original_audit_inputs': {str(path): file_sha256(path) for path in (inventory_path, declarations_path)},
        'limits': {'EEG_or_weight_reads': 0, 'archive_members_loaded': 0, 'maximum_source_archive_bytes': 8_000_000,
            'source_archive_digest_checks_requested': verify_archives, 'historical_records_modified': 0},
        'tier_counts': dict(Counter(record['source_recovery_tier'] for record in records)), 'runs': records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('index', help='Build a new small retrieval index from original G8 metadata.')
    create.add_argument('--repository', type=Path, required=True)
    create.add_argument('--verify-source-archives', action='store_true')
    create.add_argument('--output', type=Path, required=True, help='New derived JSON index file; existing output is refused.')
    query = commands.add_parser('query', help='Read the saved index, with no dataset or archive access.')
    query.add_argument('--index', type=Path, required=True)
    query.add_argument('--run', help='Run name or substring; omitted returns compact counts.')
    query.add_argument('--tier', choices=TIERS)
    verify = commands.add_parser('verify-source', help='Stream-check one original source-freeze archive, without loading members.')
    verify.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'index':
        if args.output.exists():
            parser.error(f'index output already exists: {args.output}')
        result = build_index(args.repository, verify_archives=args.verify_source_archives)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps({'index': str(args.output), 'runs': len(result['runs']), 'tier_counts': result['tier_counts']}))
    elif args.command == 'verify-source':
        reference = source_archive_reference(args.manifest, verify=True)
        if reference is None:
            raise ValueError('not an original supported source-freeze manifest')
        print(json.dumps(reference, indent=2))
    else:
        result = json.loads(args.index.read_bytes())
        if result.get('schema') != SCHEMA:
            raise ValueError('unsupported lineage retrieval index schema')
        records = [row for row in result['runs'] if (args.run is None or args.run in row['run'])
            and (args.tier is None or args.tier == row['source_recovery_tier'])]
        print(json.dumps(records if args.run or args.tier else {'runs': len(records), 'tier_counts': result['tier_counts']}, indent=2))


if __name__ == '__main__':
    main()
