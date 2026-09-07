"""Migrate historical caches through a frozen loader, verifying retained rows before retirement."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-source-root', type=Path, required=True)
    parser.add_argument('--policy-file', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, action='append', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--delete-original', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.legacy_source_root.resolve() / 'src'))
    from data.epochs import load_epoch_dataset, materialize_dataset_lineage, save_epoch_dataset
    from data.lineage import DataLineage

    spec = importlib.util.spec_from_file_location('retirement_policy', args.policy_file)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    report = {'policy': policy.PARTICIPANT_EXCLUSION_POLICY,
              'author_source': policy.PARTICIPANT_EXCLUSION_SOURCE, 'caches': []}
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')

    paths = sorted({p.resolve() for root in args.cache_root for p in root.rglob('*.npz')})
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if 'identity_json' not in archive:
                continue
            from data.identity import DatasetIdentityTable
            identity = DatasetIdentityTable.from_payload(json.loads(str(archive['identity_json'].item())))
            excluded = policy.excluded_identity_subjects(identity)
        if not excluded:
            continue
        entry = {'path': str(path), 'excluded_local_subjects': list(excluded),
                 'old_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'status': 'pending'}
        report['caches'].append(entry)
        persist()
        try:
            ds = load_epoch_dataset(path, require_labels=False, validation='attested')
            keep = ~np.isin(ds.subject_ids.astype(str), excluded)
            if not keep.any():
                raise ValueError('no eligible rows remain')
            retained_subjects = sorted(set(ds.subject_ids[keep].astype(str)))
            parent = materialize_dataset_lineage(ds)
            provenance = dict(ds.provenance)
            if 'subject_dirs' in provenance:
                provenance['subject_dirs'] = [p for p in provenance['subject_dirs']
                    if not policy.source_participant_excluded('BI2014a', Path(p).name)]
            decision = {'policy': policy.PARTICIPANT_EXCLUSION_POLICY,
                        'author_source': policy.PARTICIPANT_EXCLUSION_SOURCE,
                        'parent_cache_sha256': entry['old_sha256'],
                        'excluded_origin_keys': [key for record in identity.records
                            if record.local_subject_id in excluded for key in record.origin_subject_keys],
                        'removed_rows': int((~keep).sum()), 'retained_rows_unchanged': True}
            provenance['participant_exclusion'] = decision
            migrated = replace(ds, name=ds.name.replace('64subj', '61subj') + '-eligible',
                X=ds.X[keep], y=None if ds.y is None else ds.y[keep], subject_ids=ds.subject_ids[keep],
                metadata=ds.metadata.loc[keep].reset_index(drop=True) if not ds.metadata.empty else ds.metadata.copy(),
                event_timeline=ds.event_timeline.subset_rows(
                    ~np.isin(ds.event_timeline.subject_ids.astype(str), excluded)),
                trial_channel_mask=None if ds.trial_channel_mask is None else ds.trial_channel_mask[keep],
                qc_features=None if ds.qc_features is None else ds.qc_features.subset(np.flatnonzero(keep)),
                identity_table=identity.subset(retained_subjects), provenance=provenance,
                lineage=DataLineage.derive([parent], operation='exclude_author_rejected_participants', parameters=decision),
                verified_cache_attestation=None)
            policy.assert_identity_participants_allowed(migrated.identity_table)
            output = path.with_name(path.stem.replace('64subj', '61subj') + '_eligible61.npz')
            if output.exists():
                raise FileExistsError(output)
            save_epoch_dataset(output, migrated)
            reloaded = load_epoch_dataset(output, validation='attested')
            np.testing.assert_array_equal(reloaded.X, ds.X[keep])
            if ds.y is not None:
                np.testing.assert_array_equal(reloaded.y, ds.y[keep])
            np.testing.assert_array_equal(reloaded.subject_ids, ds.subject_ids[keep])
            policy.assert_identity_participants_allowed(reloaded.identity_table)
            entry.update(status='replacement_verified', replacement=str(output),
                         sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                         original_rows=ds.n_epochs, retained_rows=reloaded.n_epochs,
                         retained_subjects=retained_subjects, removed_rows=int((~keep).sum()))
            persist()
            if args.delete_original:
                # Only the explicitly inventoried cache bytes are retired; old records retain their identity.
                path.unlink()
                entry['status'] = 'original_deleted_after_verified_replacement'
                persist()
            print(json.dumps(entry), flush=True)
            del ds, migrated, reloaded
        except Exception as error:
            entry.update(status='requires_review', error=str(error))
            persist()
            print(json.dumps(entry), flush=True)
    persist()
    if any(row['status'] == 'requires_review' for row in report['caches']):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
