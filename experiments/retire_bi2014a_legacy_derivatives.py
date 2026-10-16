"""Delete superseded BI caches without identity contracts and obsolete frozen features."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, action='append', required=True)
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
records = []
for root in args.root:
    root = root.resolve(strict=True)
    for path in sorted(root.rglob('*.npz')):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f'Unexpected external cache link: {path}')
        if not path.name.startswith(('bi2014a_', 'bi_bnci_')):
            continue
        with np.load(path, allow_pickle=False) as archive:
            if 'identity_json' in archive:
                continue
            if 'X' in archive and 'subject_ids' in archive:
                # These explicitly retired v1/v2 files predate authority identity.
                subjects = archive['subject_ids'].astype(str)
                excluded = [s for s in np.unique(subjects) if s in {'62', '63', '64', 'BI::62', 'BI::63', 'BI::64', 'BI2014a::62', 'BI2014a::63', 'BI2014a::64'}]
                if not excluded:
                    continue
                reason = 'superseded pre-v3 cache containing author-rejected BI participants; current eligible61 caches retained'
            elif set(archive.files) <= {'features', 'feature_key', 'schema', 'labels', 'groups'}:
                excluded = []
                reason = 'obsolete frozen BI feature/label/group bundle depends on retired cache or checkpoint; rebuild from eligible61 source'
            else:
                continue
        record = {'path': str(resolved), 'bytes': path.stat().st_size,
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'excluded_local_subjects': excluded, 'reason': reason, 'status': 'pending'}
        records.append(record)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(records, indent=2))
        path.unlink()
        record['status'] = 'deleted'
        args.report.write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
print(f'Deleted {len(records)} superseded derivatives', flush=True)
