import os
import json
import collections
from pathlib import Path

root = Path(r'D:\BCI\n2p3-net')
out = Path(r'D:\Temp\n2p3_cleanup_20261110')
out.mkdir(parents=True, exist_ok=True)
rows, links, errors = [], [], []
stack = [root]
while stack:
    directory = stack.pop()
    try:
        with os.scandir(directory) as entries:
            for e in entries:
                try:
                    s = e.stat(follow_symlinks=False)
                    rel = str(Path(e.path).relative_to(root)).replace('\\', '/')
                    if e.is_symlink() or getattr(s, 'st_file_attributes', 0) & 1024:
                        links.append({'path': rel, 'target': os.readlink(e.path)})
                    elif e.is_dir(follow_symlinks=False):
                        stack.append(Path(e.path))
                    else:
                        rows.append({'path': rel, 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns})
                except OSError as ex:
                    errors.append({'path': e.path, 'error': str(ex)})
    except OSError as ex:
        errors.append({'path': str(directory), 'error': str(ex)})
groups = collections.defaultdict(lambda: {'files': 0, 'bytes': 0})
for row in rows:
    parts = row['path'].split('/')
    key = '/'.join(parts[:2]) if len(parts) > 2 else parts[0]
    groups[key]['files'] += 1
    groups[key]['bytes'] += row['bytes']
result = {'root': str(root), 'files': rows, 'links': links, 'errors': errors}
(out / 'inventory_before.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'files': len(rows), 'bytes': sum(r['bytes'] for r in rows), 'groups': sorted([dict(path=k, **v) for k, v in groups.items()], key=lambda x: -x['bytes'])[:85], 'links': links, 'errors': errors}, ensure_ascii=False, indent=2))
