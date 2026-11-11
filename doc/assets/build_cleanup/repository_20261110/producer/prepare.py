import hashlib
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(r'D:\BCI\n2p3-net')
OUT = ROOT / 'doc/assets/build_cleanup/repository_20261110'
OUT.mkdir(parents=True, exist_ok=True)
inventory = json.loads(Path(r'D:\Temp\n2p3_cleanup_20261110\inventory_before.json').read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

protected = []
for row in inventory['files']:
    p = row['path']
    if p.startswith(('src/', 'tests/', 'configs/', 'scripts/')) or (p.startswith('experiments/') and len(p.split('/')) == 2) or (p.startswith('formal/lean/') and '/.lake/' not in p and '/artifacts/' not in p) or p in ('pyproject.toml','uv.lock','requirements.txt','conftest.py','AGENTS.md') or p.startswith(('models/', 'artifacts/')) or (p.startswith('doc/assets/') and ('checkpoint' in p.lower() or 'handover_20261107/' in p)):
        if '__pycache__' in p or p.endswith(('.pyc','.pyo')):
            continue
        protected.append({'path': p, 'bytes': row['bytes'], 'sha256': sha(ROOT / p)})
(OUT / 'protected_files.json').write_text(json.dumps({'schema':'n2p3_cleanup_protected_files/1','files':protected}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
state = subprocess.check_output(['git', 'status', '--porcelain=v1', '-uall'], cwd=ROOT)
(OUT / 'git_status_before.txt').write_bytes(state)
groups = {}
for row in inventory['files']:
    top = row['path'].split('/')[0]
    group = groups.setdefault(top, {'files':0,'bytes':0})
    group['files'] += 1
    group['bytes'] += row['bytes']
(OUT / 'inventory_before.json').write_text(json.dumps({'schema':'n2p3_cleanup_inventory/1','root':str(ROOT),'files':len(inventory['files']),'bytes':sum(r['bytes'] for r in inventory['files']),'groups':groups,'reparse_points':inventory['links'],'errors':inventory['errors']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'protected_files':len(protected),'groups':groups},ensure_ascii=False,indent=2))
