import hashlib
import json
import os
from pathlib import Path

ROOT = Path(r'D:\BCI\n2p3-net')
OUT = ROOT / 'doc/assets/build_cleanup/repository_20261110'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def safe(path):
    assert path.resolve().is_relative_to(ROOT.resolve()) and path != ROOT
    for p in (path,*path.parents):
        assert not p.is_symlink() and not (p.lstat().st_file_attributes & 1024), p

targets=[]
def add(relative, reason, manifest=None):
    path=ROOT/relative
    if not path.exists():return
    safe(path)
    members=[]
    if path.is_dir():
        for parent,dirs,files in os.walk(path,followlinks=False):
            for name in dirs+files:safe(Path(parent)/name)
            members.extend(Path(parent)/name for name in files)
    else:members=[path]
    rows=[]
    for p in members:
        s=p.stat()
        rel=p.relative_to(ROOT).as_posix()
        row={'path':rel,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
        if manifest:
            expected=manifest[rel]
            assert s.st_size==expected['bytes'] and sha(p)==expected['sha256'],rel
            row['sha256']=expected['sha256']
        rows.append(row)
    targets.append({'path':relative,'absolute_path':str(path),'kind':'directory' if path.is_dir() else 'file','reason':reason,'files':rows,'bytes':sum(r['bytes'] for r in rows)})

for name in ('training_inputs','research_records'):
    m=json.loads((OUT/(name+'.manifest.json')).read_text())
    assert sha(ROOT/m['archive'])==m['sha256']
    bypath={r['path']:r for r in m['members']}
    paths=(['experiments/cache/four_dataset_20260907','experiments/cache/history','experiments/cache/legacy'] if name=='training_inputs' else ['experiments/results_history','tmp/audit','tmp/worktree_retirement_20261016','tmp/retire_worktrees_20261016.py'])
    for path in paths:add(path,'lossless_archive_verified:'+name,bypath)

m=json.loads((OUT/'documents.manifest.json').read_text())
archive=ROOT/'frozen/repository_consolidation_20261110/documents.zip'
assert sha(archive)==m['archive_sha256']
bydoc={r['path']:r for r in m['members']}
for row in json.loads(Path(r'D:\Temp\n2p3_cleanup_20261110\docs_retire.json').read_text())['members']:
    add(row['path'],'historical_document_original_archived',bydoc)

for p in ['.pytest_cache','.ruff_cache','__pycache__','tmp/cloud_ssh','tmp/data_export','formal/lean/.lake/build']:
    add(p,'regenerable_build_or_dependency_copy')
packages=ROOT/'formal/lean/.lake/packages'
for pkg in sorted(packages.iterdir()):
    p=pkg/'.lake/build'
    if p.is_dir():add(p.relative_to(ROOT).as_posix(),'regenerable_dependency_build')
for top in ('src','tests','experiments','scripts'):
    for parent,dirs,files in os.walk(ROOT/top,followlinks=False):
        if '__pycache__' in dirs:
            add((Path(parent)/'__pycache__').relative_to(ROOT).as_posix(),'regenerable_python_bytecode')
            dirs.remove('__pycache__')

target_names=[t['path'] for t in targets]
assert len(target_names)==len(set(target_names))
for a in target_names:
    assert not any(a.startswith(b+'/') for b in target_names if b!=a),(a,target_names)
result={'schema':'n2p3_verified_retirement_targets/1','root':str(ROOT),'target_count':len(targets),'logical_bytes':sum(t['bytes'] for t in targets),'targets':targets,'restore_after_audit_retirement':['tmp/audit/scratch/g8_inventory.json','tmp/audit/scratch/g8_decl.json'],'lean_validation':'lean_pilot_build.log: Build completed successfully (2406 jobs); compiled outputs are reproducible caches, formal/lean/artifacts and scheduler state retained.'}
(OUT/'targets_before.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'targets':len(targets),'files':sum(len(t['files']) for t in targets),'bytes':result['logical_bytes'],'directories':[{'path':t['path'],'files':len(t['files']),'bytes':t['bytes']} for t in targets if t['kind']=='directory']},ensure_ascii=False,indent=2))
