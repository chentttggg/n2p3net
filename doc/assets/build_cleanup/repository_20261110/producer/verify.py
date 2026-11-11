import hashlib
import importlib.util
import json
import os
import zipfile
from pathlib import Path

ROOT=Path(r'D:\BCI\n2p3-net')
OUT=ROOT/'doc/assets/build_cleanup/repository_20261110'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

before=json.loads(Path(r'D:\Temp\n2p3_cleanup_20261110\inventory_before.json').read_text())
protected=json.loads((OUT/'protected_files.json').read_text())['files']
bad=[]
intentional=[]
for row in protected:
    path=ROOT/row['path']
    if not path.is_file() or sha(path)!=row['sha256']:
        if row['path']=='formal/lean/README.md':intentional.append(row['path'])
        else:bad.append(row['path'])

archive_results=[]
for name in ('documents','training_inputs','research_records'):
    m=json.loads((OUT/(name+'.manifest.json')).read_text())
    path=ROOT/m['archive']
    assert sha(path)==m.get('sha256',m.get('archive_sha256')),name
    errors=[]
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist())==len(m['members'])==len(set(z.namelist()))
        for member in m['members']:
            h=hashlib.sha256()
            size=0
            with z.open(member['path']) as f:
                for b in iter(lambda:f.read(4*1024*1024),b''):
                    h.update(b)
                    size+=len(b)
            if h.hexdigest()!=member['sha256'] or size!=member['bytes']:errors.append(member['path'])
    archive_results.append({'archive':m['archive'],'verified_members':len(m['members']),'member_errors':errors,'bytes':path.stat().st_size})

retired=json.loads((OUT/'targets_before.json').read_text())
remaining=[]
for t in retired['targets']:
    if t['path']=='tmp/audit':continue
    if (ROOT/t['path']).exists():remaining.append(t['path'])
research=json.loads((OUT/'research_records.manifest.json').read_text())
for name in retired['restore_after_audit_retirement']:
    expected=next(r for r in research['members'] if r['path']==name)
    assert sha(ROOT/name)==expected['sha256'],name
actual_audit=sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'tmp/audit').rglob('*') if p.is_file())
assert actual_audit==sorted(retired['restore_after_audit_retirement']),actual_audit

metadata_roots=('mne_data/','cloud_inputs/','models/','Paper/','frozen/','doc/assets/','formal/lean/artifacts/')
metadata_checked=0
metadata_bad=[]
for row in before['files']:
    if row['path'].startswith(metadata_roots):
        p=ROOT/row['path']
        metadata_checked+=1
        if not p.is_file():metadata_bad.append(row['path'])
        else:
            s=p.stat()
            if (s.st_size,s.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):metadata_bad.append(row['path'])

spec=importlib.util.spec_from_file_location('n2p3_docs_check',ROOT/'experiments/check_docs.py')
checker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
checker.ACTIVE_DOCUMENTS=tuple(dict.fromkeys((*checker.ACTIVE_DOCUMENTS,'doc/project_handover_20261107.zh.md','doc/build_cleanup_maintenance.zh.md','doc/training_data.zh.md','doc/evidence/history.zh.md','formal/lean/README.md','frozen/repository_consolidation_20261110/README.zh.md')))
links=checker.check_links(ROOT)

groups={}
stack=[ROOT]
links_after=[]
while stack:
    directory=stack.pop()
    with os.scandir(directory) as entries:
        for e in entries:
            s=e.stat(follow_symlinks=False)
            rel=Path(e.path).relative_to(ROOT).as_posix()
            if e.is_symlink() or s.st_file_attributes&1024:
                links_after.append({'path':rel,'target':os.readlink(e.path)})
            elif e.is_dir(follow_symlinks=False):stack.append(Path(e.path))
            else:
                g=groups.setdefault(rel.split('/')[0],{'files':0,'bytes':0})
                g['files']+=1
                g['bytes']+=s.st_size
assert sorted(links_after,key=lambda x:x['path'])==sorted(before['links'],key=lambda x:x['path'])
after_bytes=sum(g['bytes'] for g in groups.values())
report={'schema':'n2p3_cleanup_verification/1','protected_files_checked':len(protected),'protected_files_unchanged':len(protected)-len(intentional)-len(bad),'intentional_document_updates':intentional,'protected_file_errors':bad,'archives':archive_results,'remaining_retirement_targets':remaining,'retained_lineage_inputs':actual_audit,'preserved_file_metadata_checked':metadata_checked,'preserved_file_metadata_errors':metadata_bad,'document_links':links,'reparse_points_unchanged':True,'inventory_after':{'groups':groups,'files':sum(g['files'] for g in groups.values()),'bytes':after_bytes},'inventory_before_bytes':sum(f['bytes'] for f in before['files']),'logical_reduction_bytes_at_verification':sum(f['bytes'] for f in before['files'])-after_bytes,'note':'Metadata checks compare size and mtime, not full-byte hashes. Source/model/checkpoint protection and all archive members have SHA-256 checks. Logical reduction is not measured physical disk release.'}
report['passed']=not (bad or metadata_bad or remaining or any(x['member_errors'] for x in archive_results) or not links['passed'])
(OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('inventory_after','document_links')},ensure_ascii=False,indent=2))
print(json.dumps({'links':links['repository_links_checked'],'documents':links['documents'],'missing':links['missing'],'after_files':report['inventory_after']['files'],'after_bytes':after_bytes},ensure_ascii=False))
assert report['passed']
