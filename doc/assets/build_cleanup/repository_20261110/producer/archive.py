import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

ROOT = Path(r'D:\BCI\n2p3-net')
DEST = ROOT / 'frozen/repository_consolidation_20261110'
EVIDENCE = ROOT / 'doc/assets/build_cleanup/repository_20261110'
DEST.mkdir(parents=True, exist_ok=True)

def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def files_in(relative):
    path = ROOT / relative
    assert path.resolve().is_relative_to(ROOT.resolve())
    if path.is_file():
        yield path
    else:
        for parent, directories, files in os.walk(path, followlinks=False):
            for name in directories + files:
                child = Path(parent) / name
                assert not child.is_symlink() and not (child.lstat().st_file_attributes & 1024), child
            for name in files:
                yield Path(parent) / name

def archive(name, relatives):
    target = DEST / (name + '.zip')
    if target.exists():
        raise FileExistsError(target)
    members = []
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for relative in relatives:
            for file in sorted(files_in(relative)):
                member = file.relative_to(ROOT).as_posix()
                before = file.stat()
                sha = digest_file(file)
                z.write(file, member)
                after = file.stat()
                assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns), member
                members.append({'path':member,'bytes':before.st_size,'sha256':sha})
    with zipfile.ZipFile(target) as z:
        assert len(z.namelist()) == len(members) == len(set(z.namelist()))
        for row in members:
            h = hashlib.sha256()
            with z.open(row['path']) as f:
                for b in iter(lambda:f.read(4*1024*1024),b''):
                    h.update(b)
            assert h.hexdigest() == row['sha256'], row['path']
    manifest = {'schema':'n2p3_lossless_retirement_archive/1','archive':target.relative_to(ROOT).as_posix(),'sha256':digest_file(target),'bytes':target.stat().st_size,'source_bytes':sum(x['bytes'] for x in members),'members':members,'verified_members':len(members),'restore':'Extract members into repository root to restore original relative paths; archive itself does not establish scientific acceptance.'}
    (EVIDENCE / (name+'.manifest.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='members'},ensure_ascii=False),flush=True)

p=argparse.ArgumentParser()
p.add_argument('name')
p.add_argument('paths',nargs='+')
args=p.parse_args()
archive(args.name,args.paths)
