"""Preserve completed runs without SQLite files or credentials; verify source hashes."""
import hashlib,json,shutil
from pathlib import Path
root=Path('artifacts/scoped-cancel-goals-20261117')
out=Path('reports/evidence/scoped-cancel-goals-20261117');out.mkdir(parents=True,exist_ok=True)
for label in ['pilot','dev-B2']:
 src=root/label;dst=out/label;dst.mkdir(exist_ok=True)
 manifest=json.loads((src/'manifest.json').read_text())
 for name,digest in manifest['file_hashes'].items():
  p=Path(name);assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
  target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 index=[]
 for p in src.rglob('*.json'):
  target=dst/p.relative_to(src);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  if p.parent.name=='traces':index.append({'original_path':str(p),'archived_path':str(target),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 for p in src.glob('*.jsonl'):shutil.copy2(p,dst/p.name)
 (dst/'trace-index.json').write_text(json.dumps(index,indent=2)+'\n')
for p in root.glob('*.py'):shutil.copy2(p,out/p.name)
shutil.copy2(root/'judgments.json',out/'judgments.json')
print('Manifests verified and evidence archived')
