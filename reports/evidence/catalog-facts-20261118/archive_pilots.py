import hashlib,json,shutil,sqlite3
from pathlib import Path
from retailflow.evaluation import apply_review
from retailflow.dataset import check_final_state
from retailflow.upstream.paths import RETAIL_DB_PATH
root=Path('artifacts/catalog-facts-20261118');out=Path('reports/evidence/catalog-facts-20261118');out.mkdir(parents=True,exist_ok=True)
task=next(t for t in map(json.loads,Path('data/derived/dev.jsonl').read_text().splitlines()) if t['task_id']=='tau-retail-2')
initial=json.loads(RETAIL_DB_PATH.read_text())
for label,graph,runtime in [('pilot','graph-before-continuation.py','runtime-before-question-focus.py'),('pilot-v2','graph-recovery.py','runtime-before-question-focus.py'),('pilot-v3','graph-question-focus.py','runtime-question-focus.py')]:
 src=root/label;dst=out/label;dst.mkdir(exist_ok=True)
 result=json.loads((src/'result.json').read_text())
 with sqlite3.connect(f'file:{src/"business.sqlite"}?mode=ro',uri=True) as db:
  after=json.loads(db.execute('select business_json from sessions where thread=?',(result['thread_id'],)).fetchone()[0])
 digest=hashlib.sha256(json.dumps(after,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 assert digest==result['state_hash_after']
 checked=check_final_state(task,initial,after,result['events'])
 assert [f for f in checked['failures'] if f!='missing_communication']==result['state_failures']
 audit={'state_hash_verified':True,'state_hash_after':digest,'state_failures':result['state_failures'],'changed_orders':{k:v for k,v in after['orders'].items() if v!=initial['orders'][k]}}
 (dst/'state-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 reason=('No native return proposal; requested count absent.' if label=='pilot' else 'Return was committed correctly after confirmation, but no successful catalog lookup and no answer to requested T-shirt count. Full task fails.')
 review={'decision':'fail','reason':reason,'reviewer':'Codex assistant-assisted offline semantic review','rubric_version':'t6-v1','source_result_sha256':hashlib.sha256((src/'result.json').read_bytes()).hexdigest()}
 (dst/'reviewed-result.json').write_text(json.dumps(apply_review(result,review),indent=2)+'\n')
 manifest=json.loads((src/'manifest.json').read_text())
 for name,digest in manifest['file_hashes'].items():
  path=root/graph if name=='src/react_agent/graph.py' else root/runtime if name=='src/retailflow/runtime.py' else Path(name)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,(label,name)
 for p in src.glob('*.json'):shutil.copy2(p,dst/p.name)
 shutil.copytree(src/'traces',dst/'traces',dirs_exist_ok=True)
 shutil.copy2(root/graph,dst/'graph-at-run.py');shutil.copy2(root/runtime,dst/'runtime-at-run.py')
print('All three pilot state and source hashes verified; all full-task failures retained')
