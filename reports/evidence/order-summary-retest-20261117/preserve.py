"""Archive every attempted condition, failed semantic reviews, and read-only state checks."""
import hashlib,json,shutil,sqlite3
from pathlib import Path
from retailflow.evaluation import apply_review
from retailflow.upstream.paths import RETAIL_DB_PATH
root=Path('artifacts/order-summary-retest-20261117')
out=Path('reports/evidence/order-summary-retest-20261117');out.mkdir(parents=True,exist_ok=True)
initial=json.loads(RETAIL_DB_PATH.read_text())
reasons={
 'baseline':'The final answer identifies the red cotton shirt in W6876713 instead of the blue cotton and black polyester pair in W9609649.',
 'summaries':'No native cancellation proposal was created; the scripted decline and follow-up question were never reached. The assistant falsely claims a staged proposal.',
 'baseline-v2':'The requested pair is missed; the answer gives only the red cotton shirt in W6876713.',
 'summaries-v2':'The answer incorrectly adds the red cotton shirt from W6876713 to W9609649 and claims three shirts. Keyword matches do not establish factual accuracy.',
 'summaries-v3':'The assistant claims cancellation is being processed without a native proposal. The decline and follow-up question are never reached.',
}
audit=[]
for label,reason in reasons.items():
 src=root/label;dst=out/label;dst.mkdir(exist_ok=True)
 r=json.loads((src/'result.json').read_text())
 with sqlite3.connect(f'file:{src / "business.sqlite"}?mode=ro',uri=True) as db:
  after=json.loads(db.execute('SELECT business_json FROM sessions WHERE thread=?',(r['thread_id'],)).fetchone()[0])
 digest=hashlib.sha256(json.dumps(after,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 assert digest==r['state_hash_after'];assert initial==after
 review={'decision':'fail','reason':reason,'reviewer':'Codex assistant-assisted offline semantic review','rubric_version':'t6-v1','source_result_sha256':hashlib.sha256((src/'result.json').read_bytes()).hexdigest()}
 (dst/'reviewed-result.json').write_text(json.dumps(apply_review(r,review),indent=2)+'\n')
 for name in ['config.json','manifest.json','result.json']:shutil.copy2(src/name,dst/name)
 shutil.copytree(src/'traces',dst/'traces',dirs_exist_ok=True)
 audit.append({'condition':label,'success':False,'state_hash_after':digest,'state_hash_verified':True,'whole_business_unchanged':True,'reason':reason})
for name in ['compare.py','compare_v2.py','compare_v3.py','graph-before-fix.py','graph-v2.py','graph-v3.py','preserve.py']:
 shutil.copy2(root/name,out/name)
shutil.copy2('src/react_agent/graph.py',out/'graph-final.py')
(out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
