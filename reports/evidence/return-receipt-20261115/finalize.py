from pathlib import Path
import json,hashlib,sqlite3,shutil
from retailflow.evaluation import apply_review,aggregate_results,score_episode
from retailflow.environment import _json
from retailflow.upstream.paths import RETAIL_DB_PATH
root=Path('artifacts/return-receipt-20261115/dev-B2');out=Path('reports/evidence/return-receipt-20261115/dev-B2')
lines=(root/'results.jsonl').read_text().splitlines(); assert len(lines)==9
rows=[json.loads(l) for l in lines]
reasons={
 'tau-retail-24':('fail','Only the red cotton shirt in #W6876713 is described; the blue cotton and black polyester pair in #W9609649 is omitted.'),
 'tau-retail-48':('pass','Actual air purifier return, original PayPal method and confirmation match the receipt. The application-generated answer correctly states no settled refund or promised arrival date and answers that both remaining vacuum cleaners in the same order cannot be returned again. No other items were requested for return.'),
 'tau-retail-69':('pass','Confirmed cancellation of #W2417020 and immediate $2674.40 credit to gift_card_8541487 match the committed state and final answer.')}
reviews=[];revised=[];audits=[];index=[]
tasks={d['task_id']:d for d in map(json.loads,Path('data/derived/dev.jsonl').read_text().splitlines())}
before=json.loads(RETAIL_DB_PATH.read_text())
db=sqlite3.connect(f'file:{root.resolve()}/business.sqlite?mode=ro',uri=True)
for raw,r in zip(lines,rows):
 after=json.loads(db.execute('SELECT business_json FROM sessions WHERE thread=?',(r['thread_id'],)).fetchone()[0])
 h=hashlib.sha256(_json(after).encode()).hexdigest();assert h==r['state_hash_after']
 checked=score_episode(tasks[r['task_id']],before,after,r['events'],r['status']);assert checked['state_failures']==r['state_failures']
 changed={k:v for k,v in after['orders'].items() if v!=before['orders'][k]}
 payments={k:{'before':before['users'][k]['payment_methods'],'after':v['payment_methods']} for k,v in after['users'].items() if v['payment_methods']!=before['users'][k]['payment_methods']}
 audits.append({'task_id':r['task_id'],'state_hash_after':h,'state_failures':r['state_failures'],'changed_orders':changed,'changed_payment_methods':payments})
 if r['objective_passed']:
  decision,reason=reasons[r['task_id']]
  review={'task_id':r['task_id'],'decision':decision,'reason':reason,'reviewer':'Codex assistant-assisted offline semantic review','rubric_version':'t6-v1','source_result_sha256':hashlib.sha256(raw.encode()).hexdigest()}
  reviews.append(review);r=apply_review(r,review)
 revised.append(r)
 for path in r['trace_paths']:
  p=Path(path);dst=out/'traces'/p.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
  index.append({'original':path,'public':'traces/'+p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for name in ('results.jsonl','manifest.json','report.json'):shutil.copyfile(root/name,out/name)
(out/'reviewed-results.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in revised)+'\n')
for name,obj in [('communication-reviews.json',{'raw_results_sha256':hashlib.sha256((root/'results.jsonl').read_bytes()).hexdigest(),'reviews':reviews}),('reviewed-report.json',aggregate_results(revised)),('final-state-audit.json',audits),('trace-index.json',index)]:
 (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
print(json.dumps(aggregate_results(revised),indent=2))
