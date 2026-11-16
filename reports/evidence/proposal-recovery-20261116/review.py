"""Apply recorded semantic judgments and independently audit the final SQLite states."""
import hashlib,json,sqlite3
from pathlib import Path
from retailflow.evaluation import apply_review,aggregate_results
from retailflow.dataset import check_final_state
from retailflow.upstream.paths import RETAIL_DB_PATH
root=Path('artifacts/proposal-recovery-20261116/dev-B2')
source=root/'results.jsonl';lines=source.read_text().splitlines()
assert len(lines)==9, 'Wait for the complete batch'
tasks={d['task_id']:d for d in map(json.loads,Path('data/derived/dev.jsonl').read_text().splitlines())}
judgments={
 'tau-retail-2':('fail','The confirmed return is correct, but the user-visible answer omits the requested count of available T-shirt options (10).'),
 'tau-retail-24':('fail','The answer describes only the red cotton T-shirt in #W6876713 and denies a second shirt, omitting the blue cotton and black polyester pair in #W9609649.'),
 'tau-retail-48':('pass','Confirmed air purifier return to original PayPal method; receipt correctly states no settled refund or arrival date and no further return of remaining items in this order.'),
 'tau-retail-69':('pass','Confirmed laptop cancellation and immediate $2674.40 gift-card credit match the final answer and business state.'),
 'tau-retail-73':('pass','Confirmed return of all four non-coffee items to the original credit card. Final receipt explicitly excludes Espresso Machine, states refund not settled, no promised arrival date, and no real email sent.'),
}
initial=json.loads(RETAIL_DB_PATH.read_text());results=[];reviews=[];audits=[]
with sqlite3.connect(f'file:{root / "business.sqlite"}?mode=ro',uri=True) as db:
 for line in lines:
  result=json.loads(line);task=tasks[result['task_id']]
  after=json.loads(db.execute('SELECT business_json FROM sessions WHERE thread=?',(result['thread_id'],)).fetchone()[0])
  digest=hashlib.sha256(json.dumps(after,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  assert digest==result['state_hash_after']
  checked=check_final_state(task,initial,after,result['events'])
  assert [f for f in checked['failures'] if f!='missing_communication']==result['state_failures']
  changed={key:value for key,value in after['orders'].items() if value!=initial['orders'][key]}
  payments={key:value['payment_methods'] for key,value in after['users'].items() if value['payment_methods']!=initial['users'][key]['payment_methods']}
  audits.append({'task_id':result['task_id'],'state_hash_verified':True,'state_hash_after':digest,'state_failures':result['state_failures'],'changed_orders':changed,'changed_payment_methods':payments})
  if result['review_required']:
   decision,reason=judgments[result['task_id']]
   review={'task_id':result['task_id'],'decision':decision,'reason':reason,'reviewer':'Codex assistant-assisted offline semantic review','rubric_version':'t6-v1','source_result_sha256':hashlib.sha256(line.encode()).hexdigest()}
   reviews.append(review);result=apply_review(result,review)
  results.append(result)
(root/'communication-reviews.json').write_text(json.dumps({'raw_results_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'reviews':reviews},indent=2)+'\n')
(root/'reviewed-results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
report={**aggregate_results(results),'n_planned':9,'run_complete':True}
(root/'reviewed-report.json').write_text(json.dumps(report,indent=2)+'\n')
(root/'final-state-audit.json').write_text(json.dumps(audits,indent=2)+'\n')
print(json.dumps(report,indent=2))
