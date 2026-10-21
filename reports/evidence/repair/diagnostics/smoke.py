import asyncio,json,os
from pathlib import Path
from retailflow.runtime import RetailRuntime
from retailflow.dataset import check_final_state
os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
root=Path('artifacts/repair-v1/smoke');root.mkdir(parents=True,exist_ok=True);config=json.loads(Path('configs/local.json').read_text())
async def main():
 worker=RetailRuntime(root/'business.sqlite',root/'traces');events=[];event={'kind':'message','text':'Please cancel order #W5056519. I ordered it by mistake.'};outputs=[]
 for _ in range(5):
  result=await worker.run_turn('missing-identity-cancel',event,config);outputs.append(result);events.append({'input':event,'output':result['status'],'text':result['text']});(root/'events.json').write_text(json.dumps(events,indent=2)+'\n');print(result['status'],result['text'],flush=True)
  if result['status']=='needs_input':
   answers={'identity':'I am Yara Muller, ZIP 85041.','order':'#W5056519','cancel_reason':'I ordered it by mistake.'}
   if any(f not in answers for f in result['request']['fields']):break
   event={'kind':'message','text':' '.join(answers[f] for f in result['request']['fields'])}
  elif result['status']=='needs_confirmation':
   p=result['request']
   if p['name']!='cancel_pending_order' or p['args']!={'order_id':'#W5056519','reason':'ordered by mistake'}:break
   event={'kind':'confirmation','proposal_id':p['proposal_id'],'accepted':True}
  else:break
 s=worker.load_session('missing-identity-cancel');before=json.loads(Path('vendor/tau_retail/source/data/tau2/domains/retail/db.json').read_text());after=s['business'];confirmed=[{'type':'confirmed_write','order_id':'#W5056519'}] if any(e['input']['kind']=='confirmation' for e in events) else []
 checked=check_final_state({'group_id':'source-user:yara_muller_8652','expected_checks':{'orders':[{'order_id':'#W5056519','status':'cancelled','cancel_reason':'ordered by mistake'}],'communicate_info':[],'transfer_to_human':False}},before,after,confirmed)
 (root/'result.json').write_text(json.dumps({'outputs':outputs,'check':checked,'source':'synthetic dialogue smoke; not a benchmark task'},indent=2)+'\n');print('FINAL STATE',checked,flush=True)
asyncio.run(main())
