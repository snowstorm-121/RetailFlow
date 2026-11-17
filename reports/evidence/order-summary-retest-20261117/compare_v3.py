import asyncio,json,os
from pathlib import Path
from retailflow.evaluation import evaluate_episode,build_manifest
os.environ['RETAILFLOW_LANGFUSE_ENABLED']='false'
async def main():
 root=Path('artifacts/order-summary-retest-20261117')
 path=Path('configs/proposal-recovery.json'); config=json.loads(path.read_text())
 task=next(json.loads(x) for x in Path('data/derived/dev.jsonl').read_text().splitlines() if json.loads(x)['task_id']=='tau-retail-24')
 for label,enabled in [('summaries-v3',True)]:
  out=root/label;out.mkdir(exist_ok=False)
  cfg={**config,'execution':{**config['execution'],'profile_order_summaries':enabled}}
  (out/'config.json').write_text(json.dumps(cfg,indent=2))
  (out/'manifest.json').write_text(json.dumps(build_manifest(out/'config.json',cfg),indent=2))
  result=await evaluate_episode(task,'B2',{**cfg,'output_dir':str(out)})
  (out/'result.json').write_text(json.dumps(result,indent=2))
  print(label,result['status'],result['failure_reason'],result['objective_passed'],flush=True)
asyncio.run(main())
