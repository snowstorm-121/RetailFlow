import asyncio,json
from pathlib import Path
from retailflow.evaluation import evaluate_episode,build_manifest
async def main():
 task=next(json.loads(l) for l in Path('data/derived/dev.jsonl').read_text().splitlines() if json.loads(l)['task_id']=='tau-retail-48')
 for variant in ['baseline','review']:
  out=Path('artifacts/receipt-20261115')/variant;out.mkdir(exist_ok=False)
  config=json.loads(Path('configs/repair-v5.json').read_text());config['execution']['receipt_answer_review']=variant=='review'
  cp=out/'config.json';cp.write_text(json.dumps(config,indent=2))
  manifest=build_manifest(cp,config);manifest['assigned_task_ids']=[task['task_id']]
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
  r=await evaluate_episode(task,'B2',{**config,'output_dir':str(out)})
  (out/'result.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(variant,r['status'],r['objective_passed'],r['events'][-2:],flush=True)
asyncio.run(main())
