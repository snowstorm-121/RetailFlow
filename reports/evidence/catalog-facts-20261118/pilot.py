import asyncio,json,os
from pathlib import Path
from retailflow.evaluation import build_manifest,evaluate_episode
os.environ['RETAILFLOW_LANGFUSE_ENABLED']='false'
async def main():
 root=Path('artifacts/catalog-facts-20261118/pilot');root.mkdir(exist_ok=False)
 path=Path('configs/catalog-facts.json');cfg=json.loads(path.read_text())
 (root/'manifest.json').write_text(json.dumps(build_manifest(path,cfg),indent=2))
 task=next(json.loads(x) for x in Path('data/derived/dev.jsonl').read_text().splitlines() if json.loads(x)['task_id']=='tau-retail-2')
 result=await evaluate_episode(task,'B2',{**cfg,'output_dir':str(root)})
 (root/'result.json').write_text(json.dumps(result,indent=2))
 print(result['status'],result['objective_passed'],result['failure_reason'],flush=True)
asyncio.run(main())
