import asyncio,json,os
from pathlib import Path
os.environ['RETAILFLOW_LANGFUSE_ENABLED']='false'
from retailflow.evaluation import build_manifest,evaluate_episode
root=Path.cwd(); cfg_path=root/'configs/exchange-evaluation.json'; cfg=json.loads(cfg_path.read_text())
out=root/'artifacts/exchange-20261118/pilot-75-receipt'
out.mkdir(exist_ok=False)
tasks=[json.loads(l) for l in (root/cfg['dataset_root']/'data/derived/dev.jsonl').read_text().splitlines()]
task=next(t for t in tasks if t['task_id']=='tau-retail-75')
manifest=build_manifest(cfg_path,cfg);manifest.update(assigned_task_ids=[task['task_id']],split='dev',variant='B2')
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
result=asyncio.run(evaluate_episode(task,'B2',{**cfg,'output_dir':str(out)}))
(out/'results.jsonl').write_text(json.dumps(result,ensure_ascii=False)+'\n')
print(json.dumps({k:result.get(k) for k in ['task_id','objective_passed','success','failure_reason','elapsed_seconds','usage','events']},ensure_ascii=False,indent=2))
