import asyncio,json,os
from pathlib import Path
os.environ['RETAILFLOW_LANGFUSE_ENABLED']='false'
from retailflow.evaluation import build_manifest,evaluate_episode
root=Path.cwd();path=root/'configs/coverage-evaluation.json';cfg=json.loads(path.read_text())
out=root/'artifacts/coverage-20261118/pilot-44';out.mkdir(exist_ok=False)
task=next(json.loads(line) for line in (root/cfg['dataset_root']/'data/derived/dev.jsonl').read_text().splitlines() if json.loads(line)['source_task_id']=='44')
m=build_manifest(path,cfg);m.update(assigned_task_ids=[task['task_id']],split='dev',variant='B2');(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
r=asyncio.run(evaluate_episode(task,'B2',{**cfg,'output_dir':str(out)}));(out/'results.jsonl').write_text(json.dumps(r,ensure_ascii=False)+'\n')
print(json.dumps({k:r.get(k) for k in ['task_id','objective_passed','failure_reason','elapsed_seconds','usage','events']},ensure_ascii=False,indent=2))
