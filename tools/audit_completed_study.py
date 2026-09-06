from pathlib import Path
import json,gzip,hashlib
import numpy as np
import torch
root=Path('docs/experiments/ten-rounds')
steps=0;baseline_reference=None;world_references={}
for n in range(1,11):
 r=json.loads((root/f'r{n:02d}/results.json').read_text())
 assert not r['source']['dirty']
 c=r['config']['environment'];assert c['scenario']=='generalized' and c['finite_horizon'] and not c['enforce_distance_budget']
 for trial in r['trials']:
  checkpoint=Path(f'output/fixed-window/r{n:02d}/seed{trial["seed"]}/last.pt')
  state=torch.load(checkpoint,map_location='cpu',weights_only=True)['environment']
  world={k:state[k] for k in ('rng','users','positions','measurements','bounds','scenario_metadata')}
  if trial['seed'] not in world_references:world_references[trial['seed']]=world
  else:assert world==world_references[trial['seed']],(n,trial['seed'],'world RNG mismatch')
  m=trial['manifest'];assert m['device']=='mps' and not m['source']['dirty']
  assert m['source']['commit']==r['source']['commit']
  raw=gzip.decompress((root/f'r{n:02d}/seed{trial["seed"]}-metrics.jsonl.gz').read_bytes())
  assert hashlib.sha256(raw).hexdigest()==trial['training_log_sha256']
  rows=[json.loads(line) for line in raw.splitlines()]
  assert rows[-1]['steps']==32768 and sum(len(x['completed_episodes']) for x in rows)==512
  steps+=rows[-1]['steps']
  assert all(np.isfinite([x[k] for k in ('policy_loss','value_loss','entropy','approx_kl','grad_norm')]).all() for x in rows)
  for kind,e in trial['evaluations'].items():
   assert e['evaluation_device']=='mps'
   for policy,p in e['results'].items():
    assert p['seeds']==list(range(410000,410008))
    for episode in p['episodes']:
     assert episode['steps']==64 and episode['duration_seconds']==64
     assert 0<=episode['horizon_coverage']<=1 and episode['distance_m']<=384+1e-6
  b=trial['evaluations']['best']['results']
  reference={p:[{k:e[k] for k in ('seed','horizon_throughput_bps','horizon_coverage','distance_m','collisions')} for e in b[p]['episodes']] for p in ('random','waypoint','stationary')}
  if baseline_reference is None:baseline_reference=reference
  else:assert reference==baseline_reference,(n,trial['seed'])
assert steps==983040
final=json.loads((root/'final-test.json').read_text())
assert not final['evaluation_source']['dirty']
for r in final['rounds']:
 for t in r['trials']:
  for suite,start in [('standard',510000),('elongated',610000)]:
   for p in t['suites'][suite]['results'].values():
    assert p['seeds']==list(range(start,start+8))
    assert all(e['steps']==64 for e in p['episodes'])
   assert all(e['steps']==64 for e in t['last_suites'][suite]['episodes'])
print(json.dumps(dict(rounds=10,training_runs=30,training_steps=steps,training_episodes=30*512,all_mps=True,all_clean_source=True,full_logs_sha256='verified',training_world_rng='same end state per seed across all rounds',map_only_baseline_pairing='identical service across rounds',final_suites='verified'),indent=2))
