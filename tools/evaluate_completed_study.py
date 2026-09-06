"""Open the sealed test suites only after all ten optimization rounds exist."""
from dataclasses import replace
import hashlib,json,sys,copy
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from run_optimization_round import load_trainer
from build_training_report import build
from localgp.trainer import atomic_json,source_metadata
from localgp.evaluation import compare,episode,evaluate
from threadpoolctl import threadpool_limits

records=ROOT/'docs/experiments/ten-rounds'
assert all((records/f'r{n:02d}/results.json').exists() for n in range(1,11))
source=source_metadata();assert not source['dirty']
selected=max(range(1,11),key=lambda n:json.loads((records/f'r{n:02d}/results.json').read_text())['aggregate']['best']['horizon_throughput_bps']['mean'])
output=ROOT/'output/fixed-window/final-test';output.mkdir(parents=True,exist_ok=False)
results=[]
for n in range(1,11):
    trials=[]
    for seed in (7,17,27):
        checkpoint=ROOT/f'output/fixed-window/r{n:02d}/seed{seed}/best.pt'
        trainer,payload=load_trainer(checkpoint)
        suites={}
        for name,family,start in [('standard','standard',510000),('elongated','elongated',610000)]:
            trainer.ec=replace(trainer.ec,map_family=family)
            with threadpool_limits(limits=1):result=compare(trainer,8,start)
            cfg=copy.deepcopy(payload['configuration']);cfg['environment']=trainer.ec.__dict__
            result.update(evaluation_configuration=trainer.ec.__dict__,evaluation_device=str(trainer.device),configuration=cfg,
                          checkpoint_step=payload['total_steps'],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),evaluation_source=source)
            suites[name]=result
            if n==selected and seed==7:
                replay=output/f'replay-{name}';replay.mkdir()
                atomic_json(replay/'comparison.json',result)
                for policy in ('trained','greedy'):
                    with threadpool_limits(limits=1):trace=episode(trainer,start,policy,trace=True)
                    trace.update(schema_version=1,configuration=cfg,checkpoint_sha256=result['checkpoint_sha256'])
                    atomic_json(replay/f'{policy}-episode.json',trace)
                build(replay)
        last_checkpoint=checkpoint.with_name('last.pt');last_trainer,last_payload=load_trainer(last_checkpoint)
        last_suites={}
        for name,family,start in [('standard','standard',510000),('elongated','elongated',610000)]:
            last_trainer.ec=replace(last_trainer.ec,map_family=family)
            with threadpool_limits(limits=1):last_suites[name]=evaluate(last_trainer,8,start)
        trials.append(dict(seed=seed,checkpoint_step=payload['total_steps'],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),suites=suites,
                           last_checkpoint_sha256=hashlib.sha256(last_checkpoint.read_bytes()).hexdigest(),last_checkpoint_step=last_payload['total_steps'],last_suites=last_suites))
        print(f'round {n}, seed {seed}: sealed test complete',flush=True)
    results.append(dict(round=n,trials=trials))
    atomic_json(output/f'r{n:02d}.json',results[-1])
atomic_json(records/'final-test.json',dict(evaluation_source=source,selected_by_tuning_validation=selected,
           interpretation='All ten rounds evaluated after optimization; no further tuning. Finite scenario families, not universal proof.',rounds=results))
