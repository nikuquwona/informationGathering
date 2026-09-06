"""Evaluate a checkpoint against paired observation-only baselines."""
import argparse
from dataclasses import replace
from pathlib import Path
import torch
from threadpoolctl import threadpool_limits
from .config import EnvConfig,TrainConfig
from .trainer import Trainer,atomic_json
from .evaluation import compare,episode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='auto',choices=('auto','cpu','mps','cuda'))
    parser.add_argument('--episodes',type=int,default=8)
    parser.add_argument('--seed',type=int,default=200_000)
    args=parser.parse_args()
    if args.episodes<1:
        parser.error('--episodes must be positive')
    output=Path(args.output)
    if output.exists():
        parser.error('Output already exists; choose a new experiment directory')
    cfg=torch.load(args.checkpoint,map_location='cpu',weights_only=True)['configuration']
    trainer=Trainer(EnvConfig(**cfg['environment']),replace(TrainConfig(**cfg['training']),device=args.device))
    trainer.restore(args.checkpoint)
    output.mkdir(parents=True)
    with threadpool_limits(limits=trainer.tc.cpu_threads):
        result=compare(trainer,args.episodes,args.seed)
        import hashlib
        result['checkpoint_sha256']=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        result['configuration']=cfg
        atomic_json(output/'comparison.json',result)
        for policy in ('trained','greedy'):
            trace=episode(trainer,args.seed,policy,trace=True)
            trace.update(schema_version=1,configuration=cfg,checkpoint_sha256=result['checkpoint_sha256'])
            atomic_json(output/f'{policy}-episode.json',trace)
    print(output/'comparison.json')


if __name__=='__main__':
    main()
