"""python -m localgp.train --config configs/mac.json --output output/run-001"""
import argparse
from dataclasses import replace
from .config import load_config
from .trainer import Trainer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'))
    parser.add_argument('--steps',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--resume')
    args=parser.parse_args()
    ec,tc=load_config(args.config)
    overrides={k:v for k,v in dict(device=args.device,total_steps=args.steps,seed=args.seed).items() if v is not None}
    trainer=Trainer(ec,replace(tc,**overrides))
    trainer.run(args.output,args.resume)


if __name__=='__main__':
    main()
