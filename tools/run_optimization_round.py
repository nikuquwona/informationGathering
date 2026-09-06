"""Run three fresh MPS training seeds and record a complete tuning-only round."""
import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from localgp.config import EnvConfig, TrainConfig
from localgp.trainer import Trainer, atomic_json, source_metadata
from localgp.evaluation import compare


def load_trainer(checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = payload['configuration']
    trainer = Trainer(EnvConfig(**cfg['environment']), replace(TrainConfig(**cfg['training']), device='mps'))
    trainer.restore(checkpoint)
    return trainer, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('round', type=int)
    args = parser.parse_args()
    label = f'r{args.round:02d}'
    config = ROOT / f'configs/optimization/{label}.json'
    record = ROOT / f'docs/experiments/ten-rounds/{label}'
    record.mkdir(parents=True, exist_ok=True)
    output = ROOT / f'output/optimization/{label}'
    output.mkdir(parents=True, exist_ok=True)
    source = source_metadata()
    if source['dirty']:
        raise RuntimeError('Commit round configuration/code before running')
    trials = []
    start = time.time()
    for seed in (7, 17, 27):
        destination = output / f'seed{seed}'
        print(f'{label}: seed {seed} training started', flush=True)
        with (output / f'seed{seed}-stdout.jsonl').open('w') as log:
            subprocess.run([sys.executable, '-m', 'localgp.train', '--config', str(config),
                            '--seed', str(seed), '--output', str(destination)], cwd=ROOT,
                           stdout=log, check=True)
        logs = [json.loads(line) for line in (destination/'metrics.jsonl').read_text().splitlines()]
        assert logs[-1]['steps'] == 32768
        assert all(np.isfinite([r[k] for k in ('policy_loss','value_loss','entropy','approx_kl','grad_norm')]).all() for r in logs)
        evaluations = {}
        for kind in ('best', 'last'):
            checkpoint = destination / f'{kind}.pt'
            trainer, payload = load_trainer(checkpoint)
            with threadpool_limits(limits=1):
                result = compare(trainer, 8, 410_000)
            result.update(checkpoint_step=payload['total_steps'], checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          evaluation_source=source, evaluation_device=str(trainer.device))
            atomic_json(destination/f'tuning-{kind}.json', result)
            evaluations[kind] = result
        # All update rows and completed training episodes are versioned losslessly.
        with gzip.open(record/f'seed{seed}-metrics.jsonl.gz','wb') as target:
            target.write((destination/'metrics.jsonl').read_bytes())
        manifest = json.loads((destination/'manifest.json').read_text())
        assert manifest['source']['commit'] == source['commit'] and not manifest['source']['dirty']
        trials.append(dict(seed=seed, manifest=manifest, evaluations=evaluations,
                           training_log_sha256=hashlib.sha256((destination/'metrics.jsonl').read_bytes()).hexdigest(),
                           curve=[{k:v for k,v in row.items() if k!='completed_episodes'} for row in logs]))
        print(f'{label}: seed {seed} complete; tuning Mbps best={evaluations["best"]["results"]["trained"]["summary"]["horizon_throughput_bps"]["mean"]/1e6:.3f}, last={evaluations["last"]["results"]["trained"]["summary"]["horizon_throughput_bps"]["mean"]/1e6:.3f}',flush=True)
    aggregate = {}
    for kind in ('best','last'):
        aggregate[kind] = {}
        for metric in ('horizon_throughput_bps','horizon_coverage','collisions','delivered_megabits'):
            a = np.array([t['evaluations'][kind]['results']['trained']['summary'][metric]['mean'] for t in trials])
            aggregate[kind][metric] = dict(mean=float(a.mean()), std_training_seeds=float(a.std(ddof=1)), seed_means=a.tolist())
    atomic_json(record/'results.json',dict(round=args.round,source=source,config=json.loads(config.read_text()),
                 tuning_seeds=list(range(410000,410008)),elapsed_seconds=time.time()-start,trials=trials,aggregate=aggregate))
    print(json.dumps(dict(round=args.round,aggregate=aggregate),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
