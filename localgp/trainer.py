"""Small on-policy MAPPO trainer with episode-aware GAE and exact CPU resume."""
from dataclasses import asdict
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_limits
from .config import config_dict, EnvConfig, TrainConfig
from .environment import DeploymentEnv
from .model import Actor, Critic, as_tensor, select_device
from .rollout import Rollout


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temp, path)


def source_metadata():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.run(['git','-C',str(root),*args],capture_output=True,check=True).stdout
    return dict(commit=git('rev-parse','HEAD').decode().strip(),
                dirty=bool(git('status','--porcelain')), diff_sha256=hashlib.sha256(git('diff','HEAD')).hexdigest(),
                source_files_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'localgp').glob('*.py'))},
                python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform())


def clip_actor_critic(actor_parameters, critic_parameters, maximum, separate):
    actor_parameters,critic_parameters=list(actor_parameters),list(critic_parameters)
    if separate:
        actor_norm=torch.nn.utils.clip_grad_norm_(actor_parameters,maximum,error_if_nonfinite=True)
        critic_norm=torch.nn.utils.clip_grad_norm_(critic_parameters,maximum,error_if_nonfinite=True)
        return torch.hypot(actor_norm,critic_norm)
    return torch.nn.utils.clip_grad_norm_(actor_parameters+critic_parameters,maximum,error_if_nonfinite=True)


class Trainer:
    def __init__(self, environment_config, training_config):
        self.ec, self.tc = environment_config, training_config
        torch.set_num_threads(training_config.cpu_threads)
        self.device = select_device(training_config.device)
        torch.manual_seed(training_config.seed)
        # Sampling on CPU avoids unsupported distribution operations on MPS and
        # gives us explicit, separately checkpointed sampling/permutation streams.
        self.action_rng = torch.Generator(device='cpu').manual_seed(training_config.seed+1)
        self.shuffle_rng = torch.Generator(device='cpu').manual_seed(training_config.seed+2)
        self.env = DeploymentEnv(environment_config,training_config.seed)
        obs, central = self.env.observe(), self.env.central_state()
        self.actor = Actor(obs['context'].shape[-1], training_config.hidden_size, obs['maps'].shape[1]).to(self.device)
        self.critic = Critic(central['context'].shape[-1], training_config.hidden_size, central['maps'].shape[1]).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()),lr=training_config.learning_rate,eps=1e-5)
        self.total_steps,self.updates,self.episodes = 0,0,0
        self.episode_return = 0.0
        self.best_eval = -float('inf')
        self.last_rollout_bytes = 0

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        result = self.actor.sample(as_tensor(observation['maps'],self.device),as_tensor(observation['context'],self.device),self.action_rng,deterministic)
        return tuple(t.cpu().numpy() for t in result)

    @torch.no_grad()
    def value(self, state):
        return self.critic(as_tensor(state['maps'],self.device),as_tensor(state['context'],self.device)).cpu().numpy()

    def collect(self):
        obs,central = self.env.observe(),self.env.central_state()
        rollout = Rollout(self.tc.rollout_steps,obs,central)
        finished=[]
        for _ in range(self.tc.rollout_steps):
            actions,latent,log_prob = self.act(obs)
            values = self.value(central)
            next_obs,rewards,terminated,truncated,info = self.env.step(actions)
            # Critically: bootstrap using the final state BEFORE a reset.
            next_central = self.env.central_state()
            next_values = self.value(next_central)
            rollout.add(obs,central,latent,log_prob,rewards,values,next_values,terminated,truncated)
            self.total_steps += 1
            self.episode_return += info['team_reward']
            if terminated or truncated:
                self.episodes += 1
                finished.append(dict(episode=self.episodes,team_return=self.episode_return,**self.env.metrics()))
                self.episode_return=0.0
                obs=self.env.reset()
                central=self.env.central_state()
            else:
                obs,central=next_obs,next_central
        self.last_rollout_bytes=rollout.nbytes
        return rollout.finish(self.tc.gamma,self.tc.gae_lambda),finished

    def update(self, batch):
        c=self.tc
        tensors={key:as_tensor(value,self.device) for key,value in batch.items() if key not in ('terminated','truncated')}
        count=len(batch['rewards'])
        metrics=[]
        early=False
        parameters=list(self.actor.parameters())+list(self.critic.parameters())
        for _ in range(c.epochs):
            permutation=torch.randperm(count,generator=self.shuffle_rng)
            for start in range(0,count,c.minibatch_size):
                idx=permutation[start:start+c.minibatch_size].to(self.device)
                mean,log_std=self.actor(tensors['actor_maps'][idx],tensors['actor_context'][idx])
                log_prob=self.actor.log_prob(tensors['latent_actions'][idx],mean,log_std)
                log_ratio=log_prob-tensors['log_prob'][idx]
                ratio=torch.exp(log_ratio)
                approximate_kl=((ratio-1)-log_ratio).mean()
                if not torch.isfinite(approximate_kl):
                    raise FloatingPointError('Non-finite policy KL')
                if approximate_kl.item()>c.target_kl:
                    early=True
                    break
                advantages=tensors['advantages'][idx]
                policy_loss=-torch.minimum(ratio*advantages,ratio.clamp(1-c.clip_ratio,1+c.clip_ratio)*advantages).mean()
                # Monte Carlo entropy of the squashed policy, with its Jacobian.
                noise=torch.randn(mean.shape,generator=self.action_rng).to(self.device)
                fresh=mean+log_std.exp()*noise
                entropy=-self.actor.log_prob(fresh,mean,log_std).mean()
                values=self.critic(tensors['critic_maps'][idx],tensors['critic_context'][idx])
                value_loss=.5*(values-tensors['returns'][idx]).square().mean()
                loss=policy_loss+c.value_coef*value_loss-c.entropy_coef*entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite PPO loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm=clip_actor_critic(self.actor.parameters(),self.critic.parameters(),c.max_grad_norm,c.separate_grad_clip)
                self.optimizer.step()
                metrics.append(dict(policy_loss=policy_loss.item(),value_loss=value_loss.item(),entropy=entropy.item(),
                                    approx_kl=approximate_kl.item(),grad_norm=grad_norm.item(),
                                    clip_fraction=((ratio-1).abs()>c.clip_ratio).float().mean().item()))
            if early:
                break
        self.updates+=1
        if not metrics:
            raise RuntimeError('PPO stopped before any optimizer step; check behavior log probabilities')
        result={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        result.update(kl_early_stop=early,optimizer_steps=len(metrics),rollout_bytes=self.last_rollout_bytes)
        return result

    def checkpoint(self, path):
        payload=dict(schema_version=1,configuration=config_dict(self.ec,self.tc),
                     actor=self.actor.state_dict(),critic=self.critic.state_dict(),optimizer=self.optimizer.state_dict(),
                     action_rng=self.action_rng.get_state(),shuffle_rng=self.shuffle_rng.get_state(),
                     torch_rng=torch.get_rng_state(),environment=self.env.state_dict(),
                     total_steps=self.total_steps,updates=self.updates,episodes=self.episodes,
                     episode_return=self.episode_return,best_eval=self.best_eval,device=str(self.device),
                     source=source_metadata())
        if self.device.type=='mps':
            payload['mps_rng']=torch.mps.get_rng_state()
        if self.device.type=='cuda':
            payload['cuda_rng']=torch.cuda.get_rng_state_all()
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.tmp')
        torch.save(payload,temp)
        os.replace(temp,path)

    def restore(self, path):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('schema_version')!=1:
            raise ValueError('Unsupported checkpoint schema')
        saved=config_dict(EnvConfig(**payload['configuration']['environment']),TrainConfig(**payload['configuration']['training']))
        current=config_dict(self.ec,self.tc)
        # Permit extending a run or moving it between devices; every semantic
        # hyperparameter and scenario assumption must otherwise remain identical.
        for cfg in (saved,current):
            for key in ('total_steps','device'):
                cfg['training'].pop(key)
        if saved!=current:
            raise ValueError('Checkpoint configuration differs from requested run')
        self.actor.load_state_dict(payload['actor'])
        self.critic.load_state_dict(payload['critic'])
        self.optimizer.load_state_dict(payload['optimizer'])
        self.action_rng.set_state(payload['action_rng'])
        self.shuffle_rng.set_state(payload['shuffle_rng'])
        torch.set_rng_state(payload['torch_rng'])
        if self.device.type=='mps' and 'mps_rng' in payload:
            torch.mps.set_rng_state(payload['mps_rng'])
        if self.device.type=='cuda' and 'cuda_rng' in payload:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
        self.env.load_state_dict(payload['environment'])
        for name in ('total_steps','updates','episodes','episode_return','best_eval'):
            setattr(self,name,payload[name])
        return payload

    def run(self, output, resume=None):
        from .evaluation import evaluate
        output=Path(output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Use a new output directory; existing experiment evidence is never overwritten')
        output.mkdir(parents=True,exist_ok=True)
        inherited_best=None
        if resume:
            self.restore(resume)
            if self.total_steps>=self.tc.total_steps:
                raise ValueError('Resume target must exceed the checkpoint step count')
            # A last checkpoint knows the best score, but contains current rather
            # than best parameters. Preserve a compatible sibling best snapshot.
            candidate=Path(resume).resolve().with_name('best.pt')
            if candidate.is_file():
                best=torch.load(candidate,map_location='cpu',weights_only=True)
                saved=config_dict(EnvConfig(**best['configuration']['environment']),TrainConfig(**best['configuration']['training']))
                current=config_dict(self.ec,self.tc)
                for cfg in (saved,current):
                    for key in ('total_steps','device'):
                        cfg.get('training',{}).pop(key,None)
                if saved==current and best.get('total_steps',float('inf'))<=self.total_steps and best.get('best_eval')==self.best_eval:
                    shutil.copyfile(candidate,output/'best.pt')
                    inherited_best=dict(path=str(candidate),sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
            if inherited_best is None:
                # Portable last.pt without its sibling: select among this new
                # segment's evaluations instead of retaining an unattainable score.
                self.best_eval=-float('inf')
        manifest=dict(configuration=config_dict(self.ec,self.tc),source=source_metadata(),device=str(self.device),
                      started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                      inherited_best=inherited_best,best_selection_scope='inherited and current' if inherited_best else 'current segment',
                      resumed_from=str(Path(resume).resolve()) if resume else None,
                      checkpoint_sha256=hashlib.sha256(Path(resume).read_bytes()).hexdigest() if resume else None)
        atomic_json(output/'manifest.json',manifest)
        started=time.perf_counter()
        with threadpool_limits(limits=self.tc.cpu_threads), (output/'metrics.jsonl').open('a',buffering=1) as log:
            while self.total_steps<self.tc.total_steps:
                tick=time.perf_counter()
                batch,episodes=self.collect()
                collection_seconds=time.perf_counter()-tick
                tick=time.perf_counter()
                metrics=self.update(batch)
                row=dict(steps=self.total_steps,update=self.updates,episodes=self.episodes,
                         elapsed_seconds=time.perf_counter()-started,collection_seconds=collection_seconds,
                         update_seconds=time.perf_counter()-tick,completed_episodes=episodes,**metrics)
                if self.updates%self.tc.eval_every==0 or self.total_steps==self.tc.total_steps:
                    evaluation=evaluate(self,self.tc.eval_episodes,self.tc.eval_seed)
                    row['evaluation']=evaluation['summary']
                    atomic_json(output/f'evaluation-{self.total_steps}.json',evaluation)
                    # Select by deploy-time task utility, not the training proxy.
                    score=evaluation['summary']['mean_throughput_bps']['mean']
                    if score>self.best_eval:
                        self.best_eval=score
                        self.checkpoint(output/'best.pt')
                self.checkpoint(output/'last.pt')
                log.write(json.dumps(row,allow_nan=False)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k!='completed_episodes'},allow_nan=False),flush=True)
        return output
