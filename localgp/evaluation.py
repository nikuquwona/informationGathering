"""Paired evaluation on explicit seeds; baselines only consume actor information."""
import hashlib
import numpy as np
from .environment import DeploymentEnv


def greedy_action(env):
    c=env.config
    mu,var=env.belief.predict()
    # Fixed observation-only UCB heuristic, not a tuned competitor or an oracle.
    score=mu+.5*np.sqrt(var)
    goals=[]
    for i in range(c.agents):
        current=score.copy()
        for goal in goals:
            current-=2*np.exp(-np.sum((env.query-goal)**2,axis=1)/(2*c.gp_length_scale**2))
        # Penalize travel to break the unobserved prior tie locally.
        current-=.05*np.linalg.norm(env.query-env.positions[i],axis=1)/c.area_size
        goals.append(env.query[np.argmax(current)])
    delta=np.asarray(goals)-env.positions
    angle=np.arctan2(delta[:,1],delta[:,0])%(2*np.pi)
    speed=np.minimum(np.linalg.norm(delta,axis=1)/(c.max_speed*c.dt),1)
    return np.column_stack((angle/np.pi-1,2*speed-1))


def episode(trainer, seed, policy='trained', trace=False):
    env=DeploymentEnv(trainer.ec,seed)
    rng=np.random.default_rng(seed+10_000_000)
    rows=[]
    frames=[]
    total_reward=0.
    def record(info=None,actions=None):
        mu,var=env.belief.predict()
        return dict(step=env.steps,positions=env.positions.tolist(),users=env.users.tolist(),
                    measurements=env.measurements.tolist(),distance=env.distance.tolist(),
                    mean=mu.reshape(env.config.grid_size,env.config.grid_size).tolist(),
                    std=np.sqrt(var).reshape(env.config.grid_size,env.config.grid_size).tolist(),
                    reward=info,actions=actions.tolist() if actions is not None else None,metrics=env.metrics())
    if trace:
        frames.append(record())
    while not env.ended:
        if policy=='trained':
            action=trainer.act(env.observe(),deterministic=True)[0]
        elif policy=='random':
            action=rng.uniform(-1,1,(env.config.agents,2))
        elif policy=='greedy':
            action=greedy_action(env)
        elif policy=='stationary':
            action=np.full((env.config.agents,2),-1.)
        else:
            raise ValueError(f'Unknown policy: {policy}')
        _,_,_,_,info=env.step(action)
        total_reward+=info['team_reward']
        rows.append(env.metrics())
        if trace:
            frames.append(record(info,action))
    result=dict(seed=seed,policy=policy,steps=env.steps,team_return=total_reward,
                mean_coverage=float(np.mean([r['coverage'] for r in rows])),
                mean_throughput_bps=float(np.mean([r['throughput_bps'] for r in rows])),
                mean_prediction_rmse=float(np.mean([r['prediction_rmse'] for r in rows])),
                final_coverage=rows[-1]['coverage'],distance_m=rows[-1]['distance_m'],collisions=rows[-1]['collisions'])
    if trace:
        result['frames']=frames
    return result


def summarize(episodes):
    result={}
    for metric in ('team_return','mean_coverage','mean_throughput_bps','mean_prediction_rmse','final_coverage','distance_m','collisions'):
        values=np.array([row[metric] for row in episodes],float)
        result[metric]=dict(mean=float(values.mean()),std=float(values.std(ddof=1)) if len(values)>1 else 0.,n=len(values))
    return result


def evaluate(trainer, count, seed, policy='trained'):
    episodes=[episode(trainer,seed+i,policy) for i in range(count)]
    return dict(policy=policy,seeds=list(range(seed,seed+count)),summary=summarize(episodes),episodes=episodes)


def compare(trainer, count, seed):
    results={policy:evaluate(trainer,count,seed,policy) for policy in ('trained','random','greedy','stationary')}
    # Paired scene differences, not confidence intervals over training seeds.
    trained=results['trained']['episodes']
    differences={}
    for policy in ('random','greedy','stationary'):
        differences[policy]={metric:[a[metric]-b[metric] for a,b in zip(trained,results[policy]['episodes'])]
                             for metric in ('mean_coverage','mean_throughput_bps')}
    return dict(results=results,paired_differences=differences,
                interpretation='Same scene seeds, one trained checkpoint. This is not a multi-training-seed significance study.')
