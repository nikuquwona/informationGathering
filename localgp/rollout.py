"""On-policy time × agent storage; never recurse GAE across agents or resets."""
import numpy as np


def generalized_advantage(rewards, values, next_values, terminated, truncated, gamma, lam):
    arrays = [np.asarray(x) for x in (rewards, values, next_values, terminated, truncated)]
    if arrays[0].ndim != 2 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('All rollout arrays must have identical (time, agent) shapes')
    rewards, values, next_values, terminated, truncated = arrays
    delta = rewards + gamma * (~terminated.astype(bool)) * next_values - values
    advantages = np.zeros_like(rewards, dtype=np.float32)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in range(len(rewards)-1, -1, -1):
        continuation = ~(terminated[t].astype(bool) | truncated[t].astype(bool))
        carry = delta[t] + gamma*lam*continuation*carry
        advantages[t] = carry
    return advantages, (advantages+values).astype(np.float32)


class Rollout:
    def __init__(self, steps, observation, central):
        n = len(observation['maps'])
        self.arrays = {}
        for prefix, source in (('actor', observation), ('critic', central)):
            for name, value in source.items():
                self.arrays[f'{prefix}_{name}'] = np.empty((steps, *value.shape), np.float32)
        self.arrays['latent_actions'] = np.empty((steps,n,2), np.float32)
        for key in ('log_prob','rewards','values','next_values'):
            self.arrays[key] = np.empty((steps,n), np.float32)
        for key in ('terminated','truncated'):
            self.arrays[key] = np.empty((steps,n), bool)
        self.steps, self.count = steps, 0

    def add(self, observation, central, latent, log_prob, rewards, values, next_values, terminated, truncated):
        if self.count >= self.steps:
            raise RuntimeError('Rollout is full')
        t = self.count
        for prefix, source in (('actor', observation), ('critic', central)):
            for name, value in source.items():
                self.arrays[f'{prefix}_{name}'][t] = value
        for key, value in dict(latent_actions=latent, log_prob=log_prob, rewards=rewards,
                               values=values, next_values=next_values,
                               terminated=terminated, truncated=truncated).items():
            self.arrays[key][t] = value
        self.count += 1

    def finish(self, gamma, lam):
        if self.count != self.steps:
            raise RuntimeError('Cannot train on uninitialized rollout entries')
        a = self.arrays
        advantages, returns = generalized_advantage(a['rewards'],a['values'],a['next_values'],a['terminated'],a['truncated'],gamma,lam)
        # Normalize over this fresh on-policy batch only; not over future runs.
        a['advantages'] = (advantages-advantages.mean()) / (advantages.std()+1e-8)
        a['returns'] = returns
        if any(not np.isfinite(value).all() for value in a.values()):
            raise FloatingPointError('Non-finite rollout')
        return {k: v.reshape((-1, *v.shape[2:])) for k,v in a.items()}

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.arrays.values())
