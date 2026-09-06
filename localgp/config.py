"""Explicit, serializable assumptions; lengths in metres, powers in watts."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class EnvConfig:
    agents: int = 3
    users: int = 50
    clusters: int = 3
    area_size: float = 100.0
    grid_size: int = 24
    horizon: int = 64
    dt: float = 1.0
    max_speed: float = 2.0
    distance_budget: float = 100.0
    min_separation: float = 2.0
    user_speed_std: float = 0.3
    altitude: float = 10.0
    mu_power: float = 0.2
    aebs_power: float = 1.0
    bandwidth: float = 1e6
    noise_power: float = 1e-10
    association_threshold: float = 1e-8
    los_c: float = 9.61
    los_d: float = 0.16
    reference_loss_db: float = 30.0
    path_loss_exponent: float = 2.0
    nlos_extra_db: float = 20.0
    measurement_noise: float = 1e-8
    signal_scale: float = 1e-6
    gp_length_scale: float = 15.0
    gp_noise: float = 0.03
    gp_axis: int = 3
    gp_max_samples: int = 48
    gp_max_age: int = 64
    reward: str = 'balanced'
    signal_weight: float = 1.0
    information_weight: float = 0.5
    movement_weight: float = 0.03
    collision_weight: float = 1.0

    def __post_init__(self):
        integer_names = ('agents', 'users', 'clusters', 'grid_size', 'horizon', 'gp_axis', 'gp_max_samples', 'gp_max_age')
        for name in integer_names:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.agents < 2 or self.grid_size < 8 or self.clusters > self.users:
            raise ValueError('Need >=2 agents, grid_size>=8, and clusters<=users')
        nonnegative = {'user_speed_std', 'measurement_noise', 'nlos_extra_db', 'signal_weight', 'information_weight', 'movement_weight', 'collision_weight'}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name not in integer_names and field.name != 'reward' and (type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (value == 0 and field.name not in nonnegative)):
                raise ValueError(f'{field.name} must be finite and positive (or a permitted zero)')
        if self.min_separation * (self.agents - 1) >= self.area_size * 0.7:
            raise ValueError('Initial fleet cannot fit with the requested separation')
        if self.reward not in ('balanced', 'mean_change'):
            raise ValueError('reward must be balanced or mean_change')


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 7
    device: str = 'auto'
    total_steps: int = 4096
    rollout_steps: int = 64
    epochs: int = 4
    minibatch_size: int = 96
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    eval_every: int = 16
    eval_episodes: int = 4
    eval_seed: int = 100_000
    hidden_size: int = 64
    cpu_threads: int = 1

    def __post_init__(self):
        for name in ('total_steps','rollout_steps','epochs','minibatch_size','eval_every','eval_episodes','hidden_size','cpu_threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.device not in ('auto', 'cpu', 'mps', 'cuda'):
            raise ValueError('Unknown device')
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('gamma and gae_lambda must be in [0,1]')
        for name in ('learning_rate','clip_ratio','max_grad_norm','target_kl','value_coef'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not math.isfinite(self.entropy_coef) or self.entropy_coef < 0:
            raise ValueError('entropy_coef must be finite and nonnegative')
        if self.total_steps % self.rollout_steps:
            raise ValueError('total_steps must be divisible by rollout_steps for resumable updates')


def load_config(path):
    raw = json.loads(Path(path).read_text()) if path else {}
    unknown = set(raw) - {'environment', 'training'}
    if unknown:
        raise ValueError(f'Unknown config sections: {unknown}')
    return EnvConfig(**raw.get('environment', {})), TrainConfig(**raw.get('training', {}))


def config_dict(env, train):
    return {'environment': asdict(env), 'training': asdict(train)}
