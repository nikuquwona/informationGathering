"""Continuous multi-agent sensing with explicit observation and truth boundaries."""
from dataclasses import asdict
import copy
import numpy as np
from .config import EnvConfig
from .gp import LocalBelief
from .radio import channel_gain, service_metrics


def resolve_motion(positions, deltas, area_size, min_separation):
    """Reject unsafe moves until all simultaneous straight paths are separated.

    Once a move is rejected, that agent is stationary; recheck everyone against
    this new path. This monotone loop terminates in at most N rejection rounds.
    """
    positions, deltas = np.asarray(positions, float), np.asarray(deltas, float)
    goals = positions + deltas
    rejected = np.any((goals < 0) | (goals > area_size), axis=1)
    for _ in range(len(positions) + 1):
        moves = deltas.copy()
        moves[rejected] = 0
        newly = rejected.copy()
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                relative = positions[i] - positions[j]
                velocity = moves[i] - moves[j]
                denom = velocity @ velocity
                time = np.clip(-(relative @ velocity) / denom, 0, 1) if denom > 1e-15 else 0
                if np.linalg.norm(relative + time * velocity) < min_separation - 1e-8:
                    newly[i] = newly[j] = True
        if np.array_equal(newly, rejected):
            return positions + moves, rejected
        rejected = newly
    raise RuntimeError('Collision resolution did not converge')


class DeploymentEnv:
    def __init__(self, config=EnvConfig(), seed=0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        axis = (np.arange(config.grid_size) + 0.5) * config.area_size / config.grid_size
        self.query = np.array(np.meshgrid(axis, axis, indexing='ij')).reshape(2, -1).T
        self.reset()

    def encode_signal(self, signal):
        # Fixed physical reference. Never normalize by hidden ground-truth extrema.
        return np.log1p(np.maximum(signal, 0) / self.config.signal_scale)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.config
        self.steps = 0
        self.ended = False
        centers = self.rng.uniform(0.2, 0.8, (c.clusters, 2)) * c.area_size
        labels = self.rng.integers(c.clusters, size=c.users)
        self.users = np.clip(centers[labels] + self.rng.normal(0, c.area_size * .07, (c.users, 2)), 0, c.area_size)
        self.positions = np.column_stack((np.full(c.agents, c.area_size * .12), np.linspace(.15, .85, c.agents) * c.area_size))
        self.distance = np.zeros(c.agents)
        self.collisions = 0
        self.belief = LocalBelief(c, self.query)
        self.measurements = self._measure()
        self.belief.update(self.positions, self.encode_signal(self.measurements), self.steps)
        return self.observe()

    def _measure(self):
        power = self.config.mu_power * channel_gain(self.positions, self.users, self.config).sum(axis=1)
        return power + self.rng.normal(0, self.config.measurement_noise, len(self.positions))

    def observe(self):
        c = self.config
        mu, var = self.belief.predict()
        shared = np.stack((mu.reshape(c.grid_size, c.grid_size), np.sqrt(var).reshape(c.grid_size, c.grid_size))).astype(np.float32)
        coords = self.positions / c.area_size
        sensor = self.encode_signal(self.measurements)
        # Own position, remaining budget, own reading, relative fleet positions.
        context = []
        for i in range(c.agents):
            others = np.delete(coords - coords[i], i, axis=0).reshape(-1)
            context.append(np.r_[coords[i], 1-self.distance[i]/c.distance_budget, sensor[i], 1-self.steps/c.horizon, others])
        return dict(maps=np.repeat(shared[None], c.agents, axis=0), context=np.asarray(context, np.float32))

    def central_state(self):
        c = self.config
        actor = self.observe()
        truth = self.encode_signal(c.mu_power * channel_gain(self.query, self.users, c).sum(axis=1)).reshape(c.grid_size, c.grid_size)
        maps = np.concatenate((actor['maps'][0], truth[None]), axis=0).astype(np.float32)
        global_context = np.r_[self.positions.reshape(-1)/c.area_size, 1-self.distance/c.distance_budget, self.steps/c.horizon]
        contexts = [np.r_[global_context, np.eye(c.agents)[i]] for i in range(c.agents)]
        return dict(maps=np.repeat(maps[None], c.agents, axis=0), context=np.asarray(contexts, np.float32))

    def step(self, actions):
        c = self.config
        actions = np.asarray(actions, dtype=float)
        if self.ended:
            raise RuntimeError('Call reset after termination or truncation')
        if actions.shape != (c.agents, 2) or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1 + 1e-6):
            raise ValueError('Actions must be finite (agents, 2), in [-1,1]')
        actions = np.clip(actions, -1, 1)
        heading = np.pi * (actions[:, 0] + 1)
        length = np.minimum((actions[:, 1]+1)*.5*c.max_speed*c.dt, c.distance_budget-self.distance)
        deltas = np.column_stack((np.cos(heading), np.sin(heading))) * length[:, None]
        previous = self.positions.copy()
        self.positions, rejected = resolve_motion(previous, deltas, c.area_size, c.min_separation)
        travelled = np.linalg.norm(self.positions-previous, axis=1)
        self.distance += travelled
        self.collisions += int(rejected.sum())
        self.steps += 1
        # Reflect rather than accumulate/clamp users at boundaries.
        moved = self.users + self.rng.normal(0, c.user_speed_std*c.dt, self.users.shape)
        folded = np.mod(moved, 2*c.area_size)
        self.users = np.minimum(folded, 2*c.area_size-folded)
        self.belief.age(self.steps)
        before_mu, before_var = self.belief.predict()
        self.measurements = self._measure()
        self.belief.update(self.positions, self.encode_signal(self.measurements), self.steps)
        after_mu, after_var = self.belief.predict()
        # Average avoids rewarding a denser numerical grid for the same scene.
        information = float(np.maximum(0, np.log(before_var)-np.log(after_var)).mean())
        mean_change = float(np.abs(after_mu-before_mu).mean())
        sensed = self.encode_signal(self.measurements)
        distances = np.linalg.norm(self.positions[:, None]-self.positions[None, :], axis=-1)
        redundancy = np.exp(-.5*(distances/c.gp_length_scale)**2).sum(axis=1)
        signal = float((sensed/redundancy).mean())
        movement = float((travelled/(c.max_speed*c.dt)).mean())
        collision = float(rejected.mean())
        utility = c.signal_weight*signal+c.information_weight*information if c.reward=='balanced' else mean_change
        team_reward = utility-c.movement_weight*movement-c.collision_weight*collision
        reward = np.full(c.agents, team_reward, dtype=np.float32)
        terminated = bool(np.any(self.distance >= c.distance_budget-1e-7))
        truncated = bool(self.steps >= c.horizon and not terminated)
        self.ended = terminated or truncated
        info = dict(signal=signal, information=information, mean_change=mean_change,
                    movement=movement, collision=collision, rejected=rejected.tolist(),
                    team_reward=team_reward)
        return self.observe(), reward, terminated, truncated, info

    def metrics(self):
        result = service_metrics(self.positions, self.users, self.config)
        mu, _ = self.belief.predict()
        truth = self.encode_signal(self.config.mu_power*channel_gain(self.query, self.users, self.config).sum(axis=1))
        result.update(prediction_rmse=float(np.sqrt(np.mean((mu-truth)**2))),
                      distance_m=float(self.distance.sum()), collisions=self.collisions)
        return result

    def state_dict(self):
        return dict(config=asdict(self.config), rng=copy.deepcopy(self.rng.bit_generator.state),
                    steps=self.steps, ended=self.ended, collisions=self.collisions,
                    positions=self.positions.tolist(), users=self.users.tolist(),
                    distance=self.distance.tolist(), measurements=self.measurements.tolist(), belief=self.belief.state_dict())

    def load_state_dict(self, state):
        if state['config'] != asdict(self.config):
            raise ValueError('Environment configuration differs from checkpoint')
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        for name in ('steps','ended','collisions'):
            setattr(self, name, state[name])
        for name in ('positions','users','distance','measurements'):
            setattr(self, name, np.asarray(state[name], dtype=float))
        self.belief.load_state_dict(state['belief'])
