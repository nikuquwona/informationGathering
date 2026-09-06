"""Bounded local RBF regressors with explicit aging and fixed hyperparameters.

The fused variance is an uncertainty surrogate, not a calibrated joint posterior.
Factorizations and geographic query caches stay on CPU, outside the policy graph.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist


class LocalBelief:
    def __init__(self, config, query):
        self.config, self.query = config, np.asarray(query, dtype=float)
        axis = (np.arange(config.gp_axis) + 0.5) * config.area_size / config.gp_axis
        self.centers = np.array(np.meshgrid(axis, axis, indexing='ij')).reshape(2, -1).T
        self.radius = config.area_size / config.gp_axis
        self.points = np.empty((0, 2))
        self.values = np.empty(0)
        self.times = np.empty(0, dtype=int)
        distances = cdist(self.query, self.centers)
        weights = np.exp(-0.5 * (distances / self.radius)**2)
        self.weights = weights / weights.sum(axis=1, keepdims=True)
        self._cached = None

    def age(self, time):
        keep = self.times >= time - self.config.gp_max_age
        if not keep.all():
            self.points, self.values, self.times = self.points[keep], self.values[keep], self.times[keep]
            self._cached = None

    def update(self, points, values, time):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(points) != len(values) or not np.isfinite(points).all() or not np.isfinite(values).all():
            raise ValueError('Need one finite value per finite 2D measurement position')
        self.points = np.vstack((self.points, points))
        self.values = np.concatenate((self.values, values))
        self.times = np.concatenate((self.times, np.full(len(points), time, dtype=int)))
        # Centimetre-scale duplicate keys avoid singular repeated measurements.
        _, reversed_index = np.unique(np.round(self.points[::-1], 2), axis=0, return_index=True)
        keep = np.sort(len(self.points) - 1 - reversed_index)
        # Bound both the global history and each local factorization.
        limit = self.config.gp_max_samples * len(self.centers)
        keep = keep[-limit:]
        self.points, self.values, self.times = self.points[keep], self.values[keep], self.times[keep]
        self._cached = None

    def predict(self):
        if self._cached is not None:
            return tuple(a.copy() for a in self._cached)
        mean, variance = np.zeros(len(self.query)), np.zeros(len(self.query))
        ell = self.config.gp_length_scale
        for i, center in enumerate(self.centers):
            indices = np.flatnonzero(np.linalg.norm(self.points - center, axis=1) <= self.radius * 1.5)
            indices = indices[-self.config.gp_max_samples:]
            if not len(indices):
                local_mean, local_var = 0.0, 1.0
            else:
                x, y = self.points[indices], self.values[indices]
                kernel = np.exp(-0.5 * cdist(x, x, 'sqeuclidean') / ell**2)
                kernel.flat[::len(x)+1] += self.config.gp_noise**2 + 1e-6
                factor = cho_factor(kernel, lower=True, check_finite=False)
                cross = np.exp(-0.5 * cdist(x, self.query, 'sqeuclidean') / ell**2)
                local_mean = cross.T @ cho_solve(factor, y, check_finite=False)
                projected = solve_triangular(factor[0], cross, lower=True, check_finite=False)
                local_var = np.maximum(1e-6, 1 - np.sum(projected**2, axis=0))
            mean += self.weights[:, i] * local_mean
            variance += self.weights[:, i] * local_var
        self._cached = (mean, np.clip(variance, 1e-6, 1))
        return tuple(a.copy() for a in self._cached)

    def state_dict(self):
        return {name: getattr(self, name).tolist() for name in ('points', 'values', 'times')}

    def load_state_dict(self, state):
        self.points = np.asarray(state['points'], dtype=float).reshape(-1, 2)
        self.values = np.asarray(state['values'], dtype=float)
        self.times = np.asarray(state['times'], dtype=int)
        self._cached = None
