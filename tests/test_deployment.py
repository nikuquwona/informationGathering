from dataclasses import replace
import numpy as np
import pytest
from localgp.config import EnvConfig
from localgp.environment import DeploymentEnv, resolve_motion
from localgp.radio import channel_gain, service_metrics


@pytest.fixture
def config():
    return EnvConfig(users=12, grid_size=8, gp_axis=2, gp_max_samples=8, horizon=5)


def test_rng_isolation_and_seed_reproducibility(config):
    np.random.seed(19)
    expected = np.random.random(4)
    np.random.seed(19)
    a, b = DeploymentEnv(config, 42), DeploymentEnv(config, 42)
    np.testing.assert_array_equal(np.random.random(4), expected)
    other = DeploymentEnv(config, 99)
    for _ in range(3):
        other.step(np.zeros((3,2)))
        oa, ra, *_ = a.step(np.zeros((3,2)))
        ob, rb, *_ = b.step(np.zeros((3,2)))
        np.testing.assert_array_equal(a.users, b.users)
        np.testing.assert_array_equal(oa['maps'], ob['maps'])
        np.testing.assert_array_equal(ra, rb)
    assert len(a.users) == 12


def test_actor_never_reads_hidden_truth(config):
    env = DeploymentEnv(config, 7)
    obs = env.observe()
    central = env.central_state()
    env.users[:] = 50
    for key in obs:
        np.testing.assert_array_equal(obs[key], env.observe()[key])
    assert not np.array_equal(central['maps'], env.central_state()['maps'])


@pytest.mark.parametrize('positions,deltas', [
    ([[10,10],[14,10]], [[4,0],[-4,0]]),  # swapped endpoints
    ([[10,10],[12,8]], [[4,0],[0,4]]),   # crossing paths
    ([[95,10],[99,10]], [[4,0],[4,0]]),  # second rejected at boundary
])
def test_entire_motion_is_collision_safe(positions, deltas):
    positions, deltas = np.array(positions, float), np.array(deltas, float)
    final, rejected = resolve_motion(positions, deltas, 100, 2)
    assert rejected.any()
    for t in np.linspace(0,1,101):
        p = positions + t*(final-positions)
        assert np.linalg.norm(p[0]-p[1]) >= 2-1e-8


def test_rejected_moves_do_not_spend_distance_and_budget_is_exact(config):
    env = DeploymentEnv(replace(config, distance_budget=.25), 7)
    _, _, terminated, truncated, _ = env.step(np.tile([-1.,1.], (3,1)))
    assert terminated and not truncated
    np.testing.assert_allclose(env.distance, .25)
    with pytest.raises(RuntimeError):
        env.step(np.zeros((3,2)))
    env = DeploymentEnv(config, 7)
    env.positions[0] = [.1, 15]
    _, _, _, _, info = env.step(np.array([[0,1],[-1,-1],[-1,-1]]))
    assert info['rejected'][0]
    assert env.distance[0] == 0


def test_time_limit_is_truncation(config):
    env = DeploymentEnv(replace(config, horizon=1), 0)
    _, _, terminated, truncated, _ = env.step(np.zeros((3,2)))
    assert not terminated and truncated


def test_restore_matches_uninterrupted_environment(config):
    env = DeploymentEnv(config, 1)
    env.step(np.zeros((3,2)))
    restored = DeploymentEnv(config, 999)
    restored.load_state_dict(env.state_dict())
    for _ in range(2):
        a, ar, *_ = env.step(np.full((3,2), .2))
        b, br, *_ = restored.step(np.full((3,2), .2))
        np.testing.assert_array_equal(env.users, restored.users)
        np.testing.assert_allclose(a['maps'], b['maps'], atol=1e-7)
        np.testing.assert_allclose(ar, br, atol=1e-7)


def test_radio_power_decreases_with_distance_and_unique_association(config):
    users = np.array([[10.,10.],[90.,90.]])
    gains = channel_gain([[10,10],[50,50]], users, config)
    assert gains[0,0] > gains[1,0]
    metrics = service_metrics([[10,10],[90,90]], users, config)
    assert sum(metrics['loads']) == metrics['served_users'] <= len(users)
    assert metrics['association'] == [0,1]
    assert metrics['throughput_bps'] > 0
    none = service_metrics([[10,10]], users, replace(config, association_threshold=1e9))
    assert none['throughput_bps'] == 0 and none['served_users'] == 0


def test_bounded_belief_and_actual_measurement_noise(config):
    from localgp.gp import LocalBelief
    c = replace(config, gp_max_samples=2, gp_max_age=2)
    belief = LocalBelief(c, np.array([[0.,0.],[50.,50.]]))
    for t in range(20):
        belief.age(t)
        belief.update([[t, t]], [t/20], t)
    assert len(belief.points) <= c.gp_max_samples*c.gp_axis**2
    assert belief.times.min() >= 17
    env = DeploymentEnv(config, 2)
    assert not np.array_equal(env._measure(), env._measure())


def test_invalid_actions_do_not_advance_environment(config):
    env = DeploymentEnv(config, 1)
    for a in (np.zeros((2,2)), np.full((3,2), np.nan), np.full((3,2), 1.01)):
        with pytest.raises(ValueError):
            env.step(a)
    assert env.steps == 0
