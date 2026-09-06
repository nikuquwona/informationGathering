import numpy as np
import pytest
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from forth.GPmodel import LocalGaussianProcessCoordinator, GlobalGaussianProcessCoordinator


def make(local=True, dtype=int):
    kwargs = dict(scenario_map=np.ones((7, 9), dtype=dtype),
                  kernel=ConstantKernel(1.0, 'fixed') * RBF(2.0, 'fixed'), alpha=1e-6)
    if local:
        return LocalGaussianProcessCoordinator(gp_positions=np.array([[2, 2], [4, 6]]),
                                                distance_threshold=3, **kwargs)
    return GlobalGaussianProcessCoordinator(**kwargs)


@pytest.mark.parametrize('local', [True, False])
def test_fractional_predictions_survive_integer_navigation_map(local):
    gp = make(local)
    mu, sigma = gp.update([[2, 2]], [0.4])
    assert np.issubdtype(mu.dtype, np.floating)
    assert mu[2, 2] == pytest.approx(0.4, abs=0.01)
    # Consensus includes an unobserved neighboring expert, so its
    # uncertainty need not equal the observed expert's near-zero value.
    assert 0 < sigma[2, 2] < 0.1
    assert np.isfinite(mu).all() and np.isfinite(sigma).all()


@pytest.mark.parametrize('local', [True, False])
def test_revisiting_a_location_uses_latest_measurement(local):
    gp = make(local)
    gp.update([[2, 2]], [0.2])
    gp.update([[2, 2]], [0.8])
    assert gp.mu_map[2, 2] == pytest.approx(0.8, abs=0.02)
    models = gp.gp_models if local else [gp.gp_models]
    assert len(models[0].x) == 1
    assert models[0].y.item() == 0.8


@pytest.mark.parametrize('local', [True, False])
def test_reset_matches_fresh_instance_and_clears_reward(local):
    gp, fresh = make(local), make(local)
    gp.update([[2, 2], [4, 6]], [0.2, 0.7])
    gp.reset()
    np.testing.assert_allclose(gp.sigma_map, fresh.sigma_map)
    for changes in gp.get_changes():
        assert not changes.any()
    gp.update([[3, 3]], [0.3])
    fresh.update([[3, 3]], [[0.3]])
    np.testing.assert_allclose(gp.mu_map, fresh.mu_map)
    np.testing.assert_allclose(gp.sigma_map, fresh.sigma_map)


@pytest.mark.parametrize('local', [True, False])
def test_empty_collision_batch_leaves_maps_unchanged(local):
    gp = make(local)
    gp.update([[2, 2]], [0.4])
    mu, sigma = gp.mu_map.copy(), gp.sigma_map.copy()
    gp.update([], [])
    np.testing.assert_allclose(gp.mu_map, mu)
    np.testing.assert_allclose(gp.sigma_map, sigma)
    assert not gp.get_changes()[0].any()


def test_global_uncertainty_is_standard_deviation_not_mean():
    gp = make(False)
    gp.update([[2, 2]], [0.0])
    assert gp.mu_map[6, 8] == 0
    assert gp.sigma_map[6, 8] > 0.9
    mu, sigma = gp.generate_nearest_map()
    np.testing.assert_allclose(sigma, gp.gp_models.gp.predict(gp.X, return_std=True)[1])


def test_fusion_uses_only_supported_experts_and_preserves_unobserved_prior():
    gp = LocalGaussianProcessCoordinator(np.array([[1, 1], [8, 8]]), np.ones((12, 12)),
                                         RBF(2, 'fixed'), distance_threshold=1)
    np.testing.assert_allclose(gp.weight[:, gp.covered].sum(axis=0), 1)
    assert not gp.weight[:, ~gp.covered].any()
    gp.update([[1, 1]], [0.5])
    assert gp.mu_map[1, 1] == pytest.approx(0.5, abs=1e-5)
    assert gp.sigma_map[5, 5] == 1
    assert gp.sigma_map[8, 8] == 1


@pytest.mark.parametrize('local', [True, False])
@pytest.mark.parametrize('x,y', [([[1, 2], [2, 3]], [0.1]), ([[1, np.nan]], [0.1]), ([[1, 2]], [np.inf])])
def test_invalid_measurements_rejected_before_mutation(local, x, y):
    gp = make(local)
    with pytest.raises(ValueError):
        gp.update(x, y)
    assert gp.x is None


def test_empty_expert_layout_has_clear_error():
    with pytest.raises(ValueError, match='GP position'):
        LocalGaussianProcessCoordinator(np.empty((0, 2)), np.ones((5, 5)), RBF())
