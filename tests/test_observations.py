from types import SimpleNamespace
import numpy as np
import pytest
from forth.InformationGatheringEnvironment import MultiagentInformationGathering


@pytest.mark.parametrize('shape,count', [((12, 17), 1), ((12, 17), 4), ((100, 100), 3)])
def test_observation_shape_dtype_and_unique_agent_identity(shape, count):
    env = MultiagentInformationGathering.__new__(MultiagentInformationGathering)
    env.scenario_map = np.ones(shape, dtype=int)
    env.number_of_agents = count
    env.fleet = SimpleNamespace(agent_positions=np.array([[i + 1, i + 2] for i in range(count)]))
    env.gp_coordinator = SimpleNamespace(mu_map=np.full(shape, 0.25), sigma_map=np.full(shape, 0.75))
    env.update_state()
    for i, state in env.state.items():
        assert state.shape == (6, *shape)
        assert state.dtype == np.float32
        assert state[3].sum() == 1
        assert state[2].sum() == count - 1
        np.testing.assert_allclose(state[5], i / max(1, count - 1))
        assert np.isfinite(state).all()
