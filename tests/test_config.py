from dataclasses import replace
import pytest
from localgp.config import EnvConfig,TrainConfig


@pytest.mark.parametrize('values',[{'max_speed':0},{'signal_scale':0},{'dt':float('nan')},{'grid_size':3},{'agents':1},{'gp_max_samples':-1},{'measurement_noise':-1}])
def test_invalid_physical_or_memory_parameters_are_rejected(values):
    with pytest.raises(ValueError):
        EnvConfig(**values)


@pytest.mark.parametrize('values',[{'total_steps':65},{'rollout_steps':0},{'gamma':1.1},{'device':'metal'},{'entropy_coef':-1}])
def test_invalid_training_parameters_are_rejected(values):
    with pytest.raises(ValueError):
        TrainConfig(**values)
