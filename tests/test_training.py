from dataclasses import replace
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from localgp.config import EnvConfig,TrainConfig
from localgp.model import Actor,select_device
from localgp.trainer import Trainer
from localgp.evaluation import evaluate
from threadpoolctl import threadpool_limits


def configs(device='cpu'):
    return (EnvConfig(users=12,grid_size=8,horizon=5,gp_axis=2,gp_max_samples=8),
            TrainConfig(device=device,total_steps=16,rollout_steps=8,epochs=2,minibatch_size=12,eval_episodes=1))


@pytest.fixture(autouse=True)
def bounded_threads():
    with threadpool_limits(limits=1):
        yield


def test_squashed_logprob_matches_torch_transform_and_extremes_are_finite():
    z=torch.tensor([[.2,-.7],[1.3,-1.6]])
    mean=torch.full_like(z,.1)
    log_std=torch.full_like(z,-.5)
    distribution=torch.distributions.TransformedDistribution(torch.distributions.Normal(mean,log_std.exp()),[torch.distributions.TanhTransform(cache_size=1)])
    expected=distribution.log_prob(z.tanh()).sum(-1)
    torch.testing.assert_close(Actor.log_prob(z,mean,log_std),expected)
    assert torch.isfinite(Actor.log_prob(torch.tensor([[30.,-30.]]),mean[:1],log_std[:1])).all()


def test_collected_actions_have_unit_ppo_ratio_before_update():
    trainer=Trainer(*configs())
    batch,_=trainer.collect()
    with torch.no_grad():
        mean,std=trainer.actor(torch.from_numpy(batch['actor_maps']),torch.from_numpy(batch['actor_context']))
        logs=Actor.log_prob(torch.from_numpy(batch['latent_actions']),mean,std).numpy()
    np.testing.assert_allclose(logs,batch['log_prob'],atol=2e-6)
    before={k:v.clone() for k,v in trainer.actor.state_dict().items()}
    metrics=trainer.update(batch)
    assert metrics['optimizer_steps']>0
    assert any(not torch.equal(before[k],v) for k,v in trainer.actor.state_dict().items())
    assert all(torch.isfinite(p).all() for p in trainer.actor.parameters())


def test_cpu_resume_matches_uninterrupted_next_update(tmp_path):
    trainer=Trainer(*configs())
    batch,_=trainer.collect()
    trainer.update(batch)
    trainer.checkpoint(tmp_path/'checkpoint.pt')
    next_batch,_=trainer.collect()
    trainer.update(next_batch)
    restored=Trainer(*configs())
    restored.restore(tmp_path/'checkpoint.pt')
    actual_batch,_=restored.collect()
    for key in next_batch:
        np.testing.assert_array_equal(actual_batch[key],next_batch[key])
    restored.update(actual_batch)
    for first,second in zip(trainer.actor.parameters(),restored.actor.parameters()):
        torch.testing.assert_close(first,second,rtol=0,atol=0)
    for first,second in zip(trainer.critic.parameters(),restored.critic.parameters()):
        torch.testing.assert_close(first,second,rtol=0,atol=0)


def test_evaluation_does_not_change_training_environment_or_sampling_rng():
    trainer=Trainer(*configs())
    env=trainer.env.state_dict()
    rng=trainer.action_rng.get_state().clone()
    first=evaluate(trainer,2,1000)
    second=evaluate(trainer,2,1000)
    assert first==second
    assert env==trainer.env.state_dict()
    assert torch.equal(rng,trainer.action_rng.get_state())


def test_restore_rejects_changed_scenario_and_allows_extended_budget(tmp_path):
    ec,tc=configs()
    trainer=Trainer(ec,tc)
    trainer.checkpoint(tmp_path/'c.pt')
    wrong=Trainer(replace(ec,users=15),tc)
    with pytest.raises(ValueError,match='configuration'):
        wrong.restore(tmp_path/'c.pt')
    extended=Trainer(ec,replace(tc,total_steps=32))
    extended.restore(tmp_path/'c.pt')


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable on this host')
def test_mps_real_forward_backward_and_checkpoint(tmp_path):
    trainer=Trainer(*configs('mps'))
    assert next(trainer.actor.parameters()).device.type=='mps'
    batch,_=trainer.collect()
    metrics=trainer.update(batch)
    assert np.isfinite(metrics['policy_loss'])
    trainer.checkpoint(tmp_path/'mps.pt')
    restored=Trainer(*configs('mps'))
    restored.restore(tmp_path/'mps.pt')
    for a,b in zip(trainer.actor.parameters(),restored.actor.parameters()):
        torch.testing.assert_close(a,b)


def test_explicit_mps_request_cannot_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='no silent'):
        select_device('mps')
