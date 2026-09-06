from dataclasses import replace
import numpy as np
import pytest
from localgp.config import EnvConfig,TrainConfig
from localgp.environment import DeploymentEnv,resolve_motion
from localgp.scenarios import inside


def config(**kwargs):return EnvConfig(scenario='generalized',**kwargs)


def test_random_maps_starts_and_clusters_vary_reproducibly():
    seen=[]
    for seed in range(12):
        a=DeploymentEnv(config(),seed);b=DeploymentEnv(config(),seed)
        np.testing.assert_array_equal(a.bounds,b.bounds)
        np.testing.assert_array_equal(a.positions,b.positions)
        np.testing.assert_array_equal(a.users,b.users)
        assert inside(a.positions,a.bounds).all() and inside(a.users,a.bounds).all()
        distances=np.linalg.norm(a.positions[:,None]-a.positions[None,:],axis=-1)+np.eye(3)*1000
        assert distances.min()>=a.config.min_separation
        seen.append(a)
    assert len({a.bounds.tobytes() for a in seen})==12
    assert len({len(a.users) for a in seen})>1
    assert np.std([a.positions[0,0] for a in seen])>10
    assert np.std([a.positions[0,1] for a in seen])>10


def test_random_map_bounds_reject_motion_and_altitude_is_validated():
    bounds=np.array([[10,20],[70,90]])
    final,rejected=resolve_motion([[69,50],[20,30]],[[2,0],[0,1]],100,2,bounds)
    assert rejected.tolist()==[True,False]
    np.testing.assert_array_equal(final[0],[69,50])
    with pytest.raises(ValueError,match='min_altitude'):config(altitude=4,min_altitude=5)


def test_actor_knows_bounds_but_not_users_and_restores():
    env=DeploymentEnv(config(),7)
    obs=env.observe();assert obs['maps'].shape==(3,2,24,24)
    np.testing.assert_allclose(obs['context'][0,-4:],env.bounds.ravel()/100)
    env.users[:]=50
    for key in obs:np.testing.assert_array_equal(obs[key],env.observe()[key])
    original=DeploymentEnv(config(),8)
    other=DeploymentEnv(config(),9);other.load_state_dict(original.state_dict())
    for _ in range(3):
        oa,ra,*_=original.step(np.zeros((3,2)));ob,rb,*_=other.step(np.zeros((3,2)))
        for key in oa:np.testing.assert_array_equal(oa[key],ob[key])
        np.testing.assert_array_equal(ra,rb)
        assert inside(original.users,original.bounds).all()


def test_heldout_elongated_maps_are_valid():
    for seed in range(5):
        env=DeploymentEnv(config(map_family='elongated'),seed)
        span=env.bounds[1]-env.bounds[0]
        assert max(span)/min(span)>=2
        assert inside(env.positions,env.bounds).all()


def test_generalized_training_update_and_restore(tmp_path):
    pytest.importorskip('torch')
    from localgp.trainer import Trainer
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        trainer=Trainer(config(users=12,grid_size=12),TrainConfig(device='cpu',rollout_steps=8,total_steps=16))
        batch,_=trainer.collect();trainer.update(batch);trainer.checkpoint(tmp_path/'model.pt')
        restored=Trainer(trainer.ec,trainer.tc);restored.restore(tmp_path/'model.pt')
        a,_=trainer.collect();b,_=restored.collect()
        for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_fixed_window_has_no_flight_budget_and_zero_terminal_bootstrap():
    env=DeploymentEnv(config(enforce_distance_budget=False,finite_horizon=True,distance_budget=.01,horizon=3),7)
    for step in range(3):
        _,_,terminated,truncated,_=env.step(np.zeros((3,2)))
        assert terminated==(step==2)
        assert not truncated
    assert env.distance.sum()>.01
    np.testing.assert_array_equal(env.observe()['context'][:,2],1.)
