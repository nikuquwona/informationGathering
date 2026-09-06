import numpy as np
import pytest
from localgp.rollout import generalized_advantage, Rollout


def test_gae_never_propagates_across_agents():
    reward=np.array([[1,100],[2,200],[3,300]],np.float32)
    zero=np.zeros_like(reward)
    done=np.zeros_like(reward,bool)
    advantages,returns=generalized_advantage(reward,zero,zero,done,done,1,1)
    np.testing.assert_allclose(advantages,[[6,600],[5,500],[3,300]])
    np.testing.assert_array_equal(advantages,returns)


def test_terminal_drops_bootstrap_but_truncation_keeps_final_value():
    r=np.ones((2,2),np.float32)
    v=np.zeros_like(r)
    next_v=np.full_like(r,10)
    terminal=np.array([[True,False],[False,False]])
    truncated=np.array([[False,True],[False,False]])
    a,_=generalized_advantage(r,v,next_v,terminal,truncated,.9,1)
    # Neither first step may inherit the next episode's advantage.
    np.testing.assert_allclose(a,[[1,10],[10,10]])


def test_gae_bootstraps_at_rollout_boundary():
    a,_=generalized_advantage(np.ones((1,2)),np.full((1,2),2),np.full((1,2),3),np.zeros((1,2),bool),np.zeros((1,2),bool),.9,.95)
    np.testing.assert_allclose(a,1+.9*3-2)


def test_unfilled_buffer_cannot_train_and_size_is_bounded():
    obs=dict(maps=np.zeros((3,2,24,24)),context=np.zeros((3,9)))
    central=dict(maps=np.zeros((3,3,24,24)),context=np.zeros((3,13)))
    rollout=Rollout(64,obs,central)
    assert rollout.nbytes < 3_000_000
    with pytest.raises(RuntimeError,match='uninitialized'):
        rollout.finish(.99,.95)


def test_gae_rejects_flattened_input():
    x=np.zeros(6)
    with pytest.raises(ValueError):
        generalized_advantage(x,x,x,x,x,.99,.95)
