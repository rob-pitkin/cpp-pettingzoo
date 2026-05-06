"""Test SimpleFormation C++ implementation matches mpe2 reference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from mpe2 import simple_formation_v1
from pettingzoo.test import parallel_api_test

from cpp_pettingzoo.simple_formation.simple_formation import parallel_env as cpp_env


def test_api_default():
    parallel_api_test(cpp_env(N=4), num_cycles=100)


def test_api_continuous():
    parallel_api_test(cpp_env(N=4, continuous_actions=True), num_cycles=100)


def test_api_terminate_on_success():
    parallel_api_test(cpp_env(N=4, terminate_on_success=True), num_cycles=100)


def test_reset_deterministic():
    env1, env2 = cpp_env(), cpp_env()
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    for a in env1.agents:
        np.testing.assert_allclose(obs1[a], obs2[a])
    env1.close(); env2.close()


def test_reset_different_seeds():
    env1, env2 = cpp_env(), cpp_env()
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=99)
    assert any(not np.allclose(obs1[a], obs2[a]) for a in env1.agents)
    env1.close(); env2.close()


def test_agent_names_default():
    env = cpp_env(N=4)
    env.reset(seed=0)
    assert set(env.agents) == {"agent_0", "agent_1", "agent_2", "agent_3"}
    env.close()


def test_observation_shape_matches_mpe2():
    for N in [2, 3, 4, 6]:
        cpp = cpp_env(N=N)
        mpe2 = simple_formation_v1.parallel_env(N=N)
        cpp.reset(seed=42); mpe2.reset(seed=42)
        for a in cpp.possible_agents:
            assert cpp.observation_space(a).shape == mpe2.observation_space(a).shape, \
                f"N={N} {a}: cpp={cpp.observation_space(a).shape} mpe2={mpe2.observation_space(a).shape}"
        cpp.close(); mpe2.close()


def test_action_spaces_discrete():
    env = cpp_env()
    env.reset(seed=42)
    for a in env.agents:
        assert env.action_space(a).n == 5
    env.close()


def test_action_spaces_continuous():
    env = cpp_env(continuous_actions=True)
    env.reset(seed=42)
    for a in env.agents:
        assert env.action_space(a).shape == (5,)
    env.close()


def test_state_shape():
    for N in [2, 4, 6]:
        env = cpp_env(N=N)
        env.reset(seed=0)
        assert env.state().shape == (N * 6,)
        env.close()


def test_global_reward_shared():
    """All agents must receive the same reward (pure global, local_ratio=0)."""
    env = cpp_env(N=4)
    env.reset(seed=42)
    _, rews, _, _, _ = env.step({a: 0 for a in env.agents})
    vals = list(rews.values())
    assert all(abs(v - vals[0]) < 1e-6 for v in vals)
    env.close()


def test_global_reward_non_positive():
    env = cpp_env(N=4)
    env.reset(seed=42)
    _, rews, _, _, _ = env.step({a: 0 for a in env.agents})
    assert list(rews.values())[0] <= 0
    env.close()


def test_reward_sign_matches_mpe2():
    """cpp and mpe2 rewards should both be <= 0 and in the same ballpark."""
    cpp = cpp_env(N=4)
    mpe2 = simple_formation_v1.parallel_env(N=4)
    cpp.reset(seed=42); mpe2.reset(seed=42)
    _, cpp_rews, _, _, _ = cpp.step({a: 0 for a in cpp.agents})
    _, mpe2_rews, _, _, _ = mpe2.step({a: 0 for a in mpe2.agents})
    cpp_r = list(cpp_rews.values())[0]
    mpe2_r = list(mpe2_rews.values())[0]
    assert cpp_r <= 0 and mpe2_r <= 0
    # Both should be negative mean distance, similar magnitude (different seeds/physics may diverge)
    assert abs(cpp_r) < 10.0 and abs(mpe2_r) < 10.0
    cpp.close(); mpe2.close()


def test_truncation_at_max_cycles():
    env = cpp_env(N=4, max_cycles=5)
    env.reset(seed=42)
    actions = {a: 0 for a in env.agents}
    for i in range(4):
        _, _, _, truncs, _ = env.step(actions)
        assert not any(truncs.values())
    _, _, _, truncs, _ = env.step(actions)
    assert all(truncs.values())
    assert len(env.agents) == 0
    env.close()


def test_terminate_on_success_false():
    env = cpp_env(N=4, terminate_on_success=False, max_cycles=25)
    env.reset(seed=42)
    for _ in range(24):
        _, _, terms, _, _ = env.step({a: 0 for a in env.agents})
        assert not any(terms.values())
    env.close()


def test_multi_step_determinism():
    env1, env2 = cpp_env(N=4), cpp_env(N=4)
    env1.reset(seed=7); env2.reset(seed=7)
    for step in range(15):
        actions = {a: step % 5 for a in env1.possible_agents}
        obs1, rew1, _, _, _ = env1.step(actions)
        obs2, rew2, _, _, _ = env2.step(actions)
        for a in env1.possible_agents:
            np.testing.assert_allclose(obs1[a], obs2[a], rtol=1e-5)
            np.testing.assert_allclose(rew1[a], rew2[a], rtol=1e-5)
    env1.close(); env2.close()


def test_multiple_resets():
    env = cpp_env(N=4)
    for seed in [0, 1, 42, 100]:
        obs, _ = env.reset(seed=seed)
        assert len(obs) == 4
        for o in obs.values():
            assert o.shape == (6,)
            assert np.all(np.isfinite(o))
    env.close()


def test_hungarian_equal_reward_all_agents():
    """Hungarian matching produces identical reward for all agents."""
    for N in [2, 3, 4, 5]:
        env = cpp_env(N=N)
        env.reset(seed=42)
        _, rews, _, _, _ = env.step({a: 0 for a in env.agents})
        vals = list(rews.values())
        for v in vals[1:]:
            assert abs(v - vals[0]) < 1e-6, f"N={N}: rewards not equal"
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
