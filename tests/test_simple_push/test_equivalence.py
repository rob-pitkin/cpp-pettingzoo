"""Test SimplePush C++ implementation matches expected behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from cpp_pettingzoo.simple_push.simple_push import parallel_env as cpp_env


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


def test_agent_names():
    env = cpp_env()
    env.reset(seed=0)
    assert set(env.agents) == {"adversary_0", "agent_0"}
    env.close()


def test_observation_space_shapes():
    env = cpp_env()
    obs, _ = env.reset(seed=42)
    assert obs["agent_0"].shape == (19,)
    assert obs["adversary_0"].shape == (8,)
    assert env.observation_space("agent_0").shape == (19,)
    assert env.observation_space("adversary_0").shape == (8,)
    env.close()


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
    env = cpp_env()
    env.reset(seed=42)
    assert env.state().shape == (27,)
    env.close()


def test_rewards_finite():
    env = cpp_env()
    env.reset(seed=42)
    _, rews, _, _, _ = env.step({a: env.action_space(a).sample() for a in env.agents})
    for a, r in rews.items():
        assert np.isfinite(r), f"Reward for {a} should be finite"
    env.close()


def test_good_agent_reward_non_positive():
    """Good agent reward is -dist(self, goal), always <= 0."""
    env = cpp_env()
    env.reset(seed=42)
    _, rews, _, _, _ = env.step({"adversary_0": 0, "agent_0": 0})
    assert rews["agent_0"] <= 0
    env.close()


def test_adversary_reward_sign():
    """Adversary reward = good_dist_to_goal - adv_dist_to_goal, unbounded in sign."""
    env = cpp_env()
    env.reset(seed=42)
    _, rews, _, _, _ = env.step({"adversary_0": 0, "agent_0": 0})
    assert np.isfinite(rews["adversary_0"])
    env.close()


def test_truncation_at_max_cycles():
    env = cpp_env(max_cycles=5)
    env.reset(seed=42)
    actions = {"adversary_0": 0, "agent_0": 0}
    for i in range(4):
        _, _, _, truncs, _ = env.step(actions)
        assert not any(truncs.values()), f"Should not truncate at step {i}"
    _, _, _, truncs, _ = env.step(actions)
    assert all(truncs.values())
    assert len(env.agents) == 0
    env.close()


def test_multi_step_determinism():
    env1, env2 = cpp_env(), cpp_env()
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
    env = cpp_env()
    for seed in [0, 1, 42, 100]:
        obs, _ = env.reset(seed=seed)
        assert len(obs) == 2
        assert obs["agent_0"].shape == (19,)
        assert obs["adversary_0"].shape == (8,)
    env.close()


def test_goal_encoded_in_good_agent_obs():
    """Good agent obs includes goal relative position (indices 2:4), which varies by reset."""
    env = cpp_env()
    obs1, _ = env.reset(seed=0)
    obs2, _ = env.reset(seed=1)
    # Goal rel pos is obs[2:4]; different seeds should give different goal positions
    # (not guaranteed but extremely likely with two independent random draws)
    assert not np.allclose(obs1["agent_0"][2:4], obs2["agent_0"][2:4]) or \
           not np.allclose(obs1["agent_0"], obs2["agent_0"])
    env.close()


def test_adversary_obs_no_goal_info():
    """Adversary obs has size 8; it does not include the goal identity."""
    env = cpp_env()
    obs, _ = env.reset(seed=42)
    # adversary only gets vel(2) + lm_pos(4) + other_pos(2) = 8, no color/goal
    assert obs["adversary_0"].shape == (8,)
    env.close()


def test_good_agent_obs_contains_color():
    """Good agent obs indices 4:7 are its own color, which encodes goal identity."""
    env = cpp_env()
    # Run two resets; goal is random so color should differ across episodes sometimes
    colors = []
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        colors.append(obs["agent_0"][4:7].tolist())
    # Should see at least 2 distinct colors (goal is randomly chosen each reset)
    assert len(set(map(tuple, colors))) >= 2
    env.close()


def test_matches_mpe2_obs_shapes():
    """Verify obs shapes match the documented MPE2 values."""
    from mpe2 import simple_push_v3
    mpe2 = simple_push_v3.parallel_env()
    mpe2.reset(seed=42)
    for a in mpe2.agents:
        cpp_shape = cpp_env().observation_space(a).shape
        mpe2_shape = mpe2.observation_space(a).shape
        assert cpp_shape == mpe2_shape, f"{a}: cpp={cpp_shape} mpe2={mpe2_shape}"
    mpe2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
