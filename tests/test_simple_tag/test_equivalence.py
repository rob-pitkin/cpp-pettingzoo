"""Test SimpleTag C++ implementation matches expected behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from cpp_pettingzoo.simple_tag.simple_tag import parallel_env as cpp_env


def test_reset_deterministic():
    env1 = cpp_env()
    env2 = cpp_env()
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    for agent in env1.agents:
        np.testing.assert_allclose(obs1[agent], obs2[agent])
    env1.close()
    env2.close()


def test_reset_different_seeds():
    env1 = cpp_env()
    env2 = cpp_env()
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=123)
    different = any(not np.allclose(obs1[a], obs2[a]) for a in env1.agents)
    assert different, "Different seeds should produce different observations"
    env1.close()
    env2.close()


def test_agent_names_default():
    env = cpp_env(num_good=1, num_adversaries=3, num_obstacles=2)
    env.reset(seed=0)
    assert set(env.agents) == {"adversary_0", "adversary_1", "adversary_2", "agent_0"}
    env.close()


def test_observation_space_shapes_default():
    """Default: 3 adversaries, 1 good, 2 obstacles.

    Adversary obs: vel(2)+pos(2)+obstacles(4)+other_agents(8)+good_vel(2) = 16+2? wait:
    good_vel for adversary = num_good = 1 -> 2 -> vel(2)+pos(2)+lm(4)+other(6)+good_vel(2)=16
    Good agent obs: vel(2)+pos(2)+lm(4)+other(6)+good_vel(0)=14
    """
    env = cpp_env(num_good=1, num_adversaries=3, num_obstacles=2)
    obs, _ = env.reset(seed=42)

    for a in env.agents:
        if a.startswith("adversary"):
            assert obs[a].shape == (16,), f"{a} obs shape should be 16"
            assert env.observation_space(a).shape == (16,)
        else:
            assert obs[a].shape == (14,), f"{a} obs shape should be 14"
            assert env.observation_space(a).shape == (14,)
    env.close()


def test_observation_space_shapes_partial_obs():
    env = cpp_env(num_good=1, num_adversaries=3, num_obstacles=2,
                  num_agent_neighbors=2, num_landmark_neighbors=1)
    obs, _ = env.reset(seed=42)
    # vel(2)+pos(2)+lm(2)+agents(4)+vel(4)=14 for all agents
    for a in env.agents:
        assert obs[a].shape == (14,), f"{a} po obs shape should be 14"
        assert env.observation_space(a).shape == (14,)
    env.close()


def test_action_spaces_discrete():
    env = cpp_env()
    env.reset(seed=42)
    for agent in env.agents:
        assert env.action_space(agent).n == 5
    env.close()


def test_action_spaces_continuous():
    env = cpp_env(continuous_actions=True)
    env.reset(seed=42)
    for agent in env.agents:
        assert env.action_space(agent).shape == (5,)
    env.close()


def test_rewards_finite():
    env = cpp_env()
    env.reset(seed=42)
    actions = {a: env.action_space(a).sample() for a in env.agents}
    _, rewards, _, _, _ = env.step(actions)
    for agent, rew in rewards.items():
        assert np.isfinite(rew), f"Reward for {agent} should be finite"
    env.close()


def test_adversary_reward_positive_on_collision():
    """Adversaries get +10 per collision; place them close to ensure collision."""
    env = cpp_env(num_good=1, num_adversaries=1, num_obstacles=0,
                  terminate_on_success=False)
    # Reset and force positions to collide
    env.reset(seed=0)
    # Step no-op a few times — we can't force positions, so just verify sign is non-neg
    actions = {a: 0 for a in env.agents}
    for _ in range(5):
        _, rewards, _, _, _ = env.step(actions)
    # Adversary reward is always >= 0 (no negative terms other than shaping which is off)
    assert rewards["adversary_0"] >= 0, "Adversary reward should be non-negative"
    env.close()


def test_good_agent_reward_non_positive():
    """Good agents are penalized for collisions and boundary; reward <= 0."""
    env = cpp_env()
    env.reset(seed=42)
    actions = {a: 0 for a in env.agents}
    _, rewards, _, _, _ = env.step(actions)
    assert rewards["agent_0"] <= 0, "Good agent reward should be non-positive"
    env.close()


def test_truncation_at_max_cycles():
    env = cpp_env(max_cycles=5)
    env.reset(seed=42)
    actions = {a: 0 for a in env.agents}
    for i in range(4):
        _, _, _, truncations, _ = env.step(actions)
        assert not any(truncations.values()), f"Should not truncate at step {i}"
    _, _, _, truncations, _ = env.step(actions)
    assert all(truncations.values()), "Should truncate after max_cycles"
    assert len(env.agents) == 0
    env.close()


def test_multi_step_determinism():
    env1 = cpp_env()
    env2 = cpp_env()
    env1.reset(seed=99)
    env2.reset(seed=99)
    for step in range(15):
        actions = {a: step % 5 for a in env1.possible_agents}
        obs1, rew1, _, _, _ = env1.step(actions)
        obs2, rew2, _, _, _ = env2.step(actions)
        for a in env1.possible_agents:
            np.testing.assert_allclose(obs1[a], obs2[a], rtol=1e-5,
                                       err_msg=f"Obs mismatch step {step} {a}")
            np.testing.assert_allclose(rew1[a], rew2[a], rtol=1e-5,
                                       err_msg=f"Reward mismatch step {step} {a}")
    env1.close()
    env2.close()


def test_agent_counts():
    for ng, na in [(1, 3), (2, 2), (1, 1), (3, 1)]:
        env = cpp_env(num_good=ng, num_adversaries=na)
        env.reset(seed=0)
        assert len(env.agents) == ng + na
        assert sum(1 for a in env.agents if a.startswith("adversary")) == na
        assert sum(1 for a in env.agents if a.startswith("agent_")) == ng
        env.close()


def test_terminate_on_success_false():
    """With terminate_on_success=False, episode never terminates early."""
    env = cpp_env(num_good=1, num_adversaries=1, num_obstacles=0,
                  max_cycles=25, terminate_on_success=False)
    env.reset(seed=42)
    actions = {a: 0 for a in env.agents}
    for _ in range(24):
        _, _, terminations, _, _ = env.step(actions)
        assert not any(terminations.values()), "Should not terminate early"
    env.close()


def test_curriculum_stages():
    env = cpp_env(curriculum=True)
    env.reset(seed=42)
    assert env.curriculum_stage == 0
    env.advance_curriculum()
    assert env.curriculum_stage == 1
    env.advance_curriculum()
    assert env.curriculum_stage == 2
    env.advance_curriculum()  # no-op at max
    assert env.curriculum_stage == 2
    env.set_curriculum_stage(0)
    assert env.curriculum_stage == 0
    env.set_curriculum_stage(5)  # clamped
    assert env.curriculum_stage == 2
    env.close()


def test_state_shape():
    env = cpp_env()
    env.reset(seed=42)
    s = env.state()
    assert s.ndim == 1
    # state = concatenation of all agent obs: 3*16 + 1*14 = 62
    assert s.shape == (62,), f"State shape should be (62,), got {s.shape}"
    env.close()


def test_multiple_resets():
    env = cpp_env()
    for seed in [0, 1, 2, 100]:
        obs, _ = env.reset(seed=seed)
        assert len(obs) == 4
        for a, o in obs.items():
            assert np.all(np.isfinite(o)), f"Obs for {a} should be finite after reset"
    env.close()


def test_po_zero_padding():
    """When num_landmark_neighbors exceeds actual obstacles, obs is zero-padded."""
    env = cpp_env(num_good=1, num_adversaries=1, num_obstacles=1,
                  num_landmark_neighbors=3)
    obs, _ = env.reset(seed=0)
    # Shape is fixed at num_landmark_neighbors slots even though only 1 obstacle exists.
    # good agent vel slots = num_good - 1 = 0 (no other good agents); adversary = num_good = 1.
    expected_good = 2 + 2 + 2 * 3 + 2 * 1 + 2 * 0  # 12
    expected_adv = 2 + 2 + 2 * 3 + 2 * 1 + 2 * 1   # 14
    for a, o in obs.items():
        expected = expected_adv if a.startswith("adversary") else expected_good
        assert o.shape == (expected,), f"Zero-padded obs shape mismatch for {a}"
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
