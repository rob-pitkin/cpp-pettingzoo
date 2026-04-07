"""Test SimpleSpread AEC (Agent Environment Cycle) API compatibility with MPE2."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
from cpp_pettingzoo.simple_spread import env as simple_spread_env
from mpe2 import simple_spread_v3


def test_simple_spread_aec_basic_api():
    """Test that SimpleSpread AEC wrapper has correct API structure."""
    env = simple_spread_env()
    env.reset()

    # Check AEC-specific attributes
    assert hasattr(env, 'agents')
    assert hasattr(env, 'num_agents')
    assert hasattr(env, 'agent_selection')
    assert hasattr(env, 'rewards')
    assert hasattr(env, 'terminations')
    assert hasattr(env, 'truncations')
    assert hasattr(env, 'infos')

    env.close()


def test_simple_spread_aec_matches_mpe2_structure():
    """Test that SimpleSpread AEC structure matches MPE2."""
    cpp_env = simple_spread_env()
    mpe2_env = simple_spread_v3.env()

    cpp_env.reset(seed=42)
    mpe2_env.reset(seed=42)

    # Check agent list matches
    assert len(cpp_env.agents) == len(mpe2_env.agents)
    assert cpp_env.num_agents == mpe2_env.num_agents

    cpp_env.close()
    mpe2_env.close()


def test_simple_spread_aec_reset():
    """Test AEC reset for SimpleSpread."""
    env = simple_spread_env()
    env.reset(seed=42)

    assert len(env.agents) == 3
    assert env.agent_selection is not None
    assert env.agent_selection in env.agents

    env.close()


def test_simple_spread_aec_step_cycle():
    """Test AEC step cycle for SimpleSpread."""
    env = simple_spread_env(max_cycles=5)
    env.reset(seed=42)

    steps = 0
    max_steps = 5 * 3  # max_cycles * num_agents

    while env.agents and steps < max_steps:
        agent = env.agent_selection
        obs, reward, term, trunc, info = env.last()

        # Take action
        action = env.action_space(agent).sample()
        env.step(action)
        steps += 1

        # If truncated, pass None for remaining agents
        if trunc:
            break

    # Check that we completed or truncated
    assert steps <= max_steps

    env.close()


def test_simple_spread_aec_deterministic():
    """Test that AEC with same seed is deterministic for SimpleSpread."""
    env1 = simple_spread_env()
    env2 = simple_spread_env()

    env1.reset(seed=123)
    env2.reset(seed=123)

    # First observation should be identical
    obs1, _, _, _, _ = env1.last()
    obs2, _, _, _, _ = env2.last()

    assert np.allclose(obs1, obs2)

    env1.close()
    env2.close()


def test_simple_spread_aec_agent_iter():
    """Test agent iteration in AEC mode for SimpleSpread."""
    env = simple_spread_env(max_cycles=2)
    env.reset(seed=42)

    agents_seen = []
    for agent in env.agent_iter():
        agents_seen.append(agent)
        obs, reward, term, trunc, info = env.last()

        # If agent is done, pass None, otherwise take action
        if term or trunc:
            action = None
        else:
            action = 0
        env.step(action)

    # Should see each agent 3 times (2 cycles + 1 done state) - matches MPE2
    assert agents_seen.count("agent_0") == 3
    assert agents_seen.count("agent_1") == 3
    assert agents_seen.count("agent_2") == 3

    env.close()


def test_simple_spread_aec_rewards_structure():
    """Test that rewards dict is properly maintained in AEC for SimpleSpread."""
    env = simple_spread_env()
    env.reset(seed=42)

    # After reset, rewards should exist
    assert hasattr(env, 'rewards')
    assert isinstance(env.rewards, dict)

    # Step through one complete cycle
    for _ in range(3):
        agent = env.agent_selection
        env.step(0)
        # Rewards should have entry for the agent that acted
        assert agent in env.rewards

    env.close()


def test_simple_spread_aec_observation_action_spaces():
    """Test observation and action spaces in AEC for SimpleSpread."""
    env = simple_spread_env()
    env.reset()

    for agent in ["agent_0", "agent_1", "agent_2"]:
        # Check observation space
        obs_space = env.observation_space(agent)
        assert obs_space.shape == (18,)

        # Check action space
        action_space = env.action_space(agent)
        assert action_space.n == 5

    env.close()


def test_simple_spread_aec_continuous_actions():
    """Test AEC with continuous actions for SimpleSpread."""
    env = simple_spread_env(continuous_actions=True)
    env.reset(seed=42)

    for _ in range(3):  # One complete cycle
        agent = env.agent_selection
        action_space = env.action_space(agent)

        # Should be Box space
        assert hasattr(action_space, 'shape')
        assert action_space.shape == (5,)

        # Take a continuous action
        action = np.array([0.5, 0.2, 0.3, 0.1, 0.4])
        env.step(action)

    env.close()


def test_simple_spread_aec_local_ratio():
    """Test SimpleSpread AEC with different local_ratio values."""
    # Test with pure global reward
    env = simple_spread_env(local_ratio=0.0, max_cycles=2)
    env.reset(seed=42)

    # Run one complete cycle
    for _ in range(3):
        env.step(0)

    # All agents should have similar rewards (pure global)
    rewards_list = list(env.rewards.values())
    assert np.allclose(rewards_list[0], rewards_list[1])
    assert np.allclose(rewards_list[1], rewards_list[2])

    env.close()


def test_simple_spread_aec_curriculum():
    """Test SimpleSpread AEC with curriculum enabled."""
    env = simple_spread_env(curriculum=True, curriculum_stage=0)
    env.reset(seed=42)

    # Run a few steps
    for _ in range(6):  # 2 complete cycles
        if env.agents:
            env.step(0)

    # Environment should work with curriculum stage 0
    assert True  # If we got here, no crashes

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
