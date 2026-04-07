"""Test Simple AEC (Agent Environment Cycle) API compatibility with MPE2."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
from cpp_pettingzoo.simple import env as simple_env
from mpe2 import simple_v3


def test_aec_basic_api():
    """Test that Simple AEC wrapper has correct API structure."""
    env = simple_env()
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


def test_aec_matches_mpe2_structure():
    """Test that Simple AEC structure matches MPE2."""
    cpp_env = simple_env()
    mpe2_env = simple_v3.env()

    cpp_env.reset(seed=42)
    mpe2_env.reset(seed=42)

    # Check agent list matches
    assert len(cpp_env.agents) == len(mpe2_env.agents)
    assert cpp_env.num_agents == mpe2_env.num_agents

    cpp_env.close()
    mpe2_env.close()


def test_aec_reset():
    """Test AEC reset for Simple."""
    env = simple_env()
    env.reset(seed=42)

    assert len(env.agents) == 1
    assert env.agent_selection is not None
    assert env.agent_selection in env.agents

    env.close()


def test_aec_step_cycle():
    """Test AEC step cycle for Simple."""
    env = simple_env(max_cycles=5)
    env.reset(seed=42)

    for _ in range(5):  # max_cycles steps
        agent = env.agent_selection
        obs, reward, term, trunc, info = env.last()

        # Take action
        action = env.action_space(agent).sample()
        env.step(action)

    # Check that the episode is truncated
    agent = env.agent_selection
    _, _, _, trunc, _ = env.last()
    assert trunc, "Episode should be truncated after max_cycles"

    env.close()


def test_aec_deterministic():
    """Test that AEC with same seed is deterministic for Simple."""
    env1 = simple_env()
    env2 = simple_env()

    env1.reset(seed=123)
    env2.reset(seed=123)

    # First observation should be identical
    obs1, _, _, _, _ = env1.last()
    obs2, _, _, _, _ = env2.last()

    assert np.allclose(obs1, obs2)

    env1.close()
    env2.close()


def test_aec_agent_iter():
    """Test agent iteration in AEC mode for Simple."""
    env = simple_env(max_cycles=3)
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

    # Should have seen agent_0 exactly 4 times (3 cycles + 1 done state) - matches MPE2
    assert agents_seen.count("agent_0") == 4

    env.close()


def test_aec_rewards_structure():
    """Test that rewards dict is properly maintained in AEC."""
    env = simple_env()
    env.reset(seed=42)

    # After reset, rewards should exist
    assert hasattr(env, 'rewards')
    assert isinstance(env.rewards, dict)

    # Step once
    agent = env.agent_selection
    env.step(0)

    # Rewards should have entry for the agent that acted
    assert agent in env.rewards

    env.close()


def test_aec_observation_action_spaces():
    """Test observation and action spaces in AEC for Simple."""
    env = simple_env()
    env.reset()

    agent = env.agent_selection

    # Check observation space
    obs_space = env.observation_space(agent)
    assert obs_space.shape == (4,)

    # Check action space
    action_space = env.action_space(agent)
    assert action_space.n == 5

    env.close()


def test_aec_continuous_actions():
    """Test AEC with continuous actions for Simple."""
    env = simple_env(continuous_actions=True)
    env.reset(seed=42)

    agent = env.agent_selection
    action_space = env.action_space(agent)

    # Should be Box space
    assert hasattr(action_space, 'shape')
    assert action_space.shape == (5,)

    # Take a continuous action
    action = np.array([0.5, 0.2, 0.3, 0.1, 0.4])
    env.step(action)

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
