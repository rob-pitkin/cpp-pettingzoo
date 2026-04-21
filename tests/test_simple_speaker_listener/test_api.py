"""Test PettingZoo API compliance for SimpleSpeakerListener environment."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from pettingzoo.test import parallel_api_test
from cpp_pettingzoo.simple_speaker_listener import parallel_env


def test_api():
    """Test that environment complies with PettingZoo parallel API."""
    env = parallel_env()
    parallel_api_test(env, num_cycles=100)


def test_discrete_action_space():
    """Test that discrete action spaces are asymmetric."""
    env = parallel_env(continuous_actions=False)
    env.reset(seed=42)

    speaker_space = env.action_space("speaker_0")
    listener_space = env.action_space("listener_0")

    assert speaker_space.n == 3, "Speaker discrete action space should be 3 (communication)"
    assert listener_space.n == 5, "Listener discrete action space should be 5 (movement)"

    env.close()


def test_continuous_action_space():
    """Test that continuous action spaces are asymmetric."""
    env = parallel_env(continuous_actions=True)
    env.reset(seed=42)

    speaker_space = env.action_space("speaker_0")
    listener_space = env.action_space("listener_0")

    assert speaker_space.shape == (3,), "Speaker continuous action space should be (3,) for communication"
    assert listener_space.shape == (5,), "Listener continuous action space should be (5,) for movement"

    env.close()


def test_observation_space():
    """Test that observation spaces are asymmetric."""
    env = parallel_env()
    env.reset(seed=42)

    speaker_obs_space = env.observation_space("speaker_0")
    listener_obs_space = env.observation_space("listener_0")

    assert speaker_obs_space.shape == (3,), \
        "Speaker observation should be (3,) = goal_color"
    assert listener_obs_space.shape == (11,), \
        "Listener observation should be (11,) = vel(2) + landmarks(6) + comm(3)"

    env.close()


def test_num_agents():
    """Test that environment has exactly 2 agents."""
    env = parallel_env()
    env.reset(seed=42)

    assert len(env.agents) == 2, "SimpleSpeakerListener should have 2 agents"
    assert "speaker_0" in env.agents, "Should have speaker_0"
    assert "listener_0" in env.agents, "Should have listener_0"

    env.close()


def test_local_ratio_parameter():
    """Test that local_ratio parameter is accepted."""
    env1 = parallel_env(local_ratio=0.0)
    env2 = parallel_env(local_ratio=1.0)

    env1.reset(seed=42)
    env2.reset(seed=42)

    env1.close()
    env2.close()


def test_max_cycles():
    """Test that max_cycles parameter controls episode length."""
    env = parallel_env(max_cycles=10)
    env.reset(seed=42)

    for _ in range(10):
        actions = {"speaker_0": 0, "listener_0": 0}
        _, _, _, truncations, _ = env.step(actions)

    assert all(truncations.values()), "Episode should truncate after max_cycles"

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
