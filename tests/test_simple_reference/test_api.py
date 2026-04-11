"""Test PettingZoo API compliance for SimpleReference environment."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from pettingzoo.test import parallel_api_test
from cpp_pettingzoo.simple_reference import parallel_env


def test_api():
    """Test that environment complies with PettingZoo parallel API."""
    env = parallel_env()
    parallel_api_test(env, num_cycles=100)


def test_discrete_action_space():
    """Test that discrete action space is Discrete(50)."""
    env = parallel_env(continuous_actions=False)
    env.reset(seed=42)

    for agent in env.agents:
        action_space = env.action_space(agent)
        assert action_space.n == 50, "Discrete action space should be 50 (10 comm × 5 movement)"

    env.close()


def test_continuous_action_space():
    """Test that continuous action space is Box(15,)."""
    env = parallel_env(continuous_actions=True)
    env.reset(seed=42)

    for agent in env.agents:
        action_space = env.action_space(agent)
        assert action_space.shape == (15,), "Continuous action space should be (15,) = 5 movement + 10 comm"

    env.close()


def test_observation_space():
    """Test that observation space is Box(21,)."""
    env = parallel_env()
    env.reset(seed=42)

    for agent in env.agents:
        obs_space = env.observation_space(agent)
        assert obs_space.shape == (21,), \
            "Observation should be (21,) = vel(2) + landmarks(6) + goal_color(3) + other_comm(10)"

    env.close()


def test_num_agents():
    """Test that environment has exactly 2 agents."""
    env = parallel_env()
    env.reset(seed=42)

    assert len(env.agents) == 2, "SimpleReference should have 2 agents"
    assert "agent_0" in env.agents, "Should have agent_0"
    assert "agent_1" in env.agents, "Should have agent_1"

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
        actions = {agent: 0 for agent in env.agents}
        _, _, _, truncations, _ = env.step(actions)

    assert all(truncations.values()), "Episode should truncate after max_cycles"

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
