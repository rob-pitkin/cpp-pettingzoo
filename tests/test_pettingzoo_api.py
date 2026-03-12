"""Test PettingZoo API compatibility and wrapper functionality."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from cpp_pettingzoo.simple import parallel_env
from mpe2 import simple_v3


def test_wrapper_api_compatibility():
    """Test that wrapper matches MPE2 API structure."""
    cpp = parallel_env(max_cycles=10)
    mpe2 = simple_v3.parallel_env(max_cycles=10, continuous_actions=False)

    # Check attributes exist
    assert hasattr(cpp, 'possible_agents')
    assert hasattr(cpp, 'agents')
    assert hasattr(cpp, 'action_space')
    assert hasattr(cpp, 'observation_space')

    # Check values match
    assert cpp.possible_agents == mpe2.possible_agents
    assert cpp.action_space('agent_0').n == mpe2.action_space('agent_0').n
    assert cpp.observation_space('agent_0').shape == mpe2.observation_space('agent_0').shape


def test_reset_returns_correct_format():
    """Test that reset returns (obs, info) tuple."""
    env = parallel_env(max_cycles=10)
    result = env.reset(seed=42)

    assert isinstance(result, tuple)
    assert len(result) == 2

    obs, infos = result
    assert isinstance(obs, dict)
    assert isinstance(infos, dict)
    assert "agent_0" in obs
    assert "agent_0" in infos


def test_step_returns_correct_format():
    """Test that step returns 5-tuple."""
    env = parallel_env(max_cycles=10)
    env.reset(seed=42)
    result = env.step({"agent_0": 0})

    assert isinstance(result, tuple)
    assert len(result) == 5

    obs, rewards, terms, truncs, infos = result
    assert isinstance(obs, dict)
    assert isinstance(rewards, dict)
    assert isinstance(terms, dict)
    assert isinstance(truncs, dict)
    assert isinstance(infos, dict)


def test_agents_management_reset():
    """Test that agents list is properly managed on reset."""
    env = parallel_env(max_cycles=5)

    # Initial state
    assert env.agents == ["agent_0"]

    # After reset
    env.reset(seed=42)
    assert env.agents == ["agent_0"]

    # Multiple resets
    for seed in [1, 2, 3]:
        env.reset(seed=seed)
        assert env.agents == ["agent_0"], f"Agents not reset correctly with seed={seed}"


def test_agents_management_episode_end():
    """Test that agents list is cleared when episode ends."""
    env = parallel_env(max_cycles=5)
    env.reset(seed=42)

    # Step until truncation
    for i in range(5):
        obs, rewards, terms, truncs, infos = env.step({"agent_0": 0})
        if i < 4:
            assert env.agents == ["agent_0"], f"Agents cleared too early at step {i}"
            assert not truncs["agent_0"]
        else:
            assert env.agents == [], "Agents not cleared at truncation"
            assert truncs["agent_0"]


def test_agents_management_matches_mpe2():
    """Test that agent management behavior matches MPE2."""
    cpp = parallel_env(max_cycles=5)
    mpe2 = simple_v3.parallel_env(max_cycles=5, continuous_actions=False)

    cpp.reset(seed=42)
    mpe2.reset(seed=42)

    assert cpp.agents == mpe2.agents, "Agents differ after reset"

    # Step through episode
    for i in range(6):
        cpp.step({"agent_0": 0})
        mpe2.step({"agent_0": 0})
        assert cpp.agents == mpe2.agents, f"Agents differ at step {i}"


def test_empty_actions():
    """Test that empty actions dict is handled correctly."""
    env = parallel_env(max_cycles=5)
    env.reset(seed=42)

    obs, rewards, terms, truncs, infos = env.step({})

    assert obs == {}
    assert rewards == {}
    assert terms == {}
    assert truncs == {}
    assert infos == {}
    assert env.agents == []


def test_constructor_parameters():
    """Test that constructor parameters work correctly."""
    # Test max_cycles
    env = parallel_env(max_cycles=10)
    assert env.max_cycles == 10

    env = parallel_env(max_cycles=50)
    assert env.max_cycles == 50

    # Test unsupported parameters raise errors
    with pytest.raises(NotImplementedError):
        parallel_env(continuous_actions=True)

    with pytest.raises(NotImplementedError):
        parallel_env(render_mode="human")

    with pytest.raises(NotImplementedError):
        parallel_env(dynamic_rescaling=True)


def test_action_space_values():
    """Test that action space has correct range."""
    env = parallel_env(max_cycles=10)
    action_space = env.action_space("agent_0")

    assert action_space.n == 5  # 5 discrete actions
    assert action_space.contains(0)
    assert action_space.contains(4)
    assert not action_space.contains(5)
    assert not action_space.contains(-1)


def test_observation_space_values():
    """Test that observation space has correct properties."""
    env = parallel_env(max_cycles=10)
    obs_space = env.observation_space("agent_0")

    assert obs_space.shape == (4,)
    assert obs_space.dtype == "float32"

    # Test that actual observations are in the space
    obs, _ = env.reset(seed=42)
    assert obs_space.contains(obs["agent_0"])


def test_multiple_episodes():
    """Test that environment works correctly across multiple episodes."""
    env = parallel_env(max_cycles=5)

    for episode in range(3):
        obs, _ = env.reset(seed=episode)
        assert env.agents == ["agent_0"]
        assert "agent_0" in obs

        for step in range(5):
            obs, rewards, terms, truncs, infos = env.step({"agent_0": 1})

        # Episode should be done
        assert env.agents == []
        assert truncs["agent_0"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
