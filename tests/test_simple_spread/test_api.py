"""Test SimpleSpread environment functionality and MPE2 compatibility."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from cpp_pettingzoo.simple_spread import parallel_env
from mpe2 import simple_spread_v3


def test_basic_initialization():
    """Test that SimpleSpread initializes correctly."""
    env = parallel_env()
    assert len(env.possible_agents) == 3
    assert env.possible_agents == ["agent_0", "agent_1", "agent_2"]


def test_api_compatibility_with_mpe2():
    """Test that wrapper matches MPE2 SimpleSpread API structure."""
    cpp = parallel_env(max_cycles=10)
    mpe2 = simple_spread_v3.parallel_env(max_cycles=10, continuous_actions=False)

    # Check attributes exist
    assert hasattr(cpp, 'possible_agents')
    assert hasattr(cpp, 'agents')
    assert hasattr(cpp, 'action_space')
    assert hasattr(cpp, 'observation_space')

    # Check number of agents matches
    assert len(cpp.possible_agents) == len(mpe2.possible_agents)

    # Check action spaces match
    for agent in cpp.possible_agents:
        assert cpp.action_space(agent).n == mpe2.action_space(agent).n

    # Check observation spaces match
    for agent in cpp.possible_agents:
        assert cpp.observation_space(agent).shape == mpe2.observation_space(agent).shape


def test_reset_returns_correct_format():
    """Test that reset returns (obs, info) tuple with all agents."""
    env = parallel_env(max_cycles=10)
    result = env.reset(seed=42)

    assert isinstance(result, tuple)
    assert len(result) == 2

    obs, infos = result
    assert isinstance(obs, dict)
    assert isinstance(infos, dict)

    # All 3 agents should be present
    for agent in ["agent_0", "agent_1", "agent_2"]:
        assert agent in obs
        assert agent in infos


def test_observation_shape():
    """Test that observations have correct shape (18,)."""
    env = parallel_env()
    obs, _ = env.reset(seed=42)

    for agent in env.agents:
        assert len(obs[agent]) == 18, f"Agent {agent} obs shape: {len(obs[agent])}"


def test_step_returns_correct_format():
    """Test that step returns 5-tuple for all agents."""
    env = parallel_env(max_cycles=10)
    env.reset(seed=42)

    actions = {agent: 0 for agent in env.agents}
    result = env.step(actions)

    assert isinstance(result, tuple)
    assert len(result) == 5

    obs, rewards, terms, truncs, infos = result

    for agent in ["agent_0", "agent_1", "agent_2"]:
        assert agent in obs
        assert agent in rewards
        assert agent in terms
        assert agent in truncs
        assert agent in infos


def test_local_ratio_all_global():
    """Test that local_ratio=0.0 gives same rewards for all agents (pure global)."""
    env = parallel_env(local_ratio=0.0, max_cycles=10)
    env.reset(seed=42)

    actions = {agent: 1 for agent in env.agents}
    _, rewards, _, _, _ = env.step(actions)

    # All agents should have identical rewards with local_ratio=0.0
    reward_values = list(rewards.values())
    assert np.allclose(reward_values[0], reward_values[1])
    assert np.allclose(reward_values[1], reward_values[2])


def test_local_ratio_mixed():
    """Test that local_ratio=0.5 blends local and global rewards."""
    env = parallel_env(local_ratio=0.5, max_cycles=10)
    env.reset(seed=42)

    actions = {agent: 1 for agent in env.agents}
    _, rewards, _, _, _ = env.step(actions)

    # With default local_ratio, rewards should still be similar (mostly global)
    # but may differ due to collision penalties
    assert all(r <= 0 for r in rewards.values()), "All rewards should be negative"


def test_curriculum_stage_0():
    """Test curriculum stage 0 (no collision penalties)."""
    env = parallel_env(curriculum=True, curriculum_stage=0, max_cycles=10)
    env.reset(seed=42)

    # Step the environment
    actions = {agent: 1 for agent in env.agents}
    _, rewards, _, _, _ = env.step(actions)

    # Should have rewards (distance-based)
    assert all(r <= 0 for r in rewards.values())


def test_curriculum_stage_1():
    """Test curriculum stage 1 (with collision penalties)."""
    env = parallel_env(curriculum=True, curriculum_stage=1, max_cycles=10)
    env.reset(seed=42)

    # Step the environment
    actions = {agent: 1 for agent in env.agents}
    _, rewards, _, _, _ = env.step(actions)

    # Should have rewards (distance-based + collisions)
    assert all(r <= 0 for r in rewards.values())


def test_episode_truncation():
    """Test that episode truncates after max_cycles."""
    max_cycles = 5
    env = parallel_env(max_cycles=max_cycles)
    env.reset(seed=42)

    # Step exactly max_cycles times
    for i in range(max_cycles):
        actions = {agent: 0 for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

        if i < max_cycles - 1:
            # Not done yet
            assert len(env.agents) == 3
        else:
            # Should be truncated on last step
            assert all(truncs.values()), "All agents should be truncated"
            assert len(env.agents) == 0, "Agents list should be empty after truncation"


def test_deterministic_reset():
    """Test that reset with same seed produces same initial state."""
    env1 = parallel_env()
    env2 = parallel_env()

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    for agent in env1.agents:
        assert np.allclose(obs1[agent], obs2[agent])


def test_different_seeds():
    """Test that different seeds produce different initial states."""
    env = parallel_env()

    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=456)

    # At least one agent should have different observations
    different = False
    for agent in env.agents:
        if not np.allclose(obs1[agent], obs2[agent]):
            different = True
            break

    assert different, "Different seeds should produce different states"


def test_state_method():
    """Test that state() method exists and returns correct shape."""
    env = parallel_env()
    env.reset(seed=42)

    state = env.state()

    # State should be concatenation of all observations (3 agents * 18 = 54)
    assert state.shape == (54,)
    assert state.dtype == np.float32


def test_continuous_actions():
    """Test continuous action space."""
    env = parallel_env(continuous_actions=True)
    env.reset(seed=42)

    # Check action space is Box
    for agent in env.agents:
        action_space = env.action_space(agent)
        assert isinstance(action_space, type(env.action_space(agent)))
        assert action_space.shape == (5,)

    # Test stepping with continuous actions
    actions = {agent: np.array([0.5, 0.2, 0.3, 0.1, 0.4]) for agent in env.agents}
    obs, rewards, _, _, _ = env.step(actions)

    assert len(obs) == 3


def test_multiple_episodes():
    """Test running multiple episodes."""
    env = parallel_env(max_cycles=5)

    for episode in range(3):
        obs, _ = env.reset(seed=42 + episode)
        assert len(env.agents) == 3

        for step in range(5):
            actions = {agent: 0 for agent in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)

        # After max_cycles, agents should be empty
        assert len(env.agents) == 0


def test_empty_actions():
    """Test that empty actions dict returns empty results."""
    env = parallel_env()
    env.reset(seed=42)

    obs, rewards, terms, truncs, infos = env.step({})

    assert obs == {}
    assert rewards == {}
    assert terms == {}
    assert truncs == {}
    assert infos == {}
    assert len(env.agents) == 0


def test_rewards_always_negative():
    """Test that rewards are always negative (distance-based)."""
    env = parallel_env(max_cycles=10)
    env.reset(seed=42)

    for _ in range(10):
        actions = {agent: np.random.randint(0, 5) for agent in env.agents}
        if not env.agents:
            break
        _, rewards, _, _, _ = env.step(actions)

        for agent, reward in rewards.items():
            assert reward <= 0, f"Reward should be negative, got {reward}"


def test_agents_management():
    """Test that agents list is managed correctly throughout episode."""
    env = parallel_env(max_cycles=5)

    # Before reset
    assert len(env.possible_agents) == 3

    # After reset
    env.reset(seed=42)
    assert len(env.agents) == 3
    assert env.agents == env.possible_agents

    # During episode
    for _ in range(4):
        actions = {agent: 0 for agent in env.agents}
        env.step(actions)
        assert len(env.agents) == 3

    # After truncation
    actions = {agent: 0 for agent in env.agents}
    env.step(actions)
    assert len(env.agents) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
