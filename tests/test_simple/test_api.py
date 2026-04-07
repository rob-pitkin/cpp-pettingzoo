"""Test PettingZoo API compatibility and wrapper functionality."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import gymnasium
from cpp_pettingzoo.simple import parallel_env
from mpe2 import simple_v3
import numpy as np


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

    # continuous_actions and render_mode are now implemented
    # Test that they don't raise errors
    env = parallel_env(continuous_actions=True)
    env = parallel_env(render_mode="human")


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


def test_state_method_exists():
    """Test that the global state() method exists."""
    env = parallel_env(max_cycles=5)
    env.reset(1)

    state = env.state()
    assert state is not None, "state() should not return None after reset()"
    assert isinstance(state, np.ndarray), f"state() should return a numpy array, got {type(state)}"


def test_state_shape():
    """Test the global state() shape."""
    env = parallel_env(max_cycles=5)
    env.reset(1)

    state = env.state()
    assert state.shape == (4,), f"state() shape should be (4,) for Simple env, got {state.shape}"


def test_state_dtype():
    """Tests the dtype of the global state() method."""
    env = parallel_env(max_cycles=5)
    env.reset(1)

    state = env.state()
    assert state.dtype == np.float32, f"state() dtype should be np.float32, got {state.dtype}"


def test_state_before_reset():
    """Tests calling state() before reset()."""
    env = parallel_env(max_cycles=5)
    with pytest.raises(AssertionError, match="reset.*must be called"):
        env.state()


def test_state_matches_observation():
    """Test that state() matches the regular observation for Simple."""
    env = parallel_env(max_cycles=5)
    env.reset(1)
    obs, _, _, _, _  = env.step({"agent_0": 1})
    state = env.state()
    assert np.array_equal(obs["agent_0"], state), "state() should match the regular observation for Simple"


def test_state_changes_after_step():
    """Test that state() returns a different value after a step()."""
    env = parallel_env(max_cycles=5)
    env.reset(23)
    state_before = env.state().copy()
    env.step({"agent_0": 1})
    state_after = env.state().copy()

    assert not np.array_equal(state_before, state_after), "state() should change after a step()"


def test_state_deterministic():
    """Test that the state() method is deterministic."""
    env = parallel_env(max_cycles=5)
    env.reset(1)
    state_1 = env.state()
    env.reset(1)
    state_2 = env.state()

    assert np.array_equal(state_1, state_2), "state() should be deterministic"


def test_dynamic_rescaling_parameter_accepted():
    """Test that dynamic_rescaling parameter is accepted."""
    # Should not raise NotImplementedError
    env_false = parallel_env(max_cycles=10, dynamic_rescaling=False)
    env_true = parallel_env(max_cycles=10, dynamic_rescaling=True)

    assert env_false.dynamic_rescaling == False, "dynamic_rescaling should be False"
    assert env_true.dynamic_rescaling == True, "dynamic_rescaling should be True"


def test_dynamic_rescaling_doesnt_affect_physics():
    """Test that dynamic_rescaling doesn't affect physics/observations."""
    env1 = parallel_env(max_cycles=10, dynamic_rescaling=False)
    env2 = parallel_env(max_cycles=10, dynamic_rescaling=True)

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    np.testing.assert_array_equal(obs1["agent_0"], obs2["agent_0"],
                                  err_msg="dynamic_rescaling should not affect observations")

    # Test after step as well
    step_obs1, _, _, _, _ = env1.step({"agent_0": 2})
    step_obs2, _, _, _, _ = env2.step({"agent_0": 2})

    np.testing.assert_array_equal(step_obs1["agent_0"], step_obs2["agent_0"],
                                  err_msg="dynamic_rescaling should not affect step observations")


def test_continuous_actions_parameter_accepted():
    """Test that continuous_actions parameter is accepted."""
    # Should not raise NotImplementedError
    env = parallel_env(max_cycles=10, continuous_actions=True)
    assert env.continuous_actions == True, "continuous_actions should be True"


def test_continuous_action_space_shape():
    """Test that continuous action space has correct shape."""
    env = parallel_env(max_cycles=10, continuous_actions=True)
    action_space = env.action_space("agent_0")

    assert isinstance(action_space, gymnasium.spaces.Box), "Action space should be Box for continuous"
    assert action_space.shape == (5,), f"Action space shape should be (5,), got {action_space.shape}"
    assert np.all(action_space.low == 0.0), "Action space low should be 0.0"
    assert np.all(action_space.high == 1.0), "Action space high should be 1.0"


def test_continuous_actions_movement():
    """Test that continuous actions produce correct movement."""
    env = parallel_env(max_cycles=10, continuous_actions=True)
    env.reset(seed=42)

    # Move right: action[2]=1.0 (right), others=0
    action = {"agent_0": np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)}
    obs, _, _, _, _ = env.step(action)

    # Velocity should increase in +x direction
    assert obs["agent_0"][0] > 0.4, f"vel_x should be positive, got {obs['agent_0'][0]}"


def test_continuous_actions_match_discrete():
    """Test that continuous and discrete produce same physics from same force."""
    # Discrete: action 2 = move right
    env_discrete = parallel_env(max_cycles=10, continuous_actions=False)
    env_discrete.reset(seed=42)
    obs_d, _, _, _, _ = env_discrete.step({"agent_0": 2})

    # Continuous: [0, 0, 1, 0, 0] = move right
    env_continuous = parallel_env(max_cycles=10, continuous_actions=True)
    env_continuous.reset(seed=42)
    action_c = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    obs_c, _, _, _, _ = env_continuous.step({"agent_0": action_c})

    # Velocities should match
    np.testing.assert_array_almost_equal(obs_d["agent_0"][:2], obs_c["agent_0"][:2],
                                         err_msg="Discrete and continuous should produce same velocities")


def test_continuous_actions_partial_forces():
    """Test that partial continuous actions work (e.g., 0.5 force)."""
    env = parallel_env(max_cycles=10, continuous_actions=True)
    env.reset(seed=42)

    # Half force to the right
    action = {"agent_0": np.array([0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)}
    obs, _, _, _, _ = env.step(action)

    # Velocity should be positive but less than full force
    assert 0.2 < obs["agent_0"][0] < 0.3, f"Half force should give ~0.25 vel_x, got {obs['agent_0'][0]}"


def test_continuous_actions_opposing_forces():
    """Test that opposing forces cancel out correctly (right-left, up-down)."""
    env = parallel_env(max_cycles=10, continuous_actions=True)
    env.reset(seed=42)

    # Right=0.8, Left=0.3 -> net force = 0.5 in +x
    # Up=0.6, Down=0.2 -> net force = 0.4 in +y
    action = {"agent_0": np.array([0.0, 0.3, 0.8, 0.2, 0.6], dtype=np.float32)}
    obs, _, _, _, _ = env.step(action)

    # Net force_x = (0.8 - 0.3) = 0.5, so vel_x should be ~0.25
    # Net force_y = (0.6 - 0.2) = 0.4, so vel_y should be ~0.20
    assert 0.2 < obs["agent_0"][0] < 0.3, f"Net right force 0.5 should give vel_x~0.25, got {obs['agent_0'][0]}"
    assert 0.15 < obs["agent_0"][1] < 0.25, f"Net up force 0.4 should give vel_y~0.20, got {obs['agent_0'][1]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
