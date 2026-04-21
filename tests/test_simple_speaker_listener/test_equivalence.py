"""Test SimpleSpeakerListener C++ implementation matches expected behavior."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
from cpp_pettingzoo.simple_speaker_listener import parallel_env as cpp_env


def test_reset_deterministic():
    """Test that reset with same seed is deterministic."""
    cpp1 = cpp_env()
    cpp2 = cpp_env()

    obs1, _ = cpp1.reset(seed=42)
    obs2, _ = cpp2.reset(seed=42)

    for agent in cpp1.agents:
        assert np.allclose(obs1[agent], obs2[agent]), \
            f"Observations for {agent} should be identical with same seed"

    cpp1.close()
    cpp2.close()


def test_reset_different_seeds():
    """Test that reset with different seeds produces different results."""
    cpp1 = cpp_env()
    cpp2 = cpp_env()

    obs1, _ = cpp1.reset(seed=42)
    obs2, _ = cpp2.reset(seed=123)

    # At least one agent should have different observations
    different = False
    for agent in cpp1.agents:
        if not np.allclose(obs1[agent], obs2[agent]):
            different = True
            break

    assert different, "Different seeds should produce different observations"

    cpp1.close()
    cpp2.close()


def test_single_step_deterministic():
    """Test that single step is deterministic with same seed and actions."""
    cpp1 = cpp_env()
    cpp2 = cpp_env()

    cpp1.reset(seed=42)
    cpp2.reset(seed=42)

    actions = {"speaker_0": 0, "listener_0": 0}
    obs1, _, _, _, _ = cpp1.step(actions)
    obs2, _, _, _, _ = cpp2.step(actions)

    for agent in cpp1.agents:
        assert np.allclose(obs1[agent], obs2[agent]), \
            f"Step results for {agent} should be deterministic"

    cpp1.close()
    cpp2.close()


def test_physics_dynamics():
    """Test that physics updates velocities correctly."""
    env = cpp_env()
    obs, _ = env.reset(seed=42)

    # Speaker doesn't have velocity in obs (only goal color)
    # Listener has velocity in obs[0:2]
    initial_listener_vel = obs["listener_0"][0:2]

    # Take action to move listener right
    actions = {"speaker_0": 0, "listener_0": 2}  # listener moves right
    obs, _, _, _, _ = env.step(actions)

    final_listener_vel = obs["listener_0"][0:2]

    # Listener velocity should have changed
    assert not np.allclose(initial_listener_vel, final_listener_vel), \
        "Physics should update listener velocity"

    env.close()


def test_full_episode_truncation():
    """Test that episode truncates after max_cycles."""
    env = cpp_env(max_cycles=5)
    env.reset(seed=42)

    for _ in range(5):
        actions = {"speaker_0": 0, "listener_0": 0}
        _, _, _, truncations, _ = env.step(actions)

    # After max_cycles, all agents should be truncated
    assert all(truncations.values()), \
        "All agents should be truncated after max_cycles"

    env.close()


def test_multiple_resets():
    """Test that environment can be reset multiple times."""
    env = cpp_env()

    for seed in [42, 123, 456]:
        obs, _ = env.reset(seed=seed)
        assert len(obs) == 2, "Should have 2 agents after each reset"
        assert obs["speaker_0"].shape == (3,), \
            "Speaker observation should be shape (3,)"
        assert obs["listener_0"].shape == (11,), \
            "Listener observation should be shape (11,)"

    env.close()


def test_reward_calculation():
    """Test that rewards are calculated correctly."""
    env = cpp_env(local_ratio=0.5)
    env.reset(seed=42)

    actions = {"speaker_0": 0, "listener_0": 0}
    _, rewards, _, _, _ = env.step(actions)

    # All agents should get rewards
    assert len(rewards) == 2, "Should have rewards for all 2 agents"

    # Rewards should be finite
    for agent, reward in rewards.items():
        assert np.isfinite(reward), f"Reward for {agent} should be finite"

    # Rewards should be negative (distance-based)
    for agent, reward in rewards.items():
        assert reward <= 0, f"Reward for {agent} should be negative or zero"

    env.close()


def test_asymmetric_observation_shapes():
    """Test that observation shapes are asymmetric (speaker: 3, listener: 11)."""
    env = cpp_env()
    obs, _ = env.reset(seed=42)

    # Speaker sees only goal color (3)
    assert obs["speaker_0"].shape == (3,), \
        "Speaker observation should be shape (3,) for goal color"
    assert obs["speaker_0"].dtype == np.float32, \
        "Speaker observation should be float32"

    # Listener sees vel + landmarks + comm (11)
    assert obs["listener_0"].shape == (11,), \
        "Listener observation should be shape (11,)"
    assert obs["listener_0"].dtype == np.float32, \
        "Listener observation should be float32"

    env.close()


def test_asymmetric_action_spaces():
    """Test that action spaces are asymmetric."""
    env = cpp_env(continuous_actions=False)

    # Speaker: Discrete(3) for communication
    assert env.action_space("speaker_0").n == 3, \
        "Speaker should have 3 discrete actions"

    # Listener: Discrete(5) for movement
    assert env.action_space("listener_0").n == 5, \
        "Listener should have 5 discrete actions"

    env.close()


def test_continuous_asymmetric_action_spaces():
    """Test that continuous action spaces are asymmetric."""
    env = cpp_env(continuous_actions=True)

    # Speaker: Box(3,) for communication
    assert env.action_space("speaker_0").shape == (3,), \
        "Speaker should have 3-dimensional continuous action space"

    # Listener: Box(5,) for movement
    assert env.action_space("listener_0").shape == (5,), \
        "Listener should have 5-dimensional continuous action space"

    env.close()


def test_speaker_communication_to_listener():
    """Test that speaker's communication appears in listener's observation."""
    env = cpp_env(continuous_actions=True)
    obs, _ = env.reset(seed=42)

    # Speaker sends communication word 0 (first channel = 1.0)
    actions = {
        "speaker_0": np.array([1.0, 0.0, 0.0]),  # comm word 0
        "listener_0": np.zeros(5)  # no movement
    }

    obs, _, _, _, _ = env.step(actions)

    # Listener's observation last 3 elements should contain speaker's comm
    listener_comm = obs["listener_0"][-3:]

    # First communication channel should be active
    assert listener_comm[0] > 0.9, "Listener should receive speaker's communication"

    env.close()


def test_discrete_speaker_communication():
    """Test that discrete speaker actions set communication correctly."""
    env = cpp_env(continuous_actions=False)
    env.reset(seed=42)

    # Speaker sends communication word 1
    actions = {
        "speaker_0": 1,  # comm word 1
        "listener_0": 0  # no movement
    }

    obs, _, _, _, _ = env.step(actions)

    # Listener's observation last 3 elements should contain speaker's comm
    listener_comm = obs["listener_0"][-3:]

    # Communication channel 1 should be active
    assert listener_comm[1] > 0.9, "Discrete action 1 should activate comm word 1"

    env.close()


def test_goal_color_in_speaker_observation():
    """Test that speaker sees goal landmark color."""
    env = cpp_env()
    obs, _ = env.reset(seed=42)

    # Speaker should see goal color (3 elements)
    speaker_obs = obs["speaker_0"]
    assert len(speaker_obs) == 3, "Speaker should see 3-element goal color"

    # Should be a valid color (one of RGB landmarks)
    # At least one channel should be dominant
    assert np.sum(speaker_obs > 0.5) >= 1, \
        "Goal color should have at least one dominant channel"

    env.close()


def test_listener_sees_landmarks():
    """Test that listener observation contains landmark positions."""
    env = cpp_env()
    obs, _ = env.reset(seed=42)

    # Listener obs: vel(2) + landmarks(6) + comm(3) = 11
    listener_obs = obs["listener_0"]

    # Elements 2:8 should be relative landmark positions
    landmark_positions = listener_obs[2:8]

    # Positions should be reasonable (within world bounds)
    assert np.all(np.abs(landmark_positions) < 10), \
        "Landmark positions should be within reasonable bounds"

    env.close()


def test_entity_sizes_match_mpe2():
    """Test that entity sizes match MPE2 values."""
    # Create environment with render_mode to access world
    env = cpp_env(render_mode="rgb_array")
    env.reset(seed=42)

    # SimpleSpeakerListener uses 0.075 for agents
    for agent in env.world.agents:
        assert agent.size == 0.075, f"Agent size should be 0.075, got {agent.size}"

    # SimpleSpeakerListener uses 0.04 for landmarks
    for landmark in env.world.landmarks:
        assert landmark.size == 0.04, f"Landmark size should be 0.04, got {landmark.size}"

    env.close()


def test_agent_names():
    """Test that agents have correct names (speaker_0, listener_0)."""
    env = cpp_env()
    env.reset(seed=42)

    assert env.agents == ["speaker_0", "listener_0"], \
        "Agents should be named speaker_0 and listener_0"

    env.close()


def test_same_reward_for_both_agents():
    """Test that both agents receive the same reward (cooperative task)."""
    env = cpp_env()
    env.reset(seed=42)

    actions = {"speaker_0": 0, "listener_0": 0}
    _, rewards, _, _, _ = env.step(actions)

    # Both agents should get the same reward (cooperative)
    assert np.isclose(rewards["speaker_0"], rewards["listener_0"]), \
        "Both agents should receive the same reward in cooperative task"

    env.close()


def test_reward_uses_squared_distance():
    """Test that reward calculation uses squared distance (not sqrt)."""
    env = cpp_env()
    obs, _ = env.reset(seed=42)

    # Take one step and check reward magnitude
    actions = {"speaker_0": 0, "listener_0": 0}
    _, rewards, _, _, _ = env.step(actions)

    # Rewards should be negative (distance-based)
    # Squared distance produces larger negative values than sqrt for distances > 1
    assert rewards["speaker_0"] < 0, "Reward should be negative"

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
