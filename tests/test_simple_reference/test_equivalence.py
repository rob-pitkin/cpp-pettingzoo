"""Test SimpleReference C++ implementation matches expected behavior."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
from cpp_pettingzoo.simple_reference import parallel_env as cpp_env


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

    actions = {agent: 0 for agent in cpp1.agents}
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

    initial_velocities = [obs[agent][0:2] for agent in env.agents]

    # Take action to move agent 0 right
    actions = {env.agents[0]: 2, env.agents[1]: 0}
    obs, _, _, _, _ = env.step(actions)

    final_velocities = [obs[agent][0:2] for agent in env.agents]

    # Velocity should have changed for at least one agent
    changed = any(
        not np.allclose(initial_velocities[i], final_velocities[i])
        for i in range(2)
    )
    assert changed, "Physics should update velocities"

    env.close()


def test_full_episode_truncation():
    """Test that episode truncates after max_cycles."""
    env = cpp_env(max_cycles=5)
    env.reset(seed=42)

    for _ in range(5):
        actions = {agent: 0 for agent in env.agents}
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
        assert all(len(obs[agent]) == 21 for agent in env.agents), \
            "Observations should be correct size after reset"

    env.close()


def test_reward_calculation():
    """Test that rewards are calculated correctly."""
    env = cpp_env(local_ratio=0.5)
    env.reset(seed=42)

    actions = {agent: 0 for agent in env.agents}
    _, rewards, _, _, _ = env.step(actions)

    # All agents should get rewards
    assert len(rewards) == 2, "Should have rewards for all 2 agents"

    # Rewards should be finite
    for agent, reward in rewards.items():
        assert np.isfinite(reward), f"Reward for {agent} should be finite"

    env.close()


def test_local_ratio_affects_rewards():
    """Test that local_ratio parameter affects reward calculation."""
    env_local = cpp_env(local_ratio=1.0)
    env_global = cpp_env(local_ratio=0.0)

    env_local.reset(seed=42)
    env_global.reset(seed=42)

    actions = {agent: 0 for agent in env_local.agents}
    _, rewards_local, _, _, _ = env_local.step(actions)
    _, rewards_global, _, _, _ = env_global.step(actions)

    # With different local_ratio, at least one agent should have different rewards
    different = any(
        not np.isclose(rewards_local[agent], rewards_global[agent])
        for agent in env_local.agents
    )
    assert different, "Different local_ratio should affect rewards"

    env_local.close()
    env_global.close()


def test_observation_shape():
    """Test that observation shape matches expected dimensions."""
    env = cpp_env()
    obs, _ = env.reset(seed=42)

    for agent in env.agents:
        assert obs[agent].shape == (21,), \
            f"Observation for {agent} should be shape (21,)"
        assert obs[agent].dtype == np.float32, \
            f"Observation for {agent} should be float32"

    env.close()


def test_communication_in_observations():
    """Test that communication from other agent appears in observations."""
    env = cpp_env(continuous_actions=True)
    obs, _ = env.reset(seed=42)

    # In continuous mode, set agent_0's communication explicitly
    actions = {
        "agent_0": np.array([0.0, 0.0, 0.0, 0.0, 0.0,  # movement (5)
                            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # comm (10)
        "agent_1": np.zeros(15)
    }

    obs, _, _, _, _ = env.step(actions)

    # agent_1's observation should include agent_0's communication in last 10 elements
    agent_1_obs_comm = obs["agent_1"][-10:]

    # First communication channel should be 1.0 from agent_0
    assert agent_1_obs_comm[0] > 0.9, "agent_1 should receive agent_0's communication"

    env.close()


def test_discrete_action_decomposition():
    """Test that discrete actions properly decompose into movement and communication."""
    env = cpp_env(continuous_actions=False)
    env.reset(seed=42)

    # Action 7 = (7 // 5) * 5 + (7 % 5) = comm=1, move=2 (right)
    actions = {
        "agent_0": 7,  # comm word 1, move right
        "agent_1": 0   # comm word 0, no movement
    }

    obs, _, _, _, _ = env.step(actions)

    # agent_1's observation should show agent_0's comm word 1 in the last 10 elements
    agent_1_obs_comm = obs["agent_1"][-10:]

    # Communication channel 1 should be active
    assert agent_1_obs_comm[1] > 0.9, "Discrete action 7 should activate comm word 1"

    env.close()


def test_goal_color_in_observation():
    """Test that goal landmark color appears in observations."""
    env = cpp_env()
    obs, _ = env.reset(seed=42)

    for agent in env.agents:
        # Elements 8:11 should be the goal landmark color (one of RGB landmarks)
        goal_color = obs[agent][8:11]

        # Should be a valid color (one channel dominant)
        assert np.sum(goal_color > 0.5) >= 1, \
            f"Goal color for {agent} should have at least one dominant channel"

    env.close()


def test_entity_sizes_match_mpe2():
    """Test that entity sizes match MPE2 defaults."""
    # Create environment with render_mode to access world
    env = cpp_env(render_mode="rgb_array")
    env.reset(seed=42)

    # SimpleReference uses default size 0.050 for both agents and landmarks
    for agent in env.world.agents:
        assert agent.size == 0.050, f"Agent size should be 0.050, got {agent.size}"

    for landmark in env.world.landmarks:
        assert landmark.size == 0.050, f"Landmark size should be 0.050, got {landmark.size}"

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
