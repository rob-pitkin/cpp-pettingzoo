"""Test SimpleFormation AEC API compatibility."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from pettingzoo.test import api_test as aec_api_test

from cpp_pettingzoo.simple_formation.simple_formation import env as formation_env


def test_aec_api():
    aec_api_test(formation_env(N=4), num_cycles=10, verbose_progress=False)


def test_aec_basic_attributes():
    env = formation_env(N=4)
    env.reset()
    assert hasattr(env, 'agents')
    assert hasattr(env, 'num_agents')
    assert hasattr(env, 'agent_selection')
    assert hasattr(env, 'rewards')
    assert hasattr(env, 'terminations')
    assert hasattr(env, 'truncations')
    assert hasattr(env, 'infos')
    env.close()


def test_aec_reset():
    env = formation_env(N=4)
    env.reset(seed=42)
    assert len(env.agents) == 4
    assert env.agent_selection in env.agents
    env.close()


def test_aec_step_cycle():
    env = formation_env(N=4, max_cycles=5)
    env.reset(seed=42)
    steps = 0
    while env.agents and steps < 5 * 4:
        agent = env.agent_selection
        obs, _, term, trunc, _ = env.last()
        env.step(None if (term or trunc) else env.action_space(agent).sample())
        steps += 1
    env.close()


def test_aec_deterministic():
    env1 = formation_env(N=4)
    env2 = formation_env(N=4)
    env1.reset(seed=123); env2.reset(seed=123)
    obs1, _, _, _, _ = env1.last()
    obs2, _, _, _, _ = env2.last()
    np.testing.assert_allclose(obs1, obs2)
    env1.close(); env2.close()


def test_aec_agent_iter():
    env = formation_env(N=4, max_cycles=2)
    env.reset(seed=42)
    agents_seen = []
    for agent in env.agent_iter():
        agents_seen.append(agent)
        _, _, term, trunc, _ = env.last()
        env.step(None if (term or trunc) else 0)
    assert agents_seen.count("agent_0") == 3  # 2 cycles + done state
    env.close()


def test_aec_observation_action_spaces():
    env = formation_env(N=4)
    env.reset()
    for agent in [f"agent_{i}" for i in range(4)]:
        assert env.observation_space(agent).shape == (6,)
        assert env.action_space(agent).n == 5
    env.close()


def test_aec_global_reward_shared():
    """In AEC mode, after a full cycle all agents should have the same reward."""
    env = formation_env(N=4)
    env.reset(seed=42)
    for _ in range(4):  # one full cycle
        env.step(0)
    vals = list(env.rewards.values())
    assert all(abs(v - vals[0]) < 1e-6 for v in vals)
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
