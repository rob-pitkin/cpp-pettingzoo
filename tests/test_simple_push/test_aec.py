"""Test SimplePush AEC API compatibility."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from pettingzoo.test import api_test as aec_api_test

from cpp_pettingzoo.simple_push.simple_push import env as push_env


def test_aec_api():
    aec_api_test(push_env(), num_cycles=10, verbose_progress=False)


def test_aec_api_continuous():
    aec_api_test(push_env(continuous_actions=True), num_cycles=10, verbose_progress=False)


def test_aec_basic_attributes():
    env = push_env()
    env.reset()
    assert hasattr(env, 'agents')
    assert hasattr(env, 'agent_selection')
    assert hasattr(env, 'rewards')
    assert hasattr(env, 'terminations')
    assert hasattr(env, 'truncations')
    env.close()


def test_aec_reset():
    env = push_env()
    env.reset(seed=42)
    assert set(env.agents) == {"adversary_0", "agent_0"}
    assert env.agent_selection in env.agents
    env.close()


def test_aec_step_cycle():
    env = push_env(max_cycles=5)
    env.reset(seed=42)
    steps = 0
    while env.agents and steps < 5 * 2:
        agent = env.agent_selection
        _, _, term, trunc, _ = env.last()
        env.step(None if (term or trunc) else env.action_space(agent).sample())
        steps += 1
    env.close()


def test_aec_deterministic():
    env1 = push_env()
    env2 = push_env()
    env1.reset(seed=7); env2.reset(seed=7)
    obs1, _, _, _, _ = env1.last()
    obs2, _, _, _, _ = env2.last()
    np.testing.assert_allclose(obs1, obs2)
    env1.close(); env2.close()


def test_aec_observation_spaces():
    env = push_env()
    env.reset()
    assert env.observation_space("adversary_0").shape == (8,)
    assert env.observation_space("agent_0").shape == (19,)
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
