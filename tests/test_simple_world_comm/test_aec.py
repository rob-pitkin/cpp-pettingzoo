"""Test SimpleWorldComm AEC API compatibility."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from pettingzoo.test import api_test as aec_api_test

from cpp_pettingzoo.simple_world_comm.simple_world_comm import env as swc_env


def test_aec_api():
    aec_api_test(swc_env(), num_cycles=10, verbose_progress=False)


def test_aec_basic_attributes():
    env = swc_env()
    env.reset()
    for attr in ['agents', 'num_agents', 'agent_selection', 'rewards',
                 'terminations', 'truncations', 'infos']:
        assert hasattr(env, attr)
    env.close()


def test_aec_reset():
    env = swc_env()
    env.reset(seed=42)
    assert len(env.agents) == 6  # 4 adversaries + 2 good
    assert env.agent_selection in env.agents
    env.close()


def test_aec_step_cycle():
    env = swc_env(max_cycles=5)
    env.reset(seed=42)
    steps = 0
    while env.agents and steps < 5 * 6:
        agent = env.agent_selection
        _, _, term, trunc, _ = env.last()
        env.step(None if (term or trunc) else env.action_space(agent).sample())
        steps += 1
    env.close()


def test_aec_deterministic():
    env1 = swc_env()
    env2 = swc_env()
    env1.reset(seed=123); env2.reset(seed=123)
    obs1, _, _, _, _ = env1.last()
    obs2, _, _, _, _ = env2.last()
    np.testing.assert_allclose(obs1, obs2)
    env1.close(); env2.close()


def test_aec_observation_action_spaces():
    env = swc_env()
    env.reset()
    assert env.observation_space("leadadversary_0").shape == (34,)
    assert env.action_space("leadadversary_0").n == 20
    for i in range(3):
        assert env.observation_space(f"adversary_{i}").shape == (34,)
        assert env.action_space(f"adversary_{i}").n == 5
    for i in range(2):
        assert env.observation_space(f"agent_{i}").shape == (28,)
        assert env.action_space(f"agent_{i}").n == 5
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
