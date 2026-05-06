"""Test PettingZoo API compliance for SimpleTag environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from pettingzoo.test import parallel_api_test

from cpp_pettingzoo.simple_tag.simple_tag import parallel_env


def test_api_default():
    env = parallel_env()
    parallel_api_test(env, num_cycles=100)


def test_api_partial_obs():
    env = parallel_env(num_agent_neighbors=2, num_landmark_neighbors=1)
    parallel_api_test(env, num_cycles=100)


def test_api_continuous():
    env = parallel_env(continuous_actions=True)
    parallel_api_test(env, num_cycles=100)


def test_api_curriculum():
    env = parallel_env(curriculum=True)
    parallel_api_test(env, num_cycles=100)


def test_api_terminate_on_success():
    env = parallel_env(terminate_on_success=True)
    parallel_api_test(env, num_cycles=100)


def test_max_cycles():
    env = parallel_env(max_cycles=10)
    env.reset(seed=42)
    actions = {a: 0 for a in env.agents}
    for _ in range(10):
        _, _, _, truncations, _ = env.step(actions)
    assert all(truncations.values()), "Should truncate after max_cycles"
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
