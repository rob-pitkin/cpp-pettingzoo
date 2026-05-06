"""Test PettingZoo API compliance for SimplePush environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from pettingzoo.test import parallel_api_test

from cpp_pettingzoo.simple_push.simple_push import parallel_env


def test_api_default():
    parallel_api_test(parallel_env(), num_cycles=100)


def test_api_continuous():
    parallel_api_test(parallel_env(continuous_actions=True), num_cycles=100)


def test_max_cycles():
    env = parallel_env(max_cycles=10)
    env.reset(seed=42)
    for _ in range(10):
        _, _, _, truncs, _ = env.step({"adversary_0": 0, "agent_0": 0})
    assert all(truncs.values())
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
