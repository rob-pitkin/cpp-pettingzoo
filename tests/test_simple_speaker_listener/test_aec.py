"""Test AEC wrapper for SimpleSpeakerListener environment."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from pettingzoo.test import api_test
from cpp_pettingzoo.simple_speaker_listener import env as aec_env


def test_aec_api():
    """Test that AEC wrapper complies with PettingZoo API."""
    environment = aec_env()
    api_test(environment, num_cycles=100, verbose_progress=False)


def test_aec_discrete():
    """Test AEC wrapper with discrete actions."""
    environment = aec_env(continuous_actions=False)
    environment.reset(seed=42)

    for agent in environment.agent_iter(max_iter=20):
        observation, reward, termination, truncation, info = environment.last()

        if termination or truncation:
            action = None
        else:
            action = environment.action_space(agent).sample()

        environment.step(action)

    environment.close()


def test_aec_continuous():
    """Test AEC wrapper with continuous actions."""
    environment = aec_env(continuous_actions=True)
    environment.reset(seed=42)

    for agent in environment.agent_iter(max_iter=20):
        observation, reward, termination, truncation, info = environment.last()

        if termination or truncation:
            action = None
        else:
            action = environment.action_space(agent).sample()

        environment.step(action)

    environment.close()


def test_aec_local_ratio():
    """Test AEC wrapper with different local_ratio values."""
    for local_ratio in [0.0, 0.5, 1.0]:
        environment = aec_env(local_ratio=local_ratio)
        environment.reset(seed=42)

        for agent in environment.agent_iter(max_iter=10):
            observation, reward, termination, truncation, info = environment.last()

            if termination or truncation:
                action = None
            else:
                # Use appropriate action for each agent
                if agent == "speaker_0":
                    action = 0  # comm word 0
                else:
                    action = 0  # no movement

            environment.step(action)

        environment.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
