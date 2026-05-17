"""Test rendering functionality for SimpleCrypto environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from cpp_pettingzoo.simple_crypto.simple_crypto import parallel_env


def test_rgb_array_render():
    env = parallel_env(render_mode="rgb_array")
    env.reset(seed=42)
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (700, 700, 3)
    assert frame.dtype == np.uint8
    env.close()


def test_human_render():
    env = parallel_env(render_mode="human")
    env.reset(seed=42)
    result = env.render()
    assert result is None
    env.close()


def test_no_render_mode():
    env = parallel_env()
    env.reset(seed=42)
    result = env.render()
    assert result is None
    env.close()


def test_render_after_steps():
    env = parallel_env(render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(5):
        env.step({a: env.action_space(a).sample() for a in env.agents})
    frame = env.render()
    assert frame.shape == (700, 700, 3)
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
