"""Test rendering functionality for SimpleSpread environment."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
from cpp_pettingzoo.simple_spread import parallel_env


def test_rgb_array_render():
    """Test that rgb_array rendering produces correct output."""
    env = parallel_env(render_mode="rgb_array")
    env.reset(seed=42)

    # Render should return numpy array
    frame = env.render()
    assert isinstance(frame, np.ndarray), "Render should return numpy array"
    assert frame.shape == (700, 700, 3), "Frame should be 700x700 RGB"
    assert frame.dtype == np.uint8, "Frame should be uint8"

    env.close()


def test_human_render():
    """Test that human rendering initializes correctly."""
    env = parallel_env(render_mode="human")
    env.reset(seed=42)

    # Should not crash
    result = env.render()
    assert result is None, "Human render should return None"

    env.close()


def test_no_render_mode():
    """Test that rendering without mode produces warning."""
    env = parallel_env()  # No render_mode
    env.reset(seed=42)

    # Should not crash, but may warn
    result = env.render()
    assert result is None, "Render without mode should return None"

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
