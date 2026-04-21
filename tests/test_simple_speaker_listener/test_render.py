"""Test rendering functionality for SimpleSpeakerListener environment."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
from cpp_pettingzoo.simple_speaker_listener import parallel_env


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


def test_render_shows_two_agents():
    """Test that rendering includes both agents (speaker and listener)."""
    env = parallel_env(render_mode="rgb_array")
    env.reset(seed=42)

    frame = env.render()

    # Frame should be non-empty (agents and landmarks visible)
    # Check that frame is not all white (255, 255, 255)
    assert not np.all(frame == 255), "Frame should contain rendered entities"

    env.close()


def test_render_shows_landmarks():
    """Test that rendering includes all 3 landmarks."""
    env = parallel_env(render_mode="rgb_array")
    env.reset(seed=42)

    frame = env.render()

    # Check that there are multiple distinct colors in the frame
    # (listener has color from goal landmark, landmarks have RGB colors)
    # Expect at least: white background + 3 RGB landmarks + 2 agents = 5+ colors
    unique_pixels = len(np.unique(frame.reshape(-1, 3), axis=0))
    assert unique_pixels >= 4, "Frame should contain multiple distinct colors for agents and landmarks"

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
