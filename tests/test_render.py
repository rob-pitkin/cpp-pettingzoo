"""Quick test script to verify rendering works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cpp_pettingzoo.simple import parallel_env
import numpy as np


def test_rgb_array_render():
    """Test that rgb_array rendering works."""
    print("Testing rgb_array rendering...")
    env = parallel_env(max_cycles=10, render_mode="rgb_array")
    env.reset(seed=42)

    # Take a few steps and render
    for i in range(5):
        actions = {"agent_0": env.action_space("agent_0").sample()}
        obs, rewards, terms, truncs, infos = env.step(actions)

        img = env.render()
        assert img is not None, "render() should return an image for rgb_array mode"
        assert isinstance(img, np.ndarray), f"Expected numpy array, got {type(img)}"
        assert img.shape == (700, 700, 3), f"Expected (700, 700, 3), got {img.shape}"
        print(f"  Step {i}: rendered image shape = {img.shape}")

    env.close()
    print("✓ rgb_array rendering works!\n")


def test_human_render():
    """Test that human rendering initializes correctly."""
    print("Testing human rendering...")
    env = parallel_env(max_cycles=10, render_mode="human")
    env.reset(seed=42)

    # Take a few steps with rendering
    # Note: This won't actually display a window in headless mode, but should not crash
    for i in range(5):
        actions = {"agent_0": 2}  # Move right
        obs, rewards, terms, truncs, infos = env.step(actions)
        env.render()
        print(f"  Step {i}: reward = {rewards['agent_0']:.4f}")

    env.close()
    print("✓ human rendering initialized successfully!\n")


def test_no_render_mode():
    """Test that environment works without render_mode."""
    print("Testing without render_mode...")
    env = parallel_env(max_cycles=10, render_mode=None)
    env.reset(seed=42)

    for i in range(5):
        actions = {"agent_0": 1}  # Move left
        obs, rewards, terms, truncs, infos = env.step(actions)

    env.close()
    print("✓ No render mode works!\n")


if __name__ == "__main__":
    test_no_render_mode()
    test_rgb_array_render()
    test_human_render()
    print("All rendering tests passed! 🎉")
