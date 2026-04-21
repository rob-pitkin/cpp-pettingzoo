"""Benchmark C++ vs Python PettingZoo SimpleSpeakerListener environments.

Compares cpp-pettingzoo wrapper (C++ backend) vs mpe2 (Python backend).
"""

import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpp_pettingzoo.simple_speaker_listener import parallel_env as cpp_parallel_env
from mpe2 import simple_speaker_listener_v4


def benchmark_resets(env_factory, n_resets):
    env = env_factory()
    start_time = time.perf_counter()
    for _ in range(n_resets):
        env.reset()
    end_time = time.perf_counter()
    return (end_time - start_time)


def benchmark_steps(env_factory, n_steps):
    env = env_factory()
    start_time = time.perf_counter()
    env.reset()
    for _ in range(n_steps):
        # SimpleSpeakerListener has 2 agents: speaker_0 and listener_0
        actions = {"speaker_0": 0, "listener_0": 0}
        obs, rewards, terms, truncs, infos = env.step(actions)
        # Reset if episode is done
        if any(terms.values()) or any(truncs.values()):
            env.reset()
    end_time = time.perf_counter()
    return (end_time - start_time)


def benchmark_episodes(env_factory, n_episodes, max_cycles):
    env = env_factory()
    start_time = time.perf_counter()
    for _ in range(n_episodes):
        env.reset()
        for _ in range(max_cycles):
            actions = {"speaker_0": 0, "listener_0": 0}
            env.step(actions)
    end_time = time.perf_counter()
    return (end_time - start_time)


if __name__ == "__main__":
    n_resets = 1000000
    n_steps = 1000000
    n_episodes = 100000
    max_cycles = 25

    print("=" * 60)
    print("PettingZoo SimpleSpeakerListener Environment Benchmark")
    print("=" * 60)
    print(f"Configuration: {n_resets} resets, {n_steps} steps, {n_episodes} episodes")
    print(f"Max cycles per episode: {max_cycles}")
    print(f"Agents: 2 (speaker_0, listener_0), Landmarks: 3, Communication: 3 words")
    print()

    # C++ wrapper benchmarks (cpp-pettingzoo with C++ backend)
    cpp_factory = lambda: cpp_parallel_env(max_cycles=max_cycles, continuous_actions=False)
    print("cpp-pettingzoo (C++ backend with Python wrapper):")
    cpp_reset_time = benchmark_resets(cpp_factory, n_resets)
    print(f"  Resets:   {n_resets / cpp_reset_time:>10.1f} resets/sec  ({cpp_reset_time:.3f}s total)")

    cpp_step_time = benchmark_steps(cpp_factory, n_steps)
    print(f"  Steps:    {n_steps / cpp_step_time:>10.1f} steps/sec   ({cpp_step_time:.3f}s total)")

    cpp_episode_time = benchmark_episodes(cpp_factory, n_episodes, max_cycles)
    print(f"  Episodes: {n_episodes / cpp_episode_time:>10.1f} episodes/sec ({cpp_episode_time:.3f}s total)")

    # MPE2 benchmarks (pure Python)
    mpe2_factory = lambda: simple_speaker_listener_v4.parallel_env(max_cycles=max_cycles, continuous_actions=False)
    print("\nmpe2 (Pure Python backend):")
    mpe2_reset_time = benchmark_resets(mpe2_factory, n_resets)
    print(f"  Resets:   {n_resets / mpe2_reset_time:>10.1f} resets/sec  ({mpe2_reset_time:.3f}s total)")

    mpe2_step_time = benchmark_steps(mpe2_factory, n_steps)
    print(f"  Steps:    {n_steps / mpe2_step_time:>10.1f} steps/sec   ({mpe2_step_time:.3f}s total)")

    mpe2_episode_time = benchmark_episodes(mpe2_factory, n_episodes, max_cycles)
    print(f"  Episodes: {n_episodes / mpe2_episode_time:>10.1f} episodes/sec ({mpe2_episode_time:.3f}s total)")

    # Speedup comparison
    print("\n" + "=" * 60)
    print("Speedup (C++ / Python):")
    print("=" * 60)
    print(f"  Resets:   {mpe2_reset_time / cpp_reset_time:.2f}x faster")
    print(f"  Steps:    {mpe2_step_time / cpp_step_time:.2f}x faster")
    print(f"  Episodes: {mpe2_episode_time / cpp_episode_time:.2f}x faster")
    print("=" * 60)
