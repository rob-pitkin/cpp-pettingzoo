"""Benchmark C++ vs Python PettingZoo SimpleTag environments.

Compares cpp-pettingzoo wrapper (C++ backend) vs mpe2 (Python backend).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cpp_pettingzoo.simple_tag.simple_tag import parallel_env as cpp_parallel_env
from mpe2 import simple_tag_v3

NUM_GOOD = 1
NUM_ADVERSARIES = 3
NUM_OBSTACLES = 2
MAX_CYCLES = 25

AGENTS = (
    [f"adversary_{i}" for i in range(NUM_ADVERSARIES)]
    + [f"agent_{i}" for i in range(NUM_GOOD)]
)


def benchmark_resets(env_factory, n_resets):
    env = env_factory()
    start = time.perf_counter()
    for _ in range(n_resets):
        env.reset()
    return time.perf_counter() - start


def benchmark_steps(env_factory, n_steps):
    env = env_factory()
    env.reset()
    start = time.perf_counter()
    for _ in range(n_steps):
        actions = {a: 0 for a in AGENTS}
        _, _, terms, truncs, _ = env.step(actions)
        if any(terms.values()) or any(truncs.values()):
            env.reset()
    return time.perf_counter() - start


def benchmark_episodes(env_factory, n_episodes):
    env = env_factory()
    start = time.perf_counter()
    for _ in range(n_episodes):
        env.reset()
        for _ in range(MAX_CYCLES):
            actions = {a: 0 for a in AGENTS}
            env.step(actions)
    return time.perf_counter() - start


if __name__ == "__main__":
    n_resets = 1_000_000
    n_steps = 1_000_000
    n_episodes = 100_000

    print("=" * 60)
    print("PettingZoo SimpleTag Environment Benchmark")
    print("=" * 60)
    print(f"Configuration: {n_resets} resets, {n_steps} steps, {n_episodes} episodes")
    print(f"Max cycles per episode: {MAX_CYCLES}")
    print(f"Agents: {NUM_ADVERSARIES} adversaries + {NUM_GOOD} good, "
          f"Obstacles: {NUM_OBSTACLES}")
    print()

    cpp_factory = lambda: cpp_parallel_env(
        num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES,
        num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES,
        continuous_actions=False,
    )
    print("cpp-pettingzoo (C++ backend with Python wrapper):")
    cpp_reset = benchmark_resets(cpp_factory, n_resets)
    print(f"  Resets:   {n_resets / cpp_reset:>10.1f} resets/sec  ({cpp_reset:.3f}s total)")
    cpp_step = benchmark_steps(cpp_factory, n_steps)
    print(f"  Steps:    {n_steps / cpp_step:>10.1f} steps/sec   ({cpp_step:.3f}s total)")
    cpp_ep = benchmark_episodes(cpp_factory, n_episodes)
    print(f"  Episodes: {n_episodes / cpp_ep:>10.1f} episodes/sec ({cpp_ep:.3f}s total)")

    mpe2_factory = lambda: simple_tag_v3.parallel_env(
        num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES,
        num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES,
        continuous_actions=False,
    )
    print("\nmpe2 (Pure Python backend):")
    mpe2_reset = benchmark_resets(mpe2_factory, n_resets)
    print(f"  Resets:   {n_resets / mpe2_reset:>10.1f} resets/sec  ({mpe2_reset:.3f}s total)")
    mpe2_step = benchmark_steps(mpe2_factory, n_steps)
    print(f"  Steps:    {n_steps / mpe2_step:>10.1f} steps/sec   ({mpe2_step:.3f}s total)")
    mpe2_ep = benchmark_episodes(mpe2_factory, n_episodes)
    print(f"  Episodes: {n_episodes / mpe2_ep:>10.1f} episodes/sec ({mpe2_ep:.3f}s total)")

    print("\n" + "=" * 60)
    print("Speedup (C++ / Python):")
    print("=" * 60)
    print(f"  Resets:   {mpe2_reset / cpp_reset:.2f}x faster")
    print(f"  Steps:    {mpe2_step / cpp_step:.2f}x faster")
    print(f"  Episodes: {mpe2_ep / cpp_ep:.2f}x faster")
    print("=" * 60)
