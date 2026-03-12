import time
import sys
sys.path.insert(0, 'build')
import _simple_core
from mpe2 import simple_v3

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
    result = env.step({"agent_0": 1})
    # Reset if episode is done (check truncations/terminations)
    if len(result) == 4:  # Tuple: (obs, rewards, terms, truncs)
      if result[2].get("agent_0", False) or result[3].get("agent_0", False):
        env.reset()
  end_time = time.perf_counter()
  return (end_time - start_time)

def benchmark_episodes(env_factory, n_episodes, max_cycles):
  env = env_factory()
  start_time = time.perf_counter()
  for _ in range(n_episodes):
    env.reset()
    for _ in range(max_cycles):
      env.step({"agent_0": 0})
  end_time = time.perf_counter()
  return (end_time - start_time)

if __name__ == "__main__":
  n_resets = 10000
  n_steps = 100000
  n_episodes = 1000
  max_cycles = 25

  # C++ benchmarks
  cpp_factory = lambda: _simple_core.SimpleEnv(seed=1, max_cycles=max_cycles)
  print("C++ implementation")
  cpp_reset_time = benchmark_resets(cpp_factory, n_resets)
  print(f"  Resets: {n_resets / cpp_reset_time:.1f} resets/sec")

  cpp_step_time = benchmark_steps(cpp_factory, n_steps)
  print(f"  Steps: {n_steps / cpp_step_time:.1f} steps/sec")

  cpp_episode_time = benchmark_episodes(cpp_factory, n_episodes, max_cycles)
  print(f"  Episodes: {n_episodes / cpp_episode_time:.1f} episodes/sec")

  # MPE2 benchmarks
  mpe2_factory = lambda: simple_v3.parallel_env(max_cycles=max_cycles, continuous_actions=False)
  print("\nMPE2 implementation:")
  mpe2_reset_time = benchmark_resets(mpe2_factory, n_resets)
  print(f"  Resets: {n_resets / mpe2_reset_time:.1f} resets/sec")

  mpe2_step_time = benchmark_steps(mpe2_factory, n_steps)
  print(f"  Steps: {n_steps / mpe2_step_time:.1f} steps/sec")

  mpe2_episode_time = benchmark_episodes(mpe2_factory, n_episodes, max_cycles)
  print(f"  Episodes: {n_episodes / mpe2_episode_time:.1f} episodes/sec")

  # Speedup comparison
  print("\nSpeedup (C++ / Python):")
  print(f"  Resets: {mpe2_reset_time / cpp_reset_time:.2f}x")
  print(f"  Steps: {mpe2_step_time / cpp_step_time:.2f}x")
  print(f"  Episodes: {mpe2_episode_time / cpp_episode_time:.2f}x")
