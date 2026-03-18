"""Compare rendering between MPE2 and cpp-pettingzoo visually."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cpp_pettingzoo.simple import parallel_env as cpp_parallel_env
from mpe2 import simple_v3
import time


def watch_cpp(episodes=3, steps_per_episode=25):
    """Watch cpp-pettingzoo rendering."""
    print("=" * 60)
    print("WATCHING CPP-PETTINGZOO")
    print("=" * 60)
    print("Gray = agent, Red = landmark")
    print(f"Running {episodes} episodes\n")

    env = cpp_parallel_env(max_cycles=steps_per_episode, render_mode="human")

    for episode in range(episodes):
        print(f"CPP Episode {episode + 1}/{episodes}")
        env.reset(seed=episode)

        for step in range(steps_per_episode):
            action = env.action_space("agent_0").sample()
            obs, rewards, terms, truncs, _ = env.step({"agent_0": action})
            env.render()
            time.sleep(0.1)

            if terms["agent_0"] or truncs["agent_0"]:
                break

        time.sleep(0.5)

    env.close()
    print("\nCPP rendering done!\n")


def watch_mpe2(episodes=3, steps_per_episode=25):
    """Watch MPE2 rendering."""
    print("=" * 60)
    print("WATCHING MPE2")
    print("=" * 60)
    print("Gray = agent, Red = landmark")
    print(f"Running {episodes} episodes\n")

    # Use raw_env for MPE2 to get render support
    from mpe2.simple_v3 import raw_env
    env = raw_env(max_cycles=steps_per_episode, continuous_actions=False, render_mode="human")

    for episode in range(episodes):
        print(f"MPE2 Episode {episode + 1}/{episodes}")
        env.reset(seed=episode)

        for agent in env.agent_iter(max_iter=steps_per_episode):
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                action = None
            else:
                action = env.action_space(agent).sample()

            env.step(action)
            time.sleep(0.1)

        time.sleep(0.5)

    env.close()
    print("\nMPE2 rendering done!\n")


if __name__ == "__main__":
    # First watch cpp-pettingzoo
    watch_cpp()

    input("\nPress Enter to watch MPE2 rendering...")

    # Then watch MPE2
    watch_mpe2()
