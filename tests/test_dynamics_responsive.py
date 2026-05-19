"""Regression test: every env's reward must respond to dynamics.

This test would have caught the SimpleFormation / SimpleLine stale-cache bug
where global_reward() returned a frozen value computed at step 1 regardless
of subsequent agent movement.

For each environment, apply the same non-zero force action for several steps
and require that the reward varies across at least 2 distinct values. If a
scenario caches reward state across steps without invalidation (the
SimpleFormation/SimpleLine bug), this test fires.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from cpp_mpe2.simple.simple import parallel_env as simple_env
from cpp_mpe2.simple_spread.simple_spread import parallel_env as simple_spread_env
from cpp_mpe2.simple_reference.simple_reference import parallel_env as simple_reference_env
from cpp_mpe2.simple_speaker_listener.simple_speaker_listener import (
    parallel_env as simple_speaker_listener_env,
)
from cpp_mpe2.simple_adversary.simple_adversary import parallel_env as simple_adversary_env
from cpp_mpe2.simple_tag.simple_tag import parallel_env as simple_tag_env
from cpp_mpe2.simple_push.simple_push import parallel_env as simple_push_env
from cpp_mpe2.simple_formation.simple_formation import parallel_env as simple_formation_env
from cpp_mpe2.simple_line.simple_line import parallel_env as simple_line_env
from cpp_mpe2.collect_treasure.collect_treasure import parallel_env as collect_treasure_env
from cpp_mpe2.simple_world_comm.simple_world_comm import (
    parallel_env as simple_world_comm_env,
)
from cpp_mpe2.simple_crypto.simple_crypto import parallel_env as simple_crypto_env


# Each entry: (env factory, "action to apply each step")
# Action 2 = move right for movement-capable agents
# For pure-comm envs (simple_crypto), use 1 (toggle a comm word that creates non-zero delta).
ENVS_WITH_DYNAMICS = [
    ("simple", simple_env, 2),
    ("simple_spread", simple_spread_env, 2),
    ("simple_reference", simple_reference_env, 2),
    ("simple_speaker_listener", simple_speaker_listener_env, 2),
    ("simple_adversary", simple_adversary_env, 2),
    ("simple_tag", simple_tag_env, 2),
    ("simple_push", simple_push_env, 2),
    ("simple_formation", simple_formation_env, 2),
    ("simple_line", simple_line_env, 2),
    ("collect_treasure", collect_treasure_env, 2),
    ("simple_world_comm", simple_world_comm_env, 2),
]


# SimpleTag has a sparse reward (only fires on collision or out-of-bounds);
# under random actions agents tend to stay near origin so reward stays 0.
# Skip the general test for it — it's covered by the specific test below.
SPARSE_REWARD_ENVS = {"simple_tag"}


@pytest.mark.parametrize(
    "name,factory,action",
    [e for e in ENVS_WITH_DYNAMICS if e[0] not in SPARSE_REWARD_ENVS],
    ids=[e[0] for e in ENVS_WITH_DYNAMICS if e[0] not in SPARSE_REWARD_ENVS],
)
def test_reward_responds_to_dynamics(name, factory, action):
    """Reward must vary across steps when state changes — guards against stale caches.

    Uses random actions per step so the env state genuinely changes. Then
    asserts that at least one agent's reward varies across steps.
    """
    env = factory(max_cycles=25)
    env.reset(seed=42)
    agents = list(env.agents)

    rewards_per_agent = {a: [] for a in agents}
    for step_i in range(10):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        _, rew, terms, truncs, _ = env.step(actions)
        for a in agents:
            if a in rew:
                rewards_per_agent[a].append(float(rew[a]))
        if any(terms.values()) or any(truncs.values()):
            break
    env.close()

    varied = any(len(set(rs)) > 1 for rs in rewards_per_agent.values() if rs)
    distinct_counts = {a: len(set(rs)) for a, rs in rewards_per_agent.items()}
    assert varied, (
        f"{name}: reward was constant across 10 random-action steps for every "
        f"agent (distinct-value counts: {distinct_counts}). Likely a stale-cache "
        f"bug in scenario reward / global_reward."
    )


def test_simple_tag_reward_responds_via_bound_penalty():
    """SimpleTag's reward is sparse — only fires on collision or out-of-bounds.

    Force the good agent to fly out of bounds with sustained right-thrust, then
    verify its reward changes from 0 to negative as bound() penalty kicks in.
    """
    env = simple_tag_env(max_cycles=40)
    env.reset(seed=42)
    good_rewards = []
    for _ in range(40):
        # All-right actions; good agent is faster, will exit bounds first.
        _, rew, _, truncs, _ = env.step({a: 2 for a in env.agents})
        if "agent_0" in rew:
            good_rewards.append(float(rew["agent_0"]))
        if any(truncs.values()):
            break
    env.close()
    assert len(set(good_rewards)) > 1, (
        f"SimpleTag good_agent reward stayed constant at {good_rewards[:3]}... "
        f"across {len(good_rewards)} steps of sustained right-thrust. "
        f"Expected bound() penalty to engage as agent exits [-0.9, 0.9]."
    )


def test_simple_crypto_reward_responds():
    """SimpleCrypto is pure-comm: eve's reward depends on alice's broadcast.

    Goal is a random one-hot; eve's reward = -||eve.c - goal||^2. With eve sending
    a fixed comm word (action=0 sets eve.c[0]=1), her reward is constant. But
    when alice varies, eve sees alice's comm in her obs — different game.
    To test the env actually responds, vary EVE's comm action across 4 distinct
    values and assert her reward varies (since she'll match the goal for one
    value and miss for others).
    """
    env = simple_crypto_env(max_cycles=10)
    env.reset(seed=42)
    rews = []
    for k in range(4):
        # Vary eve's broadcast — her reward varies based on whether eve.c matches goal.
        _, rew, _, _, _ = env.step({"eve_0": k, "bob_0": 0, "alice_0": 0})
        rews.append(rew["eve_0"])
    env.close()
    assert len(set(rews)) > 1, (
        f"SimpleCrypto eve reward was constant across 4 distinct eve comm words: "
        f"{rews}. Expected variation because eve's reward = -||eve.c - goal||^2 "
        f"changes with eve.c."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
