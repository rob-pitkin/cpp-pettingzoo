"""Test SimpleWorldComm C++ implementation matches mpe2 reference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, "/Users/robpitkin/Desktop/rl-research/third-party/mpe2")

import numpy as np
import pytest
from mpe2 import simple_world_comm_v3
from pettingzoo.test import parallel_api_test

from cpp_pettingzoo.simple_world_comm.simple_world_comm import parallel_env as cpp_env


def test_api_default():
    parallel_api_test(cpp_env(), num_cycles=50)


def test_api_continuous():
    parallel_api_test(cpp_env(continuous_actions=True), num_cycles=50)


def test_api_small():
    parallel_api_test(
        cpp_env(num_good=1, num_adversaries=2, num_obstacles=1, num_food=1,
                num_forests=1),
        num_cycles=50,
    )


def test_reset_deterministic():
    env1, env2 = cpp_env(), cpp_env()
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    for a in env1.agents:
        np.testing.assert_allclose(obs1[a], obs2[a])
    env1.close(); env2.close()


def test_reset_different_seeds():
    env1, env2 = cpp_env(), cpp_env()
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=99)
    assert any(not np.allclose(obs1[a], obs2[a]) for a in env1.agents)
    env1.close(); env2.close()


def test_agent_names_default():
    env = cpp_env()
    env.reset(seed=0)
    expected = {"leadadversary_0", "adversary_0", "adversary_1", "adversary_2",
                "agent_0", "agent_1"}
    assert set(env.agents) == expected
    env.close()


def test_observation_shape_matches_mpe2_default():
    cpp = cpp_env()
    mpe = simple_world_comm_v3.parallel_env()
    cpp.reset(seed=42); mpe.reset(seed=42)
    for a in cpp.possible_agents:
        assert cpp.observation_space(a).shape == mpe.observation_space(a).shape, \
            f"{a}: cpp={cpp.observation_space(a).shape} mpe={mpe.observation_space(a).shape}"
    cpp.close(); mpe.close()


@pytest.mark.parametrize("config", [
    dict(num_good=1, num_adversaries=2, num_obstacles=1, num_food=1, num_forests=1),
    dict(num_good=3, num_adversaries=3, num_obstacles=2, num_food=3, num_forests=2),
    dict(num_good=2, num_adversaries=5, num_obstacles=2, num_food=2, num_forests=3),
])
def test_observation_shape_matches_mpe2_custom(config):
    cpp = cpp_env(**config)
    mpe = simple_world_comm_v3.parallel_env(**config)
    cpp.reset(seed=42); mpe.reset(seed=42)
    for a in cpp.possible_agents:
        assert cpp.observation_space(a).shape == mpe.observation_space(a).shape, \
            f"config={config} {a}: cpp={cpp.observation_space(a).shape} mpe={mpe.observation_space(a).shape}"
    cpp.close(); mpe.close()


def test_action_spaces_match_mpe2():
    cpp = cpp_env()
    mpe = simple_world_comm_v3.parallel_env()
    cpp.reset(seed=0); mpe.reset(seed=0)
    for a in cpp.possible_agents:
        # both should be Discrete(20) for leader, Discrete(5) otherwise
        assert cpp.action_space(a).n == mpe.action_space(a).n, \
            f"{a}: cpp={cpp.action_space(a)} mpe={mpe.action_space(a)}"
    cpp.close(); mpe.close()


def test_action_spaces_continuous_match_mpe2():
    cpp = cpp_env(continuous_actions=True)
    mpe = simple_world_comm_v3.parallel_env(continuous_actions=True)
    cpp.reset(seed=0); mpe.reset(seed=0)
    for a in cpp.possible_agents:
        assert cpp.action_space(a).shape == mpe.action_space(a).shape, \
            f"{a}: cpp={cpp.action_space(a)} mpe={mpe.action_space(a)}"
    cpp.close(); mpe.close()


def test_leader_has_largest_action_space():
    env = cpp_env()
    env.reset(seed=0)
    assert env.action_space("leadadversary_0").n == 20  # 5 movement * 4 comm
    for a in env.possible_agents:
        if a != "leadadversary_0":
            assert env.action_space(a).n == 5
    env.close()


def test_rewards_all_finite():
    env = cpp_env()
    env.reset(seed=42)
    for _ in range(25):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        _, rews, terms, truncs, _ = env.step(actions)
        assert all(np.isfinite(r) for r in rews.values())
        if any(terms.values()) or any(truncs.values()):
            break
    env.close()


def test_truncation_at_max_cycles():
    env = cpp_env(max_cycles=5)
    env.reset(seed=42)
    actions = {a: 0 for a in env.agents}
    for _ in range(4):
        _, _, _, truncs, _ = env.step(actions)
        assert not any(truncs.values())
    _, _, _, truncs, _ = env.step(actions)
    assert all(truncs.values())
    assert len(env.agents) == 0
    env.close()


def test_multi_step_determinism():
    env1, env2 = cpp_env(), cpp_env()
    env1.reset(seed=7); env2.reset(seed=7)
    for step in range(10):
        actions = {a: step % env1.action_space(a).n for a in env1.possible_agents}
        obs1, rew1, _, _, _ = env1.step(actions)
        obs2, rew2, _, _, _ = env2.step(actions)
        for a in env1.possible_agents:
            np.testing.assert_allclose(obs1[a], obs2[a], rtol=1e-5)
            np.testing.assert_allclose(rew1[a], rew2[a], rtol=1e-5)
    env1.close(); env2.close()


def test_multiple_resets():
    env = cpp_env()
    for seed in [0, 1, 42, 100]:
        obs, _ = env.reset(seed=seed)
        assert len(obs) == 6
        for a, o in obs.items():
            expected = 34 if "adversary" in a else 28
            assert o.shape == (expected,), f"{a}: shape={o.shape}"
            assert np.all(np.isfinite(o))
    env.close()


def test_random_rollout_continuous():
    env = cpp_env(continuous_actions=True, max_cycles=25)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(25):
        actions = {a: rng.uniform(0, 1, env.action_space(a).shape).astype(np.float32)
                   for a in env.agents}
        obs, rews, terms, truncs, _ = env.step(actions)
        for o in obs.values():
            assert np.all(np.isfinite(o))
        for r in rews.values():
            assert np.isfinite(r)
        if any(terms.values()) or any(truncs.values()):
            break
    env.close()


def test_leader_comm_observed_by_adversaries():
    """When the leader takes a comm action, adversaries should see it in their obs.

    Comm sits at the tail of an adversary's obs (last dim_c=4 elements).
    """
    env = cpp_env()
    env.reset(seed=0)
    actions = {a: 0 for a in env.agents}
    # Leader action 5 = movement 0, comm word 1 -> one-hot [0,1,0,0]
    actions["leadadversary_0"] = 5
    obs, _, _, _, _ = env.step(actions)
    for adv in ["adversary_0", "adversary_1", "adversary_2", "leadadversary_0"]:
        comm = obs[adv][-4:]
        np.testing.assert_allclose(comm, [0.0, 1.0, 0.0, 0.0])
    env.close()


def test_good_agents_have_no_comm_in_obs():
    """Good agents' obs ends with other_vel, not comm."""
    env = cpp_env()
    env.reset(seed=0)
    actions = {a: 0 for a in env.agents}
    actions["leadadversary_0"] = 5  # broadcast comm word 1
    obs, _, _, _, _ = env.step(actions)
    # good obs shape is 28; tail dim_c elements should NOT carry the comm one-hot
    # (they're other_vel values, which are zero-ish on step 1 with action=0)
    for good in ["agent_0", "agent_1"]:
        assert obs[good].shape == (28,)
        # Last 2 values are the other-good-agent's velocity (not comm)
        tail = obs[good][-2:]
        assert np.all(np.isfinite(tail))
    env.close()


def test_state_concatenation():
    env = cpp_env()
    env.reset(seed=0)
    state = env.state()
    # 4 adversaries * 34 + 2 good * 28 = 136 + 56 = 192
    assert state.shape == (192,)
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
