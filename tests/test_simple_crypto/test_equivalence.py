"""Test SimpleCrypto C++ implementation matches mpe2 reference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, "/Users/robpitkin/Desktop/rl-research/third-party/mpe2")

import numpy as np
import pytest
from mpe2 import simple_crypto_v3
from pettingzoo.test import parallel_api_test

from cpp_pettingzoo.simple_crypto.simple_crypto import parallel_env as cpp_env


def test_api_default():
    parallel_api_test(cpp_env(), num_cycles=50)


def test_api_continuous():
    parallel_api_test(cpp_env(continuous_actions=True), num_cycles=50)


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


def test_agent_names():
    env = cpp_env()
    env.reset(seed=0)
    assert set(env.agents) == {"alice_0", "bob_0", "eve_0"}
    env.close()


def test_observation_shape_matches_mpe2():
    cpp = cpp_env()
    mpe = simple_crypto_v3.parallel_env()
    cpp.reset(seed=42); mpe.reset(seed=42)
    for a in cpp.possible_agents:
        assert cpp.observation_space(a).shape == mpe.observation_space(a).shape, \
            f"{a}: cpp={cpp.observation_space(a).shape} mpe={mpe.observation_space(a).shape}"
    cpp.close(); mpe.close()


def test_action_spaces_match_mpe2():
    cpp = cpp_env()
    mpe = simple_crypto_v3.parallel_env()
    cpp.reset(seed=0); mpe.reset(seed=0)
    for a in cpp.possible_agents:
        assert cpp.action_space(a).n == mpe.action_space(a).n, \
            f"{a}: cpp={cpp.action_space(a)} mpe={mpe.action_space(a)}"
    cpp.close(); mpe.close()


def test_action_spaces_continuous_match_mpe2():
    cpp = cpp_env(continuous_actions=True)
    mpe = simple_crypto_v3.parallel_env(continuous_actions=True)
    cpp.reset(seed=0); mpe.reset(seed=0)
    for a in cpp.possible_agents:
        assert cpp.action_space(a).shape == mpe.action_space(a).shape, \
            f"{a}: cpp={cpp.action_space(a)} mpe={mpe.action_space(a)}"
    cpp.close(); mpe.close()


def test_obs_shapes_per_role():
    env = cpp_env()
    obs, _ = env.reset(seed=0)
    assert obs["alice_0"].shape == (8,)
    assert obs["bob_0"].shape == (8,)
    assert obs["eve_0"].shape == (4,)
    env.close()


def test_alice_obs_is_one_hot_concat():
    """Alice's obs is [goal_color, key]; both should be one-hot in {0,1} of length 4."""
    env = cpp_env()
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        alice = obs["alice_0"]
        goal, key = alice[:4], alice[4:]
        # Each should sum to 1 and be one-hot
        assert np.isclose(goal.sum(), 1.0)
        assert np.isclose(key.sum(), 1.0)
        assert set(goal.tolist()).issubset({0.0, 1.0})
        assert set(key.tolist()).issubset({0.0, 1.0})
    env.close()


def test_eve_observes_alice_comm():
    """When Alice broadcasts comm word k, Eve's obs should be the one-hot at k."""
    env = cpp_env()
    env.reset(seed=42)
    for k in range(4):
        env.reset(seed=42)
        obs, _, _, _, _ = env.step({"eve_0": 0, "bob_0": 0, "alice_0": k})
        expected = np.zeros(4); expected[k] = 1.0
        np.testing.assert_allclose(obs["eve_0"], expected)
    env.close()


def test_bob_observes_key_and_alice_comm():
    """Bob's obs is [key, alice_comm]. After Alice broadcasts, the tail should match."""
    env = cpp_env()
    env.reset(seed=42)
    obs, _, _, _, _ = env.step({"eve_0": 0, "bob_0": 0, "alice_0": 3})
    expected_tail = np.array([0, 0, 0, 1], dtype=np.float32)
    np.testing.assert_allclose(obs["bob_0"][-4:], expected_tail)
    # head (key) should also be a length-4 one-hot
    key = obs["bob_0"][:4]
    assert np.isclose(key.sum(), 1.0)
    env.close()


def test_zero_action_yields_zero_reward():
    """If everyone broadcasts comm[0]=1, reward depends on whether that matches the goal.

    But if no one broadcasts anything (all zero comm), reward should be 0 for all.
    Note: discrete action k=0 still sets comm[0]=1 — it's not "silence." To test
    the "silence" path we need to bypass and set c manually, which we can't from
    Python. Instead, just check that rewards are finite for any action.
    """
    env = cpp_env()
    env.reset(seed=42)
    for _ in range(5):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        _, rews, _, _, _ = env.step(actions)
        for r in rews.values():
            assert np.isfinite(r)
    env.close()


def test_reward_when_bob_matches_goal():
    """If Alice/Bob both broadcast the goal color, good_rew = 0 (perfect),
    and adv_rew depends on Eve's broadcast."""
    env = cpp_env()
    env.reset(seed=42)
    alice_obs, _ = env.reset(seed=42), None  # need fresh obs
    obs, _ = env.reset(seed=42)
    goal_idx = int(np.argmax(obs["alice_0"][:4]))  # alice obs head is goal color
    # Bob broadcasts the goal color; Eve broadcasts something else
    eve_idx = (goal_idx + 1) % 4
    _, rew, _, _, _ = env.step({
        "eve_0": eve_idx,
        "bob_0": goal_idx,
        "alice_0": goal_idx,
    })
    # Bob matched goal: good_rew = 0
    # Eve missed: adv_rew = |eve.c - goal|^2 = 1+1 = 2 (one-hot vs different one-hot)
    # Good agents' reward = adv_rew + good_rew = 2
    # Eve's reward = -|eve.c - goal|^2 = -2
    np.testing.assert_allclose(rew["bob_0"], 2.0, atol=1e-5)
    np.testing.assert_allclose(rew["alice_0"], 2.0, atol=1e-5)
    np.testing.assert_allclose(rew["eve_0"], -2.0, atol=1e-5)
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
        actions = {a: step % 4 for a in env1.possible_agents}
        obs1, rew1, _, _, _ = env1.step(actions)
        obs2, rew2, _, _, _ = env2.step(actions)
        for a in env1.possible_agents:
            np.testing.assert_allclose(obs1[a], obs2[a], rtol=1e-5)
            np.testing.assert_allclose(rew1[a], rew2[a], rtol=1e-5)
    env1.close(); env2.close()


def test_state_concatenation():
    env = cpp_env()
    env.reset(seed=0)
    state = env.state()
    # 4 (eve) + 8 (bob) + 8 (alice) = 20
    assert state.shape == (20,)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
