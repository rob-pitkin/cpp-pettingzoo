#include <algorithm>
#include <random>

#include "simple_crypto_scenario.h"

namespace cpp_pettingzoo::simple_crypto {

// Agent role convention (matches mpe2 indexing):
//   index 0: eve   (adversary)
//   index 1: bob   (good listener)
//   index 2: alice (good speaker)
constexpr int EVE_IDX = 0;
constexpr int BOB_IDX = 1;
constexpr int ALICE_IDX = 2;
constexpr int NUM_LANDMARKS = 2;

SimpleCryptoScenario::SimpleCryptoScenario() {}

void SimpleCryptoScenario::make_world(core::World& w) {
  w.dim_c = 4;

  // 3 agents — all immovable, non-colliding. Only difference is role.
  w.agents.reserve(3);

  core::Agent eve("eve_0", w.dim_c);
  eve.adversary = true;
  eve.collide = false;
  eve.movable = false;
  eve.silent = false;  // Eve speaks (her comm is checked vs goal)
  w.agents.push_back(std::move(eve));

  core::Agent bob("bob_0", w.dim_c);
  bob.adversary = false;
  bob.collide = false;
  bob.movable = false;
  bob.silent = false;  // Bob speaks (his comm is checked vs goal)
  w.agents.push_back(std::move(bob));

  core::Agent alice("alice_0", w.dim_c);
  alice.adversary = false;
  alice.collide = false;
  alice.movable = false;
  alice.silent = false;  // Alice speaks (encrypted message)
  w.agents.push_back(std::move(alice));

  // 2 landmarks — also immovable, non-colliding. Used only as comm-color carriers.
  w.landmarks.reserve(NUM_LANDMARKS);
  for (int i = 0; i < NUM_LANDMARKS; ++i) {
    core::Landmark lm("landmark " + std::to_string(i));
    lm.collide = false;
    lm.movable = false;
    w.landmarks.push_back(std::move(lm));
  }

  // Precompute the per-landmark one-hot comm colors (length dim_c).
  landmark_comm_colors_.resize(NUM_LANDMARKS,
                               std::vector<float>(w.dim_c, 0.0f));
  for (int i = 0; i < NUM_LANDMARKS; ++i) {
    landmark_comm_colors_[i][i] = 1.0f;
  }

  key_.assign(w.dim_c, 0.0f);
}

void SimpleCryptoScenario::reset_world(core::World& w) {
  auto& rng = w.get_rng();
  auto pos_dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
  std::uniform_int_distribution<int> lm_idx(0, NUM_LANDMARKS - 1);

  // Choose goal landmark and speaker key (independent draws — mpe2 does the same).
  goal_index_ = lm_idx(rng);
  int key_index = lm_idx(rng);
  key_ = landmark_comm_colors_[key_index];

  for (auto& agent : w.agents) {
    agent.color = agent.adversary ? std::array<float, 3>{0.75f, 0.25f, 0.25f}
                                  : std::array<float, 3>{0.25f, 0.25f, 0.25f};
    agent.state.p_pos = {pos_dist(rng), pos_dist(rng)};
    agent.state.p_vel = {0.0f, 0.0f};
    agent.c = std::vector<float>(w.dim_c, 0.0f);
  }
  // Bob takes the goal landmark's color as his render color (mpe2 line 183).
  // Truncate the dim_c-length one-hot to 3 channels for the render color slot.
  const auto& goal_color = landmark_comm_colors_[goal_index_];
  w.agents[BOB_IDX].color = {goal_color[0], goal_color[1], goal_color[2]};

  for (int i = 0; i < NUM_LANDMARKS; ++i) {
    auto& lm = w.landmarks[i];
    const auto& cc = landmark_comm_colors_[i];
    lm.color = {cc[0], cc[1], cc[2]};
    lm.state.p_pos = {pos_dist(rng), pos_dist(rng)};
    lm.state.p_vel = {0.0f, 0.0f};
  }
}

bool SimpleCryptoScenario::is_comm_zero(const std::vector<float>& c) const {
  for (float v : c) {
    if (v != 0.0f) return false;
  }
  return true;
}

float SimpleCryptoScenario::reward(const core::Agent& agent,
                                   const core::World& world) const {
  const auto& goal = landmark_comm_colors_[goal_index_];
  const auto& alice = world.agents[ALICE_IDX];
  const auto& bob = world.agents[BOB_IDX];
  const auto& eve = world.agents[EVE_IDX];

  if (agent.adversary) {
    // Eve: penalized by squared distance between her comm and the goal color
    // (only if she actually said something).
    float rew = 0.0f;
    if (!is_comm_zero(eve.c)) {
      for (size_t k = 0; k < eve.c.size(); ++k) {
        float d = eve.c[k] - goal[k];
        rew -= d * d;
      }
    }
    return rew;
  }

  // Good agents (Alice + Bob) share a reward:
  //   good_rew = -sum_square(bob.c - goal)            if bob spoke
  //   adv_rew  = +sum_square(eve.c - goal)            if eve spoke
  //   return adv_rew + good_rew
  float good_rew = 0.0f;
  if (!is_comm_zero(bob.c)) {
    for (size_t k = 0; k < bob.c.size(); ++k) {
      float d = bob.c[k] - goal[k];
      good_rew -= d * d;
    }
  }
  float adv_rew = 0.0f;
  if (!is_comm_zero(eve.c)) {
    for (size_t k = 0; k < eve.c.size(); ++k) {
      float d = eve.c[k] - goal[k];
      adv_rew += d * d;
    }
  }
  return adv_rew + good_rew;
}

std::vector<float> SimpleCryptoScenario::observation(
    const core::Agent& agent, const core::World& world) const {
  const auto& goal = landmark_comm_colors_[goal_index_];
  const auto& alice_c = world.agents[ALICE_IDX].c;
  bool is_speaker = (&agent == &world.agents[ALICE_IDX]);
  bool is_adversary = agent.adversary;

  if (is_speaker) {
    // Alice: [goal_color, key]  -> 4 + 4 = 8
    std::vector<float> obs;
    obs.reserve(goal.size() + key_.size());
    obs.insert(obs.end(), goal.begin(), goal.end());
    obs.insert(obs.end(), key_.begin(), key_.end());
    return obs;
  }
  if (!is_adversary) {
    // Bob: [key, alice.c]  -> 4 + 4 = 8
    std::vector<float> obs;
    obs.reserve(key_.size() + alice_c.size());
    obs.insert(obs.end(), key_.begin(), key_.end());
    obs.insert(obs.end(), alice_c.begin(), alice_c.end());
    return obs;
  }
  // Eve: [alice.c]  -> 4
  return alice_c;
}

}  // namespace cpp_pettingzoo::simple_crypto
