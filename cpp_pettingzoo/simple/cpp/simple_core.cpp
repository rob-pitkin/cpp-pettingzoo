#include "simple_core.h"
#include <algorithm>
#include <cassert>
#include <random>

namespace cpp_pettingzoo {

SimpleEnv::SimpleEnv(std::optional<int> seed, int max_cycles)
    : dist_(std::uniform_real_distribution<float>(-1.0, 1.0)), has_reset_(false) {
  if (seed.has_value()) {
    gen_ = std::mt19937(seed.value());
  } else {
    std::random_device rd;
    gen_ = std::mt19937(rd());
  }
  max_cycles_ = max_cycles;
  timesteps_ = 0;
}

std::array<float, 2> SimpleEnv::action_to_force(int action) const {
  switch (action) {
  case 0:
    return {0.0, 0.0};
  case 1:
    return {-SENSITIVITY, 0.0};
  case 2:
    return {SENSITIVITY, 0.0};
  case 3:
    return {0.0, -SENSITIVITY};
  case 4:
    return {0.0, SENSITIVITY};
  default:
    return {0.0, 0.0};
  }
}

void SimpleEnv::clamp_position() {
  p_pos_[0] = std::clamp(p_pos_[0], -1.0f, 1.0f);
  p_pos_[1] = std::clamp(p_pos_[1], -1.0f, 1.0f);
}

float SimpleEnv::calculate_reward() const {
  float dx = p_pos_[0] - landmark_pos_[0];
  float dy = p_pos_[1] - landmark_pos_[1];
  return -(dx * dx + dy * dy);
}

std::vector<float> SimpleEnv::get_observation() const {
  return {p_vel_[0], p_vel_[1], landmark_pos_[0] - p_pos_[0],
          landmark_pos_[1] - p_pos_[1]};
}

ObservationMap SimpleEnv::reset(std::optional<int> seed) {
  if (seed.has_value()) {
    gen_ = std::mt19937(seed.value());
  } else {
    std::random_device rd;
    gen_ = std::mt19937(rd());
  }

  p_pos_ = {dist_(gen_), dist_(gen_)};
  p_vel_ = {0.0f, 0.0f};
  landmark_pos_ = {dist_(gen_), dist_(gen_)};
  timesteps_ = 0;
  has_reset_ = true;

  return {{"agent_0", get_observation()}};
}

State SimpleEnv::step(const ActionMap &actions) {
  assert(has_reset_ && "reset() must be called before step()");

  // If episode is already done, return current state without computing physics
  if (timesteps_ >= max_cycles_) {
    return {{{"agent_0", get_observation()}},
            {{"agent_0", calculate_reward()}},
            {{"agent_0", false}},
            {{"agent_0", true}}};
  }

  int selected_action = actions.at("agent_0");
  std::array<float, 2> force = action_to_force(selected_action);

  // Update position
  p_pos_[0] += p_vel_[0] * DT;
  p_pos_[1] += p_vel_[1] * DT;

  // Apply damping to velocity
  p_vel_[0] *= (1 - DAMPING);
  p_vel_[1] *= (1 - DAMPING);

  // Apply force to velocity
  p_vel_[0] += (force[0] / MASS) * DT;
  p_vel_[1] += (force[1] / MASS) * DT;

  // Clamp position
  clamp_position();

  timesteps_++;

  return {{{"agent_0", get_observation()}},
          {{"agent_0", calculate_reward()}},
          {{"agent_0", false}},
          {{"agent_0", timesteps_ >= max_cycles_}}};
}

} // namespace cpp_pettingzoo
