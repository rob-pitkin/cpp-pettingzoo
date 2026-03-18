#include "simple_core.h"
#include <algorithm>
#include <cassert>
#include <random>
#include <stdexcept>

namespace cpp_pettingzoo {

SimpleEnv::SimpleEnv(int max_cycles, bool dynamic_rescaling,
                     bool continuous_actions)
    : dist_(std::uniform_real_distribution<float>(-1.0, 1.0)),
      has_reset_(false) {
  max_cycles_ = max_cycles;
  timesteps_ = 0;
  dynamic_rescaling_ = dynamic_rescaling;
  continuous_actions_ = continuous_actions;
  agents_ = {"agent_0"}; // Initialize active agents
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

std::vector<float> SimpleEnv::get_state() const {
  if (!has_reset_) {
    throw std::runtime_error("reset() must be called before state()");
  }
  return get_observation();
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
  agents_ = {"agent_0"}; // Reset active agents

  return {{"agent_0", get_observation()}};
}

State SimpleEnv::step(const ActionMap &actions) {
  if (!has_reset_) {
    throw std::runtime_error("reset() must be called before step()");
  }

  // If episode is already done, clear agents and return current state
  if (timesteps_ >= max_cycles_) {
    agents_.clear();
    return {{{"agent_0", get_observation()}},
            {{"agent_0", calculate_reward()}},
            {{"agent_0", false}},
            {{"agent_0", true}}};
  }

  std::vector<float> selected_action = actions.at("agent_0");
  std::array<float, 2> force;
  if (continuous_actions_) {
    force = action_to_force_continuous(selected_action);
  } else {
    int action_idx = static_cast<int>(selected_action[0]);
    force = action_to_force(action_idx);
  }

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

  // Clear agents if episode ends
  if (timesteps_ >= max_cycles_) {
    agents_.clear();
  }

  return {{{"agent_0", get_observation()}},
          {{"agent_0", calculate_reward()}},
          {{"agent_0", false}},
          {{"agent_0", timesteps_ >= max_cycles_}}};
}

std::vector<std::string> SimpleEnv::get_agents() const { return agents_; }

std::array<float, 2>
SimpleEnv::action_to_force_continuous(const std::vector<float> &action) const {
  // action is[no - op, left, right, down, up]
  // force_x = (right - left) * sensitivity
  // force_y = (up - down) * sensitivity
  assert(action.size() == 5 && "Action must be of size 5");
  float force_x = (action[2] - action[1]) * SENSITIVITY;
  float force_y = (action[4] - action[3]) * SENSITIVITY;
  return {force_x, force_y};
}

RenderState SimpleEnv::get_render_state() const {
  if (!has_reset_) {
    throw std::runtime_error("reset() must be called before render_state()");
  }
  return {{"agent_pos", p_pos_},
          {"agent_vel", p_vel_},
          {"landmark_pos", landmark_pos_}};
}

} // namespace cpp_pettingzoo
