#include <algorithm>
#include <cassert>
#include <random>
#include <stdexcept>

#include "simple_core.h"

namespace cpp_pettingzoo {

SimpleEnv::SimpleEnv(int max_cycles, bool dynamic_rescaling,
                     bool continuous_actions)
    : max_cycles_(max_cycles),
      timesteps_(0),
      has_reset_(false),
      dynamic_rescaling_(dynamic_rescaling),
      continuous_actions_(continuous_actions),
      agents_{"agent_0"} {}

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

std::vector<float> SimpleEnv::get_state() const {
  if (!has_reset_) {
    throw std::runtime_error("reset() must be called before state()");
  }
  return scenario_.observation(world_.agents[0], world_);
}

ObservationMap SimpleEnv::reset(std::optional<int> seed) {
  if (seed.has_value()) {
    world_ = core::World(seed.value());
  } else {
    world_ = core::World();
  }

  scenario_.make_world(world_);
  scenario_.reset_world(world_);
  timesteps_ = 0;
  has_reset_ = true;
  agents_ = {"agent_0"};  // Reset active agents

  return {{"agent_0", scenario_.observation(world_.agents[0], world_)}};
}

State SimpleEnv::step(const ActionMap& actions) {
  if (!has_reset_) {
    throw std::runtime_error("reset() must be called before step()");
  }

  // If episode is already done, clear agents and return current state
  if (timesteps_ >= max_cycles_) {
    agents_.clear();
    return {{{"agent_0", scenario_.observation(world_.agents[0], world_)}},
            {{"agent_0", scenario_.reward(world_.agents[0], world_)}},
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

  world_.agents[0].action.u = force;
  world_.step();

  timesteps_++;

  // Clear agents if episode ends
  if (timesteps_ >= max_cycles_) {
    agents_.clear();
  }

  return {{{"agent_0", scenario_.observation(world_.agents[0], world_)}},
          {{"agent_0", scenario_.reward(world_.agents[0], world_)}},
          {{"agent_0", false}},
          {{"agent_0", timesteps_ >= max_cycles_}}};
}

std::vector<std::string> SimpleEnv::get_agents() const { return agents_; }

std::array<float, 2> SimpleEnv::action_to_force_continuous(
    const std::vector<float>& action) const {
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
  return {{"agent_pos", world_.agents[0].state.p_pos},
          {"agent_vel", world_.agents[0].state.p_vel},
          {"landmark_pos", world_.landmarks[0].state.p_pos}};
}

}  // namespace cpp_pettingzoo
