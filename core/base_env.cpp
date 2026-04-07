#include "base_env.h"

#include <cassert>
#include <stdexcept>

namespace cpp_pettingzoo::core {

BaseEnv::BaseEnv(Scenario& scenario, World& world, int max_cycles,
                     bool dynamic_rescaling, bool continuous_actions,
                     std::optional<float> local_ratio)
    : max_cycles_(max_cycles),
      timesteps_(0),
      has_reset_(false),
      dynamic_rescaling_(dynamic_rescaling),
      continuous_actions_(continuous_actions),
      local_ratio_(local_ratio),
      scenario_(scenario),
      world_(world) {
  // Agent list will be built after make_world() is called by derived class
}

std::array<float, 2> BaseEnv::action_to_force(int action) const {
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

std::vector<float> BaseEnv::get_state() const {
  if (!has_reset_) {
    throw std::runtime_error("reset() must be called before state()");
  }
  // Concatenate all agent observations
  std::vector<float> state;
  for (const auto& agent : world_.agents) {
    auto obs = scenario_.observation(agent, world_);
    state.insert(state.end(), obs.begin(), obs.end());
  }
  return state;
}

ObservationMap BaseEnv::reset(std::optional<int> seed) {
  // Reseed RNG if seed provided
  if (seed.has_value()) {
    world_.reseed(seed.value());
  }

  // Randomize positions
  scenario_.reset_world(world_);
  timesteps_ = 0;
  has_reset_ = true;

  // Reset active agents list
  agents_.clear();
  agents_.reserve(world_.agents.size());
  for (const auto& agent : world_.agents) {
    agents_.push_back(agent.name);
  }

  // Return observations for all agents
  ObservationMap observations;
  for (const auto& agent : world_.agents) {
    observations[agent.name] = scenario_.observation(agent, world_);
  }
  return observations;
}

State BaseEnv::step(const ActionMap& actions) {
  if (!has_reset_) {
    throw std::runtime_error("reset() must be called before step()");
  }

  // If episode is already done, clear agents and return current state
  if (timesteps_ >= max_cycles_) {
    agents_.clear();
    ObservationMap observations;
    RewardMap rewards;
    BoolMap terminations;
    BoolMap truncations;
    for (const auto& agent : world_.agents) {
      observations[agent.name] = scenario_.observation(agent, world_);
      rewards[agent.name] = scenario_.reward(agent, world_);
      terminations[agent.name] = false;
      truncations[agent.name] = true;
    }
    return {observations, rewards, terminations, truncations};
  }

  // Apply actions for all agents
  for (auto& agent : world_.agents) {
    if (actions.find(agent.name) == actions.end()) {
      continue;  // Agent didn't provide action
    }
    std::vector<float> selected_action = actions.at(agent.name);
    std::array<float, 2> force;
    if (continuous_actions_) {
      force = action_to_force_continuous(selected_action);
    } else {
      int action_idx = static_cast<int>(selected_action[0]);
      force = action_to_force(action_idx);
    }
    agent.action.u = force;
  }

  // Step physics
  world_.step();
  timesteps_++;

  // Collect observations and rewards for all agents
  ObservationMap observations;
  RewardMap rewards;
  BoolMap terminations;
  BoolMap truncations;

  for (const auto& agent : world_.agents) {
    observations[agent.name] = scenario_.observation(agent, world_);

    // Calculate reward based on local_ratio
    float reward = scenario_.reward(agent, world_);
    if (local_ratio_.has_value()) {
      // Blend local and global reward
      float global_reward = scenario_.global_reward(world_);
      reward = local_ratio_.value() * reward +
               (1.0f - local_ratio_.value()) * global_reward;
    }
    rewards[agent.name] = reward;

    terminations[agent.name] = false;
    truncations[agent.name] = (timesteps_ >= max_cycles_);
  }

  // Clear agents if episode ends
  if (timesteps_ >= max_cycles_) {
    agents_.clear();
  }

  return {observations, rewards, terminations, truncations};
}

std::vector<std::string> BaseEnv::get_agents() const { return agents_; }

std::array<float, 2> BaseEnv::action_to_force_continuous(
    const std::vector<float>& action) const {
  // action is [no-op, left, right, down, up]
  // force_x = (right - left) * sensitivity
  // force_y = (up - down) * sensitivity
  assert(action.size() == 5 && "Action must be of size 5");
  float force_x = (action[2] - action[1]) * SENSITIVITY;
  float force_y = (action[4] - action[3]) * SENSITIVITY;
  return {force_x, force_y};
}

RenderState BaseEnv::get_render_state() const {
  if (!has_reset_) {
    throw std::runtime_error("reset() must be called before render_state()");
  }
  RenderState state;
  // Add all agent positions and velocities
  for (const auto& agent : world_.agents) {
    state[agent.name + "_pos"] = agent.state.p_pos;
    state[agent.name + "_vel"] = agent.state.p_vel;
  }
  // Add all landmark positions
  for (size_t i = 0; i < world_.landmarks.size(); ++i) {
    state["landmark_" + std::to_string(i) + "_pos"] =
        world_.landmarks[i].state.p_pos;
  }
  return state;
}

}  // namespace cpp_pettingzoo::core
