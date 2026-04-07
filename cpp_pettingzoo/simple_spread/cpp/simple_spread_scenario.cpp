#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "simple_spread_scenario.h"

namespace cpp_pettingzoo::simple_spread {

SimpleSpreadScenario::SimpleSpreadScenario(bool curriculum_enabled,
                                           int curriculum_stage)
    : curriculum_enabled_(curriculum_enabled),
      curriculum_stage_(curriculum_stage) {
  curriculum_stage_ = std::clamp(curriculum_stage_, 0, 1);
}

void SimpleSpreadScenario::make_world(core::World& w) {
  w.dim_c = 2;

  // Create three agents
  for (int i = 0; i < 3; ++i) {
    core::Agent a = core::Agent("agent_" + std::to_string(i), w.dim_c);
    a.size = 0.15f;
    a.color = {0.35f, 0.35f, 0.85f};
    a.collide = true;
    a.silent = true;
    w.agents.push_back(std::move(a));
  }

  // Create three landmarks
  for (int i = 0; i < 3; ++i) {
    core::Landmark l = core::Landmark("landmark " + std::to_string(i));
    l.size = 0.15f;
    l.color = {0.25f, 0.25f, 0.25f};
    l.collide = false;
    l.movable = false;
    w.landmarks.push_back(std::move(l));
  }
}

void SimpleSpreadScenario::reset_world(core::World& w) {
  auto& rng = w.get_rng();
  auto dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);

  // Init agents
  for (int i = 0; i < 3; ++i) {
    auto& agent = w.agents[i];
    agent.state.p_pos = {dist(rng), dist(rng)};
    agent.state.p_vel = {0.0f, 0.0f};
    agent.c = std::vector<float>(w.dim_c, 0.0);
  }

  for (int i = 0; i < 3; ++i) {
    auto& landmark = w.landmarks[i];
    landmark.state.p_pos = {dist(rng), dist(rng)};
    landmark.state.p_vel = {0.0f, 0.0f};
  }
}

float SimpleSpreadScenario::reward(const core::Agent& a,
                                   const core::World& w) const {
  // Local reward: only collision penalties for this agent
  float reward = 0.0f;

  // Apply collision penalty only if curriculum allows it
  if (!curriculum_enabled_ || curriculum_stage_ == 1) {
    for (const core::Agent& agent : w.agents) {
      if (&a == &agent) {
        continue;
      }
      float dx = a.state.p_pos[0] - agent.state.p_pos[0];
      float dy = a.state.p_pos[1] - agent.state.p_pos[1];
      float agent_dist = std::sqrt(dx * dx + dy * dy);

      // Check if distance between agents is < sum of sizes --> collision
      if (agent_dist < a.size + agent.size) {
        reward -= 1.0f;
      }
    }
  }

  return reward;
}

float SimpleSpreadScenario::global_reward(const core::World& w) const {
  float reward = 0.0f;
  // For each landmark, find minimum distance to any agent
  for (const auto& landmark : w.landmarks) {
    float min_dist = std::numeric_limits<float>::max();
    for (const auto& agent : w.agents) {
      float dx = agent.state.p_pos[0] - landmark.state.p_pos[0];
      float dy = agent.state.p_pos[1] - landmark.state.p_pos[1];
      float dist = dx * dx + dy * dy;
      min_dist = std::min(min_dist, dist);
    }
    reward -= min_dist;
  }
  return reward;
}

std::vector<float> SimpleSpreadScenario::observation(
    const core::Agent& a, const core::World& w) const {
  std::vector<float> obs = {a.state.p_vel[0], a.state.p_vel[1],
                            a.state.p_pos[0], a.state.p_pos[1]};
  for (const core::Landmark& landmark : w.landmarks) {
    float rel_x = landmark.state.p_pos[0] - a.state.p_pos[0];
    float rel_y = landmark.state.p_pos[1] - a.state.p_pos[1];
    obs.push_back(rel_x);
    obs.push_back(rel_y);
  }

  for (const core::Agent& agent : w.agents) {
    if (&agent == &a) {
      continue;
    }
    float rel_x = agent.state.p_pos[0] - a.state.p_pos[0];
    float rel_y = agent.state.p_pos[1] - a.state.p_pos[1];
    obs.push_back(rel_x);
    obs.push_back(rel_y);
    obs.insert(obs.end(), agent.c.begin(), agent.c.end());
  }

  return obs;
}

int SimpleSpreadScenario::get_curriculum_stage() const {
  return curriculum_stage_;
}

void SimpleSpreadScenario::advance_curriculum() {
  if (curriculum_stage_ < 1) {
    curriculum_stage_++;
  }
}

void SimpleSpreadScenario::set_curriculum_stage(int stage) {
  curriculum_stage_ = std::clamp(stage, 0, 1);
}

}  // namespace cpp_pettingzoo::simple_spread
