#include "simple_scenario.h"

#include <random>

namespace cpp_pettingzoo::simple {

void SimpleScenario::make_world(core::World& w) {
  // Create an agent with size 0.15 and add it to the world
  core::Agent a = core::Agent("agent_0", 0);  // dim_c=0 (no communication)
  a.size = 0.15f;
  a.color = {0.25f, 0.25f, 0.25f};
  a.collide = false;
  a.silent = true;
  w.agents.push_back(std::move(a));

  // Create a landmark with size 0.15 and add it to the world
  core::Landmark l = core::Landmark("landmark_0");
  l.size = 0.15f;
  l.color = {0.75f, 0.25f, 0.25f};
  l.collide = false;
  l.movable = false;
  w.landmarks.push_back(std::move(l));
}

void SimpleScenario::reset_world(core::World& w) {
  core::Agent& agent = w.agents[0];
  core::Landmark& landmark = w.landmarks[0];
  auto& rng = w.get_rng();
  auto dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);

  // Initialize agent
  agent.state.p_pos = {dist(rng), dist(rng)};
  agent.state.p_vel = {0.0f, 0.0f};
  agent.c = std::vector<float>(w.dim_c, 0.0);

  // Initialize landmark
  landmark.state.p_pos = {dist(rng), dist(rng)};
  landmark.state.p_vel = {0.0f, 0.0f};
}

float SimpleScenario::reward(const core::Agent& agent,
                             const core::World& world) const {
  const core::Landmark& landmark = world.landmarks[0];
  float dx = agent.state.p_pos[0] - landmark.state.p_pos[0];
  float dy = agent.state.p_pos[1] - landmark.state.p_pos[1];
  float dist = dx * dx + dy * dy;
  return -dist;
}

std::vector<float> SimpleScenario::observation(const core::Agent& agent,
                                               const core::World& world) const {
  const core::Landmark& landmark = world.landmarks[0];
  float rel_x = landmark.state.p_pos[0] - agent.state.p_pos[0];
  float rel_y = landmark.state.p_pos[1] - agent.state.p_pos[1];
  return {agent.state.p_vel[0], agent.state.p_vel[1], rel_x, rel_y};
}

}  // namespace cpp_pettingzoo::simple
