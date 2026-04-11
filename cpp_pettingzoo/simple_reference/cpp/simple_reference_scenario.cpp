#include <cmath>
#include <random>

#include "core/entity.h"
#include "simple_reference_scenario.h"

namespace cpp_pettingzoo::simple_reference {

void SimpleReferenceScenario::make_world(core::World& w) {
  w.dim_c = 10;

  for (int i = 0; i < 2; ++i) {
    core::Agent a = core::Agent("agent_" + std::to_string(i), w.dim_c);
    a.size = 0.05f;
    a.silent = false;
    a.collide = false;
    a.color = {0.25f, 0.25f, 0.25f};
    w.agents.push_back(std::move(a));
  }

  for (int i = 0; i < 3; ++i) {
    core::Landmark l = core::Landmark("landmark " + std::to_string(i));
    l.size = 0.05f;
    l.collide = false;
    l.movable = false;
    if (i == 0)
      l.color = {0.75f, 0.25f, 0.25f};  // red
    else if (i == 1)
      l.color = {0.25f, 0.75f, 0.25f};  // green
    else
      l.color = {0.25f, 0.25f, 0.75f};  // blue
    w.landmarks.push_back(std::move(l));
  }
}

void SimpleReferenceScenario::reset_world(core::World& w) {
  w.agents[0].goal_a = &w.agents[1];
  w.agents[1].goal_a = &w.agents[0];

  auto& rng = w.get_rng();
  auto dist = std::uniform_int_distribution<>(0, 2);

  w.agents[0].goal_b = &w.landmarks[dist(rng)];
  w.agents[1].goal_b = &w.landmarks[dist(rng)];

  auto float_dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
  w.agents[0].state.p_pos = {float_dist(rng), float_dist(rng)};
  w.agents[0].state.p_vel = {0.0f, 0.0f};
  w.agents[0].c = std::vector<float>(w.dim_c, 0.0f);
  w.agents[0].color = w.agents[1].goal_b->color;

  w.agents[1].state.p_pos = {float_dist(rng), float_dist(rng)};
  w.agents[1].state.p_vel = {0.0f, 0.0f};
  w.agents[1].c = std::vector<float>(w.dim_c, 0.0f);
  w.agents[1].color = w.agents[0].goal_b->color;

  for (int i = 0; i < 3; ++i) {
    auto& landmark = w.landmarks[i];
    landmark.state.p_pos = {float_dist(rng), float_dist(rng)};
    landmark.state.p_vel = {0.0f, 0.0f};
  }
}

float SimpleReferenceScenario::reward(const core::Agent& agent,
                                      const core::World& world) const {
  if (agent.goal_a == nullptr || agent.goal_b == nullptr) {
    return 0.0f;
  }

  const auto& goal_a_pos = agent.goal_a->state.p_pos;
  const auto& goal_b_pos = agent.goal_b->state.p_pos;

  float dx = goal_a_pos[0] - goal_b_pos[0];
  float dy = goal_a_pos[1] - goal_b_pos[1];
  return -std::sqrt(dx * dx + dy * dy);
}

float SimpleReferenceScenario::global_reward(const core::World& w) const {
  float glob_reward = 0.0f;
  for (const auto& agent : w.agents) {
    glob_reward += reward(agent, w);
  }
  return glob_reward / static_cast<float>(w.agents.size());
}

std::vector<float> SimpleReferenceScenario::observation(
    const core::Agent& agent, const core::World& world) const {
  // TODO(human): Implement observation
  // Concatenate:
  // - agent velocity (2 elements)
  // - relative positions of all 3 landmarks (6 elements)
  // - goal color (3 elements) - agent.goal_b->color if not null, else zeros
  // - communication from other agent (10 elements)
  // Total: 21 elements
  std::vector<float> obs;
  obs.reserve(21);
  obs.push_back(agent.state.p_vel[0]);
  obs.push_back(agent.state.p_vel[1]);
  for (const auto& landmark : world.landmarks) {
    float rel_x = landmark.state.p_pos[0] - agent.state.p_pos[0];
    float rel_y = landmark.state.p_pos[1] - agent.state.p_pos[1];
    obs.push_back(rel_x);
    obs.push_back(rel_y);
  }

  if (agent.goal_b != nullptr) {
    obs.insert(obs.end(), agent.goal_b->color.begin(),
               agent.goal_b->color.end());
  } else {
    for (int i = 0; i < 3; ++i) {
      obs.push_back(0.0f);
    }
  }

  if (agent.goal_a != nullptr) {
    obs.insert(obs.end(), static_cast<core::Agent*>(agent.goal_a)->c.begin(),
               static_cast<core::Agent*>(agent.goal_a)->c.end());
  } else {
    obs.insert(obs.end(), world.dim_c, 0.0f);
  }
  return obs;
}

}  // namespace cpp_pettingzoo::simple_reference
