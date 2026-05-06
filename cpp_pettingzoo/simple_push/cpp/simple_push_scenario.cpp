#include <cmath>
#include <random>

#include "core/entity.h"
#include "simple_push_scenario.h"

namespace cpp_pettingzoo::simple_push {

void SimplePushScenario::make_world(core::World& w) {
  w.dim_c = 2;

  // adversary_0 then agent_0
  w.agents.reserve(2);
  core::Agent adversary("adversary_0", w.dim_c);
  adversary.adversary = true;
  adversary.collide = true;
  adversary.silent = true;
  w.agents.push_back(std::move(adversary));

  core::Agent good("agent_0", w.dim_c);
  good.adversary = false;
  good.collide = true;
  good.silent = true;
  w.agents.push_back(std::move(good));

  // 2 non-colliding, immovable landmarks
  w.landmarks.reserve(2);
  for (int i = 0; i < 2; ++i) {
    core::Landmark lm("landmark " + std::to_string(i));
    lm.collide = false;
    lm.movable = false;
    lm.index = i;
    w.landmarks.push_back(std::move(lm));
  }
}

void SimplePushScenario::reset_world(core::World& w) {
  auto& rng = w.get_rng();
  auto dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);

  // Landmark colors: landmark 0 = dark green tint, landmark 1 = dark blue tint
  // Matches Python: color = [0.1,0.1,0.1]; color[i+1] += 0.8
  w.landmarks[0].color = {0.1f, 0.9f, 0.1f};
  w.landmarks[1].color = {0.1f, 0.1f, 0.9f};

  // Pick a random goal landmark (index 0 or 1)
  int goal_idx =
      std::uniform_int_distribution<int>(0, 1)(rng);
  core::Landmark* goal = &w.landmarks[goal_idx];

  for (auto& agent : w.agents) {
    if (agent.adversary) {
      agent.color = {0.75f, 0.25f, 0.25f};
    } else {
      // Good agent color: [0.25,0.25,0.25] with goal color channel tinted +0.5
      agent.color = {0.25f, 0.25f, 0.25f};
      agent.color[goal_idx + 1] += 0.5f;
    }
    agent.goal_a = goal;
    agent.state.p_pos = {dist(rng), dist(rng)};
    agent.state.p_vel = {0.0f, 0.0f};
    agent.c = std::vector<float>(w.dim_c, 0.0f);
  }

  for (auto& lm : w.landmarks) {
    lm.state.p_pos = {dist(rng), dist(rng)};
    lm.state.p_vel = {0.0f, 0.0f};
  }
}

float SimplePushScenario::agent_reward(const core::Agent& agent,
                                       const core::World&) const {
  const auto& gpos = agent.goal_a->state.p_pos;
  float dx = agent.state.p_pos[0] - gpos[0];
  float dy = agent.state.p_pos[1] - gpos[1];
  return -std::sqrt(dx * dx + dy * dy);
}

float SimplePushScenario::adversary_reward(const core::Agent& agent,
                                           const core::World& world) const {
  // pos_rew = min distance of any good agent to their goal
  float pos_rew = std::numeric_limits<float>::infinity();
  for (const auto& a : world.agents) {
    if (!a.adversary) {
      const auto& gpos = a.goal_a->state.p_pos;
      float dx = a.state.p_pos[0] - gpos[0];
      float dy = a.state.p_pos[1] - gpos[1];
      pos_rew = std::min(pos_rew, std::sqrt(dx * dx + dy * dy));
    }
  }
  // neg_rew = adversary's own distance to the goal
  const auto& gpos = agent.goal_a->state.p_pos;
  float dx = agent.state.p_pos[0] - gpos[0];
  float dy = agent.state.p_pos[1] - gpos[1];
  float neg_rew = std::sqrt(dx * dx + dy * dy);
  return pos_rew - neg_rew;
}

float SimplePushScenario::reward(const core::Agent& agent,
                                 const core::World& world) const {
  return agent.adversary ? adversary_reward(agent, world)
                         : agent_reward(agent, world);
}

std::vector<float> SimplePushScenario::observation(
    const core::Agent& agent, const core::World& world) const {
  std::vector<float> obs;

  // self velocity
  obs.push_back(agent.state.p_vel[0]);
  obs.push_back(agent.state.p_vel[1]);

  if (!agent.adversary) {
    // goal relative position (2)
    obs.push_back(agent.goal_a->state.p_pos[0] - agent.state.p_pos[0]);
    obs.push_back(agent.goal_a->state.p_pos[1] - agent.state.p_pos[1]);
    // self color (3)
    obs.push_back(agent.color[0]);
    obs.push_back(agent.color[1]);
    obs.push_back(agent.color[2]);
  }

  // all landmark relative positions (2 * 2 = 4)
  for (const auto& lm : world.landmarks) {
    obs.push_back(lm.state.p_pos[0] - agent.state.p_pos[0]);
    obs.push_back(lm.state.p_pos[1] - agent.state.p_pos[1]);
  }

  if (!agent.adversary) {
    // all landmark colors (3 * 2 = 6)
    for (const auto& lm : world.landmarks) {
      obs.push_back(lm.color[0]);
      obs.push_back(lm.color[1]);
      obs.push_back(lm.color[2]);
    }
  }

  // other agent relative positions (2 per other agent)
  for (const auto& other : world.agents) {
    if (&other == &agent) continue;
    obs.push_back(other.state.p_pos[0] - agent.state.p_pos[0]);
    obs.push_back(other.state.p_pos[1] - agent.state.p_pos[1]);
  }

  return obs;
}

}  // namespace cpp_pettingzoo::simple_push
