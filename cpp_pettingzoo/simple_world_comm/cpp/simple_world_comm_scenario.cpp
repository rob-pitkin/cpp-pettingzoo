#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "simple_world_comm_scenario.h"

namespace cpp_pettingzoo::simple_world_comm {

SimpleWorldCommScenario::SimpleWorldCommScenario(int num_good,
                                                 int num_adversaries,
                                                 int num_obstacles,
                                                 int num_food, int num_forests)
    : num_good_(num_good),
      num_adversaries_(num_adversaries),
      num_obstacles_(num_obstacles),
      num_food_(num_food),
      num_forests_(num_forests) {}

void SimpleWorldCommScenario::make_world(core::World& w) {
  w.dim_c = 4;
  int num_agents = num_adversaries_ + num_good_;
  w.agents.reserve(num_agents);

  for (int i = 0; i < num_agents; ++i) {
    bool adversary = i < num_adversaries_;
    int base_index = adversary ? std::max(i - 1, 0) : i - num_adversaries_;
    std::string base_name =
        (i == 0) ? "leadadversary" : (adversary ? "adversary" : "agent");
    std::string name = base_name + "_" + std::to_string(base_index);

    core::Agent agent(name, w.dim_c);
    agent.adversary = adversary;
    agent.leader = (i == 0);
    agent.silent = (i != 0);  // only the leader speaks
    agent.collide = true;
    agent.size = adversary ? 0.075f : 0.045f;
    agent.accel = adversary ? 3.0f : 4.0f;
    agent.max_speed = adversary ? 1.0f : 1.3f;
    w.agents.push_back(std::move(agent));
  }

  int total_landmarks = num_obstacles_ + num_food_ + num_forests_;
  w.landmarks.reserve(total_landmarks);

  for (int i = 0; i < num_obstacles_; ++i) {
    core::Landmark lm("landmark " + std::to_string(i));
    lm.collide = true;
    lm.movable = false;
    lm.size = 0.2f;
    lm.boundary = false;
    lm.subtype = LANDMARK_OBSTACLE;
    w.landmarks.push_back(std::move(lm));
  }
  for (int i = 0; i < num_food_; ++i) {
    core::Landmark lm("food " + std::to_string(i));
    lm.collide = false;
    lm.movable = false;
    lm.size = 0.03f;
    lm.boundary = false;
    lm.subtype = LANDMARK_FOOD;
    w.landmarks.push_back(std::move(lm));
  }
  for (int i = 0; i < num_forests_; ++i) {
    core::Landmark lm("forest " + std::to_string(i));
    lm.collide = false;
    lm.movable = false;
    lm.size = 0.3f;
    lm.boundary = false;
    lm.subtype = LANDMARK_FOREST;
    w.landmarks.push_back(std::move(lm));
  }
}

void SimpleWorldCommScenario::reset_world(core::World& w) {
  auto& rng = w.get_rng();
  auto agent_dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
  auto lm_dist = std::uniform_real_distribution<float>(-0.9f, 0.9f);

  for (auto& agent : w.agents) {
    if (!agent.adversary) {
      agent.color = {0.45f, 0.95f, 0.45f};
    } else if (agent.leader) {
      agent.color = {0.65f, 0.15f, 0.15f};
    } else {
      agent.color = {0.95f, 0.45f, 0.45f};
    }
    agent.state.p_pos = {agent_dist(rng), agent_dist(rng)};
    agent.state.p_vel = {0.0f, 0.0f};
    agent.c = std::vector<float>(w.dim_c, 0.0f);
  }

  for (auto& lm : w.landmarks) {
    switch (lm.subtype) {
      case LANDMARK_OBSTACLE:
        lm.color = {0.25f, 0.25f, 0.25f};
        break;
      case LANDMARK_FOOD:
        lm.color = {0.15f, 0.15f, 0.65f};
        break;
      case LANDMARK_FOREST:
        lm.color = {0.6f, 0.9f, 0.6f};
        break;
    }
    lm.state.p_pos = {lm_dist(rng), lm_dist(rng)};
    lm.state.p_vel = {0.0f, 0.0f};
  }
}

bool SimpleWorldCommScenario::is_collision(const core::Entity& a,
                                           const core::Entity& b) const {
  float dx = a.state.p_pos[0] - b.state.p_pos[0];
  float dy = a.state.p_pos[1] - b.state.p_pos[1];
  return std::sqrt(dx * dx + dy * dy) < a.size + b.size;
}

float SimpleWorldCommScenario::bound(float x) const {
  if (x < 0.9f) return 0.0f;
  if (x < 1.0f) return (x - 0.9f) * 10.0f;
  return std::min(std::exp(2.0f * x - 2.0f), 10.0f);
}

std::vector<const core::Agent*> SimpleWorldCommScenario::good_agents(
    const core::World& world) const {
  std::vector<const core::Agent*> result;
  for (const auto& a : world.agents) {
    if (!a.adversary) result.push_back(&a);
  }
  return result;
}

std::vector<const core::Agent*> SimpleWorldCommScenario::adversaries(
    const core::World& world) const {
  std::vector<const core::Agent*> result;
  for (const auto& a : world.agents) {
    if (a.adversary) result.push_back(&a);
  }
  return result;
}

float SimpleWorldCommScenario::agent_reward(const core::Agent& agent,
                                            const core::World& world) const {
  float rew = 0.0f;

  if (agent.collide) {
    for (const auto* adv : adversaries(world)) {
      if (is_collision(agent, *adv)) rew -= 5.0f;
    }
  }

  for (int p = 0; p < 2; ++p) {
    rew -= 2.0f * bound(std::abs(agent.state.p_pos[p]));
  }

  float min_food_dist = std::numeric_limits<float>::infinity();
  for (const auto& lm : world.landmarks) {
    if (lm.subtype != LANDMARK_FOOD) continue;
    if (is_collision(agent, lm)) rew += 2.0f;
    float dx = lm.state.p_pos[0] - agent.state.p_pos[0];
    float dy = lm.state.p_pos[1] - agent.state.p_pos[1];
    float d = std::sqrt(dx * dx + dy * dy);
    if (d < min_food_dist) min_food_dist = d;
  }
  if (std::isfinite(min_food_dist)) rew -= 0.05f * min_food_dist;

  return rew;
}

float SimpleWorldCommScenario::adversary_reward(
    const core::Agent& agent, const core::World& world) const {
  float rew = 0.0f;
  auto goods = good_agents(world);

  float min_dist = std::numeric_limits<float>::infinity();
  for (const auto* g : goods) {
    float dx = g->state.p_pos[0] - agent.state.p_pos[0];
    float dy = g->state.p_pos[1] - agent.state.p_pos[1];
    float d = std::sqrt(dx * dx + dy * dy);
    if (d < min_dist) min_dist = d;
  }
  if (std::isfinite(min_dist)) rew -= 0.1f * min_dist;

  if (agent.collide) {
    auto advs = adversaries(world);
    for (const auto* g : goods) {
      for (const auto* adv : advs) {
        if (is_collision(*g, *adv)) rew += 5.0f;
      }
    }
  }
  return rew;
}

float SimpleWorldCommScenario::reward(const core::Agent& agent,
                                      const core::World& world) const {
  return agent.adversary ? adversary_reward(agent, world)
                         : agent_reward(agent, world);
}

std::vector<float> SimpleWorldCommScenario::observation(
    const core::Agent& agent, const core::World& world) const {
  std::vector<float> obs;
  obs.reserve(64);

  obs.push_back(agent.state.p_vel[0]);
  obs.push_back(agent.state.p_vel[1]);
  obs.push_back(agent.state.p_pos[0]);
  obs.push_back(agent.state.p_pos[1]);

  // entity_pos: all landmarks (obstacles + food + forests), in world order
  for (const auto& lm : world.landmarks) {
    obs.push_back(lm.state.p_pos[0] - agent.state.p_pos[0]);
    obs.push_back(lm.state.p_pos[1] - agent.state.p_pos[1]);
  }

  // Forest membership for self (used both as visibility filter and in_forest field)
  std::vector<bool> self_in_forest(num_forests_, false);
  bool any_self_in_forest = false;
  {
    int fi = 0;
    for (const auto& lm : world.landmarks) {
      if (lm.subtype != LANDMARK_FOREST) continue;
      bool in_f = is_collision(agent, lm);
      self_in_forest[fi] = in_f;
      if (in_f) any_self_in_forest = true;
      ++fi;
    }
  }

  // other_pos and other_vel (other_vel only includes good agents' vels)
  std::vector<float> other_pos;
  std::vector<float> other_vel;
  other_pos.reserve((world.agents.size() - 1) * 2);
  other_vel.reserve(world.agents.size() * 2);

  for (const auto& other : world.agents) {
    if (&other == &agent) continue;

    bool any_other_in_forest = false;
    bool share_forest = false;
    {
      int fi = 0;
      for (const auto& lm : world.landmarks) {
        if (lm.subtype != LANDMARK_FOREST) continue;
        bool other_in_f = is_collision(other, lm);
        if (other_in_f) any_other_in_forest = true;
        if (self_in_forest[fi] && other_in_f) share_forest = true;
        ++fi;
      }
    }

    bool visible = share_forest ||
                   (!any_self_in_forest && !any_other_in_forest) ||
                   agent.leader;

    if (visible) {
      other_pos.push_back(other.state.p_pos[0] - agent.state.p_pos[0]);
      other_pos.push_back(other.state.p_pos[1] - agent.state.p_pos[1]);
      if (!other.adversary) {
        other_vel.push_back(other.state.p_vel[0]);
        other_vel.push_back(other.state.p_vel[1]);
      }
    } else {
      other_pos.push_back(0.0f);
      other_pos.push_back(0.0f);
      if (!other.adversary) {
        other_vel.push_back(0.0f);
        other_vel.push_back(0.0f);
      }
    }
  }

  // in_forest: +1 / -1 per forest for SELF
  std::vector<float> in_forest_field(num_forests_);
  for (int fi = 0; fi < num_forests_; ++fi) {
    in_forest_field[fi] = self_in_forest[fi] ? 1.0f : -1.0f;
  }

  if (agent.adversary) {
    // [vel, pos, entity_pos, other_pos, other_vel, in_forest, comm]
    obs.insert(obs.end(), other_pos.begin(), other_pos.end());
    obs.insert(obs.end(), other_vel.begin(), other_vel.end());
    obs.insert(obs.end(), in_forest_field.begin(), in_forest_field.end());
    // comm = leader's communication vector only (mpe2 overwrites comm = [agents[0].c])
    const auto& leader_c = world.agents[0].c;
    obs.insert(obs.end(), leader_c.begin(), leader_c.end());
  } else {
    // [vel, pos, entity_pos, other_pos, in_forest, other_vel]
    obs.insert(obs.end(), other_pos.begin(), other_pos.end());
    obs.insert(obs.end(), in_forest_field.begin(), in_forest_field.end());
    obs.insert(obs.end(), other_vel.begin(), other_vel.end());
  }

  return obs;
}

}  // namespace cpp_pettingzoo::simple_world_comm
