#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

#include "core/entity.h"
#include "simple_tag_scenario.h"

namespace cpp_pettingzoo::simple_tag {

SimpleTagScenario::SimpleTagScenario(int num_good, int num_adversaries,
                                     int num_obstacles, bool curriculum,
                                     bool terminate_on_success,
                                     std::optional<int> num_agent_neighbors,
                                     std::optional<int> num_landmark_neighbors)
    : num_good_(num_good),
      num_adversaries_(num_adversaries),
      num_obstacles_(num_obstacles),
      curriculum_(curriculum),
      curriculum_stage_(0),
      terminate_on_success_(terminate_on_success),
      num_agent_neighbors_(num_agent_neighbors),
      num_landmark_neighbors_(num_landmark_neighbors) {}

void SimpleTagScenario::make_world(core::World& w) {
  w.dim_c = 2;
  w.agents.reserve(num_adversaries_ + num_good_);

  for (int i = 0; i < num_adversaries_; ++i) {
    core::Agent adversary =
        core::Agent("adversary_" + std::to_string(i), w.dim_c);
    adversary.adversary = true;
    adversary.collide = true;
    adversary.silent = true;
    adversary.size = 0.075f;
    adversary.accel = 3.0f;
    adversary.max_speed = 1.0f;
    w.agents.push_back(std::move(adversary));
  }

  for (int i = 0; i < num_good_; ++i) {
    core::Agent good = core::Agent("agent_" + std::to_string(i), w.dim_c);
    good.adversary = false;
    good.collide = true;
    good.silent = true;
    good.size = 0.05f;
    good.accel = 4.0f;
    good.max_speed = 1.3f;
    w.agents.push_back(std::move(good));
  }

  w.landmarks.reserve(num_obstacles_);
  for (int i = 0; i < num_obstacles_; ++i) {
    core::Landmark l = core::Landmark("landmark " + std::to_string(i));
    l.collide = true;
    l.movable = false;
    l.size = 0.2f;
    l.boundary = false;
    w.landmarks.push_back(std::move(l));
  }
}

void SimpleTagScenario::reset_world(core::World& w) {
  auto& rng = w.get_rng();
  auto dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
  auto obs_dist = std::uniform_real_distribution<float>(-0.9f, 0.9f);

  for (auto& agent : w.agents) {
    agent.color = agent.adversary ? std::array<float, 3>{0.85f, 0.35f, 0.35f}
                                  : std::array<float, 3>{0.35f, 0.85f, 0.35f};
    agent.state.p_pos = {dist(rng), dist(rng)};
    agent.state.p_vel = {0.0f, 0.0f};
    agent.c = std::vector<float>(w.dim_c, 0.0f);
  }

  for (auto& landmark : w.landmarks) {
    landmark.color = {0.25f, 0.25f, 0.25f};
    if (!landmark.boundary) {
      landmark.state.p_pos = {obs_dist(rng), obs_dist(rng)};
      landmark.state.p_vel = {0.0f, 0.0f};
    }
  }

  if (curriculum_) {
    static constexpr float stage_factors[3] = {0.5f, 0.75f, 1.0f};
    float factor = stage_factors[curriculum_stage_];
    for (auto& agent : w.agents) {
      if (!agent.adversary) {
        agent.max_speed = PREY_BASE_MAX_SPEED * factor;
        agent.accel = PREY_BASE_ACCEL * factor;
      }
    }
  }
}

bool SimpleTagScenario::is_collision(const core::Agent& a,
                                     const core::Agent& b) const {
  float dx = a.state.p_pos[0] - b.state.p_pos[0];
  float dy = a.state.p_pos[1] - b.state.p_pos[1];
  return std::sqrt(dx * dx + dy * dy) < a.size + b.size;
}

float SimpleTagScenario::bound(float x) const {
  if (x < 0.9f) return 0.0f;
  if (x < 1.0f) return (x - 0.9f) * 10.0f;
  return std::min(std::exp(2.0f * x - 2.0f), 10.0f);
}

std::vector<const core::Agent*> SimpleTagScenario::good_agents(
    const core::World& world) const {
  std::vector<const core::Agent*> result;
  for (const auto& agent : world.agents) {
    if (!agent.adversary) result.push_back(&agent);
  }
  return result;
}

std::vector<const core::Agent*> SimpleTagScenario::adversaries(
    const core::World& world) const {
  std::vector<const core::Agent*> result;
  for (const auto& agent : world.agents) {
    if (agent.adversary) result.push_back(&agent);
  }
  return result;
}

float SimpleTagScenario::agent_reward(const core::Agent& agent,
                                      const core::World& world) const {
  float rew = 0.0f;
  if (agent.collide) {
    for (const auto* adv : adversaries(world)) {
      if (is_collision(agent, *adv)) rew -= 10.0f;
    }
  }
  for (int p = 0; p < 2; ++p) {
    rew -= bound(std::abs(agent.state.p_pos[p]));
  }
  return rew;
}

float SimpleTagScenario::adversary_reward(const core::Agent& agent,
                                          const core::World& world) const {
  float rew = 0.0f;
  if (agent.collide) {
    for (const auto* good : good_agents(world)) {
      for (const auto* adv : adversaries(world)) {
        if (is_collision(*good, *adv)) rew += 10.0f;
      }
    }
  }
  return rew;
}

float SimpleTagScenario::reward(const core::Agent& agent,
                                const core::World& world) const {
  return agent.adversary ? adversary_reward(agent, world)
                         : agent_reward(agent, world);
}

bool SimpleTagScenario::is_terminal(const core::World& world) const {
  if (!terminate_on_success_) return false;
  auto advs = adversaries(world);
  for (const auto* prey : good_agents(world)) {
    bool caught = false;
    for (const auto* adv : advs) {
      if (is_collision(*prey, *adv)) {
        caught = true;
        break;
      }
    }
    if (!caught) return false;
  }
  return true;
}

std::vector<size_t> SimpleTagScenario::nearest_indices(
    const core::Agent& agent,
    const std::vector<const core::Entity*>& entities,
    std::optional<int> n) const {
  std::vector<size_t> indices(entities.size());
  std::iota(indices.begin(), indices.end(), 0);
  if (!n.has_value()) return indices;

  std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
    float dxa = entities[a]->state.p_pos[0] - agent.state.p_pos[0];
    float dya = entities[a]->state.p_pos[1] - agent.state.p_pos[1];
    float dxb = entities[b]->state.p_pos[0] - agent.state.p_pos[0];
    float dyb = entities[b]->state.p_pos[1] - agent.state.p_pos[1];
    return (dxa * dxa + dya * dya) < (dxb * dxb + dyb * dyb);
  });

  size_t count = std::min(static_cast<size_t>(n.value()), indices.size());
  indices.resize(count);
  return indices;
}

std::vector<float> SimpleTagScenario::padded_relative_positions(
    const core::Agent& agent,
    const std::vector<const core::Entity*>& entities,
    std::optional<int> n) const {
  std::vector<float> result;
  auto indices = nearest_indices(agent, entities, n);

  for (size_t i : indices) {
    result.push_back(entities[i]->state.p_pos[0] - agent.state.p_pos[0]);
    result.push_back(entities[i]->state.p_pos[1] - agent.state.p_pos[1]);
  }

  if (n.has_value()) {
    size_t padding = static_cast<size_t>(n.value()) - indices.size();
    result.insert(result.end(), padding * 2, 0.0f);
  }
  return result;
}

std::vector<float> SimpleTagScenario::padded_good_agent_velocities(
    const core::Agent& agent, const std::vector<const core::Agent*>& others,
    std::optional<int> n) const {
  std::vector<float> result;

  if (!n.has_value()) {
    // Full observability: only include good agents' velocities
    for (const auto* other : others) {
      if (!other->adversary) {
        result.push_back(other->state.p_vel[0]);
        result.push_back(other->state.p_vel[1]);
      }
    }
    return result;
  }

  // PO path: fixed n slots aligned with position output
  std::vector<const core::Entity*> entities(others.begin(), others.end());
  auto indices = nearest_indices(agent, entities, n);

  for (size_t i : indices) {
    if (!others[i]->adversary) {
      result.push_back(others[i]->state.p_vel[0]);
      result.push_back(others[i]->state.p_vel[1]);
    } else {
      result.push_back(0.0f);
      result.push_back(0.0f);
    }
  }

  size_t padding = static_cast<size_t>(n.value()) - indices.size();
  result.insert(result.end(), padding * 2, 0.0f);
  return result;
}

std::vector<float> SimpleTagScenario::observation(
    const core::Agent& agent, const core::World& world) const {
  std::vector<float> obs;
  obs.push_back(agent.state.p_vel[0]);
  obs.push_back(agent.state.p_vel[1]);
  obs.push_back(agent.state.p_pos[0]);
  obs.push_back(agent.state.p_pos[1]);

  // Non-boundary landmark positions
  std::vector<const core::Entity*> obstacles;
  for (const auto& lm : world.landmarks) {
    if (!lm.boundary) obstacles.push_back(&lm);
  }
  auto lm_pos = padded_relative_positions(agent, obstacles, num_landmark_neighbors_);
  obs.insert(obs.end(), lm_pos.begin(), lm_pos.end());

  // Other agent positions
  std::vector<const core::Agent*> others;
  std::vector<const core::Entity*> others_as_entities;
  for (const auto& other : world.agents) {
    if (&other != &agent) {
      others.push_back(&other);
      others_as_entities.push_back(&other);
    }
  }
  auto other_pos = padded_relative_positions(agent, others_as_entities, num_agent_neighbors_);
  obs.insert(obs.end(), other_pos.begin(), other_pos.end());

  // Good agent velocities
  auto good_vel = padded_good_agent_velocities(agent, others, num_agent_neighbors_);
  obs.insert(obs.end(), good_vel.begin(), good_vel.end());

  return obs;
}

int SimpleTagScenario::get_curriculum_stage() const {
  return curriculum_stage_;
}

void SimpleTagScenario::advance_curriculum() {
  if (curriculum_stage_ < 2) curriculum_stage_++;
}

void SimpleTagScenario::set_curriculum_stage(int stage) {
  curriculum_stage_ = std::max(0, std::min(stage, 2));
}

}  // namespace cpp_pettingzoo::simple_tag
