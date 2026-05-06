#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "core/entity.h"
#include "core/third_party/munkres/munkres.h"
#include "simple_formation_scenario.h"

namespace cpp_pettingzoo::simple_formation {

SimpleFormationScenario::SimpleFormationScenario(bool terminate_on_success)
    : terminate_on_success_(terminate_on_success) {}

void SimpleFormationScenario::make_world(core::World& w, int N) {
  w.dim_c = 0;
  w.agents.reserve(N);
  for (int i = 0; i < N; ++i) {
    core::Agent agent("agent_" + std::to_string(i), 0);
    agent.collide = true;
    agent.silent = true;
    agent.size = 0.05f;
    w.agents.push_back(std::move(agent));
  }

  core::Landmark lm("landmark_0");
  lm.collide = false;
  lm.movable = false;
  lm.size = 0.03f;
  w.landmarks.push_back(std::move(lm));
}

void SimpleFormationScenario::reset_world(core::World& w) {
  auto& rng = w.get_rng();
  auto agent_dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
  auto lm_dist = std::uniform_real_distribution<float>(-0.5f, 0.5f);

  for (auto& agent : w.agents) {
    agent.color = {0.35f, 0.35f, 0.85f};
    agent.state.p_pos = {agent_dist(rng), agent_dist(rng)};
    agent.state.p_vel = {0.0f, 0.0f};
  }

  w.landmarks[0].color = {0.25f, 0.25f, 0.25f};
  w.landmarks[0].state.p_pos = {lm_dist(rng), lm_dist(rng)};
  w.landmarks[0].state.p_vel = {0.0f, 0.0f};

  cache_valid_ = false;
  joint_reward_ = 0.0f;
  delta_dists_.clear();
}

float SimpleFormationScenario::find_angle(float x, float y) {
  float angle = std::atan2(y, x);
  if (angle < 0.0f) angle += 2.0f * static_cast<float>(M_PI);
  return angle;
}

void SimpleFormationScenario::compute_formation(const core::World& world) const {
  if (cache_valid_) return;

  const int N = static_cast<int>(world.agents.size());
  const float ideal_sep = 2.0f * static_cast<float>(M_PI) / N;
  const auto& lm_pos = world.landmarks[0].state.p_pos;

  // Find theta_min: smallest angle of any agent relative to the landmark
  float theta_min = std::numeric_limits<float>::infinity();
  for (const auto& agent : world.agents) {
    float rx = agent.state.p_pos[0] - lm_pos[0];
    float ry = agent.state.p_pos[1] - lm_pos[1];
    theta_min = std::min(theta_min, find_angle(rx, ry));
  }

  // N ideal target positions evenly spaced on circle, anchored at theta_min
  std::vector<std::array<float, 2>> targets(N);
  for (int i = 0; i < N; ++i) {
    float angle = theta_min + i * ideal_sep;
    targets[i] = {lm_pos[0] + TARGET_RADIUS * std::cos(angle),
                  lm_pos[1] + TARGET_RADIUS * std::sin(angle)};
  }

  // N×N cost matrix: cost[i][j] = dist(agent_i, target_j)
  Matrix<float> cost(N, N);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float dx = world.agents[i].state.p_pos[0] - targets[j][0];
      float dy = world.agents[i].state.p_pos[1] - targets[j][1];
      cost(i, j) = std::sqrt(dx * dx + dy * dy);
    }
  }

  // Keep a copy of distances before munkres modifies the matrix in-place
  Matrix<float> cost_copy = cost;
  Munkres<float> solver;
  solver.solve(cost);  // cost is modified: 0 = assigned, -1 = not assigned

  delta_dists_.resize(N);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (cost(i, j) == 0) {
        delta_dists_[i] = cost_copy(i, j);
        break;
      }
    }
  }

  float sum = 0.0f;
  for (float d : delta_dists_) sum += std::min(d, 2.0f);
  joint_reward_ = -(sum / N);

  cache_valid_ = true;
}

float SimpleFormationScenario::global_reward(const core::World& world) const {
  compute_formation(world);
  return joint_reward_;
}

bool SimpleFormationScenario::is_terminal(const core::World& world) const {
  if (!terminate_on_success_) return false;
  compute_formation(world);
  for (float d : delta_dists_) {
    if (d >= DIST_THRESHOLD) return false;
  }
  return true;
}

std::vector<float> SimpleFormationScenario::observation(
    const core::Agent& agent, const core::World& world) const {
  std::vector<float> obs;
  obs.push_back(agent.state.p_vel[0]);
  obs.push_back(agent.state.p_vel[1]);
  obs.push_back(agent.state.p_pos[0]);
  obs.push_back(agent.state.p_pos[1]);
  obs.push_back(world.landmarks[0].state.p_pos[0] - agent.state.p_pos[0]);
  obs.push_back(world.landmarks[0].state.p_pos[1] - agent.state.p_pos[1]);
  return obs;
}

}  // namespace cpp_pettingzoo::simple_formation
