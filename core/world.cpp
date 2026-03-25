#include <cmath>
#include <optional>
#include <random>

#include "world.h"

namespace cpp_pettingzoo::core {

World::World(uint32_t seed) { rng_ = std::mt19937(seed); }

std::vector<Entity*> World::entities() {
  std::vector<Entity*> all_entities;
  all_entities.reserve(agents.size() + landmarks.size());
  for (auto& agent : agents) {
    all_entities.push_back(&agent);
  }
  for (auto& landmark : landmarks) {
    all_entities.push_back(&landmark);
  }
  return all_entities;
}

void World::integrate_state(const ForceVector& p_force) {
  std::vector<Entity*> all_entities = entities();
  for (size_t i = 0; i < all_entities.size(); ++i) {
    Entity* e = all_entities[i];
    if (!e->movable) {
      continue;
    }
    e->state.p_pos[0] += e->state.p_vel[0] * dt_;
    e->state.p_pos[1] += e->state.p_vel[1] * dt_;
    e->state.p_vel[0] *= (1 - damping_);
    e->state.p_vel[1] *= (1 - damping_);

    if (p_force[i].has_value()) {
      e->state.p_vel[0] += (p_force[i].value()[0] / e->mass()) * dt_;
      e->state.p_vel[1] += (p_force[i].value()[1] / e->mass()) * dt_;
    }

    if (e->max_speed.has_value()) {
      float speed = std::sqrt(e->state.p_vel[0] * e->state.p_vel[0] +
                              e->state.p_vel[1] * e->state.p_vel[1]);
      if (speed > e->max_speed.value()) {
        e->state.p_vel[0] = e->state.p_vel[0] / speed * e->max_speed.value();
        e->state.p_vel[1] = e->state.p_vel[1] / speed * e->max_speed.value();
      }
    }
  }
}

std::pair<OptionalForce, OptionalForce> World::get_collision_force(
    const Entity& a, const Entity& b) const {
  if (!a.collide || !b.collide) {
    return {std::nullopt, std::nullopt};
  }
  if (&a == &b) {
    return {std::nullopt, std::nullopt};
  }

  // Compute actual distance
  float delta_pos_x = a.state.p_pos[0] - b.state.p_pos[0];
  float delta_pos_y = a.state.p_pos[1] - b.state.p_pos[1];
  float dist = sqrt(delta_pos_x * delta_pos_x + delta_pos_y * delta_pos_y);
  float dist_min = a.size + b.size;
  float k = contact_margin_;
  float penetration = std::log1p(std::exp(-(dist - dist_min) / k)) * k;
  float force_x = contact_force_ * delta_pos_x / dist * penetration;
  float force_y = contact_force_ * delta_pos_y / dist * penetration;
  OptionalForce force_a = std::nullopt;
  if (a.movable) {
    force_a = {force_x, force_y};
  }
  OptionalForce force_b = std::nullopt;
  if (b.movable) {
    force_b = {-force_x, -force_y};
  }
  return {force_a, force_b};
}

ForceVector World::apply_action_force() {
  ForceVector p_force(agents.size() + landmarks.size(), std::nullopt);
  for (size_t i = 0; i < agents.size(); ++i) {
    if (agents[i].movable) {
      p_force[i] = agents[i].action.u;
      if (agents[i].u_noise.has_value()) {
        float noise_x = sample_normal(0.0f, agents[i].u_noise.value());
        float noise_y = sample_normal(0.0f, agents[i].u_noise.value());
        p_force[i].value()[0] += noise_x;
        p_force[i].value()[1] += noise_y;
      }
    }
  }
  return p_force;
}

void World::apply_environment_force(ForceVector& p_force) {
  auto e_vec = entities();
  for (size_t i = 0; i < e_vec.size(); ++i) {
    for (size_t j = 0; j < e_vec.size(); ++j) {
      if (j <= i) {
        continue;
      }
      auto collision_force = get_collision_force(*e_vec[i], *e_vec[j]);
      OptionalForce force_a = collision_force.first;
      OptionalForce force_b = collision_force.second;
      if (force_a.has_value()) {
        if (!p_force[i].has_value()) {
          p_force[i] = {0.0, 0.0};
        }
        p_force[i].value()[0] += force_a.value()[0];
        p_force[i].value()[1] += force_a.value()[1];
      }
      if (force_b.has_value()) {
        if (!p_force[j].has_value()) {
          p_force[j] = {0.0, 0.0};
        }
        p_force[j].value()[0] += force_b.value()[0];
        p_force[j].value()[1] += force_b.value()[1];
      }
    }
  }
}

void World::update_agent_state(Agent& agent) {
  if (agent.silent) {
    agent.c = std::vector<float>(dim_c, 0.0);
  } else {
    agent.c = agent.action.c;
    if (agent.c_noise.has_value()) {
      for (size_t i = 0; i < agent.c.size(); ++i) {
        float noise = sample_normal(0.0f, agent.c_noise.value());
        agent.c[i] += noise;
      }
    }
  }
}

void World::step() {
  ForceVector p_force = apply_action_force();
  apply_environment_force(p_force);
  integrate_state(p_force);
  for (auto& a : agents) {
    update_agent_state(a);
  }
}

float World::sample_normal(float mean, float stddev) {
  std::normal_distribution<float> dist(mean, stddev);
  return dist(rng_);
}

}  // namespace cpp_pettingzoo::core
