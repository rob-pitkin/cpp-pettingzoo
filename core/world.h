#ifndef WORLD_H_
#define WORLD_H_

#include <optional>
#include <random>
#include <vector>

#include "entity.h"

namespace cpp_pettingzoo::core {

using ForceVector = std::vector<std::optional<std::array<float, 2>>>;
using Force = std::array<float, 2>;
using OptionalForce = std::optional<std::array<float, 2>>;

class World {
 public:
  int dim_c = 0;

  std::vector<Agent> agents;
  std::vector<Landmark> landmarks;

  World(uint32_t seed = std::random_device{}());

  void step();
  std::mt19937& get_rng() { return rng_; }
  void reseed(uint32_t seed) { rng_ = std::mt19937(seed); }

 private:
  float dt_ = 0.1;
  float damping_ = 0.25;
  float contact_force_ = 1e2;
  float contact_margin_ = 1e-3;
  int dim_p_ = 2;
  std::mt19937 rng_;
  std::vector<Entity*> entities_;

  std::vector<Entity*> entities();
  ForceVector apply_action_force();
  void apply_environment_force(ForceVector& p_force);

  std::pair<OptionalForce, OptionalForce> get_collision_force(
      const Entity& a, const Entity& b) const;

  void integrate_state(const ForceVector& p_force);
  void update_agent_state(Agent& agent);
  float sample_normal(float mean, float stddev);
};

}  // namespace cpp_pettingzoo::core

#endif  // WORLD_H_
