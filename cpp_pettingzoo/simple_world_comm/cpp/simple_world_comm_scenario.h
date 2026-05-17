#ifndef SIMPLE_WORLD_COMM_SCENARIO_H_
#define SIMPLE_WORLD_COMM_SCENARIO_H_

#include <vector>

#include "../../../core/scenario.h"

namespace cpp_pettingzoo::simple_world_comm {

constexpr int LANDMARK_OBSTACLE = 0;
constexpr int LANDMARK_FOOD = 1;
constexpr int LANDMARK_FOREST = 2;

class SimpleWorldCommScenario : public core::Scenario {
 public:
  SimpleWorldCommScenario(int num_good = 2, int num_adversaries = 4,
                          int num_obstacles = 1, int num_food = 2,
                          int num_forests = 2);

  void make_world(core::World& w) override;
  void reset_world(core::World& w) override;
  float reward(const core::Agent& agent,
               const core::World& world) const override;
  std::vector<float> observation(const core::Agent& agent,
                                 const core::World& world) const override;

 private:
  int num_good_;
  int num_adversaries_;
  int num_obstacles_;
  int num_food_;
  int num_forests_;

  bool is_collision(const core::Entity& a, const core::Entity& b) const;
  float bound(float x) const;
  float agent_reward(const core::Agent& agent, const core::World& world) const;
  float adversary_reward(const core::Agent& agent,
                         const core::World& world) const;
  std::vector<const core::Agent*> good_agents(const core::World& world) const;
  std::vector<const core::Agent*> adversaries(const core::World& world) const;
};

}  // namespace cpp_pettingzoo::simple_world_comm

#endif  // SIMPLE_WORLD_COMM_SCENARIO_H_
