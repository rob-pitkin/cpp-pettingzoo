#ifndef SIMPLE_WORLD_COMM_ENV_H_
#define SIMPLE_WORLD_COMM_ENV_H_

#include "../../../core/base_env.h"
#include "simple_world_comm_scenario.h"

namespace cpp_pettingzoo::simple_world_comm {

class SimpleWorldCommEnv : public core::BaseEnv {
 public:
  SimpleWorldCommEnv(int num_good = 2, int num_adversaries = 4,
                     int num_obstacles = 1, int num_food = 2,
                     int num_forests = 2, int max_cycles = 25,
                     bool continuous_actions = false,
                     bool dynamic_rescaling = false);

 private:
  core::World world_;
  SimpleWorldCommScenario scenario_;
};

}  // namespace cpp_pettingzoo::simple_world_comm

#endif  // SIMPLE_WORLD_COMM_ENV_H_
