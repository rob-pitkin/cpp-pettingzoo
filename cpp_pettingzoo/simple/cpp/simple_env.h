#ifndef SIMPLE_ENV_H_
#define SIMPLE_ENV_H_

#include "../../../core/base_env.h"
#include "../../../core/world.h"
#include "simple_scenario.h"

namespace cpp_pettingzoo::simple {

// Simple environment - owns scenario and world, inherits from BaseEnv
class SimpleEnv : public core::BaseEnv {
 public:
  SimpleEnv(int max_cycles = 25, bool dynamic_rescaling = false,
            bool continuous_actions = false);

 private:
  core::World world_;
  SimpleScenario scenario_;
};

}  // namespace cpp_pettingzoo::simple

#endif  // SIMPLE_ENV_H_
