#ifndef SIMPLE_FORMATION_ENV_H_
#define SIMPLE_FORMATION_ENV_H_

#include "../../../core/base_env.h"
#include "simple_formation_scenario.h"

namespace cpp_pettingzoo::simple_formation {

class SimpleFormationEnv : public core::BaseEnv {
 public:
  SimpleFormationEnv(int N = 4, int max_cycles = 25,
                     bool continuous_actions = false,
                     bool dynamic_rescaling = false,
                     bool terminate_on_success = false);

 private:
  core::World world_;
  SimpleFormationScenario scenario_;
};

}  // namespace cpp_pettingzoo::simple_formation

#endif  // SIMPLE_FORMATION_ENV_H_
