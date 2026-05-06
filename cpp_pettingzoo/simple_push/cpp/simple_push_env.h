#ifndef SIMPLE_PUSH_ENV_H_
#define SIMPLE_PUSH_ENV_H_

#include "../../../core/base_env.h"
#include "simple_push_scenario.h"

namespace cpp_pettingzoo::simple_push {

class SimplePushEnv : public core::BaseEnv {
 public:
  SimplePushEnv(int max_cycles = 25, bool continuous_actions = false,
                bool dynamic_rescaling = false);

 private:
  core::World world_;
  SimplePushScenario scenario_;
};

}  // namespace cpp_pettingzoo::simple_push

#endif  // SIMPLE_PUSH_ENV_H_
