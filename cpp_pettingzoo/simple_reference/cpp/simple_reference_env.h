#ifndef SIMPLE_REFERENCE_ENV_H_
#define SIMPLE_REFERENCE_ENV_H_

#include "../../../core/base_env.h"
#include "../../../core/world.h"
#include "simple_reference_scenario.h"

namespace cpp_pettingzoo::simple_reference {

class SimpleReferenceEnv : public core::BaseEnv {
 public:
  SimpleReferenceEnv(int max_cycles = 25, bool dynamic_rescaling = false,
                     bool continuous_actions = false, float local_ratio = 0.5f);

 private:
  core::World world_;
  SimpleReferenceScenario scenario_;
};

}  // namespace cpp_pettingzoo::simple_reference

#endif  // SIMPLE_REFERENCE_ENV_H_
