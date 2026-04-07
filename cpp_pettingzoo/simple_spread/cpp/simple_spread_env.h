#ifndef SIMPLE_SPREAD_ENV_H_
#define SIMPLE_SPREAD_ENV_H_

#include "../../../core/base_env.h"
#include "../../../core/world.h"
#include "simple_spread_scenario.h"

namespace cpp_pettingzoo::simple_spread {

class SimpleSpreadEnv : public core::BaseEnv {
 public:
  SimpleSpreadEnv(int max_cycles = 25, bool dynamic_rescaling = false,
                  bool continuous_actions = false, float local_ratio = 0.5f,
                  bool curriculum = false, int curriculum_stage = 0);

  // Curriculum management methods
  int get_curriculum_stage() const;
  void advance_curriculum();
  void set_curriculum_stage(int stage);

 private:
  core::World world_;
  SimpleSpreadScenario scenario_;
};
}  // namespace cpp_pettingzoo::simple_spread

#endif  // SIMPLE_SPREAD_ENV_H_
