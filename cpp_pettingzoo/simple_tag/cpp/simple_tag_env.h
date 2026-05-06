#ifndef SIMPLE_TAG_ENV_H_
#define SIMPLE_TAG_ENV_H_

#include <optional>

#include "../../../core/base_env.h"
#include "simple_tag_scenario.h"

namespace cpp_pettingzoo::simple_tag {

class SimpleTagEnv : public core::BaseEnv {
 public:
  SimpleTagEnv(int num_good = 1, int num_adversaries = 3, int num_obstacles = 2,
               int max_cycles = 25, bool continuous_actions = false,
               bool dynamic_rescaling = false, bool curriculum = false,
               bool terminate_on_success = false,
               std::optional<int> num_agent_neighbors = std::nullopt,
               std::optional<int> num_landmark_neighbors = std::nullopt);

  int get_curriculum_stage() const;
  void advance_curriculum();
  void set_curriculum_stage(int stage);

 private:
  core::World world_;
  SimpleTagScenario scenario_;
};

}  // namespace cpp_pettingzoo::simple_tag

#endif  // SIMPLE_TAG_ENV_H_
