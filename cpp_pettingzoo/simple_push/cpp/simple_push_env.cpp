#include "simple_push_env.h"

namespace cpp_pettingzoo::simple_push {

SimplePushEnv::SimplePushEnv(int max_cycles, bool continuous_actions,
                             bool dynamic_rescaling)
    : world_(),
      scenario_(),
      BaseEnv(scenario_, world_, max_cycles, dynamic_rescaling,
              continuous_actions) {
  scenario_.make_world(world_);
  world_.cache_entities();

  agents_.reserve(world_.agents.size());
  for (const auto& agent : world_.agents) {
    agents_.push_back(agent.name);
  }
}

}  // namespace cpp_pettingzoo::simple_push
