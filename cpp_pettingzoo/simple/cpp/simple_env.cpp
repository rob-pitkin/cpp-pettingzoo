#include "simple_env.h"

namespace cpp_pettingzoo::simple {

SimpleEnv::SimpleEnv(int max_cycles, bool dynamic_rescaling,
                     bool continuous_actions)
    : world_(),
      scenario_(),
      BaseEnv(scenario_, world_, max_cycles, dynamic_rescaling,
              continuous_actions) {
  // Create world structure
  scenario_.make_world(world_);
  world_.cache_entities();

  // Build agent list after world is populated
  agents_.reserve(world_.agents.size());
  for (const auto& agent : world_.agents) {
    agents_.push_back(agent.name);
  }
}

}  // namespace cpp_pettingzoo::simple
