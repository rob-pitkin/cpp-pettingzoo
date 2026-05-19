#include "simple_formation_env.h"

namespace cpp_mpe2::simple_formation {

SimpleFormationEnv::SimpleFormationEnv(int N, int max_cycles,
                                       bool continuous_actions,
                                       bool dynamic_rescaling,
                                       bool terminate_on_success)
    : BaseEnv(scenario_, world_, max_cycles, dynamic_rescaling,
              continuous_actions, /*local_ratio=*/0.0f),
      world_(),
      scenario_(terminate_on_success) {
  scenario_.make_world(world_, N);
  world_.cache_entities();

  agents_.reserve(world_.agents.size());
  for (const auto& agent : world_.agents) {
    agents_.push_back(agent.name);
  }
}

}  // namespace cpp_mpe2::simple_formation
