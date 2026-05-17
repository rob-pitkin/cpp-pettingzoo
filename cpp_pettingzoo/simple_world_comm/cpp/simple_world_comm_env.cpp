#include "simple_world_comm_env.h"

namespace cpp_pettingzoo::simple_world_comm {

SimpleWorldCommEnv::SimpleWorldCommEnv(int num_good, int num_adversaries,
                                       int num_obstacles, int num_food,
                                       int num_forests, int max_cycles,
                                       bool continuous_actions,
                                       bool dynamic_rescaling)
    : world_(),
      scenario_(num_good, num_adversaries, num_obstacles, num_food,
                num_forests),
      BaseEnv(scenario_, world_, max_cycles, dynamic_rescaling,
              continuous_actions) {
  scenario_.make_world(world_);
  world_.cache_entities();

  agents_.reserve(world_.agents.size());
  for (const auto& agent : world_.agents) {
    agents_.push_back(agent.name);
  }
}

}  // namespace cpp_pettingzoo::simple_world_comm
