#include "simple_spread_env.h"

namespace cpp_pettingzoo::simple_spread {

SimpleSpreadEnv::SimpleSpreadEnv(int max_cycles, bool dynamic_rescaling,
                                 bool continuous_actions, float local_ratio,
                                 bool curriculum, int curriculum_stage)
    : world_(),
      scenario_(curriculum, curriculum_stage),
      BaseEnv(scenario_, world_, max_cycles, dynamic_rescaling,
              continuous_actions, local_ratio) {
  scenario_.make_world(world_);
  world_.cache_entities();

  // Build agent list after world is populated
  agents_.reserve(world_.agents.size());
  for (const auto& agent : world_.agents) {
    agents_.push_back(agent.name);
  }
}

int SimpleSpreadEnv::get_curriculum_stage() const {
  return scenario_.get_curriculum_stage();
}

void SimpleSpreadEnv::advance_curriculum() { scenario_.advance_curriculum(); }

void SimpleSpreadEnv::set_curriculum_stage(int stage) {
  scenario_.set_curriculum_stage(stage);
}

}  // namespace cpp_pettingzoo::simple_spread
