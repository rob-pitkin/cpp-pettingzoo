#include "simple_speaker_listener_env.h"

namespace cpp_pettingzoo::simple_speaker_listener {

SimpleSpeakerListenerEnv::SimpleSpeakerListenerEnv(int max_cycles,
                                                   bool dynamic_rescaling,
                                                   bool continuous_actions,
                                                   float local_ratio)
    : world_(),
      scenario_(),
      BaseEnv(scenario_, world_, max_cycles, dynamic_rescaling,
              continuous_actions, local_ratio) {
  // Create world structure
  scenario_.make_world(world_);
  world_.cache_entities();

  // Build agent list after world is populated
  agents_.reserve(world_.agents.size());
  for (const auto& agent : world_.agents) {
    agents_.push_back(agent.name);
  }
}

}  // namespace cpp_pettingzoo::simple_speaker_listener
