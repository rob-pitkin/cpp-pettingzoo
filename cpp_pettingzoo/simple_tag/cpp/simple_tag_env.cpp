#include "simple_tag_env.h"

namespace cpp_pettingzoo::simple_tag {

SimpleTagEnv::SimpleTagEnv(int num_good, int num_adversaries, int num_obstacles,
                           int max_cycles, bool continuous_actions,
                           bool dynamic_rescaling, bool curriculum,
                           bool terminate_on_success,
                           std::optional<int> num_agent_neighbors,
                           std::optional<int> num_landmark_neighbors)
    : world_(),
      scenario_(num_good, num_adversaries, num_obstacles, curriculum,
                terminate_on_success, num_agent_neighbors,
                num_landmark_neighbors),
      BaseEnv(scenario_, world_, max_cycles, dynamic_rescaling,
              continuous_actions) {
  scenario_.make_world(world_);
  world_.cache_entities();

  agents_.reserve(world_.agents.size());
  for (const auto& agent : world_.agents) {
    agents_.push_back(agent.name);
  }
}

int SimpleTagEnv::get_curriculum_stage() const {
  return scenario_.get_curriculum_stage();
}

void SimpleTagEnv::advance_curriculum() { scenario_.advance_curriculum(); }

void SimpleTagEnv::set_curriculum_stage(int stage) {
  scenario_.set_curriculum_stage(stage);
}

}  // namespace cpp_pettingzoo::simple_tag
