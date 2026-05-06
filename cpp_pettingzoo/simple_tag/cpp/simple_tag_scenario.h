
#ifndef SIMPLE_TAG_SCENARIO_H_
#define SIMPLE_TAG_SCENARIO_H_

#include <optional>
#include <vector>

#include "../../../core/scenario.h"

namespace cpp_pettingzoo::simple_tag {

class SimpleTagScenario : public core::Scenario {
 public:
  SimpleTagScenario(int num_good, int num_adversaries, int num_obstacles,
                    bool curriculum, bool terminate_on_success,
                    std::optional<int> num_agent_neighbors,
                    std::optional<int> num_landmark_neighbors);

  void make_world(core::World& w) override;
  void reset_world(core::World& w) override;
  float reward(const core::Agent& agent,
               const core::World& world) const override;
  std::vector<float> observation(const core::Agent& agent,
                                 const core::World& world) const override;
  bool is_terminal(const core::World& world) const override;

  int get_curriculum_stage() const;
  void advance_curriculum();
  void set_curriculum_stage(int stage);

 private:
  int num_good_;
  int num_adversaries_;
  int num_obstacles_;
  bool curriculum_;
  int curriculum_stage_;
  bool terminate_on_success_;
  std::optional<int> num_agent_neighbors_;
  std::optional<int> num_landmark_neighbors_;

  // Curriculum constants
  static constexpr float PREY_BASE_MAX_SPEED = 1.3f;
  static constexpr float PREY_BASE_ACCEL = 4.0f;

  bool is_collision(const core::Agent& a, const core::Agent& b) const;
  float bound(float x) const;
  float agent_reward(const core::Agent& agent, const core::World& world) const;
  float adversary_reward(const core::Agent& agent,
                         const core::World& world) const;
  std::vector<const core::Agent*> good_agents(const core::World& world) const;
  std::vector<const core::Agent*> adversaries(const core::World& world) const;

  std::vector<size_t> nearest_indices(
      const core::Agent& agent,
      const std::vector<const core::Entity*>& entities,
      std::optional<int> n) const;
  std::vector<float> padded_relative_positions(
      const core::Agent& agent,
      const std::vector<const core::Entity*>& entities,
      std::optional<int> n) const;
  std::vector<float> padded_good_agent_velocities(
      const core::Agent& agent, const std::vector<const core::Agent*>& others,
      std::optional<int> n) const;
};

}  // namespace cpp_pettingzoo::simple_tag

#endif  // SIMPLE_TAG_SCENARIO_H_
