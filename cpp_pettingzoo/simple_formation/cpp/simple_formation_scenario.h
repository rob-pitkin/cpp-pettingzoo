#ifndef SIMPLE_FORMATION_SCENARIO_H_
#define SIMPLE_FORMATION_SCENARIO_H_

#include <optional>
#include <vector>

#include "../../../core/scenario.h"

namespace cpp_pettingzoo::simple_formation {

class SimpleFormationScenario : public core::Scenario {
 public:
  explicit SimpleFormationScenario(bool terminate_on_success = false);

  void make_world(core::World& w, int N = 4);
  void make_world(core::World& w) override { make_world(w, 4); }
  void reset_world(core::World& w) override;
  float reward(const core::Agent& agent,
               const core::World& world) const override { return 0.0f; }
  float global_reward(const core::World& world) const override;
  std::vector<float> observation(const core::Agent& agent,
                                 const core::World& world) const override;
  bool is_terminal(const core::World& world) const override;

 private:
  bool terminate_on_success_;

  // Cached per-step formation result (mutable for const reward/terminal calls)
  mutable std::vector<float> delta_dists_;
  mutable float joint_reward_ = 0.0f;
  mutable bool cache_valid_ = false;

  static constexpr float TARGET_RADIUS = 0.5f;
  static constexpr float DIST_THRESHOLD = 0.05f;

  void compute_formation(const core::World& world) const;
  static float find_angle(float x, float y);
};

}  // namespace cpp_pettingzoo::simple_formation

#endif  // SIMPLE_FORMATION_SCENARIO_H_
