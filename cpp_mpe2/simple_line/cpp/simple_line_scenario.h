#ifndef SIMPLE_LINE_SCENARIO_H_
#define SIMPLE_LINE_SCENARIO_H_

#include <array>
#include <vector>

#include "../../../core/scenario.h"

namespace cpp_mpe2::simple_line {

class SimpleLineScenario : public core::Scenario {
 public:
  explicit SimpleLineScenario(bool terminate_on_success = false);

  void make_world(core::World& w, int N = 4);
  void make_world(core::World& w) override { make_world(w, 4); }
  void reset_world(core::World& w) override;
  void post_step(core::World& w) override { cache_valid_ = false; }
  float reward(const core::Agent& agent,
               const core::World& world) const override { return 0.0f; }
  float global_reward(const core::World& world) const override;
  std::vector<float> observation(const core::Agent& agent,
                                 const core::World& world) const override;
  bool is_terminal(const core::World& world) const override;

 private:
  bool terminate_on_success_;

  // Fixed-per-episode target positions (set at reset, reused each step)
  mutable std::vector<std::array<float, 2>> expected_positions_;

  // Cached per-step matching result
  mutable std::vector<float> delta_dists_;
  mutable float joint_reward_ = 0.0f;
  mutable bool cache_valid_ = false;

  static constexpr float TOTAL_SEP = 1.25f;
  static constexpr float DIST_THRESHOLD = 0.05f;

  void compute_line(const core::World& world) const;
};

}  // namespace cpp_mpe2::simple_line

#endif  // SIMPLE_LINE_SCENARIO_H_
