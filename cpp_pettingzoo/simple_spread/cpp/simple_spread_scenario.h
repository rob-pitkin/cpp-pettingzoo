#ifndef SIMPLE_SPREAD_SCENARIO_H_
#define SIMPLE_SPREAD_SCENARIO_H_

#include <random>

#include "../../../core/scenario.h"

namespace cpp_pettingzoo::simple_spread {

class SimpleSpreadScenario : public core::Scenario {
 public:
  SimpleSpreadScenario(bool curriculum = false, int curriculum_stage = 0);
  void make_world(core::World& w) override;
  void reset_world(core::World& w) override;
  float reward(const core::Agent& a, const core::World& w) const override;
  float global_reward(const core::World& w) const override;
  std::vector<float> observation(const core::Agent& a,
                                 const core::World& w) const override;
  int get_curriculum_stage() const;
  void advance_curriculum();             // Move to next stage
  void set_curriculum_stage(int stage);  // Jump to specific stage
 private:
  bool curriculum_enabled_;
  int curriculum_stage_;  // 0 or 1
};

}  // namespace cpp_pettingzoo::simple_spread

#endif  // SIMPLE_SPREAD_SCENARIO_H_
