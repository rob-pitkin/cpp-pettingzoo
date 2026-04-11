#ifndef SIMPLE_REFERENCE_SCENARIO_H_
#define SIMPLE_REFERENCE_SCENARIO_H_

#include "../../../core/scenario.h"

namespace cpp_pettingzoo::simple_reference {

class SimpleReferenceScenario : public core::Scenario {
 public:
  SimpleReferenceScenario() = default;
  void make_world(core::World& w) override;

  void reset_world(core::World& w) override;
  float reward(const core::Agent& a, const core::World& w) const override;
  float global_reward(const core::World& w) const override;
  std::vector<float> observation(const core::Agent& a,
                                 const core::World& w) const override;
};

}  // namespace cpp_pettingzoo::simple_reference

#endif  // SIMPLE_REFERENCE_SCENARIO_H_
