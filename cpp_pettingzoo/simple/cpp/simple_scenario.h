#ifndef SIMPLE_SCENARIO_H_
#define SIMPLE_SCENARIO_H_

#include <random>

#include "../../../core/scenario.h"

namespace cpp_pettingzoo::simple {

class SimpleScenario : public core::Scenario {
 public:
  void make_world(core::World& world) override;
  void reset_world(core::World& world) override;
  float reward(const core::Agent& agent,
               const core::World& world) const override;
  std::vector<float> observation(const core::Agent& agent,
                                 const core::World& world) const override;
};

}  // namespace cpp_pettingzoo::simple

#endif  // SIMPLE_SCENARIO_H_
