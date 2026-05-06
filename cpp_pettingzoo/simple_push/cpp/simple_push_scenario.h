#ifndef SIMPLE_PUSH_SCENARIO_H_
#define SIMPLE_PUSH_SCENARIO_H_

#include <vector>

#include "../../../core/scenario.h"

namespace cpp_pettingzoo::simple_push {

class SimplePushScenario : public core::Scenario {
 public:
  SimplePushScenario() = default;

  void make_world(core::World& w) override;
  void reset_world(core::World& w) override;
  float reward(const core::Agent& agent,
               const core::World& world) const override;
  std::vector<float> observation(const core::Agent& agent,
                                 const core::World& world) const override;
  bool is_terminal(const core::World& world) const override { return false; }

 private:
  float agent_reward(const core::Agent& agent,
                     const core::World& world) const;
  float adversary_reward(const core::Agent& agent,
                         const core::World& world) const;
};

}  // namespace cpp_pettingzoo::simple_push

#endif  // SIMPLE_PUSH_SCENARIO_H_
