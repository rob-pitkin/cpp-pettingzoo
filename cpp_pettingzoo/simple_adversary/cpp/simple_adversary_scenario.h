#ifndef SIMPLE_ADVERSARY_SCENARIO_H_
#define SIMPLE_ADVERSARY_SCENARIO_H_

#include "../../../core/scenario.h"

namespace cpp_pettingzoo::simple_adversary {

class SimpleAdversaryScenario : public core::Scenario {
 public:
  SimpleAdversaryScenario(size_t n);
  void make_world(core::World& w) override;

  void reset_world(core::World& w) override;
  float reward(const core::Agent& a, const core::World& w) const override;
  float global_reward(const core::World& w) const override;
  std::vector<float> observation(const core::Agent& a,
                                 const core::World& w) const override;

 private:
  size_t n_;

  // Cached per-step reward components — mutable so reward() can populate once
  mutable float cached_min_good_dist_ = 0.0f;
  mutable float cached_adv_dist_ = 0.0f;
  mutable bool reward_cache_valid_ = false;

  void compute_reward_cache(const core::World& world) const;
  float agent_reward(const core::Agent& agent, const core::World& world) const;
  float adversary_reward(const core::Agent& agent,
                         const core::World& world) const;
};

}  // namespace cpp_pettingzoo::simple_adversary

#endif  // SIMPLE_ADVERSARY_SCENARIO_H_
