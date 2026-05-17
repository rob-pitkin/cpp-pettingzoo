#ifndef SIMPLE_CRYPTO_SCENARIO_H_
#define SIMPLE_CRYPTO_SCENARIO_H_

#include <vector>

#include "../../../core/scenario.h"

namespace cpp_pettingzoo::simple_crypto {

class SimpleCryptoScenario : public core::Scenario {
 public:
  SimpleCryptoScenario();

  void make_world(core::World& w) override;
  void reset_world(core::World& w) override;
  float reward(const core::Agent& agent,
               const core::World& world) const override;
  std::vector<float> observation(const core::Agent& agent,
                                 const core::World& world) const override;

 private:
  // The one-hot color vector of each landmark (length dim_c). These act as both
  // landmark identity and the canonical message content.
  std::vector<std::vector<float>> landmark_comm_colors_;
  // Index of the goal landmark for this episode (chosen at reset).
  int goal_index_ = 0;
  // Speaker's private key: copy of some landmark's comm color (chosen at reset).
  std::vector<float> key_;

  bool is_comm_zero(const std::vector<float>& c) const;
};

}  // namespace cpp_pettingzoo::simple_crypto

#endif  // SIMPLE_CRYPTO_SCENARIO_H_
