#ifndef CORE_BASE_ENV_H_
#define CORE_BASE_ENV_H_

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "scenario.h"
#include "types.h"
#include "world.h"

namespace cpp_pettingzoo::core {

class BaseEnv {
 public:
  BaseEnv(Scenario& scenario, World& world, int max_cycles = 25,
          bool dynamic_rescaling = false, bool continuous_actions = false,
          std::optional<float> local_ratio = std::nullopt);

  ObservationMap reset(std::optional<int> seed);
  State step(const ActionMap& actions);
  std::vector<std::string> get_agents() const;
  std::vector<float> get_state() const;
  RenderState get_render_state() const;

 protected:
  int timesteps_;
  int max_cycles_;
  bool has_reset_;
  bool dynamic_rescaling_;
  bool continuous_actions_;
  std::optional<float> local_ratio_;
  std::vector<std::string> agents_;
  Scenario& scenario_;
  World& world_;

  std::array<float, 2> action_to_force(int action) const;
  std::array<float, 2> action_to_force_continuous(
      const std::vector<float>& action) const;
};

}  // namespace cpp_pettingzoo::core

#endif  // CORE_BASE_ENV_H_
