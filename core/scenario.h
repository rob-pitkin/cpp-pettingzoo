#ifndef SCENARIO_H_
#define SCENARIO_H_

#include <unordered_map>
#include <vector>
#include <string>

#include "entity.h"
#include "world.h"

namespace cpp_pettingzoo::core {

class Scenario {
public:
  virtual ~Scenario() = default;

  virtual void make_world(World& w) = 0;
  virtual void reset_world(World& w) = 0;
  virtual float reward(const Agent& a, const World& w) const = 0;
  virtual std::vector<float> observation(const Agent& a, const World& w) const = 0;

  // Optional methods with default implementations
  virtual float global_reward(const World& w) const {
    return 0.0f;  // Default: no global reward
  }
  virtual std::unordered_map<std::string, float> benchmark_data(const Agent& a, const World& w) const {
    return {};
  }
  virtual bool is_terminal(const World& w) const {
    return false;
  }
};

}  // namespace cpp_pettingzoo::core

#endif  // SCENARIO_H_
