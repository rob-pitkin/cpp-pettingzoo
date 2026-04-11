#ifndef TYPES_H_
#define TYPES_H_

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace cpp_pettingzoo {

// Action force sensitivity constant (shared across all MPE environments)
static constexpr float SENSITIVITY = 5.0;

// Type aliases for PettingZoo parallel API
typedef std::unordered_map<std::string, std::vector<float>> ObservationMap;
typedef std::unordered_map<std::string, float> RewardMap;
typedef std::unordered_map<std::string, bool> BoolMap;
typedef std::unordered_map<std::string, bool> TruncationMap;
// We use std::vector<float> for both discrete and continuous actions.
// Discrete: one-element vector, Continuous: 5-element vector.
typedef std::unordered_map<std::string, std::vector<float>> ActionMap;
typedef std::unordered_map<std::string, std::vector<float>> RenderState;

struct State {
  ObservationMap observations;
  RewardMap rewards;
  BoolMap terminations;
  BoolMap truncations;
};

}  // namespace cpp_pettingzoo

#endif  // TYPES_H_
