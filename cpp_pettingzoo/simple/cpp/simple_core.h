#ifndef SIMPLE_CORE_H_
#define SIMPLE_CORE_H_

#include <array>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace cpp_pettingzoo {

static constexpr float DT = 0.1;
static constexpr float DAMPING = 0.25;
static constexpr float MASS = 1.0;
static constexpr float SENSITIVITY = 5.0;

typedef std::unordered_map<std::string, std::vector<float>> ObservationMap;
typedef std::unordered_map<std::string, float> RewardMap;
typedef std::unordered_map<std::string, bool> BoolMap;
typedef std::unordered_map<std::string, bool> TruncationMap;
// We use std::vector float for discrete and continuous. Former case is a one
// element vector, latter is 5 float elements.
typedef std::unordered_map<std::string, std::vector<float>> ActionMap;

struct State {
  ObservationMap observations;
  RewardMap rewards;
  BoolMap terminations;
  BoolMap truncations;
};

class SimpleEnv {
public:
  SimpleEnv(int max_cycles = 25, bool dynamic_rescaling = false,
            bool continuous_actions_ = false);
  ObservationMap reset(std::optional<int> seed);
  State step(const ActionMap &actions);
  std::vector<std::string> get_agents() const;
  std::vector<float> get_state() const;

private:
  std::array<float, 2> p_pos_;
  std::array<float, 2> p_vel_;
  std::array<float, 2> landmark_pos_;
  int timesteps_;
  int max_cycles_;
  bool has_reset_;
  bool dynamic_rescaling_;
  bool continuous_actions_;
  std::vector<std::string> agents_;
  std::mt19937 gen_;
  std::uniform_real_distribution<float> dist_;

  std::array<float, 2> action_to_force(int action) const;

  std::array<float, 2>
  action_to_force_continuous(const std::vector<float> &action) const;

  void clamp_position();

  float calculate_reward() const;

  std::vector<float> get_observation() const;
};
} // namespace cpp_pettingzoo

#endif
