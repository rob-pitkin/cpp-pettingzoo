#ifndef SIMPLE_SPEAKER_LISTENER_ENV_H_
#define SIMPLE_SPEAKER_LISTENER_ENV_H_

#include "../../../core/base_env.h"
#include "../../../core/world.h"
#include "simple_speaker_listener_scenario.h"

namespace cpp_pettingzoo::simple_speaker_listener {

class SimpleSpeakerListenerEnv : public core::BaseEnv {
 public:
  SimpleSpeakerListenerEnv(int max_cycles = 25, bool dynamic_rescaling = false,
                           bool continuous_actions = false,
                           float local_ratio = 0.5f);

 private:
  core::World world_;
  SimpleSpeakerListenerScenario scenario_;
};

}  // namespace cpp_pettingzoo::simple_speaker_listener

#endif  // SIMPLE_SPEAKER_LISTENER_ENV_H_
