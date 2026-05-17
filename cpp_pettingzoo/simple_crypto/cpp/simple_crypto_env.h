#ifndef SIMPLE_CRYPTO_ENV_H_
#define SIMPLE_CRYPTO_ENV_H_

#include "../../../core/base_env.h"
#include "simple_crypto_scenario.h"

namespace cpp_pettingzoo::simple_crypto {

class SimpleCryptoEnv : public core::BaseEnv {
 public:
  SimpleCryptoEnv(int max_cycles = 25, bool continuous_actions = false,
                  bool dynamic_rescaling = false);

 private:
  core::World world_;
  SimpleCryptoScenario scenario_;
};

}  // namespace cpp_pettingzoo::simple_crypto

#endif  // SIMPLE_CRYPTO_ENV_H_
