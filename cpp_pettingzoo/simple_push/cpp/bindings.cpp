#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "simple_push_env.h"

PYBIND11_MODULE(_simple_push, m) {
  m.doc() = "Simple Push Environment C++ Implementation";

  pybind11::class_<cpp_pettingzoo::simple_push::SimplePushEnv>(
      m, "SimplePushEnv")
      .def(pybind11::init<int, bool, bool>(),
           pybind11::arg("max_cycles") = 25,
           pybind11::arg("continuous_actions") = false,
           pybind11::arg("dynamic_rescaling") = false)
      .def("get_agents",
           &cpp_pettingzoo::simple_push::SimplePushEnv::get_agents)
      .def("get_state",
           &cpp_pettingzoo::simple_push::SimplePushEnv::get_state)
      .def("get_render_state",
           &cpp_pettingzoo::simple_push::SimplePushEnv::get_render_state)
      .def(
          "reset",
          [](cpp_pettingzoo::simple_push::SimplePushEnv& self,
             std::optional<int> seed) {
            auto obs = self.reset(seed);
            pybind11::dict infos;
            for (const auto& [name, o] : obs) {
              infos[name.c_str()] = pybind11::dict();
            }
            return pybind11::make_tuple(obs, infos);
          },
          pybind11::arg("seed") = pybind11::none())
      .def("step",
           [](cpp_pettingzoo::simple_push::SimplePushEnv& self,
              const cpp_pettingzoo::ActionMap& actions) {
             auto state = self.step(actions);
             pybind11::dict infos;
             for (const auto& [name, o] : state.observations) {
               infos[name.c_str()] = pybind11::dict();
             }
             return pybind11::make_tuple(state.observations, state.rewards,
                                         state.terminations, state.truncations,
                                         infos);
           });
}
