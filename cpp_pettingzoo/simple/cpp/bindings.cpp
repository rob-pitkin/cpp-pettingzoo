#include "simple_core.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(_simple_core, m) {
  m.doc() = "Simple Environment C++ Implementation";

  pybind11::class_<cpp_pettingzoo::SimpleEnv>(m, "SimpleEnv")
      .def(pybind11::init<int, bool, bool>(), pybind11::arg("max_cycles") = 25,
           pybind11::arg("dynamic_rescaling") = false,
           pybind11::arg("continuous_actions") = false)
      .def("get_agents", &cpp_pettingzoo::SimpleEnv::get_agents)
      .def("get_state", &cpp_pettingzoo::SimpleEnv::get_state)
      .def("get_render_state", &cpp_pettingzoo::SimpleEnv::get_render_state)
      .def(
          "reset",
          [](cpp_pettingzoo::SimpleEnv &self, std::optional<int> seed) {
            auto obs = self.reset(seed);
            // Return (observations, infos) tuple to match PettingZoo API
            pybind11::dict infos;
            infos["agent_0"] = pybind11::dict();
            return pybind11::make_tuple(obs, infos);
          },
          pybind11::arg("seed") = pybind11::none())
      .def("step", [](cpp_pettingzoo::SimpleEnv &self,
                      const cpp_pettingzoo::ActionMap &actions) {
        auto state = self.step(actions);
        // Return (observations, rewards, terminations, truncations, infos)
        // tuple to match PettingZoo API
        pybind11::dict infos;
        infos["agent_0"] = pybind11::dict();
        return pybind11::make_tuple(state.observations, state.rewards,
                                    state.terminations, state.truncations,
                                    infos);
      });
}
