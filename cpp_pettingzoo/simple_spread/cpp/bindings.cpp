#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "simple_spread_env.h"

PYBIND11_MODULE(_simple_spread, m) {
  m.doc() = "Simple Spread Environment C++ Implementation";

  pybind11::class_<cpp_pettingzoo::simple_spread::SimpleSpreadEnv>(
      m, "SimpleSpreadEnv")
      .def(pybind11::init<int, bool, bool, float, bool, int>(),
           pybind11::arg("max_cycles") = 25,
           pybind11::arg("dynamic_rescaling") = false,
           pybind11::arg("continuous_actions") = false,
           pybind11::arg("local_ratio") = 0.5f,
           pybind11::arg("curriculum") = false,
           pybind11::arg("curriculum_stage") = 0)
      .def("get_agents",
           &cpp_pettingzoo::simple_spread::SimpleSpreadEnv::get_agents)
      .def("get_state",
           &cpp_pettingzoo::simple_spread::SimpleSpreadEnv::get_state)
      .def("get_render_state",
           &cpp_pettingzoo::simple_spread::SimpleSpreadEnv::get_render_state)
      .def(
          "reset",
          [](cpp_pettingzoo::simple_spread::SimpleSpreadEnv& self,
             std::optional<int> seed) {
            auto obs = self.reset(seed);
            // Return (observations, infos) tuple to match PettingZoo API
            pybind11::dict infos;
            for (const auto& [agent_name, o] : obs) {
              infos[agent_name.c_str()] = pybind11::dict();
            }
            return pybind11::make_tuple(obs, infos);
          },
          pybind11::arg("seed") = pybind11::none())
      .def("step",
           [](cpp_pettingzoo::simple_spread::SimpleSpreadEnv& self,
              const cpp_pettingzoo::ActionMap& actions) {
             auto state = self.step(actions);
             // Return (observations, rewards, terminations, truncations, infos)
             pybind11::dict infos;
             for (const auto& [agent_name, o] : state.observations) {
               infos[agent_name.c_str()] = pybind11::dict();
             }
             return pybind11::make_tuple(state.observations, state.rewards,
                                         state.terminations, state.truncations,
                                         infos);
           })
      .def(
          "get_curriculum_stage",
          &cpp_pettingzoo::simple_spread::SimpleSpreadEnv::get_curriculum_stage)
      .def("advance_curriculum",
           &cpp_pettingzoo::simple_spread::SimpleSpreadEnv::advance_curriculum)
      .def("set_curriculum_stage", &cpp_pettingzoo::simple_spread::
                                       SimpleSpreadEnv::set_curriculum_stage);
}
