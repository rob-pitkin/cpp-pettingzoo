#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "simple_world_comm_env.h"

PYBIND11_MODULE(_simple_world_comm, m) {
  m.doc() = "Simple World Comm Environment C++ Implementation";

  pybind11::class_<cpp_pettingzoo::simple_world_comm::SimpleWorldCommEnv>(
      m, "SimpleWorldCommEnv")
      .def(pybind11::init<int, int, int, int, int, int, bool, bool>(),
           pybind11::arg("num_good") = 2,
           pybind11::arg("num_adversaries") = 4,
           pybind11::arg("num_obstacles") = 1,
           pybind11::arg("num_food") = 2,
           pybind11::arg("num_forests") = 2,
           pybind11::arg("max_cycles") = 25,
           pybind11::arg("continuous_actions") = false,
           pybind11::arg("dynamic_rescaling") = false)
      .def("get_agents",
           &cpp_pettingzoo::simple_world_comm::SimpleWorldCommEnv::get_agents)
      .def("get_state",
           &cpp_pettingzoo::simple_world_comm::SimpleWorldCommEnv::get_state)
      .def("get_render_state",
           &cpp_pettingzoo::simple_world_comm::SimpleWorldCommEnv::
               get_render_state)
      .def(
          "reset",
          [](cpp_pettingzoo::simple_world_comm::SimpleWorldCommEnv& self,
             std::optional<int> seed) {
            auto obs = self.reset(seed);
            pybind11::dict infos;
            for (const auto& [agent_name, o] : obs) {
              infos[agent_name.c_str()] = pybind11::dict();
            }
            return pybind11::make_tuple(obs, infos);
          },
          pybind11::arg("seed") = pybind11::none())
      .def("step",
           [](cpp_pettingzoo::simple_world_comm::SimpleWorldCommEnv& self,
              const cpp_pettingzoo::ActionMap& actions) {
             auto state = self.step(actions);
             pybind11::dict infos;
             for (const auto& [agent_name, o] : state.observations) {
               infos[agent_name.c_str()] = pybind11::dict();
             }
             return pybind11::make_tuple(state.observations, state.rewards,
                                         state.terminations, state.truncations,
                                         infos);
           });
}
