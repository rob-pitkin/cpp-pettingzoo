#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "simple_tag_env.h"

PYBIND11_MODULE(_simple_tag, m) {
  m.doc() = "Simple Tag Environment C++ Implementation";

  pybind11::class_<cpp_pettingzoo::simple_tag::SimpleTagEnv>(m, "SimpleTagEnv")
      .def(pybind11::init<int, int, int, int, bool, bool, bool, bool,
                          std::optional<int>, std::optional<int>>(),
           pybind11::arg("num_good") = 1,
           pybind11::arg("num_adversaries") = 3,
           pybind11::arg("num_obstacles") = 2,
           pybind11::arg("max_cycles") = 25,
           pybind11::arg("continuous_actions") = false,
           pybind11::arg("dynamic_rescaling") = false,
           pybind11::arg("curriculum") = false,
           pybind11::arg("terminate_on_success") = false,
           pybind11::arg("num_agent_neighbors") = pybind11::none(),
           pybind11::arg("num_landmark_neighbors") = pybind11::none())
      .def("get_agents",
           &cpp_pettingzoo::simple_tag::SimpleTagEnv::get_agents)
      .def("get_state",
           &cpp_pettingzoo::simple_tag::SimpleTagEnv::get_state)
      .def("get_render_state",
           &cpp_pettingzoo::simple_tag::SimpleTagEnv::get_render_state)
      .def("get_curriculum_stage",
           &cpp_pettingzoo::simple_tag::SimpleTagEnv::get_curriculum_stage)
      .def("advance_curriculum",
           &cpp_pettingzoo::simple_tag::SimpleTagEnv::advance_curriculum)
      .def("set_curriculum_stage",
           &cpp_pettingzoo::simple_tag::SimpleTagEnv::set_curriculum_stage)
      .def(
          "reset",
          [](cpp_pettingzoo::simple_tag::SimpleTagEnv& self,
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
           [](cpp_pettingzoo::simple_tag::SimpleTagEnv& self,
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
