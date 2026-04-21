#include <cmath>
#include <random>

#include "core/entity.h"
#include "simple_speaker_listener_scenario.h"

namespace cpp_pettingzoo::simple_speaker_listener {

void SimpleSpeakerListenerScenario::make_world(core::World& w) {
  w.dim_c = 3;

  core::Agent speaker = core::Agent("speaker_0", w.dim_c);
  speaker.movable = false;
  speaker.size = 0.075f;
  speaker.collide = false;
  speaker.silent = false;
  w.agents.push_back(std::move(speaker));

  core::Agent listener = core::Agent("listener_0", w.dim_c);
  listener.movable = true;
  listener.size = 0.075f;
  listener.collide = false;
  listener.silent = true;
  w.agents.push_back(std::move(listener));

  for (int i = 0; i < 3; ++i) {
    core::Landmark l = core::Landmark("landmark " + std::to_string(i));
    l.size = 0.04f;
    l.collide = false;
    l.movable = false;
    w.landmarks.push_back(std::move(l));
  }
}

void SimpleSpeakerListenerScenario::reset_world(core::World& w) {
  w.agents[0].goal_a = &w.agents[1];

  auto& rng = w.get_rng();
  auto dist = std::uniform_int_distribution<>(0, 2);

  w.agents[0].goal_b = &w.landmarks[dist(rng)];

  auto float_dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
  w.agents[0].state.p_pos = {float_dist(rng), float_dist(rng)};
  w.agents[0].state.p_vel = {0.0f, 0.0f};
  w.agents[0].c = std::vector<float>(w.dim_c, 0.0f);
  w.agents[0].color = {0.25f, 0.25f, 0.25f};

  w.agents[1].state.p_pos = {float_dist(rng), float_dist(rng)};
  w.agents[1].state.p_vel = {0.0f, 0.0f};
  w.agents[1].c = std::vector<float>(w.dim_c, 0.0f);
  w.agents[1].color = w.agents[0].goal_b->color;
  for (size_t i = 0; i < w.agents[0].goal_b->color.size(); ++i) {
    w.agents[1].color[i] += 0.45f;
  }

  for (int i = 0; i < 3; ++i) {
    auto& landmark = w.landmarks[i];
    landmark.state.p_pos = {float_dist(rng), float_dist(rng)};
    landmark.state.p_vel = {0.0f, 0.0f};

    if (i == 0)
      landmark.color = {0.65f, 0.15f, 0.15f};  // red
    else if (i == 1)
      landmark.color = {0.15f, 0.65f, 0.15f};  // green
    else
      landmark.color = {0.15f, 0.15f, 0.65f};  // blue
  }
}

float SimpleSpeakerListenerScenario::reward(const core::Agent& agent,
                                            const core::World& world) const {
  const auto& speaker = world.agents[0];
  const auto& listener = speaker.goal_a;
  const auto& goal = speaker.goal_b;
  float dx = listener->state.p_pos[0] - goal->state.p_pos[0];
  float dy = listener->state.p_pos[1] - goal->state.p_pos[1];
  return -(dx * dx + dy * dy);
}

float SimpleSpeakerListenerScenario::global_reward(const core::World& w) const {
  float glob_reward = 0.0f;
  for (const auto& agent : w.agents) {
    glob_reward += reward(agent, w);
  }
  return glob_reward / static_cast<float>(w.agents.size());
}

std::vector<float> SimpleSpeakerListenerScenario::observation(
    const core::Agent& agent, const core::World& world) const {
  std::vector<float> obs;

  if (!agent.movable) {
    // speaker case
    if (agent.goal_b != nullptr) {
      obs.insert(obs.end(), agent.goal_b->color.begin(),
                 agent.goal_b->color.end());
    } else {
      obs = std::vector<float>(3, 0.0f);
    }
    return obs;
  }

  if (agent.silent) {
    // listener case
    obs.reserve(11);
    obs.push_back(agent.state.p_vel[0]);
    obs.push_back(agent.state.p_vel[1]);

    for (const auto& l : world.landmarks) {
      float rel_x = l.state.p_pos[0] - agent.state.p_pos[0];
      float rel_y = l.state.p_pos[1] - agent.state.p_pos[1];
      obs.push_back(rel_x);
      obs.push_back(rel_y);
    }

    const auto& speaker = world.agents[0];
    if (!speaker.silent) {
      obs.insert(obs.end(), speaker.c.begin(), speaker.c.end());
    }
    return obs;
  }

  return obs;
}

}  // namespace cpp_pettingzoo::simple_speaker_listener
