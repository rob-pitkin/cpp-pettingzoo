#ifndef ENTITY_H_
#define ENTITY_H_

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace cpp_pettingzoo::core {

struct EntityState {
  std::array<float, 2> p_pos;
  std::array<float, 2> p_vel;
};

struct Action {
  std::array<float, 2> u;  // physical force
  std::vector<float> c;    // Communication

  explicit Action(size_t dim_c = 0) : c(dim_c, 0.0f) {}
};

class Entity {
 public:
  std::string name;
  float size = 0.050f;
  bool movable = false;
  bool collide = true;
  float density = 25.0f;
  float initial_mass = 1.0f;
  std::optional<float> max_speed = std::nullopt;
  std::optional<float> accel = std::nullopt;
  std::array<float, 3> color = {0.5f, 0.5f, 0.5f};
  EntityState state;

  float mass() const { return initial_mass; };

  Entity() = default;
  explicit Entity(std::string n) : name(std::move(n)) {};
};

class Landmark : public Entity {
 public:
  Landmark() = default;
  explicit Landmark(std::string n) : Entity(std::move(n)) {};
};

class Agent : public Entity {
 public:
  bool silent = false;
  bool blind = false;
  std::optional<float> u_noise = std::nullopt;
  std::optional<float> c_noise = std::nullopt;
  float u_range = 1.0f;
  std::vector<float> c;  // Communication, size = dim_c
  Action action;

  Agent() { movable = true; }
  explicit Agent(std::string n, size_t dim_c)
      : Entity(std::move(n)), c(dim_c, 0.0f), action(dim_c) {
    movable = true;
  };
};

}  // namespace cpp_pettingzoo::core

#endif  // ENTITY_H_
