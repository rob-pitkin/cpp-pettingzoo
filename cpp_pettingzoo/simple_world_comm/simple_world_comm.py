"""SimpleWorldComm environment with C++ backend."""

import sys
from pathlib import Path

import gymnasium
import numpy as np
import pygame
from gymnasium.utils import EzPickle
from pettingzoo import ParallelEnv
from pettingzoo.utils.conversions import parallel_to_aec_wrapper

build_dir = Path(__file__).parent.parent.parent / "build"
sys.path.insert(0, str(build_dir))

import _simple_world_comm
from cpp_pettingzoo.core import Agent, Landmark, World


def _obs_size(num_good, num_adversaries, num_obstacles, num_food, num_forests,
              is_adversary, dim_c=4):
    """Compute observation vector length for one agent."""
    n_agents = num_good + num_adversaries
    n_landmarks = num_obstacles + num_food + num_forests
    n_others = n_agents - 1

    # other_vel includes only good-agent velocities
    if is_adversary:
        good_other_vels = num_good
    else:
        good_other_vels = num_good - 1

    base = 4 + 2 * n_landmarks + 2 * n_others + 2 * good_other_vels + num_forests
    if is_adversary:
        return base + dim_c
    return base


class parallel_env(ParallelEnv, EzPickle):
    """SimpleWorldComm parallel environment.

    Cooperative-competitive: adversaries (red) chase good agents (green), with
    a leader adversary (dark red) that can see all and broadcasts a
    communication vector to its team. Forests hide agents inside them from
    being seen by agents outside.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "simple_world_comm_v3",
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        num_good=2,
        num_adversaries=4,
        num_obstacles=1,
        num_food=2,
        max_cycles=25,
        num_forests=2,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            num_food=num_food,
            max_cycles=max_cycles,
            num_forests=num_forests,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            dynamic_rescaling=dynamic_rescaling,
        )

        self.num_good = num_good
        self.num_adversaries = num_adversaries
        self.num_obstacles = num_obstacles
        self.num_food = num_food
        self.num_forests = num_forests
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode
        self.dynamic_rescaling = dynamic_rescaling
        self.dim_c = 4

        self._cpp_env = _simple_world_comm.SimpleWorldCommEnv(
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            num_food=num_food,
            num_forests=num_forests,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )

        # Agent naming: leadadversary_0, adversary_0..adversary_{N-2},
        # agent_0..agent_{M-1}
        adv_names = ["leadadversary_0"] + [
            f"adversary_{i}" for i in range(num_adversaries - 1)
        ]
        good_names = [f"agent_{i}" for i in range(num_good)]
        self.possible_agents = adv_names + good_names
        self.agents = self.possible_agents[:]

        # Action spaces
        # leader: movement(5) x comm(dim_c) -> Discrete(5 * dim_c) or Box(5 + dim_c)
        # others: movement only -> Discrete(5) or Box(5)
        self._action_spaces = {}
        for name in self.possible_agents:
            is_leader = name == "leadadversary_0"
            if self.continuous_actions:
                if is_leader:
                    shape = (5 + self.dim_c,)
                else:
                    shape = (5,)
                self._action_spaces[name] = gymnasium.spaces.Box(
                    low=0.0, high=1.0, shape=shape, dtype="float32"
                )
            else:
                if is_leader:
                    self._action_spaces[name] = gymnasium.spaces.Discrete(
                        5 * self.dim_c
                    )
                else:
                    self._action_spaces[name] = gymnasium.spaces.Discrete(5)

        # Observation spaces (asymmetric: adversaries get +dim_c for comm)
        self._observation_spaces = {}
        for name in self.possible_agents:
            is_adv = "adversary" in name
            obs_dim = _obs_size(
                num_good, num_adversaries, num_obstacles, num_food, num_forests,
                is_adv, self.dim_c
            )
            self._observation_spaces[name] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype="float32"
            )

        if self.render_mode is not None:
            pygame.init()
            self.width = 700
            self.height = 700
            self.screen = pygame.Surface([self.width, self.height])
            self.game_font = pygame.font.Font(None, 24)
            self.viewer = None
            self.renderOn = False
            self.original_cam_range = 1.0

            self.world = World()
            self.world.agents = [Agent(name, silent=True) for name in self.possible_agents]
            for agent in self.world.agents:
                is_adv = "adversary" in agent.name
                agent.size = 0.075 if is_adv else 0.045

            self.world.landmarks = []
            for i in range(num_obstacles):
                lm = Landmark(f"landmark {i}")
                lm.size = 0.2
                lm.color = np.array([0.25, 0.25, 0.25])
                self.world.landmarks.append(lm)
            for i in range(num_food):
                lm = Landmark(f"food {i}")
                lm.size = 0.03
                lm.color = np.array([0.15, 0.15, 0.65])
                self.world.landmarks.append(lm)
            for i in range(num_forests):
                lm = Landmark(f"forest {i}")
                lm.size = 0.3
                lm.color = np.array([0.6, 0.9, 0.6])
                self.world.landmarks.append(lm)

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        observations, infos = self._cpp_env.reset(seed)
        self.agents = self.possible_agents[:]
        observations = {
            agent: np.array(obs, dtype=np.float32)
            for agent, obs in observations.items()
        }
        if self.render_mode is not None:
            self._sync_render_state()
        return observations, infos

    def step(self, actions):
        if not self.continuous_actions:
            actions = {
                agent: np.array([action], dtype=np.float32)
                for agent, action in actions.items()
            }
        else:
            actions = {
                agent: np.array(action, dtype=np.float32)
                for agent, action in actions.items()
            }

        observations, rewards, terminations, truncations, infos = (
            self._cpp_env.step(actions)
        )
        observations = {
            agent: np.array(obs, dtype=np.float32)
            for agent, obs in observations.items()
        }

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        if self.render_mode is not None:
            self._sync_render_state()

        return observations, rewards, terminations, truncations, infos

    def _sync_render_state(self):
        render_state = self._cpp_env.get_render_state()
        for i, name in enumerate(self.possible_agents):
            self.world.agents[i].state.p_pos = np.array(render_state[f"{name}_pos"])
            self.world.agents[i].state.p_vel = np.array(render_state[f"{name}_vel"])
            self.world.agents[i].color = np.array(render_state[f"{name}_color"])
        for i in range(len(self.world.landmarks)):
            self.world.landmarks[i].state.p_pos = np.array(
                render_state[f"landmark_{i}_pos"]
            )
            self.world.landmarks[i].state.p_vel = np.zeros(2)

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.clock = pygame.time.Clock()
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)
        self.draw()

        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def draw(self):
        self.screen.fill((255, 255, 255))

        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))
        scaling_factor = 0.9 * self.original_cam_range / cam_range

        for entity in self.world.entities:
            x, y = entity.state.p_pos
            y *= -1
            x = (x / cam_range) * self.width // 2 * 0.9
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2

            if self.dynamic_rescaling:
                radius = entity.size * 350 * scaling_factor
            else:
                radius = entity.size * 350

            pygame.draw.circle(self.screen, entity.color * 200, (x, y), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius, 1)

    def close(self):
        if hasattr(self, "screen") and self.screen is not None:
            pygame.quit()
            self.screen = None

    def state(self):
        return np.array(self._cpp_env.get_state(), dtype=np.float32)


# Alias for symmetry with mpe2
raw_env = parallel_env


def env(**kwargs):
    aec_env = parallel_env(**kwargs)
    aec_env = parallel_to_aec_wrapper(aec_env)
    return aec_env
