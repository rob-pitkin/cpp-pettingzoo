"""SimpleFormation environment with C++ backend."""

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

import _simple_formation
from cpp_pettingzoo.core import Agent, Landmark, World

# Each agent: vel(2) + pos(2) + landmark_rel_pos(2) = 6
_OBS_SIZE = 6


class parallel_env(ParallelEnv, EzPickle):
    """SimpleFormation parallel environment.

    N agents cooperate to arrange themselves in a circle of radius 0.5 around
    a central landmark. Reward is the negative mean assigned distance (Hungarian
    matching), shared equally by all agents.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "simple_formation_v1",
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        N=4,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
        terminate_on_success=False,
    ):
        EzPickle.__init__(
            self,
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            dynamic_rescaling=dynamic_rescaling,
            terminate_on_success=terminate_on_success,
        )

        self.N = N
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode
        self.dynamic_rescaling = dynamic_rescaling
        self.terminate_on_success = terminate_on_success

        self._cpp_env = _simple_formation.SimpleFormationEnv(
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
            terminate_on_success=terminate_on_success,
        )

        self.possible_agents = [f"agent_{i}" for i in range(N)]
        self.agents = self.possible_agents[:]

        if self.continuous_actions:
            self._action_spaces = {
                agent: gymnasium.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype="float32")
                for agent in self.possible_agents
            }
        else:
            self._action_spaces = {
                agent: gymnasium.spaces.Discrete(5)
                for agent in self.possible_agents
            }

        self._observation_spaces = {
            agent: gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(_OBS_SIZE,), dtype="float32"
            )
            for agent in self.possible_agents
        }

        if self.render_mode is not None:
            pygame.init()
            self.width = 700
            self.height = 700
            self.screen = pygame.Surface([self.width, self.height])
            self.game_font = pygame.font.Font(None, 24)
            self.renderOn = False
            self.original_cam_range = 1.0

            self.world = World()
            self.world.agents = [Agent(name, silent=True) for name in self.possible_agents]
            self.world.landmarks = [Landmark("landmark_0")]
            for agent in self.world.agents:
                agent.size = 0.05
            self.world.landmarks[0].size = 0.03
            self.world.landmarks[0].color = np.array([0.25, 0.25, 0.25])

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

        observations, rewards, terminations, truncations, infos = self._cpp_env.step(actions)
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
        self.world.landmarks[0].state.p_pos = np.array(render_state["landmark_0_pos"])
        self.world.landmarks[0].state.p_vel = np.zeros(2)

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
            radius = entity.size * 350 * scaling_factor if self.dynamic_rescaling else entity.size * 350
            pygame.draw.circle(self.screen, entity.color * 200, (x, y), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius, 1)

    def close(self):
        if hasattr(self, "screen") and self.screen is not None:
            pygame.quit()
            self.screen = None

    def state(self):
        return np.array(self._cpp_env.get_state(), dtype=np.float32)

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]


def env(**kwargs):
    aec_env = parallel_env(**kwargs)
    aec_env = parallel_to_aec_wrapper(aec_env)
    return aec_env
