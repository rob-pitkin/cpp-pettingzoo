"""SimpleTag environment with C++ backend."""

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

import _simple_tag
from cpp_pettingzoo.core import Agent, Landmark, World


def _obs_size(num_good, num_adversaries, num_obstacles, num_agent_neighbors,
              num_landmark_neighbors, is_adversary):
    """Compute observation vector length for one agent."""
    lm_slots = num_landmark_neighbors if num_landmark_neighbors is not None else num_obstacles
    other_agents = num_good + num_adversaries - 1
    agent_slots = num_agent_neighbors if num_agent_neighbors is not None else other_agents

    if num_agent_neighbors is None:
        # full obs: good-agent velocities only (excluding self if good)
        good_vel_slots = num_good if is_adversary else num_good - 1
    else:
        good_vel_slots = num_agent_neighbors

    return 2 + 2 + 2 * lm_slots + 2 * agent_slots + 2 * good_vel_slots


class parallel_env(ParallelEnv, EzPickle):
    """SimpleTag parallel environment.

    Adversaries (red) try to catch good agents (green).
    Good agents are faster but penalized for leaving the play area.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "simple_tag_v3",
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
        curriculum=False,
        terminate_on_success=False,
        num_agent_neighbors=None,
        num_landmark_neighbors=None,
    ):
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            dynamic_rescaling=dynamic_rescaling,
            curriculum=curriculum,
            terminate_on_success=terminate_on_success,
            num_agent_neighbors=num_agent_neighbors,
            num_landmark_neighbors=num_landmark_neighbors,
        )

        self.num_good = num_good
        self.num_adversaries = num_adversaries
        self.num_obstacles = num_obstacles
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode
        self.dynamic_rescaling = dynamic_rescaling
        self.curriculum = curriculum
        self.terminate_on_success = terminate_on_success
        self.num_agent_neighbors = num_agent_neighbors
        self.num_landmark_neighbors = num_landmark_neighbors

        self._cpp_env = _simple_tag.SimpleTagEnv(
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
            curriculum=curriculum,
            terminate_on_success=terminate_on_success,
            num_agent_neighbors=num_agent_neighbors,
            num_landmark_neighbors=num_landmark_neighbors,
        )

        self.possible_agents = (
            [f"adversary_{i}" for i in range(num_adversaries)]
            + [f"agent_{i}" for i in range(num_good)]
        )
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

        self._observation_spaces = {}
        for agent in self.possible_agents:
            is_adv = agent.startswith("adversary")
            obs_dim = _obs_size(
                num_good, num_adversaries, num_obstacles,
                num_agent_neighbors, num_landmark_neighbors, is_adv
            )
            self._observation_spaces[agent] = gymnasium.spaces.Box(
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
            self.world.agents = [
                Agent(name, silent=True) for name in self.possible_agents
            ]
            self.world.landmarks = [
                Landmark(f"landmark {i}") for i in range(num_obstacles)
            ]

            for agent in self.world.agents:
                agent.size = 0.075 if agent.name.startswith("adversary") else 0.05

            for landmark in self.world.landmarks:
                landmark.size = 0.2

    # ------------------------------------------------------------------
    # Curriculum API (delegated to C++)
    # ------------------------------------------------------------------

    @property
    def curriculum_stage(self):
        return self._cpp_env.get_curriculum_stage()

    def advance_curriculum(self):
        self._cpp_env.advance_curriculum()

    def set_curriculum_stage(self, stage):
        self._cpp_env.set_curriculum_stage(stage)

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

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
        for i in range(self.num_obstacles):
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

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]


def env(**kwargs):
    aec_env = parallel_env(**kwargs)
    aec_env = parallel_to_aec_wrapper(aec_env)
    return aec_env
