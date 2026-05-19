"""CollectTreasure environment with C++ backend."""

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

import _collect_treasure
from cpp_mpe2._wrappers import check_and_maybe_clip_actions, make_aec_env
from cpp_mpe2.core import Agent, Landmark, World

_TYPE_COLORS = [
    np.array([0.212, 0.408, 0.776]),  # blue
    np.array([0.945, 0.553, 0.000]),  # orange
    np.array([0.169, 0.627, 0.173]),  # green
    np.array([0.839, 0.149, 0.157]),  # red
    np.array([0.580, 0.404, 0.741]),  # purple
    np.array([0.549, 0.337, 0.294]),  # brown
]


def _obs_size(num_collectors, num_deposits, num_treasures, is_collector):
    n_agents = num_collectors + num_deposits
    n_types = num_deposits
    base = 4  # pos(2) + vel(2)
    if is_collector:
        base += n_types  # holding one-hot
    others = (n_agents - 1) * (4 + 2 * n_types)  # rel_pos + vel + encoding
    treasures = num_treasures * (2 + n_types)
    return base + others + treasures


class parallel_env(ParallelEnv, EzPickle):
    """CollectTreasure parallel environment.

    Collector agents pick up typed treasures and deliver them to matching
    deposit agents. Reward blends global pickup/delivery bonuses with local
    distance shaping and collector collision penalties.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "collect_treasure_v1",
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        num_collectors=6,
        num_deposits=2,
        num_treasures=6,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
         benchmark_data=False,
    ):
        EzPickle.__init__(
            self,
            num_collectors=num_collectors,
            num_deposits=num_deposits,
            num_treasures=num_treasures,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            dynamic_rescaling=dynamic_rescaling,
             benchmark_data=benchmark_data,
        )

        self.num_collectors = num_collectors
        self.num_deposits = num_deposits
        self.num_treasures = num_treasures
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode
        self.dynamic_rescaling = dynamic_rescaling

        self._cpp_env = _collect_treasure.CollectTreasureEnv(
            num_collectors=num_collectors,
            num_deposits=num_deposits,
            num_treasures=num_treasures,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )

        collector_names = [f"collector_{i}" for i in range(num_collectors)]
        deposit_names = [f"deposit_{i}" for i in range(num_deposits)]
        self.possible_agents = collector_names + deposit_names
        self.agents = self.possible_agents[:]

        collector_obs = _obs_size(num_collectors, num_deposits, num_treasures, True)
        deposit_obs = _obs_size(num_collectors, num_deposits, num_treasures, False)

        if self.continuous_actions:
            self._action_spaces = {
                a: gymnasium.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype="float32")
                for a in self.possible_agents
            }
        else:
            self._action_spaces = {
                a: gymnasium.spaces.Discrete(5)
                for a in self.possible_agents
            }

        self._observation_spaces = {}
        for a in collector_names:
            self._observation_spaces[a] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(collector_obs,), dtype="float32"
            )
        for a in deposit_names:
            self._observation_spaces[a] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(deposit_obs,), dtype="float32"
            )

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
            self.world.landmarks = [Landmark(f"treasure_{i}") for i in range(num_treasures)]

            for i, agent in enumerate(self.world.agents):
                if i < num_collectors:
                    agent.size = 0.05
                    agent.color = np.array([0.85, 0.85, 0.85])
                else:
                    agent.size = 0.075
                    d_i = i - num_collectors
                    agent.color = _TYPE_COLORS[d_i % len(_TYPE_COLORS)] * 0.35

            for lm in self.world.landmarks:
                lm.size = 0.025
                lm.color = _TYPE_COLORS[0].copy()

    def reset(self, seed=None, options=None):
        observations, infos = self._cpp_env.reset(seed)
        self.agents = self.possible_agents[:]
        observations = {
            a: np.array(obs, dtype=np.float32)
            for a, obs in observations.items()
        }
        if self.render_mode is not None:
            self._sync_render_state()
        return observations, infos

    def step(self, actions):
        actions = check_and_maybe_clip_actions(actions, self.action_space, self.continuous_actions)
        if not self.continuous_actions:
            actions = {
                a: np.array([action], dtype=np.float32)
                for a, action in actions.items()
            }
        else:
            actions = {
                a: np.array(action, dtype=np.float32)
                for a, action in actions.items()
            }

        observations, rewards, terminations, truncations, infos = self._cpp_env.step(actions)
        observations = {
            a: np.array(obs, dtype=np.float32)
            for a, obs in observations.items()
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
        for i in range(self.num_treasures):
            self.world.landmarks[i].state.p_pos = np.array(render_state[f"landmark_{i}_pos"])
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
        # Skip dead treasures (teleported to -999)
        visible = [
            e for e in self.world.entities
            if not (hasattr(e, 'state') and
                    np.any(np.abs(e.state.p_pos) > 100))
        ]
        if not visible:
            return

        all_poses = [e.state.p_pos for e in visible]
        cam_range = np.max(np.abs(np.array(all_poses)))
        if cam_range == 0:
            cam_range = 1.0
        scaling_factor = 0.9 * self.original_cam_range / cam_range

        for entity in visible:
            x, y = entity.state.p_pos
            y *= -1
            x = (x / cam_range) * self.width // 2 * 0.9 + self.width // 2
            y = (y / cam_range) * self.height // 2 * 0.9 + self.height // 2
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
    return make_aec_env(parallel_env(**kwargs))
