import logging
from abc import ABC

import numpy as np

from flatland.envs.malfunction_generators import ParamMalfunctionGen, \
    NoMalfunctionGen
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatlander.envs.flatland_sparse import FlatlandSparse
from flatlander.envs.utils.gym_env import FlatlandGymEnv
from flatlander.envs.utils.gym_env_wrappers import FlatlandRenderWrapper as RailEnv


class FlatlandSparseScaling(FlatlandSparse, ABC):

    def __init__(self, env_config) -> None:
        self._add_agent_interval = env_config.get("add_agent_interval", 10000)
        self._interval_growth_rate = env_config.get("interval_growth_rate", 1.1)
        self._max_agents = env_config.get("max_agents", 10)
        self._cumulative_steps = 0
        self._prev_num_agents = 1
        self._agent_intervals = np.zeros(self._max_agents)
        self._agent_intervals[0] = self._add_agent_interval
        for i in range(1, self._max_agents):
            if i == 1:
                new_interval = self._agent_intervals[i - 1] * self._interval_growth_rate
            else:
                new_interval = (self._agent_intervals[i - 1] - self._agent_intervals[i - 2]) \
                               * self._interval_growth_rate
            self._agent_intervals[i] = new_interval + self._agent_intervals[i - 1]

        super(FlatlandSparseScaling, self).__init__(env_config)

    def step(self, action_dict):
        self._cumulative_steps += 1
        return super(FlatlandSparseScaling, self).step(action_dict)

    def get_num_agents(self, steps):
        num_agents = 1
        for i, interval in enumerate(self._agent_intervals):
            if steps < interval:
                return num_agents
            else:
                num_agents += 1
        return self._max_agents

    def _launch(self):
        rail_generator = self.get_rail_generator()

        malfunction_generator = NoMalfunctionGen()
        if {'malfunction_rate', 'malfunction_min_duration', 'malfunction_max_duration'} <= self._config.keys():
            stochastic_data = {
                'malfunction_rate': self._config['malfunction_rate'],
                'min_duration': self._config['malfunction_min_duration'],
                'max_duration': self._config['malfunction_max_duration']
            }
            malfunction_generator = ParamMalfunctionGen(stochastic_data)

        speed_ratio_map = None
        if 'speed_ratio_map' in self._config:
            speed_ratio_map = {
                float(k): float(v) for k, v in self._config['speed_ratio_map'].items()
            }
        schedule_generator = sparse_schedule_generator(speed_ratio_map)

        env = None
        try:
            print("GENERATE NEW ENV WITH", self._prev_num_agents, "AGENTS")
            env = RailEnv(
                width=self._config['width'],
                height=self._config['height'],
                rail_generator=rail_generator,
                schedule_generator=schedule_generator,
                number_of_agents=self._prev_num_agents,
                malfunction_generator=malfunction_generator,
                obs_builder_object=self._observation.builder(),
                remove_agents_at_target=False,
                random_seed=self._config['seed'],
                use_renderer=self._env_config.get('render')
            )

            env.reset()
        except ValueError as e:
            logging.error("=" * 50)
            logging.error(f"Error while creating env: {e}")
            logging.error("=" * 50)

        return env

    def get_env(self):
        return FlatlandGymEnv(
            rail_env=self._launch(),
            observation_space=self._observation.observation_space(),
            render=self._env_config.get('render'),
            regenerate_rail_on_reset=self._config['regenerate_rail_on_reset'],
            regenerate_schedule_on_reset=self._config['regenerate_schedule_on_reset']
        )

    def reset(self, *args, **kwargs):
        cur_num_agents = self.get_num_agents(self._cumulative_steps)
        if cur_num_agents > self._prev_num_agents:
            self._prev_num_agents = cur_num_agents
            self._env = self.get_env()
        self._env.reset(*args, **kwargs)
        return self._env.reset(*args, **kwargs)
