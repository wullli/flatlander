import gym
import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid import grid4
from flatland.envs.observations import GlobalObsForRailEnv
from flatlander.envs.observations import Observation, register_obs
from flatlander.envs.observations.builders.perfect_information_global_obs import PerfectInformationGlobalObs

'''
A 2-d array matrix on-hot encoded similar to tf.one_hot function
https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy/36960495
'''


@register_obs("pi_global")
class PerfectInformationGlobalObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._config = config
        self._builder = PaddedGlobalObsForRailEnv(max_width=config['max_width'],
                                                  max_height=config['max_height'],
                                                  max_n_agents=config['max_n_agents'])

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        grid_shape = (self._config['max_width'], self._config['max_height'])
        return gym.spaces.Box(low=0, high=np.inf, shape=grid_shape + (36,), dtype=np.float32)


class PaddedGlobalObsForRailEnv(ObservationBuilder):

    def __init__(self, max_width, max_height, max_n_agents):
        super().__init__()
        self._max_n_agents = max_n_agents
        self._max_width = max_width
        self._max_height = max_height
        self._builder = PerfectInformationGlobalObs(max_n_agents=self._max_n_agents)

    def set_env(self, env: Environment):
        self._builder.set_env(env)

    def reset(self):
        self._builder.reset()

    def get_many(self, handle: int = 0):
        obs = self._builder.get_many()
        height, width = obs.shape[:2]
        pad_height, pad_width = self._max_height - height, self._max_width - width
        assert pad_height >= 0 and pad_width >= 0
        return np.pad(obs, ((0, pad_height), (0, pad_height), (0, 0)), constant_values=0)

    def get(self, handle: int = 0):
        return None
