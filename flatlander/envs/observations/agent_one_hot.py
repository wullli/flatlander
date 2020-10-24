import gym
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent

from flatlander.envs.observations import Observation, register_obs
from flatlander.envs.observations.common.utils import one_hot


@register_obs("agent_one_hot")
class AgentOneHotObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._max_n_agents = config.get('max_n_agents', 5)
        self._builder = AgentOneHotBuilder(max_n_agents=self._max_n_agents)

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        dim = 13
        dim += 0 if not self._concat_handle else self._max_n_agents
        return gym.spaces.Box(low=-1, high=1, shape=(dim,))


class AgentOneHotBuilder(ObservationBuilder):
    def __init__(self, max_n_agents=5):
        super().__init__()
        self.max_n_agents = max_n_agents

    def get(self, handle: int = 0):
        obs = one_hot([handle], self.max_n_agents)
        return obs

    def reset(self):
        pass
