import gym
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent

from flatlander.envs.observations import Observation, register_obs
from flatlander.envs.observations.common.utils import one_hot


@register_obs("agent_info")
class AgentInfoObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._max_n_agents = config.get('max_n_agents', 5)
        self._concat_handle = config.get('concat_hanlde', False)
        self._builder = AgentInfoBuilder(max_n_agents=self._max_n_agents,
                                         concat_handle=self._concat_handle)

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        dim = 13
        dim += 0 if not self._concat_handle else self._max_n_agents
        return gym.spaces.Box(low=-1, high=1, shape=(dim,))


class AgentInfoBuilder(ObservationBuilder):
    def __init__(self, max_n_agents=5, concat_handle=True):
        super().__init__()
        self.max_n_agents = max_n_agents
        self.concat_handle = concat_handle

    def get(self, handle: int = 0):
        agent: EnvAgent = self.env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None
        obs = np.concatenate([one_hot([agent.status.value], 4),
                              np.array(agent_virtual_position) / self.env.height,
                              np.array(agent.target) / self.env.height,
                              one_hot([agent.direction], 4),
                              [int(agent.moving)]])
        if self.concat_handle:
            obs = np.concatenate([obs, one_hot([handle], self.max_n_agents)])
        return obs

    def reset(self):
        pass
