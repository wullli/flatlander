from typing import Optional, List

import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus


class PerfectInformationGlobalObs(ObservationBuilder):
    """
    Gives a global observation of the entire rail environment.
    The observation is composed of the following elements:

        - transition map array with dimensions (env.height, env.width, 16),\
          assuming 16 bits encoding of transitions.

        - obs_agents_state: A 3D array (map_height, map_width, 5) with
            - first channel containing the agents position and direction
            - second channel containing the other agents positions and direction
            - third channel containing agent/other agent malfunctions
            - fourth channel containing agent/other agent fractional speeds
            - fifth channel containing number of other agents ready to depart

        - obs_targets: Two 2D arrays (map_height, map_width, 2) containing respectively the position of the given agent\
         target and the positions of the other agents targets (flag only, no counter!).
    """

    def __init__(self, max_n_agents: int = 5):
        super(PerfectInformationGlobalObs, self).__init__()
        self.max_n_agents = max_n_agents
        self.rail_obs = None

    def set_env(self, env: Environment):
        super().set_env(env)

    def reset(self):
        self.rail_obs = np.zeros((self.env.height, self.env.width, 16))
        for i in range(self.rail_obs.shape[0]):
            for j in range(self.rail_obs.shape[1]):
                bit_list = [int(digit) for digit in bin(self.env.rail.get_full_transitions(i, j))[2:]]
                bit_list = [0] * (16 - len(bit_list)) + bit_list
                self.rail_obs[i, j] = np.array(bit_list)

    def _one_hot_agent(self, handle, old: np.ndarray = None):
        if old is not None:
            oh = old
        else:
            oh = np.zeros(self.max_n_agents)
        oh[handle] = 1
        return oh

    def _one_hot_direction(self, direction):
        oh = np.zeros(4)
        oh[direction] = 1
        return oh

    def get(self, **kwargs):
        return None

    def get_many(self, handles: Optional[List[int]] = None) -> np.ndarray:
        obs_agent_ids = np.zeros((self.env.height, self.env.width, self.max_n_agents))
        obs_agent_directions = np.zeros((self.env.height, self.env.width, 4))
        obs_agent_malfunctions = np.zeros((self.env.height, self.env.width, 1))
        obs_agent_initials = np.zeros((self.env.height, self.env.width, self.max_n_agents))
        obs_agent_targets = np.zeros((self.env.height, self.env.width, self.max_n_agents))

        for agent in self.env.agents:

            # ignore other agents not in the grid any more
            if agent.status == RailAgentStatus.DONE_REMOVED:
                continue

            obs_agent_targets[agent.target][:] = self._one_hot_agent(agent.handle)

            # second to fourth channel only if in the grid
            if agent.position is not None:
                # second channel only for other agents
                obs_agent_ids[agent.position][:] = self._one_hot_agent(agent.handle)
                obs_agent_directions[agent.position][:] = self._one_hot_direction(agent.direction)
                obs_agent_malfunctions[agent.position][:] = agent.malfunction_data['malfunction']
            # fifth channel: all ready to depart on this position
            if agent.status == RailAgentStatus.READY_TO_DEPART:
                old = obs_agent_initials[agent.initial_position][:]
                obs_agent_initials[agent.initial_position][:] = self._one_hot_agent(agent.handle, old=old)

        obs = np.concatenate([self.rail_obs,
                              obs_agent_ids,
                              obs_agent_directions,
                              obs_agent_malfunctions,
                              obs_agent_initials,
                              obs_agent_targets], axis=-1)
        return obs
