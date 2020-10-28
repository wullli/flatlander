from typing import Optional, List, Dict

import gym
import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatlander.envs.observations import Observation, register_obs


@register_obs("simple_meta")
class SimpleMetaObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._builder = SimpleMetaObservationBuilder()

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=0, high=1, shape=(2,),
                              dtype=np.float32)  # own distance to target & nr agents at start


class SimpleMetaObservationBuilder(ObservationBuilder):

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """
        get density maps for agents and compose the observation with agent's and other's density maps
        """
        obs = {}
        for handle in handles:
            obs[handle] = self.get(handle)
        return obs

    def get(self, handle: int = 0):
        """
        compute density map for agent: a value is asigned to every cell along the shortest path between
        the agent and its target based on the distance to the agent, i.e. the number of time steps the
        agent needs to reach the cell, encoding the time information.
        """
        distance_map = self.env.distance_map.get()
        nan_inf_mask = ((distance_map != np.inf) * (np.abs(np.isnan(distance_map) - 1))).astype(np.bool)
        max_distance = np.max(distance_map[nan_inf_mask])
        agent = self.env.agents[handle]
        init_pos = agent.initial_position
        init_dir = agent.initial_direction
        nr_agents_same_start = len([a.handle for a in self.env.agents
                                    if a.initial_position == agent.initial_position])
        distance = distance_map[handle][init_pos + (init_dir,)]
        distance = max_distance if (
                distance == np.inf or np.isnan(distance)) else distance
        return np.array([distance / max_distance, nr_agents_same_start / len(self.env.agents)])

    def set_env(self, env: Environment):
        self.env: RailEnv = env

    def reset(self):
        pass
