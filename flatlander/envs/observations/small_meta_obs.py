from typing import Optional, List, Dict

import gym
import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv
from flatlander.envs.observations import Observation, register_obs
from flatlander.envs.observations.common.shortest_path_conflict_detector import ShortestPathConflictDetector


@register_obs("small_meta")
class SmallMetaObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._builder = SmallMetaObservationBuilder()

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=0, high=1, shape=(4,),
                              dtype=np.float32)


class SmallMetaObservationBuilder(ObservationBuilder):

    def get_many(self, handles: Optional[List[int]] = None):
        self.conflict_detector = ShortestPathConflictDetector()
        self.conflict_detector.set_env(self.env)
        self.conflict_detector.map_predictions()

        if handles is None:
            handles = []
        obs = {h: self.get(h) for h in handles}
        obs_matrix = np.array(list(obs.values()))
        obs_normed = obs_matrix / (np.max(obs_matrix, axis=0) + 1e-10)
        obs = {h: obs_normed[i] for i, h in enumerate(handles)}
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
        agents_same_start = [a for a in self.env.agents
                             if a.initial_position == init_pos]
        nr_agents_same_start = len(agents_same_start)
        nr_agents_same_start_and_dir = len([a.handle for a in agents_same_start
                                            if a.initial_direction == init_dir])
        distance = distance_map[handle][init_pos + (init_dir,)]
        distance = max_distance if (
                distance == np.inf or np.isnan(distance)) else distance

        return np.array([distance / max_distance,
                         nr_agents_same_start,
                         int(self.env._elapsed_steps == 0),
                         nr_agents_same_start_and_dir])

    def set_env(self, env: Environment):
        self.env: RailEnv = env

    def reset(self):
        pass
