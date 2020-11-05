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


@register_obs("simple_meta")
class SimpleMetaObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._builder = SimpleMetaObservationBuilder()

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=0, high=1, shape=(4,),
                              dtype=np.float32)  # own distance to target & nr agents at start


class SimpleMetaObservationBuilder(ObservationBuilder):

    def get_many(self, handles: Optional[List[int]] = None):
        if self.env._elapsed_steps == 0:
            self.conflict_detector = ShortestPathConflictDetector()
            self.conflict_detector.set_env(self.env)

            positions = {h: self.get_position(h) for h in handles}
            directions = {h: self.env.agents[h].direction for h in handles}
            agent_conflicts, _ = self.conflict_detector.detect_conflicts(handles=handles,
                                                                         positions=positions,
                                                                         directions=directions)
            if handles is None:
                handles = []
            obs = {h: self.get(h) for h in handles}
            for h, o in obs.items():
                o[-1] = len(set(agent_conflicts[h]))

            obs_matrix = np.array(list(obs.values()))
            obs_normed = obs_matrix / (np.max(obs_matrix, axis=0) + 1e-10)
            obs = {h: obs_normed[i] for i, h in enumerate(handles)}
            return obs
        else:
            return {h: [] for h in handles}

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
                         nr_agents_same_start_and_dir,
                         1])

    def get_position(self, handle):
        self.env: RailEnv = self.env
        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None
        return agent_virtual_position

    def set_env(self, env: Environment):
        self.env: RailEnv = env

    def reset(self):
        pass
