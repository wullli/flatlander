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
        return gym.spaces.Box(low=0, high=1, shape=(3,),
                              dtype=np.float32)  # own distance to target & nr agents at start


class SimpleMetaObservationBuilder(ObservationBuilder):

    def get_many(self, handles: Optional[List[int]] = None):
        if self.env._elapsed_steps == 0:
            self.conflict_detector = ShortestPathConflictDetector(rail_env=self.env)
            self.conflict_detector.map_predictions()

            if handles is None:
                handles = []
            return {h: self.get(h) for h in handles}
        else:
            return {h: [] for h in handles}

    def get(self, handle: int = 0):
        """
        compute density map for agent: a value is asigned to every cell along the shortest path between
        the agent and its target based on the distance to the agent, i.e. the number of time steps the
        agent needs to reach the cell, encoding the time information.
        """
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

        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        distance_map = self.env.distance_map.get()
        possible_paths = []

        for movement in range(4):
            if possible_transitions[movement]:
                pos = get_new_position(agent_virtual_position, movement)
                distance = distance_map[agent.handle][pos + (movement,)]
                distance = max_distance if (distance == np.inf or np.isnan(distance)) else distance

                conflict, malf = self.conflict_detector.detect_conflicts_multi(position=pos, direction=movement,
                                                                               agent=self.env.agents[handle])

                possible_paths.append(np.array([distance, len(conflict)]))

        possible_steps = sorted(possible_paths, key=lambda path: path[1])

        return np.array([distance / max_distance,
                         nr_agents_same_start / len(self.env.agents),
                         possible_steps[0][1] / self.env.get_num_agents()])

    def set_env(self, env: Environment):
        self.env: RailEnv = env

    def reset(self):
        pass
