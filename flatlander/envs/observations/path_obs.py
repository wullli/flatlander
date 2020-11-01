from typing import Optional, List

import gym
import numpy as np
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv

from flatlander.envs.observations import register_obs, Observation
from flatlander.envs.observations.common.shortest_path_conflict_detector import ShortestPathConflictDetector


@register_obs("path")
class PathObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._config = config
        self._builder = PathObservationBuilder()

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=0, high=1, shape=(self._builder.path_size * 2,))


class PathObservationBuilder(ObservationBuilder):
    def __init__(self, encode_one_hot=True):
        super().__init__()
        self._encode_one_hot = encode_one_hot
        self._directions = list(range(4))
        self.path_size = 3
        self.conflict_detector: Optional[ShortestPathConflictDetector] = None

    def get_many(self, handles: Optional[List[int]] = None):
        self.conflict_detector = ShortestPathConflictDetector(rail_env=self.env)
        self.conflict_detector.map_predictions()

        if handles is None:
            handles = []
        return {h: self.get(h) for h in handles}

    def reset(self):
        pass

    def get(self, handle: int = 0):
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

        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        distance_map = self.env.distance_map.get()
        nan_inf_mask = ((distance_map != np.inf) * (np.abs(np.isnan(distance_map) - 1))).astype(np.bool)
        max_distance = np.max(distance_map[nan_inf_mask])
        possible_paths = []
        conflict_handles = []

        for movement in self._directions:
            if possible_transitions[movement]:
                pos = get_new_position(agent_virtual_position, movement)
                distance = distance_map[agent.handle][pos + (movement,)]
                distance = max_distance if (distance == np.inf or np.isnan(distance)) else distance

                conflict, malf = self.conflict_detector.detect_conflicts(
                    position=pos,
                    agent=self.env.agents[handle],
                    direction=movement)

                malf = np.max(malf) if len(malf) > 0 else 0
                possible_paths.append(np.array([distance / max_distance,
                                                malf,
                                                int(len(conflict) > 0)]))

        possible_steps = sorted(possible_paths, key=lambda path: path[1])
        obs = np.full(self.path_size * 2, fill_value=0)
        for i, path in enumerate(possible_steps):
            obs[i * self.path_size:self.path_size * (i + 1)] = path

        return obs
