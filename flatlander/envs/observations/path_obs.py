from typing import Optional, List

import gym
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv

from flatlander.envs.observations import register_obs, Observation


@register_obs("path")
class PathObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._config = config
        self._builder = PathObservationBuilder(encode_one_hot=True)

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=0, high=1, shape=(20,))


class PathObservationBuilder(ObservationBuilder):
    def __init__(self, encode_one_hot=True):
        super().__init__()
        self._encode_one_hot = encode_one_hot
        self._directions = list(range(4))
        self._path_size = len(self._directions) + 6
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_agent_malfunction = {}
        self.location_has_agent_ready_to_depart = {}

    def get_many(self, handles: Optional[List[int]] = None):
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_agent_malfunction = {}
        self.location_has_agent_ready_to_depart = {}

        for _agent in self.env.agents:
            if _agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and \
                    _agent.position:
                self.location_has_agent[tuple(_agent.position)] = 1
                self.location_has_agent_direction[tuple(_agent.position)] = _agent.direction
                self.location_has_agent_malfunction[tuple(_agent.position)] = _agent.malfunction_data[
                    'malfunction']

            if _agent.status in [RailAgentStatus.READY_TO_DEPART] and \
                    _agent.initial_position:
                self.location_has_agent_ready_to_depart[tuple(_agent.initial_position)] = \
                    self.location_has_agent_ready_to_depart.get(tuple(_agent.initial_position), 0) + 1

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
        assert not np.isnan(max_distance)
        assert max_distance != np.inf
        possible_steps = []

        # look in all directions for possible moves
        for movement in self._directions:
            if possible_transitions[movement]:
                next_move = movement
                pos = get_new_position(agent_virtual_position, movement)
                distance = distance_map[agent.handle][pos + (movement,)]
                distance = max_distance if (
                        distance == np.inf or np.isnan(distance)) else distance

                loc_features = self.check_location(pos, movement)
                next_possible_moves = self.env.rail.get_transitions(*pos, movement)
                while np.count_nonzero(next_possible_moves) == 1:
                    movement = np.argmax(next_possible_moves)
                    pos = get_new_position(pos, movement)
                    loc_features = np.bitwise_or(self.check_location(pos, movement), loc_features)
                    next_possible_moves = self.env.rail.get_transitions(*pos, movement)

                if self._encode_one_hot:
                    next_move_one_hot = np.zeros(len(self._directions))
                    next_move_one_hot[next_move] = 1
                    next_move = next_move_one_hot

                possible_steps.append((next_move, [distance / max_distance], loc_features))

        possible_steps = sorted(possible_steps, key=lambda step: step[1])
        obs = np.full(self._path_size * 2, fill_value=0)
        for i, path in enumerate(possible_steps):
            obs[i * self._path_size:self._path_size * (i + 1)] = np.concatenate([arr for arr in path])

        return obs

    def check_location(self, position, direction):
        location_features = np.zeros(5, dtype=int)
        if position in self.location_has_agent:
            location_features[0] = 1
            if self.location_has_agent_malfunction[position]:
                location_features[1] = 1

            if self.location_has_agent_ready_to_depart.get(position, 0) > 0:
                location_features[2] = 1

            if self.location_has_agent_direction[position] == direction:
                location_features[3] = 1
            else:
                location_features[4] = 1
        return location_features
