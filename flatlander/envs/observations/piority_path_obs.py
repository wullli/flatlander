from typing import Optional, List

import gym
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv

from flatlander.algorithms.graph_coloring import GreedyGraphColoring
from flatlander.envs.observations import register_obs, Observation


@register_obs("priority_path")
class PriorityPathObservation(Observation):

    def builder(self) -> ObservationBuilder:
        return self._builder

    def __init__(self, config) -> None:
        super().__init__(config)
        self._config = config if config is not None else {}
        self._builder = PriorityPathObservationBuilder(encode_one_hot=True,
                                                       asserts=self._config.get("asserts", False))

    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=0, high=1, shape=(13,))


class PriorityPathObservationBuilder(ObservationBuilder):
    def reset(self):
        self._conflict_map = {}

    def __init__(self, encode_one_hot=True, asserts=False):
        super().__init__()
        self._conflict_map = {}
        self._directions = list(range(4))
        self._path_size = len(self._directions) + 2
        self._encode_one_hot = encode_one_hot
        self._asserts = asserts

    def get_many(self, handles: Optional[List[int]] = None):
        if handles is None:
            handles = []

        self._conflict_map = {handle: [] for handle in handles}
        obs_dict = {handle: self.get(handle) for handle in handles}

        # the order of the colors matters
        priorities = GreedyGraphColoring.color(colors=[1, 0],
                                               nodes=obs_dict.keys(),
                                               neighbors=self._conflict_map)
        for handle, obs in obs_dict.items():
            obs[-1] = priorities[handle]

        if self._asserts:
            assert [priorities[h] != [priorities[ch] for ch in chs]
                    for h, chs in self._conflict_map.items()]

        return obs_dict

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

                conflict = self.conflict(handle, pos, movement)
                next_possible_moves = self.env.rail.get_transitions(*pos, movement)
                while np.count_nonzero(next_possible_moves) == 1 and not conflict:
                    movement = np.argmax(next_possible_moves)
                    pos = get_new_position(pos, movement)
                    conflict = self.conflict(handle, pos, movement)
                    next_possible_moves = self.env.rail.get_transitions(*pos, movement)

                if self._encode_one_hot:
                    next_move_one_hot = np.zeros(len(self._directions))
                    next_move_one_hot[next_move] = 1
                    next_move = next_move_one_hot

                possible_steps.append((next_move, [distance / max_distance], [int(conflict)]))

        possible_steps = sorted(possible_steps, key=lambda step: step[1])
        obs = np.full(self._path_size * 2, fill_value=0)
        for i, path in enumerate(possible_steps):
            obs[i * self._path_size:self._path_size * (i + 1)] = np.concatenate([arr for arr in path])

        priority = 0.
        return np.concatenate([obs, [priority]])

    def conflict(self, handle, pos, movement):
        conflict_handles = [a.handle for a in self.env.agents
                            if pos == a.position and a.handle != handle]
        potential_conflicts = []
        if len(conflict_handles) > 0:
            for conflict_handle in conflict_handles:
                other_direction = self.env.agents[conflict_handle].direction

                other_possible_moves = self.env.rail.get_transitions(*pos, other_direction)
                other_movement = np.argmax(other_possible_moves)

                own_possible_moves = self.env.rail.get_transitions(*pos, movement)
                own_movement = np.argmax(own_possible_moves)

                own_next_pos = get_new_position(pos, own_movement)
                other_next_pos = get_new_position(pos, other_movement)
                conflict = own_next_pos != other_next_pos

                if self._asserts:
                    assert np.all(np.array(own_next_pos) > 0)
                    assert np.all(np.array(other_next_pos) > 0)

                potential_conflicts.append(conflict)

                if conflict:
                    self._conflict_map[handle].append(conflict_handle)

        return np.any(potential_conflicts)
