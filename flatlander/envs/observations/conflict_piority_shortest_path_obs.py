from typing import Optional, List

import gym
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv

from flatlander.algorithms.graph_coloring import GreedyGraphColoring
from flatlander.envs.observations import register_obs, Observation
from flatlander.envs.observations.common.malf_shortest_path_predictor import MalfShortestPathPredictorForRailEnv


@register_obs("shortest_path_priority_conflict")
class ConflictPriorityShortestPathObservation(Observation):

    def builder(self) -> ObservationBuilder:
        return self._builder

    def __init__(self, config) -> None:
        super().__init__(config)
        self._config = config if config is not None else {}
        self._builder = ConflictPriorityShortestPathObservationBuilder(
            predictor=MalfShortestPathPredictorForRailEnv(config['shortest_path_max_depth']),
            encode_one_hot=True,
            asserts=self._config.get("asserts", False))

    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=0, high=1, shape=(14,))


class ConflictPriorityShortestPathObservationBuilder(ObservationBuilder):
    def reset(self):
        self._conflict_map = {}

    def set_env(self, env):
        self.predictor.set_env(env)
        self.env = env

    def __init__(self, predictor=None, encode_one_hot=True, asserts=False):
        super().__init__()
        self._conflict_map = {}
        self.predictor = predictor
        self._directions = list(range(4))
        self._path_size = len(self._directions) + 2
        self._encode_one_hot = encode_one_hot
        self._asserts = asserts

    def get_many(self, handles: Optional[List[int]] = None):
        if handles is None:
            handles = []

        if handles is None:
            handles = []

        self._conflict_map = {handle: [] for handle in handles}

        if self.predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictions = self.predictor.get()
            if self.predictions:
                for t in range(self.predictor.max_depth + 1):
                    pos_list = []
                    dir_list = []
                    for a in handles:
                        if self.predictions[a] is None:
                            continue
                        pos_list.append(self.predictions[a][t][1:3])
                        dir_list.append(self.predictions[a][t][3])
                    self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                    self.predicted_dir.update({t: dir_list})
                self.max_prediction_depth = len(self.predicted_pos)
        # Update local lookup table for all agents' positions
        # ignore other agents not in the grid (only status active and done)
        # self.location_has_agent = {tuple(agent.position): 1 for agent in self.env.agents if
        #                         agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE]}

        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_agent_speed = {}
        self.location_has_agent_malfunction = {}
        self.location_has_agent_ready_to_depart = {}

        for _agent in self.env.agents:
            if _agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and \
                    _agent.position:
                self.location_has_agent[tuple(_agent.position)] = 1
                self.location_has_agent_direction[tuple(_agent.position)] = _agent.direction
                self.location_has_agent_speed[tuple(_agent.position)] = _agent.speed_data['speed']
                self.location_has_agent_malfunction[tuple(_agent.position)] = _agent.malfunction_data[
                    'malfunction']

            if _agent.status in [RailAgentStatus.READY_TO_DEPART] and \
                    _agent.initial_position:
                self.location_has_agent_ready_to_depart[tuple(_agent.initial_position)] = \
                    self.location_has_agent_ready_to_depart.get(tuple(_agent.initial_position), 0) + 1

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

                cell_transitions = self.env.rail.get_transitions(*pos, movement)
                _, ch = self.detect_conflicts(1,
                                              np.reciprocal(agent.speed_data["speed"]),
                                              pos,
                                              cell_transitions,
                                              handle,
                                              movement)

                conflict = ch is not None

                if conflict and len(possible_steps) == 0:
                    self._conflict_map[handle].append(ch)

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
        return np.concatenate([obs, [agent.status.value != RailAgentStatus.READY_TO_DEPART, priority]])

    def _reverse_dir(self, direction):
        return int((direction + 2) % 4)

    def detect_conflicts(self, tot_dist,
                         time_per_cell,
                         position,
                         cell_transitions,
                         handle,
                         direction):
        potential_conflict = np.inf
        conflict_handle = None
        predicted_time = int(tot_dist * time_per_cell)
        if self.predictor and predicted_time < self.max_prediction_depth:
            int_position = coordinate_to_position(self.env.width, [position])
            if tot_dist < self.max_prediction_depth:

                pre_step = max(0, predicted_time - 1)
                post_step = min(self.max_prediction_depth - 1, predicted_time + 1)

                # Look for conflicting paths at distance tot_dist
                if int_position in np.delete(self.predicted_pos[predicted_time], handle, 0):
                    conflicting_agent = np.where(self.predicted_pos[predicted_time] == int_position)
                    for ca in conflicting_agent[0]:
                        if direction != self.predicted_dir[predicted_time][ca] and cell_transitions[
                            self._reverse_dir(
                                self.predicted_dir[predicted_time][ca])] == 1 and tot_dist < potential_conflict:
                            potential_conflict = tot_dist
                            conflict_handle = ca
                        if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                            potential_conflict = tot_dist
                            conflict_handle = ca

                # Look for conflicting paths at distance num_step-1
                elif int_position in np.delete(self.predicted_pos[pre_step], handle, 0):
                    conflicting_agent = np.where(self.predicted_pos[pre_step] == int_position)
                    for ca in conflicting_agent[0]:
                        if direction != self.predicted_dir[pre_step][ca] \
                                and cell_transitions[self._reverse_dir(self.predicted_dir[pre_step][ca])] == 1 \
                                and tot_dist < potential_conflict:  # noqa: E125
                            potential_conflict = tot_dist
                            conflict_handle = ca
                        if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                            potential_conflict = tot_dist
                            conflict_handle = ca

                # Look for conflicting paths at distance num_step+1
                elif int_position in np.delete(self.predicted_pos[post_step], handle, 0):
                    conflicting_agent = np.where(self.predicted_pos[post_step] == int_position)
                    for ca in conflicting_agent[0]:
                        if direction != self.predicted_dir[post_step][ca] and cell_transitions[self._reverse_dir(
                                self.predicted_dir[post_step][ca])] == 1 \
                                and tot_dist < potential_conflict:  # noqa: E125
                            potential_conflict = tot_dist
                            conflict_handle = ca
                        if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                            potential_conflict = tot_dist
                            conflict_handle = ca

        return potential_conflict, conflict_handle
