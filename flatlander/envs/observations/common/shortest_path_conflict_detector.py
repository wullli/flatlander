import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_

from flatlander.envs.observations.common.malf_shortest_path_predictor import MalfShortestPathPredictorForRailEnv


class ShortestPathConflictDetector:

    def __init__(self, rail_env: RailEnv, multi_shortest_path=False):
        self.rail_env = rail_env
        self.distance_map = self.rail_env.distance_map.get()
        self.nan_inf_mask = ((self.distance_map != np.inf) * (np.abs(np.isnan(self.distance_map) - 1))).astype(np.bool)
        self.max_distance = np.max(self.distance_map[self.nan_inf_mask])
        self._predictor = ShortestPathPredictorForRailEnv(max_depth=int(self.max_distance) * 2)
        self._predictor.set_env(self.rail_env)
        self.max_prediction_depth = 0
        self.predicted_pos = {}
        self.predicted_dir = {}
        self.multi_shortest_path = multi_shortest_path

    def map_predictions(self, handles=None):
        if handles is None:
            handles = [a.handle for a in self.rail_env.agents]
        if self._predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            predictions = self._predictor.get()
            if predictions:
                for t in range(self._predictor.max_depth + 1):
                    pos_list = []
                    dir_list = []
                    for a in handles:
                        if predictions[a] is None:
                            continue
                        pos_list.append(predictions[a][t][1:3])
                        dir_list.append(predictions[a][t][3])
                    self.predicted_pos.update({t: coordinate_to_position(self.rail_env.width, pos_list)})
                    self.predicted_dir.update({t: dir_list})
                self.max_prediction_depth = len(self.predicted_pos)

    def detect_conflicts(self,
                         position,
                         agent,
                         direction,
                         tot_dist=1):
        conflict_handles = []
        time_per_cell = int(np.reciprocal(agent.speed_data["speed"]))
        predicted_time = int(tot_dist * time_per_cell)
        handle_mask = np.zeros(self.rail_env.get_num_agents())
        handle_mask[agent.handle] = np.inf
        malfunctions = []
        if predicted_time < self.max_prediction_depth and position != agent.target:
            int_position = coordinate_to_position(self.rail_env.width, [position])
            if tot_dist < self.max_prediction_depth:

                pred_times = [max(0, predicted_time - 1),
                              predicted_time,
                              min(self.max_prediction_depth - 1, predicted_time + 1)]

                for pred_time in pred_times:
                    masked_preds = self.predicted_pos[pred_time] + handle_mask
                    if int_position in masked_preds:
                        conflicting_agents = np.where(masked_preds == int_position)
                        for ca in conflicting_agents[0]:
                            if direction != self.predicted_dir[pred_time][ca]:
                                conflict_handles.append(ca)
                                if np.isnan(self.predicted_dir[pred_time][ca]):
                                    malf_current = self.rail_env.agents[ca].malfunction_data['malfunction']
                                    malf_remaining = max(malf_current - tot_dist, 0)
                                    malfunctions.append(min(malf_remaining, 0))

            tot_dist += 1
            if self.multi_shortest_path:
                positions, directions = self.get_all_shortest_path_positions(position=position,
                                                                             direction=direction,
                                                                             handle=agent.handle)
            else:
                positions, directions = self.get_shortest_path_position(position=position,
                                                                        direction=direction,
                                                                        handle=agent.handle)
            for pos, dir in zip(positions, directions):
                new_chs, new_malfs = self.detect_conflicts(tuple(pos), agent, dir, tot_dist=tot_dist)
                conflict_handles += new_chs
                malfunctions += new_malfs

        return conflict_handles, malfunctions

    def get_shortest_path_position(self, position, direction, handle):
        best_dist = np.inf
        best_next_action = None
        distance_map = self.rail_env.distance_map

        next_actions = get_valid_move_actions_(direction, position, distance_map.rail)

        for next_action in next_actions:
            next_action_distance = distance_map.get()[handle,
                                                      next_action.next_position[0],
                                                      next_action.next_position[1],
                                                      next_action.next_direction]
            if next_action_distance < best_dist:
                if next_action_distance <= best_dist:
                    best_dist = next_action_distance
                    best_next_action = next_action
        if best_next_action is None:
            return [position], [direction]
        return [best_next_action.next_position], [best_next_action.next_direction]

    def get_all_shortest_path_positions(self, position, direction, handle):
        """
        Its possible that there are multiple shortest paths!
        """
        possible_transitions = self.rail_env.rail.get_transitions(*position, direction)

        possible_moves = []
        possible_positions = []
        distances = []
        for movement in range(4):
            if possible_transitions[movement]:
                pos = get_new_position(position, movement)
                distance = self.distance_map[handle][pos + (movement,)]
                distances.append(distance)
                possible_moves.append(movement)
                possible_positions.append(pos)

        min_dist = np.min(distances)
        sp_indexes = np.array(distances) == min_dist
        sp_pos = np.array(possible_positions)[sp_indexes]
        sp_move = np.array(possible_moves)[sp_indexes]

        return sp_pos, sp_move
