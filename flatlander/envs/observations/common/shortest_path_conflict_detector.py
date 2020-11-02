from collections import defaultdict

import numpy as np
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_

from flatlander.envs.observations.common.malf_shortest_path_predictor import MalfShortestPathPredictorForRailEnv
from flatlander.utils.helper import get_save_agent_pos


class ShortestPathConflictDetector:

    def __init__(self, rail_env: RailEnv, multi_shortest_path=False):
        self.rail_env = rail_env
        self.distance_map = self.rail_env.distance_map.get()
        self.nan_inf_mask = ((self.distance_map != np.inf) * (np.abs(np.isnan(self.distance_map) - 1))).astype(np.bool)
        self.max_distance = np.max(self.distance_map[self.nan_inf_mask])
        max_agent_dist = np.max([self.distance_map[a.handle][a.initial_position + (a.initial_direction,)]
                                 for a in self.rail_env.agents])
        self._predictor = MalfShortestPathPredictorForRailEnv(max_depth=int(max_agent_dist))
        self._predictor.set_env(self.rail_env)
        self.predicted_pos = {}
        self.predicted_dir = {}
        self.multi_shortest_path = multi_shortest_path

    def update(self):
        agent_dists = np.array([self.distance_map[a.handle][get_save_agent_pos(a)
                                                             + (a.initial_direction,)]
                                 for a in self.rail_env.agents])
        agent_dists[agent_dists == np.inf] = 0
        max_agent_dist = np.max(agent_dists)
        self._predictor = MalfShortestPathPredictorForRailEnv(max_depth=int(max_agent_dist))
        self._predictor.set_env(self.rail_env)

    def map_predictions(self, handles=None, positions=None, directions=None):
        if handles is None:
            handles = [a.handle for a in self.rail_env.agents]
        if self._predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            predictions = self._predictor.get(positions=positions, directions=directions)
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

    def detect_conflicts(self, handles=None, positions=None, directions=None):
        if self.multi_shortest_path:
            agent_conflict_handles = defaultdict(lambda: [])
            agent_malfunctions = defaultdict(lambda: [])
            self.map_predictions(handles, positions, directions)
            for h in handles:
                agent_conflict_handles[h], agent_malfunctions[h] = self.detect_conflicts_multi(
                    agent=self.rail_env.agents[h],
                    position=positions[h],
                    direction=directions[h])
            return agent_conflict_handles, agent_malfunctions

        else:
            return self.detect_conflicts_single(handles=handles, positions=positions, directions=directions)

    def detect_conflicts_single(self, handles=None, positions=None, directions=None):
        """
        More efficient implementation to detect conflicts using a shortest path predictor
        """
        if handles is None:
            handles = [a.handle for a in self.rail_env.agents]

        num_agents = self.rail_env.get_num_agents()
        predictions = self._predictor.get(positions=positions,
                                          directions=directions)
        self.predicted_pos = {}
        self.predicted_dir = {}
        agent_conflict_handles = defaultdict(lambda: [])
        agent_malfunctions = defaultdict(lambda: [])
        for t in range(self._predictor.max_depth + 1):
            pred_times = [max(0, t - 1), t]

            for h in handles:
                for pt in pred_times:
                    self.accumulate_predictions(t, pt, handles, predictions, num_agents)
                    handle_mask = np.zeros(self.rail_env.get_num_agents())
                    handle_mask[h] = np.inf
                    masked_preds = self.predicted_pos[pt] + handle_mask
                    conf_handles = np.where(self.predicted_pos[t][h] == masked_preds)
                    conf_dirs = np.where(self.predicted_dir[t][h] != self.predicted_dir[pt][conf_handles])
                    conf_handles = conf_handles[0][conf_dirs]
                    agent_conflict_handles[h].extend(list(conf_handles))

                    for ch in conf_handles:
                        if np.isnan(self.predicted_dir[pt][ch]):
                            malf_current = self.rail_env.agents[ch].malfunction_data['malfunction']
                            malf_remaining = max(malf_current - pt, 0)
                            agent_malfunctions[h].append(min(malf_remaining, 0))

        return agent_conflict_handles, agent_malfunctions

    def detect_conflicts_multi(self,
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
                              predicted_time]

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
                                    malfunctions.append(malf_remaining)

            tot_dist += 1
            positions, directions = self.get_shortest_path_position(position=position,
                                                                    direction=direction,
                                                                    handle=agent.handle)
            for pos, dir in zip(positions, directions):
                new_chs, new_malfs = self.detect_conflicts_multi(tuple(pos), agent, dir, tot_dist=tot_dist)
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

    def accumulate_predictions(self, t, pt, handles, predictions, num_agents):
        if self.predicted_pos.get(t, None) is None:
            self.predicted_pos[t], self.predicted_dir[t] = self._get_predictions(t, handles,
                                                                                 predictions, num_agents)
        if self.predicted_pos.get(pt, None) is None:
            self.predicted_pos[pt], self.predicted_dir[pt] = self._get_predictions(pt, handles,
                                                                                   predictions, num_agents)

    def _get_predictions(self, timestep: int, handles: list, predictions, num_agents):
        positions = np.zeros(num_agents)
        directions = np.zeros(num_agents)
        for h in handles:
            positions[h] = coordinate_to_position(self.rail_env.width, [predictions[h][timestep][1:3]])[0]
            directions[h] = (predictions[h][timestep][3])
        return positions, directions
