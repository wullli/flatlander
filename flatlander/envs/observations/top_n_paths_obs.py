from typing import Optional, List

import gym
import numpy as np
from flatland.core.grid.grid_utils import coordinate_to_position

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatlander.envs.observations import register_obs, Observation


@register_obs("top_n_paths")
class TopNPathsObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._config = config
        self._builder = TopNPathsBuilder(encode_one_hot=True,
                                         predictor=ShortestPathPredictorForRailEnv(
                                             max_depth=self._config.get("max_depth", 30)))

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=0, high=1, shape=(12,))


class TopNPathsBuilder(ObservationBuilder):
    def __init__(self, encode_one_hot=True, predictor: PredictionBuilder = ShortestPathPredictorForRailEnv()):
        super().__init__()
        self._encode_one_hot = encode_one_hot
        self._directions = list(range(4))
        self._path_size = len(self._directions) + 2
        self._predictor = predictor
        self.predicted_pos = None


    def reset(self):
        pass

    def get_many(self, handles: Optional[List[int]] = None):
        if handles is None:
            handles = []
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

                conflict = self.conflict(pos, agent)
                next_possible_moves = self.env.rail.get_transitions(*pos, movement)
                while np.count_nonzero(next_possible_moves) == 1 and not conflict:
                    movement = np.argmax(next_possible_moves)
                    pos = get_new_position(pos, movement)
                    conflict = self.conflict(pos, agent)
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
        return obs

    def conflict(self, pos, agent):
        conflict = self.env.agent_positions[pos] != -1
        if conflict:
            conflict_handle = self.env.agent_positions[pos]
            other_direction = self.env.agents[conflict_handle].direction
            own_direction = self.env.agents[agent].direction
            conflict = other_direction != own_direction
        return conflict
