import gym
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv

from flatlander.envs.observations import register_obs, Observation
from flatlander.envs.observations.utils import one_hot


@register_obs("path")
class PathObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._config = config
        self._builder = PathObservationBuilder(encode_one_hot=True)

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=0, high=1, shape=(12 + 13,))


class PathObservationBuilder(ObservationBuilder):
    def __init__(self, encode_one_hot=True):
        super().__init__()
        self._encode_one_hot = encode_one_hot
        self._directions = list(range(4))
        self._path_size = len(self._directions) + 2

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

                conflict = self.conflict(pos, movement)
                next_possible_moves = self.env.rail.get_transitions(*pos, movement)
                while np.count_nonzero(next_possible_moves) == 1 and not conflict:
                    movement = np.argmax(next_possible_moves)
                    pos = get_new_position(pos, movement)
                    conflict = self.conflict(pos, movement)
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

        agent_info = np.concatenate([one_hot([agent.status.value], 4),
                                     np.array(agent_virtual_position) / self.env.height,
                                     np.array(agent.target) / self.env.height,
                                     one_hot([agent.direction], 4),
                                     [int(agent.moving)]])

        return np.concatenate([agent_info, obs])

    def conflict(self, pos, movement):
        conflict = False
        conflict_handles = [agent.handle for agent in self.env.agents if pos == agent.position]
        if len(conflict_handles) > 0:
            other_direction = self.env.agents[conflict_handles[0]].direction

            own_next_pos = get_new_position(pos, movement)
            other_next_pos = get_new_position(pos, other_direction)

            conflict = own_next_pos != other_next_pos
        return conflict
