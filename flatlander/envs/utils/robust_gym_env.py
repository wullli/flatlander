from collections import defaultdict
from copy import deepcopy
from typing import Dict, NamedTuple, Any, Optional

import gym
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent

from flatland.envs.rail_env import RailEnv, RailEnvActions

from flatlander.envs.observations.common.malf_shortest_path_predictor import MalfShortestPathPredictorForRailEnv
import numpy as np

from flatlander.envs.observations.common.utils import reverse_dir
from flatlander.utils.deadlock_check import check_if_all_blocked


class StepOutput(NamedTuple):
    obs: Any  # depends on observation builder
    reward: Any
    done: Any
    info: Any


class RobustFlatlandGymEnv(gym.Env):
    action_space = gym.spaces.Discrete(4)

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10,
        'semantics.autoreset': True
    }

    def __init__(self,
                 rail_env: RailEnv,
                 observation_space: gym.spaces.Space,
                 render: bool = False,
                 regenerate_rail_on_reset: bool = True,
                 regenerate_schedule_on_reset: bool = True,
                 allow_noop=False, **_) -> None:
        super().__init__()
        self._agents_done = []
        self._agent_scores = defaultdict(float)
        self._agent_steps = defaultdict(int)
        self._regenerate_rail_on_reset = regenerate_rail_on_reset
        self._regenerate_schedule_on_reset = regenerate_schedule_on_reset
        self.rail_env = rail_env
        self.observation_space = observation_space
        self._prev_obs = None
        self.distance_map = self.rail_env.distance_map.get()
        self.nan_inf_mask = ((self.distance_map != np.inf) * (np.abs(np.isnan(self.distance_map) - 1))).astype(np.bool)
        self.max_distance = np.max(self.distance_map[self.nan_inf_mask])
        self._predictor = MalfShortestPathPredictorForRailEnv(max_depth=int(self.max_distance))
        self._predictor.set_env(self.rail_env)
        self.max_prediction_depth = 0
        self.predicted_pos = {}
        self.predicted_dir = {}
        self.sorted_handles = []
        self.allow_noop = allow_noop

        if self.allow_noop:
            self.action_space = gym.spaces.Discrete(5)
        else:
            self.action_space = gym.spaces.Discrete(4)

        if render:
            self.rail_env.set_renderer(render)

    def get_predictions(self, action_dict):
        positions = {}
        directions = {}

        for agent in self.rail_env.agents:
            if not agent.position is None:
                _, new_cell_valid, new_direction, new_position, transition_valid = \
                    self.rail_env._check_action_on_agent(action_dict[agent.handle], agent)
                if new_cell_valid and transition_valid:
                    positions[agent.handle] = new_position
                    directions[agent.handle] = new_direction
                    continue
            positions[agent.handle] = self.get_virtual_position(agent)
            directions[agent.handle] = agent.direction

        return self._predictor.get(), positions, directions

    def map_predictions(self, predictions, handles=None):
        if handles is None:
            handles = [a.handle for a in self.rail_env.agents]
        if self._predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
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

    def prioritized_agents(self, handles):
        agent_distances = {}
        distance_map = self.rail_env.distance_map.get()
        nan_inf_mask = ((distance_map != np.inf) * (np.abs(np.isnan(distance_map) - 1))).astype(np.bool)
        max_distance = np.max(distance_map[nan_inf_mask])

        for h in handles:
            agent_virtual_position = self.get_virtual_position(self.rail_env.agents[h])
            if agent_virtual_position is not None:
                distance = distance_map[h][agent_virtual_position + (self.rail_env.agents[h].direction,)]
                distance = max_distance if (distance == np.inf or np.isnan(distance)) else distance
                agent_distances[h] = distance

        sorted_dists = {k: v for k, v in sorted(agent_distances.items(), key=lambda item: item[1])}
        return list(sorted_dists.keys())

    def get_virtual_position(self, agent: EnvAgent):
        agent_virtual_position = agent.position
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        return agent_virtual_position

    def get_rail_env_actions(self, action_dict):
        return {h: a + 1 for h, a in action_dict.items()}

    def get_robust_actions(self, action_dict, sorted_handles):
        if not self.allow_noop:
            action_dict = self.get_rail_env_actions(action_dict)

        predictions, positions, directions = self.get_predictions(action_dict)
        self.map_predictions(predictions)

        robust_actions = {}
        for i, h in enumerate(sorted_handles):
            if h in action_dict.keys():
                if positions[h] is not None:
                    c_handles = self.detect_conflicts(
                        position=positions[self.rail_env.agents[h].handle],
                        agent=self.rail_env.agents[h],
                        direction=directions[self.rail_env.agents[h].handle])
                    if len([ch for ch in c_handles if sorted_handles.index(ch) < i]) > 0:
                        robust_actions[h] = RailEnvActions.STOP_MOVING.value
                        continue

                robust_actions[h] = action_dict[h]
        return robust_actions

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
        robust_actions = self.get_robust_actions(action_dict, sorted_handles=self.sorted_handles)
        obs, rewards, dones, infos = self.rail_env.step(robust_actions)

        d, r, o = dict(), dict(), dict()
        for agent, done in dones.items():
            if agent != '__all__' and not agent in obs:
                continue  # skip agent if there is no observation

            if agent not in self._agents_done:
                if agent != '__all__':
                    if done:
                        self._agents_done.append(agent)
                    if obs[agent] is not None:
                        o[agent] = obs[agent]
                    else:
                        o[agent] = self._prev_obs[agent]
                    r[agent] = rewards[agent]
                    self._agent_scores[agent] += rewards[agent]
                    self._agent_steps[agent] += 1
                d[agent] = dones[agent]

        self._prev_obs = o

        return StepOutput(obs=o, reward=r, done=d, info={agent: {
            'max_episode_steps': self.rail_env._max_episode_steps,
            'num_agents': self.rail_env.get_num_agents(),
            'agent_done': d[agent] and agent not in self.rail_env.active_agents,
            'agent_score': self._agent_scores[agent],
            'agent_step': self._agent_steps[agent],
        } for agent in o.keys()})

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        self._prev_obs = None
        self._agents_done = []
        self._agent_scores = defaultdict(float)
        self._agent_steps = defaultdict(int)
        obs, infos = self.rail_env.reset(regenerate_rail=self._regenerate_rail_on_reset,
                                         regenerate_schedule=self._regenerate_schedule_on_reset,
                                         random_seed=random_seed)
        self.sorted_handles = self.prioritized_agents(obs.keys())
        self.distance_map = self.rail_env.distance_map.get()
        self.nan_inf_mask = ((self.distance_map != np.inf) * (np.abs(np.isnan(self.distance_map) - 1))).astype(np.bool)
        self.max_distance = np.max(self.distance_map[self.nan_inf_mask])
        self._predictor = MalfShortestPathPredictorForRailEnv(max_depth=int(self.max_distance))
        self._predictor.set_env(self.rail_env)
        return {k: o for k, o in obs.items() if not k == '__all__'}

    def detect_conflicts(self,
                         position,
                         agent,
                         direction):

        tot_dist = 1
        conflict_handles = []
        time_per_cell = np.reciprocal(agent.speed_data["speed"])
        predicted_time = int(tot_dist * time_per_cell)
        while predicted_time < self.max_prediction_depth and position != agent.target:
            predicted_time = int(tot_dist * time_per_cell)
            int_position = coordinate_to_position(self.rail_env.width, [position])
            if tot_dist < self.max_prediction_depth:

                pred_times = [max(0, predicted_time - 1),
                              predicted_time,
                              min(self.max_prediction_depth - 1, predicted_time + 1)]

                for pred_time in pred_times:
                    if int_position in np.delete(self.predicted_pos[pred_time], agent.handle, 0):
                        conflicting_agents = np.where(self.predicted_pos[pred_time] == int_position)
                        for ca in conflicting_agents[0]:
                            if direction != self.predicted_dir[pred_time][ca]:
                                conflict_handles.append(ca)

            tot_dist += 1
            position, direction = self.get_shortest_path_position(position=position,
                                                                  direction=direction,
                                                                  handle=agent.handle)

        return conflict_handles

    def get_shortest_path_position(self, position, direction, handle):
        possible_transitions = self.rail_env.rail.get_transitions(*position, direction)
        min_dist = np.inf
        sp_move = None
        sp_pos = None

        for movement in range(4):
            if possible_transitions[movement]:
                pos = get_new_position(position, movement)
                distance = self.distance_map[handle][pos + (movement,)]
                distance = self.max_distance if (distance == np.inf or np.isnan(distance)) else distance
                if distance <= min_dist:
                    min_dist = distance
                    sp_move = movement
                    sp_pos = pos

        return sp_pos, sp_move
