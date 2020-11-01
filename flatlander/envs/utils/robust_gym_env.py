from collections import defaultdict
from typing import Dict, NamedTuple, Any, Optional

import gym
import numpy as np
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.envs.rail_env import RailEnv, RailEnvActions

from flatlander.envs.observations.common.shortest_path_conflict_detector import ShortestPathConflictDetector


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
                 rail_env,
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
        if not isinstance(rail_env, RailEnv):
            self.rail_env = rail_env.unwrapped
        else:
            self.rail_env = rail_env
        self.observation_space = observation_space
        self._prev_obs = None
        self.sorted_handles = []
        self.allow_noop = allow_noop
        self.conflict_detector = ShortestPathConflictDetector(rail_env=rail_env)

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

        return positions, directions

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

        positions, directions = self.get_predictions(action_dict)
        self.conflict_detector.map_predictions()

        robust_actions = {}
        for i, h in enumerate(sorted_handles):
            if h in action_dict.keys():
                if positions[h] is not None:
                    c_handles, _ = self.conflict_detector.detect_conflicts(
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
        self.conflict_detector = ShortestPathConflictDetector(rail_env=self.rail_env)
        self._prev_obs = obs
        return {k: o for k, o in obs.items() if not k == '__all__'}