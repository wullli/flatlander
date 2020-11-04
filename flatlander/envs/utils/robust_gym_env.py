from collections import defaultdict
from typing import Dict, NamedTuple, Any, Optional

import gym
import numpy as np
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.envs.rail_env import RailEnv, RailEnvActions

from flatlander.envs.observations.common.shortest_path_conflict_detector import ShortestPathConflictDetector
from flatlander.envs.observations.common.timeless_conflict_detector import TimelessConflictDetector
from flatlander.envs.utils.priorization.helper import get_virtual_position
from flatlander.envs.utils.priorization.priorizer import Priorizer, NrAgentsWaitingPriorizer, DistToTargetPriorizer, \
    NrAgentsSameStart


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
                 max_nr_active_agents: int = 50,
                 priorizer: Priorizer = NrAgentsSameStart(),
                 conflict_detector=ShortestPathConflictDetector(),
                 allow_noop=False, **_) -> None:

        super().__init__()
        self._agents_done = []
        self._agent_scores = defaultdict(float)
        self._agent_steps = defaultdict(int)
        self._regenerate_rail_on_reset = regenerate_rail_on_reset
        self._regenerate_schedule_on_reset = regenerate_schedule_on_reset
        self._max_nr_active_agents = max_nr_active_agents
        if not isinstance(rail_env, RailEnv):
            self.rail_env = rail_env.unwrapped
        else:
            self.rail_env = rail_env
        self.observation_space = observation_space
        self._prev_obs = None
        self.sorted_handles = []
        self.priorizer = priorizer
        self.allow_noop = allow_noop
        self.conflict_detector = conflict_detector
        self.conflict_detector.set_env(rail_env=rail_env)

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
                    if new_position is not None:
                        positions[agent.handle] = new_position
                        directions[agent.handle] = new_direction
                        continue
            virt_pos = get_virtual_position(agent)
            if virt_pos is not None:
                positions[agent.handle] = virt_pos
                directions[agent.handle] = agent.direction

        return positions, directions

    def get_rail_env_actions(self, action_dict):
        return {h: a + 1 for h, a in action_dict.items()}

    def get_robust_actions(self, action_dict, sorted_handles):
        self.conflict_detector.update()
        if not self.allow_noop:
            action_dict = self.get_rail_env_actions(action_dict)

        positions, directions = self.get_predictions(action_dict)

        robust_actions = {}
        relevant_handles = []
        for h in sorted_handles:
            if positions.get(h, None) is not None:
                relevant_handles.append(h)
            if len(relevant_handles) >= self._max_nr_active_agents:
                break

        self.rail_env.obs_builder.relevant_handles = relevant_handles

        agent_conflicts, agent_malf = self.conflict_detector.detect_conflicts(handles=relevant_handles,
                                                                              positions=positions,
                                                                              directions=directions)
        for i, h in enumerate(sorted_handles):
            agent = self.rail_env.agents[h]
            if h in action_dict.keys():
                if h in relevant_handles:
                    if positions.get(h, None) is not None:
                        if agent.status == RailAgentStatus.ACTIVE \
                                and np.all([self.rail_env.agents[ch].status == RailAgentStatus.READY_TO_DEPART
                                            for ch in agent_conflicts[h]]):
                            robust_actions[h] = action_dict[h]
                            continue
                        if len([ch for ch in agent_conflicts[h] if sorted_handles.index(ch) < i]) > 0:
                            robust_actions[h] = RailEnvActions.STOP_MOVING.value
                            continue
                        if agent.status == RailAgentStatus.READY_TO_DEPART \
                                and np.any([self.rail_env.agents[ch].status == RailAgentStatus.ACTIVE
                                            for ch in agent_conflicts[h]]):
                            robust_actions[h] = RailEnvActions.STOP_MOVING.value
                            continue

                    robust_actions[h] = action_dict[h]
                if h not in relevant_handles and positions.get(h, None) is None:
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
        self.sorted_handles = self.priorizer.priorize(list(obs.keys()), self.rail_env)
        self.conflict_detector = ShortestPathConflictDetector()
        self.conflict_detector.set_env(self.rail_env)
        self._prev_obs = obs
        self.rail_env.obs_builder.relevant_handles = self.sorted_handles[:self._max_nr_active_agents]
        return {k: o for k, o in obs.items() if not k == '__all__'}
