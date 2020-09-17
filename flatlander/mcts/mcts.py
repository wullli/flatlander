import itertools
import sys
import time
import traceback
from copy import deepcopy
from typing import Callable

import numpy as np

from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv
from flatlander.mcts.node import Node


def get_random_actions(obs: dict):
    return {i: np.random.randint(0, 5) for i in range(len(obs.keys()))}


class MonteCarloTreeSearch:
    def __init__(self, time_budget: float,
                 epsilon=1,
                 rollout_depth=10,
                 rollout_policy: Callable = get_random_actions):
        self.time_budget = time_budget
        self.count = 0
        self.epsilon = epsilon
        self.rollout_policy = rollout_policy
        self.root = None
        self.rollout_depth = rollout_depth
        self.last_action = None

    def get_best_actions(self, env: RailEnv, obs):
        end_time = time.time() + self.time_budget
        if self.root is None:
            self.root = Node(None, None, self.get_possible_moves(env, obs))
        else:
            self.root = list(filter(lambda child: child.action == self.last_action, self.root.children))[0]
        self.iterate(env, obs=obs, budget=end_time)

        print("Total visits:", np.sum(list(map(lambda c: c.times_visited, self.root.children))))

        best_child = max(self.root.children, key=lambda n: n.times_visited)
        best_action = best_child.action
        self.last_action = best_action
        return best_action

    def iterate(self, env: RailEnv, obs: dict, budget: float = 0.):
        try:
            while time.time() < budget:
                new_env = deepcopy(env)
                node, obs = self.select(new_env, self.root, obs)
                new_node, obs = self.expand(node, new_env, obs)
                reward = self.simulate(new_env, obs)
                new_node.propagate_reward(reward)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise e
        return self.root

    def select(self, env: RailEnv, node: Node, o: dict) -> (Node, dict):
        while True:
            # calculate UCBs
            if len(node.valid_moves) == 0 and node.children:
                best_node = max(node.children, key=self.ucb)
                o, r, d, _ = env.step(best_node.action)
                node = best_node
            else:
                return node, o

    @staticmethod
    def get_agent_positions(env: RailEnv):
        pos = {}
        for agent in env.agents:
            if agent.status == RailAgentStatus.READY_TO_DEPART:
                agent_virtual_position = agent.initial_position
            elif agent.status == RailAgentStatus.ACTIVE:
                agent_virtual_position = agent.position
            elif agent.status == RailAgentStatus.DONE:
                agent_virtual_position = agent.target
            else:
                agent_virtual_position = None
            pos[agent.handle] = agent_virtual_position
        return pos

    @classmethod
    def get_possible_moves(cls, env: RailEnv, obs: dict):
        active_agents = []
        positions = cls.get_agent_positions(env)
        possible_actions = {}
        for handle in obs.keys():
            if positions[handle] is not None:
                possible_transitions = np.flatnonzero(env.rail.get_transitions(*positions[handle],
                                                                               env.agents[handle].direction))
                if len(possible_transitions) != 0:
                    possible_actions[handle] = possible_transitions
                    active_agents.append(handle)

        possible_moves = list(itertools.product(*possible_actions.values()))
        possible_moves = [{handle: action_list[i] for i, handle in enumerate(active_agents)}
                          for action_list in possible_moves]

        return possible_moves

    @classmethod
    def expand(cls, node: Node, env: RailEnv, obs) -> (Node, dict):
        if len(node.valid_moves) == 0:
            return node
        else:
            new_node = Node(node, node.valid_moves[0], cls.get_possible_moves(env, obs))
            node.valid_moves.pop(0)
            node.children.append(new_node)
            o, r, d, _ = env.step(new_node.action)
            return new_node, o

    def simulate(self, env: RailEnv, obs: dict) -> float:
        done = False
        reward = 0.
        count = 0
        while not done and count <= self.rollout_depth:
            o, r, d, _ = env.step(self.rollout_policy(obs))
            reward += np.sum(list(r.values()))
            done = d["__all__"]
            count += 1
        return reward

    def ucb(self, node: Node):
        return node.reward / node.times_visited + self.epsilon * \
               np.sqrt(np.log(node.parent.times_visited) / node.times_visited)
