import time
from copy import deepcopy

import numpy as np

from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv
from flatlander.mcts.node import Node


class Mcts:
    def __init__(self, time_budget: float, nr_threads=8, n_agents=5, epsilon=1):
        self.time_budget = time_budget
        self.nr_threads = nr_threads
        self.count = 0
        self.n_agents = n_agents
        self.epsilon = epsilon

    def get_best_actions(self, env: RailEnv):
        end_time = time.time() + self.time_budget
        self.root = Node()
        self.iterate(env, budget=end_time)

        lengths = [len(root.children) for root in self.roots[player_round.player]]
        best_index = self.best_child_index(lengths, player_round)
        best_action = self.roots[player_round.player][int(np.argmax(lengths))].children[best_index].action
        return best_action

    def iterate(self, env: RailEnv, budget: float = 0.):
        try:
            while time.time() < budget:
                new_env = deepcopy(env)
                node = self.select(new_env, self.root)
                new_node = self.expand(node, new_env)
                reward = self.simulate(new_env)
                new_node.propagate_reward(reward)
        except:
            pass
        return self.root

    def select(self, env: RailEnv, node: Node) -> (Node, dict):
        while True:
            valid_children = self.get_available(node, env)

            # calculate UCBs
            if node.valid_moves.sum() == 0 and valid_children:
                best_node = max(valid_children, key=self.ucb)
                o, d, r, _ = env.step(best_node.action)
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

    @staticmethod
    def get_available(node: Node, env: RailEnv, obs: dict):

        for movement in directions:
            if possible_transitions[movement]:
        return valid_children

    @staticmethod
    def expand(node: Node, env: RailEnv) -> Node:
        if node.valid_moves.sum() == 0:
            return node
        else:
            cards = np.flatnonzero(node.valid_moves)
            new_node = Node(node, cards[0], )
            node.children.append(new_node)
            env.step(new_node.action)
            return new_node

    def simulate(self, env: RailEnv) -> float:
        env = deepcopy(env)
        done = False
        r = 0
        while not done:
            _, d, r, = env.step(self.get_random_actions(self.n_agents))
            r += r
            done = d["__all__"]
        return r

    @staticmethod
    def get_random_actions(n_agents):
        return {i: np.random.randint(0, 5) for i in range(n_agents)}

    def ucb(self, node: Node):
        return node.reward / node.times_visited + self.epsilon * \
               np.sqrt(np.log(node.parent.times_visited) / node.times_visited)
