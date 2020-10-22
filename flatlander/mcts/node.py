from typing import Dict

from flatland.envs.rail_env import RailEnvActions


class Node:
    __slots__ = ["parent", "children", "action", "reward", "times_visited", "valid_moves"]

    def __init__(self, parent, action: Dict[int, RailEnvActions], valid_moves: list):
        self.parent = parent
        self.children = []
        self.valid_moves = valid_moves
        self.action = action
        self.reward = 0
        self.times_visited = 0

    def propagate_reward(self, reward):
        self.reward += reward
        self.times_visited += 1

        if self.parent:
            self.parent.propagate_reward(reward)
