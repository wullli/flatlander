from typing import Dict

import numpy as np
from flatland.envs.rail_env import RailEnvActions


class Node:
    __slots__ = ["parent", "children", "action", "player",
                 "next_player", "reward", "available", "times_visited", "terminal", "valid_moves"]

    def __init__(self, parent, action: Dict[int, RailEnvActions]):
        self.parent = parent
        self.children = []
        self.action = action
        self.reward = 0
        self.valid_moves = np.zeros([36], dtype=np.int32)
        self.times_visited = 0

    def propagate_reward(self, reward):
        self.reward += reward
        self.times_visited += 1

        if self.parent:
            self.parent.propagate_reward(reward)
