from flatland.envs.rail_env import RailEnv
from ray.rllib.agents import Trainer

from flatlander.agents.agent import Agent
from flatlander.envs.utils.gym_env_wrappers import possible_actions_sorted_by_distance


class ShortestPathRllibAgent(Agent):
    def __init__(self, trainer: Trainer, explore=False):
        self.trainer = trainer
        self.explore = explore

    def compute_actions(self, observation_dict: dict, env: RailEnv):
        obs = {h: o for h, o in observation_dict.items() if o is not None}
        actions = self.trainer.compute_actions(obs, explore=self.explore)
        return {h: possible_actions_sorted_by_distance(env, h)[a][0] for h, a in actions.items()}

    def compute_action(self, obs, env: RailEnv, handle):
        action = self.trainer.compute_action(obs[handle], explore=self.explore)
        return possible_actions_sorted_by_distance(env, handle)[action][0]
