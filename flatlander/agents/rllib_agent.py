from flatland.envs.rail_env import RailEnv
from ray.rllib.agents import Trainer

from flatlander.agents.agent import Agent


class RllibAgent(Agent):
    def __init__(self, trainer: Trainer, explore=False):
        self.trainer = trainer
        self.explore = explore

    def compute_actions(self, observation_dict: dict, env: RailEnv):
        return self.trainer.compute_actions(observation_dict, explore=self.explore)

    def compute_action(self, obs, env: RailEnv, handle):
        return self.trainer.compute_action(obs[handle], explore=self.explore)
