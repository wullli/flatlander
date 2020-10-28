from flatland.envs.rail_env import RailEnv, RailEnvActions

from flatlander.agents.agent import Agent
from flatlander.envs.utils.gym_env_wrappers import possible_actions_sorted_by_distance


class ShortestPathAgent(Agent):

    def compute_actions(self, observation_dict: dict, env: RailEnv):

        actions = {}

        for handle, obs in observation_dict.items():
            actions[handle] = self.compute_action(obs, env, handle)

        return actions

    def compute_action(self, obs, env: RailEnv, handle):
        action_3 = 1
        action = possible_actions_sorted_by_distance(env, handle)
        if action is not None:
            return action[action_3 - 1][0]
        else:
            return RailEnvActions.MOVE_FORWARD
