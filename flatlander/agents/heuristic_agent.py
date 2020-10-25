from flatland.envs.rail_env import RailEnv, RailEnvActions

from flatlander.agents.agent import Agent
from flatlander.envs.utils.gym_env_wrappers import possible_actions_sorted_by_distance


class HeuristicPriorityAgent(Agent):

    def compute_actions(self, observation_dict: dict, env: RailEnv):

        actions = {}

        for handle, obs in observation_dict.items():
            actions[handle] = self.compute_action(obs, env, handle)

        return actions

    def compute_action(self, obs, env: RailEnv, handle):
        action = RailEnvActions.STOP_MOVING
        if obs is not None:
            if obs[0][6] == 1 and not obs[0][5] == 1:
                action = 1
                action = possible_actions_sorted_by_distance(env, handle)[action - 1][0]
            elif obs[0][13] == 1 and not obs[0][12] == 1:
                action = 2
                action = possible_actions_sorted_by_distance(env, handle)[action - 1][0]
            elif obs[0][6] == 1:
                action = 1
                action = possible_actions_sorted_by_distance(env, handle)[action - 1][0]
            elif obs[0][13] == 1:
                action = 2
                action = possible_actions_sorted_by_distance(env, handle)[action - 1][0]
        else:
            action = RailEnvActions.MOVE_FORWARD

        return action
