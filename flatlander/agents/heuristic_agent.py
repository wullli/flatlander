from flatland.envs.rail_env import RailEnv, RailEnvActions

from flatlander.envs.utils.gym_env_wrappers import possible_actions_sorted_by_distance


class HeuristicPriorityAgent:

    def compute_actions(self, observation_dict: dict, env: RailEnv):

        actions = {}

        count = 0
        for handle, obs in observation_dict.items():
            if obs is not None:
                if obs[0][6] == 1 and not obs[0][5] == 1:
                    action = 1
                    actions[handle] = possible_actions_sorted_by_distance(env, handle)[action - 1][0]
                    count += 1
                elif obs[0][13] == 1 and not obs[0][12] == 1:
                    action = 2
                    actions[handle] = possible_actions_sorted_by_distance(env, handle)[action - 1][0]
                elif obs[0][6] == 1:
                    action = 1
                    actions[handle] = possible_actions_sorted_by_distance(env, handle)[action - 1][0]
                elif obs[0][13] == 1:
                    action = 2
                    actions[handle] = possible_actions_sorted_by_distance(env, handle)[action - 1][0]
                else:
                    actions[handle] = RailEnvActions.STOP_MOVING
            else:
                actions[handle] = RailEnvActions.MOVE_FORWARD

        return actions
