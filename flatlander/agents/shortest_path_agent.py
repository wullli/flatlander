from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_
import numpy as np
from flatlander.agents.agent import Agent


class ShortestPathAgent(Agent):

    def compute_actions(self, observation_dict: dict, env: RailEnv):

        actions = {}

        for handle, obs in observation_dict.items():
            actions[handle] = self.compute_action(obs, env, handle)

        return actions

    def compute_action(self, obs, env: RailEnv, handle):
        action_3 = 1
        action = self.possible_actions_sorted_by_distance(env, handle)
        if action is not None:
            return action[action_3 - 1][0]
        else:
            return RailEnvActions.MOVE_FORWARD

    @staticmethod
    def possible_actions_sorted_by_distance(env: RailEnv, handle: int):
        agent = env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        distance_map = env.distance_map
        best_dist = np.inf
        best_next_action = None
        other_dist = None
        other_action = None

        next_actions = get_valid_move_actions_(agent.direction, agent_virtual_position, distance_map.rail)

        for next_action in next_actions:
            next_action_distance = distance_map.get()[
                agent.handle, next_action.next_position[0], next_action.next_position[
                    1], next_action.next_direction]
            if next_action_distance < best_dist:
                other_dist = best_dist
                other_action = best_next_action
                best_dist = next_action_distance
                best_next_action = next_action

        # always keep iteration order to make shortest paths deterministic
        if other_action is None:
            return [(best_next_action.action, best_dist)] * 2
        else:
            return [(best_next_action.action, best_dist), (other_action.action, other_dist)]
