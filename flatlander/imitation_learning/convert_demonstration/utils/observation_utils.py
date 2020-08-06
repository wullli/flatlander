import collections
from itertools import combinations
from typing import List

import numpy as np

from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.observations import TreeObsForRailEnv


def max_lt(seq, val):
    """
    Return greatest item in seq for which item < val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    max = 0
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] < val and seq[idx] >= 0 and seq[idx] > max:
            max = seq[idx]
        idx -= 1
    return max


def min_gt(seq, val):
    """
    Return smallest item in seq for which item > val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    min = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] >= val and seq[idx] < min:
            min = seq[idx]
        idx -= 1
    return min


def norm_obs_clip(obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
    """
    This function returns the difference between min and max value of an observation
    :param obs: Observation that should be normalized
    :param clip_min: min value where observation will be clipped
    :param clip_max: max value where observation will be clipped
    :return: returnes normalized and clipped observatoin
    """
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        max_obs = max(1, max_lt(obs, 1000)) + 1

    min_obs = 0  # min(max_obs, min_gt(obs, 0))
    if normalize_to_range:
        min_obs = min_gt(obs, 0)
    if min_obs > max_obs:
        min_obs = max_obs
    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
    norm = np.abs(max_obs - min_obs)
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


def _split_node_into_feature_groups(node: TreeObsForRailEnv.Node) -> (np.ndarray, np.ndarray, np.ndarray):
    data = np.zeros(6)
    distance = np.zeros(1)
    agent_data = np.zeros(4)

    data[0] = node.dist_own_target_encountered
    data[1] = node.dist_other_target_encountered
    data[2] = node.dist_other_agent_encountered
    data[3] = node.dist_potential_conflict
    data[4] = node.dist_unusable_switch
    data[5] = node.dist_to_next_branch

    distance[0] = node.dist_min_to_target

    agent_data[0] = node.num_agents_same_direction
    agent_data[1] = node.num_agents_opposite_direction
    agent_data[2] = node.num_agents_malfunctioning
    agent_data[3] = node.speed_min_fractional

    return data, distance, agent_data


def _split_subtree_into_feature_groups(node: TreeObsForRailEnv.Node, current_tree_depth: int, max_tree_depth: int) -> (
        np.ndarray, np.ndarray, np.ndarray):
    if node == -np.inf:
        remaining_depth = max_tree_depth - current_tree_depth
        # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
        num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
        return [-np.inf] * num_remaining_nodes * 6, [-np.inf] * num_remaining_nodes, [-np.inf] * num_remaining_nodes * 4

    data, distance, agent_data = _split_node_into_feature_groups(node)

    if not node.childs:
        return data, distance, agent_data

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(node.childs[direction],
                                                                                    current_tree_depth + 1,
                                                                                    max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def split_tree_into_feature_groups(tree: TreeObsForRailEnv.Node, max_tree_depth: int) -> (
        np.ndarray, np.ndarray, np.ndarray):
    """
    This function splits the tree into three difference arrays of values
    """
    data, distance, agent_data = _split_node_into_feature_groups(tree)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(tree.childs[direction], 1,
                                                                                    max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def normalize_observation(observation: TreeObsForRailEnv.Node, tree_depth: int, observation_radius=0):
    """
    This function normalizes the observation used by the RL algorithm
    """
    data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)

    data = norm_obs_clip(data, fixed_radius=observation_radius)
    distance = norm_obs_clip(distance, normalize_to_range=True)
    agent_data = np.clip(agent_data, -1, 1)
    normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))
    return normalized_obs


def strategy_action_map(action, observation_shortest, observation_next_shortest):
    """
    convert action space from 0-2 to 0-4
    """
    if action == 2:
        return 4
    elif action == 0:
        return np.argmax(observation_shortest) + 1
    elif action == 1:
        return np.argmax(observation_next_shortest) + 1


def action_strategy_map(action, observation_shortest, observation_next_shortest, moving):
    """
    convert action space from 0-4 to 0-2 representing shortest path, deviate and stop
    """
    if action == np.argmax(observation_shortest) + 1:
        return 0
    elif action == np.argmax(observation_next_shortest) + 1:
        return 1
    elif action == 0:
        if moving:
            if np.argmax(observation_shortest) == 1:
                return 0
            elif np.argmax(observation_shortest) == 1:
                return 1
        else:
            return 2
    elif action == 4:
        return 2
    else:
        return 0


def create_agent_states(env, obs, info, action_dict, strategy_dict, n_local, max_depth):
    n_agents, x_dim, y_dim = env.get_num_agents(), env.width, env.height
    local_agent_states_all = dict()

    distance_target = np.ones(n_agents)
    # observation_shortest = []
    # observation_next_shortest = []
    extra_distance = np.zeros(n_agents)
    malfunction = np.zeros(n_agents)
    malfunction_rate = np.zeros(n_agents)
    next_malfunction = np.zeros(n_agents)
    nr_malfunctions = np.zeros(n_agents)
    speed = np.zeros(n_agents)
    position_fraction = np.zeros(n_agents)
    transition_action_on_cellexit = np.zeros(n_agents)
    num_transitions = np.zeros(n_agents)
    moving = np.zeros(n_agents)
    status = np.zeros(n_agents)
    # predictions =
    info_action_required = np.zeros(n_agents)

    for i in range(n_agents):
        if obs[i] is not None:
            custom_observations = obs[i]
            distance_target[i] = custom_observations.distance_target
            # observation_shortest[i] = np.array(custom_observations.observation_shortest)
            # observation_next_shortest[i] = np.array(custom_observations.observation_next_shortest)
            extra_distance[i] = custom_observations.extra_distance
            malfunction[i] = custom_observations.malfunction
            malfunction_rate[i] = custom_observations.malfunction_rate
            next_malfunction[i] = custom_observations.next_malfunction
            nr_malfunctions[i] = custom_observations.nr_malfunctions
            speed[i] = custom_observations.speed
            position_fraction[i] = custom_observations.position_fraction
            transition_action_on_cellexit[i] = custom_observations.transition_action_on_cellexit
            num_transitions[i] = int(custom_observations.num_transitions > 1)
            moving[i] = int(custom_observations.moving)
            status[i] = int(custom_observations.status > 0)
            info_action_required[i] = int(info['action_required'][i])

    predicted_pos = custom_observations.predicted_pos
    agent_conflicts_count_path, agent_conflicts_step_path, agent_total_step_conflicts = get_agent_conflict_prediction_matrix(
        n_agents, max_depth, predicted_pos)

    avg_dim = (x_dim * y_dim) ** 0.5
    depth = int(n_local * avg_dim / n_agents)

    agent_conflict_steps = min(max_depth - 1, depth)

    agent_conflicts = agent_conflicts_step_path[agent_conflict_steps]
    agent_counts = agent_conflicts_count_path[agent_conflict_steps]
    agent_conflicts_avg_step_count = np.average(agent_total_step_conflicts) / n_agents

    for i in range(n_agents):
        if obs is None or obs[i] is None:
            action_dict.update({i: 2})
            strategy_dict.update({i: 0})
        elif obs[i] is not None:
            n_upd_local = min(n_local, n_agents - 1)
            if n_upd_local < n_local:
                n_pad = n_local - n_upd_local
                ls_other_local_agents = np.argpartition(agent_conflicts[i, :], n_upd_local)[:n_upd_local - 1]
                for j in range(n_pad):
                    ls_other_local_agents = np.hstack([ls_other_local_agents, i])
            else:
                ls_other_local_agents = np.argpartition(agent_conflicts[i, :], n_local)[:n_local - 1]
            ls_local_agents = np.hstack([i, ls_other_local_agents])
            local_agent_states = np.hstack([distance_target[ls_local_agents], extra_distance[ls_local_agents]])

            local_agent_states = np.hstack([local_agent_states, info_action_required[ls_local_agents]])
            local_agent_states = np.hstack([local_agent_states,
                                            agent_conflicts_step_path[0][i, ls_other_local_agents],
                                            agent_conflicts_step_path[1][i, ls_other_local_agents],
                                            agent_conflicts_step_path[2][i, ls_other_local_agents]])
            local_agent_states = np.hstack([local_agent_states,
                                            agent_conflicts_count_path[0][ls_local_agents],
                                            agent_conflicts_count_path[1][ls_local_agents],
                                            agent_conflicts_count_path[2][ls_local_agents]])

            local_agent_states = np.hstack(
                [local_agent_states, malfunction[ls_local_agents], malfunction_rate[ls_local_agents],
                 next_malfunction[ls_local_agents],
                 nr_malfunctions[ls_local_agents], speed[ls_local_agents], position_fraction[ls_local_agents],
                 transition_action_on_cellexit[ls_local_agents], num_transitions[ls_local_agents],
                 moving[ls_local_agents], status[ls_local_agents]])

            for j in ls_local_agents:
                if obs[j] is None:
                    local_agent_states = np.hstack([local_agent_states, [0, 0, 0]])
                    local_agent_states = np.hstack([local_agent_states, [0, 0, 0]])
                else:
                    local_agent_states = np.hstack([local_agent_states, obs[j].observation_shortest])
                    local_agent_states = np.hstack([local_agent_states, obs[j].observation_next_shortest])

            local_agent_states = np.hstack([local_agent_states, agent_conflicts_avg_step_count])
            local_agent_states_all[i] = local_agent_states
    return local_agent_states_all, action_dict, strategy_dict


def get_agent_conflict_prediction_matrix(n_agents, max_depth, predicted_pos):
    agent_total_step_conflicts = []
    agent_conflicts_step_path = []
    agent_conflicts_count_path = []
    values = []
    counts = []
    agent_conflicts_step = max_depth * np.ones((n_agents, n_agents))

    for i in range(max_depth):
        step = i + 1
        pos = predicted_pos[i]
        val, count = np.unique(pos, return_counts=True)
        if val[0] == -1:
            val = val[1:]
            count = count[1:]
        values.append(val)
        counts.append(count)

        counter = np.zeros(n_agents)
        agent_conflicts_count = np.zeros(n_agents)

        for j, curVal in enumerate(val):
            curCount = count[j]
            if curCount > 1:
                idxs = np.argwhere(pos == curVal)
                lsIdx = [int(x) for x in idxs]
                combs = list(combinations(lsIdx, 2))
                for k, comb in enumerate(combs):
                    counter[comb[0]] += 1
                    counter[comb[1]] += 1
                    agent_conflicts_count[comb[0]] = counter[comb[0]]
                    agent_conflicts_count[comb[1]] = counter[comb[1]]
                    # if agent_conflicts_step[comb[0], comb[1]] == max_depth:
                    #     agent_conflicts_step[comb[0], comb[1]] = step
                    # else:
                    agent_conflicts_step[comb[0], comb[1]] = min(step, agent_conflicts_step[comb[0], comb[1]])
                    agent_conflicts_step[comb[1], comb[0]] = min(step, agent_conflicts_step[comb[1], comb[0]])

        # agent_conflicts_step_current = agent_conflicts_step + np.transpose(agent_conflicts_step)
        agent_conflicts_step_current = agent_conflicts_step / max_depth
        agent_conflicts_step_path.append(agent_conflicts_step_current)
        agent_conflicts_count = agent_conflicts_count / n_agents
        agent_conflicts_count_path.append(agent_conflicts_count)

    for i in range(n_agents):
        agent_total_step_conflicts.append(sum(agent_conflicts_step_current[i, :]))

    return agent_conflicts_count_path, agent_conflicts_step_path, agent_total_step_conflicts


class MultipleAgentNavigationCustomObs(TreeObsForRailEnv):
    """
    We build a representation vector with 3 binary components, indicating which of the 3 available directions
    for each agent (Left, Forward, Right) lead to the shortest path to its target.
    E.g., if taking the Left branch (if available) is the shortest route to the agent's target, the observation vector
    will be [1, 0, 0].
    """
    Node = collections.namedtuple('Node', 'distance_target '
                                          'observation_shortest '
                                          'observation_next_shortest '
                                          'extra_distance '
                                          'malfunction '
                                          'malfunction_rate '
                                          'next_malfunction '
                                          'nr_malfunctions '
    # 'moving_before_malfunction '
                                          'speed '
                                          'position_fraction '
                                          'transition_action_on_cellexit '
                                          'num_transitions '
                                          'moving '
                                          'status '
                                          'predictions '
                                          'predicted_pos')

    def __init__(self, max_depth: int, predictor: PredictionBuilder = None):
        super().__init__(max_depth, predictor)

    def reset(self):
        pass

    def get(self, handle: int = 0) -> List[int]:
        agent = self.env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        if agent.position:
            possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        else:
            possible_transitions = self.env.rail.get_transitions(*agent.initial_position, agent.direction)

        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        distance_map = self.env.distance_map.get()
        max_distance = self.env.width + self.env.height
        max_steps = int(4 * 2 * (20 + self.env.height + self.env.width))

        visited = set()
        for _idx in range(10):
            # Check if any of the other prediction overlap with agents own predictions
            x_coord = self.predictions[handle][_idx][1]
            y_coord = self.predictions[handle][_idx][2]

            # We add every observed cell to the observation rendering
            visited.add((x_coord, y_coord))

        # This variable will be access by the renderer to visualize the observation
        self.env.dev_obs_dict[handle] = visited

        min_distances = []
        for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[direction]:
                new_position = get_new_position(agent_virtual_position, direction)
                min_distances.append(
                    distance_map[handle, new_position[0], new_position[1], direction])
            else:
                min_distances.append(np.inf)

        if num_transitions == 1:
            observation1 = [0, 1, 0]
            observation2 = observation1

        elif num_transitions == 2:
            idx = np.argpartition(np.array(min_distances), 2)
            observation1 = [0, 0, 0]
            observation1[idx[0]] = 1

            observation2 = [0, 0, 0]
            observation2[idx[1]] = 1

        min_distances = np.sort(min_distances)
        incremental_distances = np.diff(np.sort(min_distances))
        incremental_distances[incremental_distances == np.inf] = 0
        incremental_distances[np.isnan(incremental_distances)] = 0
        # min_distances[min_distances == np.inf] = 0

        distance_target = distance_map[(handle, *agent_virtual_position,
                                        agent.direction)]

        root_node_observation = MultipleAgentNavigationCustomObs.Node(distance_target=distance_target / max_distance,
                                                                      observation_shortest=observation1,
                                                                      observation_next_shortest=observation2,
                                                                      extra_distance=incremental_distances[
                                                                                         0] / max_distance,
                                                                      malfunction=agent.malfunction_data[
                                                                                      'malfunction'] / max_distance,
                                                                      malfunction_rate=agent.malfunction_data[
                                                                          'malfunction_rate'],
                                                                      next_malfunction=agent.malfunction_data[
                                                                                           'next_malfunction'] / max_distance,
                                                                      nr_malfunctions=agent.malfunction_data[
                                                                          'nr_malfunctions'],
                                                                      # moving_before_malfunction=agent.malfunction_data['moving_before_malfunction'],
                                                                      speed=agent.speed_data['speed'],
                                                                      position_fraction=agent.speed_data[
                                                                          'position_fraction'],
                                                                      transition_action_on_cellexit=agent.speed_data[
                                                                          'transition_action_on_cellexit'],
                                                                      num_transitions=num_transitions,
                                                                      moving=agent.moving,
                                                                      status=agent.status,
                                                                      predictions=self.predictions[handle],
                                                                      predicted_pos=self.predicted_pos)

        return root_node_observation
