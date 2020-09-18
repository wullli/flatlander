import numpy as np

from flatland.envs.observations import Node


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


def norm_obs_clip(obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False) -> object:
    """
    This function returns the difference between min and max value of an observation
    :param obs: Observation that should be normalized
    :param clip_min: min value where observation will be clipped
    :param clip_max: max value where observation will be clipped
    :return: returnes normalized and clipped observation
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


def _get_small_node_feature_vector(node: Node) -> np.ndarray:
    data = np.zeros(3)
    distance = np.zeros(1)
    agent_data = np.zeros(2)

    data[0] = node.dist_potential_conflict
    data[1] = node.dist_unusable_switch
    data[2] = node.dist_other_agent_encountered

    distance[0] = node.dist_min_to_target

    agent_data[0] = node.num_agents_opposite_direction
    agent_data[1] = node.num_agents_malfunctioning

    data = norm_obs_clip(data, fixed_radius=10)
    distance = norm_obs_clip(distance, fixed_radius=100)
    agent_data = np.clip(agent_data, -1, 1)
    normalized_obs = np.concatenate([data, distance, agent_data])

    return normalized_obs


def _get_node_feature_vector(node: Node) -> (np.ndarray, np.ndarray, np.ndarray):
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

    data = norm_obs_clip(data, fixed_radius=10)
    distance = norm_obs_clip(distance, fixed_radius=100)
    agent_data = np.clip(agent_data, -1, 1)
    normalized_obs = np.concatenate([data, distance, agent_data])

    return normalized_obs
