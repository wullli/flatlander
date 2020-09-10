import argparse
import copy
# from gen_envs import *
import json
import os
import tarfile
from collections import deque
from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from tensorflow.python.framework.ops import enable_eager_execution
from utils.observation_utils import normalize_observation  # noqa

from flatland.core.grid import grid4
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.malfunction_generators import malfunction_from_file
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.schedule_generators import schedule_from_file

enable_eager_execution()

parser = argparse.ArgumentParser(description="Flatland Saving Experiences Parallel.")
parser.add_argument("--single", default=False, action="store_true")
parser.add_argument("--visual", default=False, action="store_true")
parser.add_argument("--globalobs", default=False, action="store_true")

## Legacy Code for the correct expert actions

# change below line in method malfunction_from_file in the file flatland.envs.malfunction_generators.py
# mean_malfunction_rate = 1/oMPD.malfunction_rate

extract = False

if extract:
    env_path = 'expert-demonstrations.tgz'
    env_names = env_path.split(".")[0]

    if not os.path.isdir(env_names):
        with tarfile.open(env_path) as tar_file:
            tar_file.extractall('.')

imitate = True

obs_type = "tree"  # global

#  Setting this parameters to True can slow down training
visuals = False

_max_height = 45
_max_width = 45

columns = ['Agents', 'X_DIM', 'Y_DIM', 'TRIAL_NO',
           'REWARD', 'NORMALIZED_REWARD',
           'DONE_RATIO', 'STEPS', 'ACTION_PROB']
# To disable parallel for debug purposes etc.
parallel = True

if parallel:
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(path="./out", max_file_size=1024 * 1024 * 1024)

'''
A 2-d array matrix on-hot encoded similar to tf.one_hot function
https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy/36960495
'''


def one_hot2d(arr, depth):
    return (np.arange(depth) == arr[..., None]).astype(int)


def create_global_observation(agent_obs):
    # Taken from the file global_obs_model - Intended to be used with Impala/CNN Architectures
    global_obs = list(agent_obs)
    height, width = global_obs[0].shape[:2]
    pad_height, pad_width = _max_height - height, _max_width - width
    global_obs[1] = global_obs[1] + 1  # get rid of -1
    assert pad_height >= 0 and pad_width >= 0

    final_obs = tuple([
        np.pad(o, ((0, pad_height), (0, pad_height), (0, 0)), constant_values=0)
        for o in global_obs
    ])

    # observations = [tf.keras.layers.Input(shape=o.shape) for o in final_obs]
    # processed_observations = preprocess_obs(tuple(observations))
    processed_observations = preprocess_obs(final_obs)
    return processed_observations


def preprocess_obs(obs):
    transition_map, agents_state, targets = obs
    new_agents_state = agents_state.transpose([2, 0, 1])
    *states, = new_agents_state
    processed_agents_state_layers = []
    for i, feature_layer in enumerate(states):
        if i in {0, 1}:  # agent direction (categorical)
            feature_layer = tf.one_hot(tf.cast(feature_layer, tf.int32), depth=len(grid4.Grid4TransitionsEnum) + 1,
                                       dtype=tf.float32).numpy()
            # Numpy Version
            # feature_layer = one_hot2d(feature_layer, depth=len(grid4.Grid4TransitionsEnum) + 1)
        elif i in {2, 4}:  # counts
            feature_layer = np.expand_dims(np.log(feature_layer + 1), axis=-1)
        else:  # well behaved scalars
            feature_layer = np.expand_dims(feature_layer, axis=-1)
        processed_agents_state_layers.append(feature_layer)

    return np.concatenate([transition_map, targets] + processed_agents_state_layers, axis=-1)


def generate_experiences(trials, start=0, tree_depth=2, max_depth=30, obs_type="tree", batch_builder=None, writer=None):
    env_file = f"envs-100-999/envs/Level_{trials}.pkl"

    # env_file = f"../env_configs/round_1-small/Test_0/Level_{trials}.mpk"
    pad_name = False

    if pad_name:
        total_size = 5
        _str_trial = str(trials)
        trials = str(0) * (total_size - len(_str_trial)) + _str_trial

    # env_file = f"./{env_names}/envs/Level_{trials}.pkl"

    # file = f"../env_configs/actions-small/Test_0/Level_{trials}.mpk"
    file = f"envs-100-999/actions/envs/Level_{trials}.json"
    # file = f"./{env_names}/actions/envs/Level_{trials}.json"

    if not os.path.isfile(env_file) or not os.path.isfile(file):
        print("Missing file!", env_file, file)
        return

    step = 0

    if obs_type == "tree":

        obs_builder_object = TreeObsForRailEnv(max_depth=tree_depth,
                                               predictor=ShortestPathPredictorForRailEnv(
                                                   max_depth))

    elif obs_type == "global":
        obs_builder_object = GlobalObsForRailEnv()

    env = RailEnv(width=1, height=1,
                  rail_generator=rail_from_file(env_file),
                  schedule_generator=schedule_from_file(env_file),
                  malfunction_generator_and_process_data=malfunction_from_file(
                      env_file),
                  obs_builder_object=obs_builder_object)

    obs, info = env.reset(
        regenerate_rail=True,
        regenerate_schedule=True,
        activate_agents=False,
        random_seed=1001
    )

    with open(file, "r") as files:
        expert_actions = json.load(files)

    n_agents = env.get_num_agents()
    x_dim, y_dim = env.width, env.height

    agent_obs = [None] * n_agents
    agent_obs_buffer = [None] * n_agents
    done = dict()
    done["__all__"] = False

    if imitate:
        agent_action_buffer = list(
            expert_actions[step].values())
    else:
        # , p=[0.2, 0, 0.5])  # [0] * n_agents
        agent_action_buffer = np.random.choice(5, n_agents, replace=True)
    update_values = [False] * n_agents

    max_steps = int(4 * 2 * (20 + env.height + env.width))

    action_size = 5  # 3

    # And some variables to keep track of the progress
    action_dict = dict()
    scores_window = deque(maxlen=100)
    reward_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    action_prob = [0] * action_size

    # agent = Agent(state_size, action_size)

    if visuals:
        from flatland.utils.rendertools import RenderTool
        env_renderer = RenderTool(env, gl="PILSVG")
        env_renderer.render_env(
            show=True, frames=True, show_observations=True)

    for a in range(n_agents):
        if obs[a]:
            if obs_type == "global":
                agent_obs[a] = create_global_observation(obs[a])
            elif obs_type == "tree":
                agent_obs[a] = normalize_observation(
                    obs[a], tree_depth, observation_radius=10)
            agent_obs_buffer[a] = copy.copy(agent_obs[a])  # agent_obs[a].copy()

    # Reset score and done
    score = 0
    agent_action_buffer = np.zeros(n_agents)
    # prev_action = np.zeros_like(envs.action_space.sample())
    prev_reward = np.zeros(n_agents)
    for step in range(max_steps):
        for a in range(n_agents):
            if info['action_required'][a]:
                if imitate:
                    if step < len(expert_actions):
                        action = expert_actions[step][str(a)]
                    else:
                        action = 0
                else:
                    action = 0

                action_prob[action] += 1
                update_values[a] = True

            else:
                update_values[a] = False
                action = 0

            action_dict.update({a: action})

        next_obs, all_rewards, done, info = env.step(action_dict)

        for a in range(n_agents):

            if next_obs[a] is not None:
                if obs_type == "global":
                    agent_obs[a] = create_global_observation(next_obs[a])
                elif obs_type == "tree":
                    agent_obs[a] = normalize_observation(
                        next_obs[a], tree_depth, observation_radius=10)

            # Only update the values when we are done or when an action
            # was taken and thus relevant information is present
            if update_values[a] or done[a]:
                start += 1

                batch_builder.add_values(
                    t=step,
                    eps_id=trials,
                    agent_index=0,
                    obs=agent_obs_buffer[a],
                    actions=action_dict[a],
                    action_prob=1.0,  # put the true action probability
                    rewards=all_rewards[a],
                    prev_actions=agent_action_buffer[a],
                    prev_rewards=prev_reward[a],
                    dones=done[a],
                    infos=info['action_required'][a],
                    new_obs=agent_obs[a])

            agent_obs_buffer[a] = copy.copy(agent_obs[a])  # agent_obs[a].copy()
            agent_action_buffer[a] = action_dict[a]
            prev_reward[a] = all_rewards[a]

            score += all_rewards[a]  # / envs.get_num_agents()

        if visuals:
            env_renderer.render_env(
                show=True, frames=True, show_observations=True)

        if done["__all__"] or step > max_steps:
            writer.write(batch_builder.build_and_reset())
            break

        # Collection information about training
        if step % 100 == 0:
            tasks_finished = 0
            for current_agent in env.agents:
                if current_agent.status == RailAgentStatus.DONE_REMOVED:
                    tasks_finished += 1
            print(
                '\rTrial No {} Training {} Agents on ({},{}).\t Steps {}\t Reward: {:.3f}\t Normalized Reward: {:.3f}\tDones: {:.2f}%\t'.format(
                    trials, env.get_num_agents(), x_dim, y_dim,
                    step,
                    score,
                    score / (max_steps + n_agents),
                    100 * np.mean(tasks_finished / max(
                        1, env.get_num_agents()))), end=" ")

    tasks_finished = 0
    for current_agent in env.agents:
        if current_agent.status == RailAgentStatus.DONE_REMOVED:
            tasks_finished += 1
    done_window.append(tasks_finished / max(1, env.get_num_agents()))
    reward_window.append(score)
    scores_window.append(score / (max_steps + n_agents))

    data = [[n_agents, x_dim, y_dim,
             trials,
             np.mean(reward_window),
             np.mean(scores_window),
             100 * np.mean(done_window),
             step, action_prob / np.sum(action_prob)]]

    df_cur = pd.DataFrame(data, columns=columns)

    print(
        '\rTrial No {} Training {} Agents on ({},{}).\t Total Steps {}\t Reward: {:.3f}\t Normalized Reward: {:.3f}\tDones: {:.2f}%\t'.format(
            trials, env.get_num_agents(), x_dim, y_dim,
            step,
            np.mean(reward_window),
            np.mean(scores_window),
            100 * np.mean(done_window)))

    if visuals:
        env_renderer.close_window()

    return df_cur


def main():
    args = parser.parse_args()
    if args.single:
        print("Running process in single process")
        global parallel
        parallel = False
    if args.visual:
        print("Rendering environment")
        global visuals
        visuals = True
    if args.globalobs:
        print("Running for global observation")
        global obs_type
        obs_type = "global"

    if visuals:
        from flatland.utils.rendertools import RenderTool

    max_depth = 30
    tree_depth = 2
    trial_start = 100
    n_trials = 999
    start = 0

    df_all_results = pd.DataFrame(columns=columns)

    all_trials = range(trial_start, n_trials + 1)

    if parallel:
        from ray.util.multiprocessing import Pool
        print(tf.executing_eagerly(), tf.__version__)
        pool = Pool(processes=None)

        # By default, Ray uses this to determine the number of CPUs
        # TODO: Check if splitting based on cores yields better performance
        # Especially for cases where there are too many trials and far less cpu cores

        # import psutil
        # n_cores = psutil.cpu_count()
        # parallel_splits = np.array_split(np.array(all_trials),n_cores)

        generate_experiences_trial = partial(generate_experiences, start=start, tree_depth=tree_depth,
                                             max_depth=max_depth, obs_type=obs_type, batch_builder=batch_builder,
                                             writer=writer)

        for df_cur in pool.map(generate_experiences_trial, all_trials):
            if df_cur is not None:
                df_all_results = pd.concat([df_all_results, df_cur])

    else:

        generate_experiences_trial = partial(generate_experiences, start=start, tree_depth=tree_depth,
                                             max_depth=max_depth, obs_type=obs_type, batch_builder=SampleBatchBuilder(),
                                             writer=JsonWriter(path="/", max_file_size=1024 * 1024 * 1024))

        for trial in all_trials:
            df_cur = generate_experiences_trial(trial)
            if df_cur is not None:
                df_all_results = pd.concat([df_all_results, df_cur])

    if imitate:
        df_all_results.to_csv(
            f'TreeImitationLearning_DQN_TrainingResults.csv', index=False)


if __name__ == '__main__':
    main()
