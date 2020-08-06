import gc
import getopt
# from gen_envs import *
import json
import os
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from utils.observation_utils import normalize_observation  # noqa

from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.malfunction_generators import malfunction_from_file
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.schedule_generators import schedule_from_file
from flatland.utils.misc import str2bool

imitate = True


## Legacy Code for the correct expert actions

# change below line in method malfunction_from_file in the file flatland.env.malfunction_generators.py
# mean_malfunction_rate = 1/oMPD.malfunction_rate

def main(args):
    try:
        opts, args = getopt.getopt(args, "", ["sleep-for-animation=", ""])
    except getopt.GetoptError as err:
        print(str(err))  # will print something like "option -a not recognized"
        sys.exit(2)
    sleep_for_animation = True
    for o, a in opts:
        if o in ("--sleep-for-animation"):
            sleep_for_animation = str2bool(a)
        else:
            assert False, "unhandled option"

    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter("./")

    #  Setting these 2 parameters to True can slow down training
    visuals = False
    sleep_for_animation = False

    if visuals:
        from flatland.utils.rendertools import RenderTool

    max_depth = 30
    tree_depth = 2
    trial_start = 100
    n_trials = 999
    start = 0

    columns = ['Agents', 'X_DIM', 'Y_DIM', 'TRIAL_NO',
               'REWARD', 'NORMALIZED_REWARD',
               'DONE_RATIO', 'STEPS', 'ACTION_PROB']
    df_all_results = pd.DataFrame(columns=columns)

    for trials in range(trial_start, n_trials + 1):

        env_file = f"envs-100-999/envs/Level_{trials}.pkl"
        # env_file = f"../env_configs/test-env-small/Test_0/Level_{trials}.mpk"

        # file = f"../env_configs/actions-small/Test_0/Level_{trials}.mpk"
        file = f"envs-100-999/actions/envs/Level_{trials}.json"

        if not os.path.isfile(env_file) or not os.path.isfile(file):
            print("Missing file!", env_file, file)
            continue

        step = 0

        obs_builder_object = TreeObsForRailEnv(max_depth=tree_depth,
                                               predictor=ShortestPathPredictorForRailEnv(
                                                   max_depth))

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
            env_renderer = RenderTool(env, gl="PILSVG")
            env_renderer.render_env(
                show=True, frames=True, show_observations=True)

        for a in range(n_agents):
            if obs[a]:
                agent_obs[a] = normalize_observation(
                    obs[a], tree_depth, observation_radius=10)
                agent_obs_buffer[a] = agent_obs[a].copy()

        # Reset score and done
        score = 0
        agent_action_buffer = np.zeros(n_agents)
        # prev_action = np.zeros_like(env.action_space.sample())
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

                agent_obs_buffer[a] = agent_obs[a].copy()
                agent_action_buffer[a] = action_dict[a]
                prev_reward[a] = all_rewards[a]

                score += all_rewards[a]  # / env.get_num_agents()

            if visuals:
                env_renderer.render_env(
                    show=True, frames=True, show_observations=True)
                if sleep_for_animation:
                    time.sleep(0.5)

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
        df_all_results = pd.concat([df_all_results, df_cur])

        if imitate:
            df_all_results.to_csv(
                f'TreeImitationLearning_DQN_TrainingResults.csv', index=False)

        print(
            '\rTrial No {} Training {} Agents on ({},{}).\t Total Steps {}\t Reward: {:.3f}\t Normalized Reward: {:.3f}\tDones: {:.2f}%\t'.format(
                trials, env.get_num_agents(), x_dim, y_dim,
                step,
                np.mean(reward_window),
                np.mean(scores_window),
                100 * np.mean(done_window)))

        if visuals:
            env_renderer.close_window()

        gc.collect()


if __name__ == '__main__':
    if 'argv' in globals():
        main(sys.argv)
    else:
        main(sys.argv[1:])
