import os
from copy import deepcopy, copy

import numpy as np
from flatland.envs.rail_env import RailEnvActions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from flatland.evaluators.client import FlatlandRemoteClient
from flatlander.envs.observations import make_obs
from flatlander.utils.deadlock_check import check_if_all_blocked
from flatlander.utils.helper import fine_tune, episode_start_info, episode_end_info, skip, init_run, get_agent
from timeit import default_timer as timer

tf.compat.v1.disable_eager_execution()
remote_client = FlatlandRemoteClient()

TUNE = False
TIME_LIMIT = 60*60*7.75


def evaluate(config, run):
    start_time = timer()
    obs_builder = make_obs(config["env_config"]['observation'],
                           config["env_config"].get('observation_config')).builder()

    evaluation_number = 0
    total_reward = 0
    all_rewards = []
    n_agents = 0

    while True:
        if timer() - start_time > TIME_LIMIT:
            remote_client.submit()
            return

        observation, info = remote_client.env_create(obs_builder_object=obs_builder)
        action_template = {handle: RailEnvActions.STOP_MOVING for handle in observation.keys()}

        if not observation:
            break

        if remote_client.env.get_num_agents() != n_agents:
            agent = get_agent(config, run,  remote_client.env.get_num_agents())
            n_agents = remote_client.env.get_num_agents()

        if TUNE and remote_client.env.get_num_agents() > 10:
            agent = fine_tune(config, run, env=remote_client.env)

        evaluation_number += 1

        episode_start_info(evaluation_number, remote_client=remote_client)

        steps = 0
        done = {}

        while True:
            if not check_if_all_blocked(env=remote_client.env):

                local_env = deepcopy(remote_client.env)

                real_actions = {}
                for handle in observation.keys():
                    actions = action_template.copy()
                    actions[handle] = agent.compute_action(observation[handle], explore=False)
                    observation, _, _, _ = local_env.step(actions)
                    real_actions[handle] = actions[handle]

                observation, all_rewards, done, info = remote_client.env_step(real_actions)
                steps += 1

                print('.', end='', flush=True)

            elif not done['__all__']:
                skip(remote_client=remote_client)

            if done['__all__']:
                total_reward = episode_end_info(all_rewards,
                                                total_reward,
                                                evaluation_number,
                                                steps, remote_client=remote_client)
                break

    print("Evaluation of all environments complete...")
    print(remote_client.submit())


if __name__ == "__main__":
    config, run = init_run()
    evaluate(config, run)
