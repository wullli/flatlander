import os
from copy import deepcopy

from flatland.envs.persistence import RailEnvPersister

from flatland.envs.rail_env import RailEnv
from ray.tune import register_env

from flatlander.envs.flatland_sparse import FlatlandSparse
from flatlander.utils.deadlock_check import check_if_all_blocked

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import os
from timeit import default_timer as timer

import ray
import yaml
from ray.rllib.agents import sac, ppo, dqn, Trainer

from flatlander.envs.observations import make_obs
from flatlander.utils.loader import load_envs, load_models

from flatland.evaluators.client import FlatlandRemoteClient

tf.compat.v1.disable_eager_execution()
remote_client = FlatlandRemoteClient()

agent_map = {"sac": sac.SACTrainer,
             "ppo": ppo.PPOTrainer,
             "dqn": dqn.DQNTrainer}

runs = {
    "apex_dqn_1": {
        "checkpoint_path": os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "model_checkpoints/apex_dqn_small_v0/checkpoint_230/checkpoint-230")),
        "agent": agent_map["dqn"]
    },
    "sac_small_v0": {
        "checkpoint_path": os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "model_checkpoints/sac_small_v0/checkpoint-51089/checkpoint-51089")),
        "agent": agent_map["sac"]
    },
    "ttf_1": {
        "checkpoint_path": os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "model_checkpoints/tree_tf_1/checkpoint_5860_scaling/checkpoint-5860")),
        "agent": agent_map["sac"]
    }}

RUN = "apex_dqn_1"
FINETUNE_BUDGET_MS = 1000
CURRENT_ENV_PATH = './current_env.pkl'


def init():
    run = runs[RUN]
    print("RUNNING", RUN)
    with open(os.path.join(os.path.dirname(run["checkpoint_path"]), "config.yaml")) as f:
        config = yaml.safe_load(f)

    load_envs("../flatlander/runner")

    load_envs(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../flatlander/runner")))
    load_models(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../flatlander/runner")))
    return config, run


def get_agent(config, run) -> Trainer:
    ray.init(local_mode=True, num_cpus=0, num_gpus=0)
    agent = run["agent"](config=config)
    agent.restore(run["checkpoint_path"])
    return agent


def skip():
    _, all_rewards, done, info = remote_client.env_step({})
    print('!', end='', flush=True)


def episode_start_info(evaluation_number):
    print("=" * 100)
    print("=" * 100)
    print("Starting evaluation #{}".format(evaluation_number))
    print("Number of agents:", len(remote_client.env.agents))
    print("Environment size:", remote_client.env.width, "x", remote_client.env.height)


def episode_end_info(all_rewards,
                     total_reward,
                     evaluation_number,
                     steps):
    reward_values = np.array(list(all_rewards.values()))
    mean_reward = np.mean((1 + reward_values) / 2)
    total_reward += mean_reward
    print("\n\nMean reward: ", mean_reward)
    print("Total reward: ", total_reward, "\n")

    print("Evaluation Number : ", evaluation_number)
    print("Current Env Path : ", remote_client.current_env_path)
    print("Number of Steps : ", steps)
    print("=" * 100)
    return total_reward


def fine_tune(config, run, env: RailEnv):
    start_time = timer()

    RailEnvPersister.save(env, CURRENT_ENV_PATH)

    def env_creator(env_config):
        return FlatlandSparse(env_config, fine_tune_env_path=CURRENT_ENV_PATH)

    register_env("flatland_sparse", env_creator)

    agent = get_agent(config, run)

    while timer() - start_time < FINETUNE_BUDGET_MS:
        agent.train()

    return agent


def evaluate(config, run):
    obs_builder = make_obs(config["env_config"]['observation'],
                           config["env_config"].get('observation_config')).builder()

    evaluation_number = 0
    total_reward = 0
    while True:

        observation, info = remote_client.env_create(obs_builder_object=obs_builder)
        agent = fine_tune(config, run, env=remote_client.env)

        evaluation_number += 1

        if not observation:
            break

        episode_start_info(evaluation_number)

        steps = 0

        while True:
            if not check_if_all_blocked(env=remote_client.env):

                obs_batch = np.array(list(observation.values()))
                action_batch = agent.get_policy().compute_actions(obs_batch, explore=False)
                actions = dict(zip(observation.keys(), action_batch[0]))

                observation, all_rewards, done, info = remote_client.env_step(actions)
                steps += 1

                while len(observation) == 0 and not done['__all__']:
                    observation, all_rewards, done, info = remote_client.env_step({})
                    print('.', end='', flush=True)
                    steps += 1

                print('.', end='', flush=True)

            elif not done['__all__']:
                skip()

            if done['__all__']:
                total_reward = episode_end_info(all_rewards,
                                                total_reward,
                                                evaluation_number,
                                                steps)
                break

    print("Evaluation of all environments complete...")
    print(remote_client.submit())


if __name__ == "__main__":
    config, run = init()
    evaluate(config, run)
