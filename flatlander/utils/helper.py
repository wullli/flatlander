import os

import numpy as np
import ray
import yaml
from ray.tune.trial import Trial

from flatland.envs.rail_env import RailEnv
from ray.tune import register_env

from flatland.envs.persistence import RailEnvPersister
from ray.rllib.agents import Trainer

from flatlander.envs.flatland_sparse import FlatlandSparse
from flatlander.utils.loader import load_models, load_envs
from flatlander.utils.submissions import RUN, CURRENT_ENV_PATH


def init_run():
    run = RUN
    print("RUNNING", RUN)
    with open(os.path.join(os.path.dirname(run["checkpoint_path"]), "config.yaml")) as f:
        config = yaml.safe_load(f)

    load_envs("../flatlander/runner")

    load_envs(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../flatlander/runner")))
    load_models(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../flatlander/runner")))
    ray.init(local_mode=False)
    return config, run


def get_agent(config, run) -> Trainer:
    agent = run["agent"](config=config)
    agent.restore(run["checkpoint_path"])
    return agent


def skip(remote_client):
    _, all_rewards, done, info = remote_client.env_step({})
    print('!', end='', flush=True)


def episode_start_info(evaluation_number, remote_client):
    print("=" * 100)
    print("=" * 100)
    print("Starting evaluation #{}".format(evaluation_number))
    print("Number of agents:", len(remote_client.env.agents))
    print("Environment size:", remote_client.env.width, "x", remote_client.env.height)


def episode_end_info(all_rewards,
                     total_reward,
                     evaluation_number,
                     steps, remote_client):
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
    """
    Fine-tune the agent on a static env at evaluation time
    """
    RailEnvPersister.save(env, CURRENT_ENV_PATH)

    def env_creator(env_config):
        return FlatlandSparse(env_config, fine_tune_env_path=CURRENT_ENV_PATH)

    # register_env("flatland_sparse", env_creator)

    exp_an = ray.tune.run(run["agent"],
                          verbose=1,
                          checkpoint_freq=1,
                          keep_checkpoints_num=1,
                          stop={"time_since_restore": 20},
                          checkpoint_score_attr="episode_reward_mean",
                          config=config, restore=run["checkpoint_path"])

    trial: Trial = exp_an.trials[0]
    agent = trial.get_trainable_cls()(env=config["env"], config=trial.config)

    return agent
