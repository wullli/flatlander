import multiprocessing
import os

import numpy as np
import ray
import yaml
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from ray.rllib.agents import Trainer
from ray.tune import register_env
from ray.tune.trial import Trial

from flatlander.envs.flatland_sparse import FlatlandSparse
from flatlander.utils.loader import load_models, load_envs
from flatlander.utils.submissions import RUN, CURRENT_ENV_PATH, get_tune_time, AGENT_MAP

n_cpu = multiprocessing.cpu_count()
print("***** NUM CPUS AVAILABLE:", n_cpu, "*****")


def get_parameters():
    run = RUN
    print("RUNNING", RUN)

    with open(os.path.join(os.path.dirname(run["checkpoint_paths"][1]), "config.yaml")) as f:
        config = yaml.safe_load(f)

    return run, config


def init_run():
    run, config = get_parameters()
    load_envs(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../runner")))
    load_models(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../runner")))

    ray.init(local_mode=True, num_cpus=n_cpu)
    return config, run


def get_agent(config, run, n_agents) -> Trainer:
    agent = AGENT_MAP[run["agent"]](config=config)
    n_agents = min(18, n_agents)
    agent.restore(run["checkpoint_paths"][n_agents])
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


def is_done(agent):
    return agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED


def get_agent_pos(agent):
    if agent.status == RailAgentStatus.READY_TO_DEPART:
        agent_virtual_position = agent.initial_position
    elif agent.status == RailAgentStatus.ACTIVE:
        agent_virtual_position = agent.position
    elif agent.status == RailAgentStatus.DONE:
        agent_virtual_position = agent.target
    else:
        return None
    return agent_virtual_position


def episode_end_info(all_rewards,
                     total_reward,
                     evaluation_number,
                     steps, remote_client):
    reward_values = np.array(list(all_rewards.values()))
    mean_reward = np.mean((1 + reward_values) / 2)
    pc = np.sum(np.array([1 for a in remote_client.env.agents if is_done(a)])) / remote_client.env.get_num_agents()
    print("EPISODE PC:", pc)
    print("\n\nMean reward: ", mean_reward)
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
    num_agents = env.get_num_agents()
    tune_time = get_tune_time(num_agents)

    def env_creator(env_config):
        return FlatlandSparse(env_config, fine_tune_env_path=CURRENT_ENV_PATH, max_steps=num_agents * 100)

    register_env("flatland_sparse", env_creator)
    config['num_workers'] = 3
    config['num_envs_per_worker'] = 1
    config['lr'] = 0.00001 * num_agents
    exp_an = ray.tune.run(run["agent"],
                          reuse_actors=True,
                          verbose=1,
                          stop={"time_since_restore": tune_time},
                          checkpoint_freq=1,
                          keep_checkpoints_num=1,
                          checkpoint_score_attr="episode_reward_mean",
                          config=config,
                          restore=run["checkpoint_path"])

    trial: Trial = exp_an.trials[0]
    agent_config = trial.config
    agent_config['num_workers'] = 0
    agent = trial.get_trainable_cls()(env=config["env"], config=trial.config)
    checkpoint = exp_an.get_trial_checkpoints_paths(trial, metric="episode_reward_mean")
    agent.restore(checkpoint[0][0])
    return agent
