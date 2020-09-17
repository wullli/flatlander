import os

import numpy as np
import ray
import yaml
from ray.rllib.agents import sac, ppo, dqn

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool
from flatlander.envs.observations import make_obs
from flatlander.envs.observations.default_observation_builder import CustomObservationBuilder
from flatlander.mcts.mcts import MonteCarloTreeSearch
from flatlander.utils.loader import load_envs, load_models

agent_map = {"sac": sac.SACTrainer,
             "ppo": ppo.PPOTrainer,
             "dqn": dqn.DQNTrainer}

runs = {
    "apex_dqn_1": {
        "checkpoint_path": os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "model_checkpoints/apex_dqn_small_v0/checkpoint_119/checkpoint-119")),
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


def init():
    run = runs[RUN]
    print("RUNNING", RUN)
    with open(os.path.join(os.path.dirname(run["checkpoint_path"]), "config.yaml")) as f:
        config = yaml.safe_load(f)

    load_envs("../flatlander/runner")

    obs_builder = make_obs(config["env_config"]['observation'],
                           config["env_config"].get('observation_config')).builder()

    load_envs(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../flatlander/runner")))
    load_models(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../flatlander/runner")))

    ray.init(local_mode=True, num_cpus=1, num_gpus=1)
    agent = run["agent"](config=config)
    agent.restore(run["checkpoint_path"])
    policy = agent.get_policy()
    return policy, obs_builder


def dqn_rollout_policy(obs, policy):
    o = np.array(list(obs.values()))
    a = policy.compute_actions(o, explore=False)
    a = dict(zip(obs.keys(), a[0]))
    return a


policy, obs_builder = init()

env = RailEnv(width=25, height=25,
              rail_generator=sparse_rail_generator(),
              number_of_agents=4,
              obs_builder_object=obs_builder)

mcts = MonteCarloTreeSearch(5, epsilon=1, rollout_depth=30,
                            rollout_policy=lambda obs: dqn_rollout_policy(obs, policy))

obs, _ = env.reset()

env_renderer = RenderTool(env)
env_renderer.render_env(show=True, frames=True, show_observations=False)
done = {"__all__": False}

episode_return = 0
while not done["__all__"]:
    action = mcts.get_best_actions(env=env, obs=obs)
    obs, all_rewards, done, _ = env.step(action)
    episode_return += np.sum(list(all_rewards.values()))
    env_renderer.render_env(show=True, frames=True, show_observations=False)
    print("Rewards: ", all_rewards, "  [done=", done, "]")

print("Episode return:", episode_return)
