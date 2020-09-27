import os
from time import sleep

import numpy as np
from flatland.envs.malfunction_generators import NoMalfunctionGen
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

from flatlander.envs.observations.tree_obs import TreeObsForRailEnvRLLibWrapper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from flatlander.envs.observations import make_obs
from flatlander.utils.helper import init_run, get_agent

tf.compat.v1.disable_eager_execution()
seed = 23


def get_env():
    schedule_generator = sparse_schedule_generator(None)

    rail_generator = sparse_rail_generator(
        seed=seed,
        max_num_cities=3,
        grid_mode=False,
        max_rails_between_cities=2,
        max_rails_in_city=4,
    )

    obs_builder = make_obs(config["env_config"]['observation'],
                           config["env_config"].get('observation_config')).builder()

    env = RailEnv(
        width=29,
        height=29,
        rail_generator=rail_generator,
        schedule_generator=schedule_generator,
        number_of_agents=10,
        malfunction_generator=NoMalfunctionGen(),
        obs_builder_object=obs_builder,
        remove_agents_at_target=False,
        random_seed=seed,
    )

    return env


def render(config, run):
    env = get_env()
    env_renderer = RenderTool(env)

    agent = get_agent(config, run)

    while True:

        obs, _ = env.reset(regenerate_schedule=False, regenerate_rail=False)
        env_renderer.render_env(show=True, frames=True, show_observations=False)

        if not obs:
            break

        steps = 0

        while True:
            obs_batch = np.array(list(obs.values()))
            action_batch = agent.get_policy().compute_actions(obs_batch, explore=False)
            actions = dict(zip(obs.keys(), action_batch[0]))

            obs, all_rewards, done, info = env.step(actions)
            env_renderer.render_env(show=True, frames=True, show_observations=False)
            steps += 1

            print('.', end='', flush=True)
            sleep(0.1)

            if done['__all__']:
                break


if __name__ == "__main__":
    config, run = init_run()
    render(config, run)
