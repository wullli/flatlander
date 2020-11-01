import multiprocessing
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from flatland.envs.malfunction_generators import NoMalfunctionGen, MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from tqdm import tqdm

from flatlander.agents.shortest_path_agent import ShortestPathAgent
from flatlander.envs.observations import make_obs
from flatlander.envs.observations.simple_meta_obs import SimpleMetaObservation
from flatlander.envs.utils.robust_gym_env import RobustFlatlandGymEnv
from flatlander.utils.helper import is_done, init_run, get_agent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
seed = 234123
RENDER = True


def get_env():
    n_agents = 50
    config, run = init_run()
    schedule_generator = sparse_schedule_generator(None)
    # trainer = get_agent(config, run, 5)

    rail_generator = sparse_rail_generator(
        seed=seed,
        max_num_cities=4,
        grid_mode=False,
        max_rails_between_cities=2,
        max_rails_in_city=4,
    )

    obs_builder = make_obs(config["env_config"]['observation'],
                           config["env_config"].get('observation_config')).builder()

    params = MalfunctionParameters(malfunction_rate=1 / 250,
                                   max_duration=50,
                                   min_duration=20)
    malfunction_generator = ParamMalfunctionGen(params)

    env = RailEnv(
        width=40,
        height=40,
        rail_generator=rail_generator,
        schedule_generator=schedule_generator,
        number_of_agents=n_agents,
        malfunction_generator=malfunction_generator,
        obs_builder_object=SimpleMetaObservation({}).builder(),
        remove_agents_at_target=True,
        random_seed=seed,
    )

    return env, None


def evaluate(n_episodes):
    env, agent = get_env()
    env_renderer = RenderTool(env)

    for _ in range(n_episodes):

        obs, _ = env.reset(regenerate_schedule=True, regenerate_rail=True)
        if RENDER:
            env_renderer.reset()
            env_renderer.render_env(show=True, frames=True, show_observations=True)

        if not obs:
            break

        steps = 0
        done = defaultdict(lambda: False)
        robust_env = RobustFlatlandGymEnv(rail_env=env, observation_space=None, allow_noop=True)
        sorted_handles = robust_env.prioritized_agents(handles=obs.keys())

        while not done['__all__']:
            actions = ShortestPathAgent().compute_actions(obs, env)
            robust_actions = robust_env.get_robust_actions(actions, sorted_handles)
            obs, all_rewards, done, info = env.step(robust_actions)
            if RENDER:
                env_renderer.render_env(show=True, frames=True, show_observations=False)
            print('.', end='', flush=True)
            steps += 1

        pc = np.sum(np.array([1 for a in env.agents if is_done(a)])) / env.get_num_agents()
        print("EPISODE PC:", pc)
        n_episodes += 1


if __name__ == "__main__":
    evaluate(10)
