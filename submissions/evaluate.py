import multiprocessing
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from flatland.envs.malfunction_generators import NoMalfunctionGen
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from tqdm import tqdm

from flatlander.envs.observations.conflict_piority_shortest_path_obs import ConflictPriorityShortestPathObservation
from flatlander.planning.epsilon_greedy_planning import epsilon_greedy_plan
from flatlander.utils.helper import is_done

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
seed = 43557


def get_env():
    schedule_generator = sparse_schedule_generator(None)

    rail_generator = sparse_rail_generator(
        seed=seed,
        max_num_cities=3,
        grid_mode=False,
        max_rails_between_cities=2,
        max_rails_in_city=4,
    )

    obs_builder = ConflictPriorityShortestPathObservation(config={'predictor': 'custom', 'asserts': False,
                                                                  'shortest_path_max_depth': 10}).builder()

    env = RailEnv(
        width=25,
        height=25,
        rail_generator=rail_generator,
        schedule_generator=schedule_generator,
        number_of_agents=10,
        malfunction_generator=NoMalfunctionGen(),
        obs_builder_object=obs_builder,
        remove_agents_at_target=True,
        random_seed=seed,
    )

    return env


def evaluate(n_episodes):
    env = get_env()

    for _ in range(n_episodes):

        obs, _ = env.reset(regenerate_schedule=True, regenerate_rail=True)
        # env_renderer = RenderTool(env)
        # env_renderer.render_env(show=True, frames=True, show_observations=False)

        if not obs:
            break

        steps = 0
        actions = epsilon_greedy_plan(env, obs)
        done = defaultdict(lambda: False)

        for a in tqdm(actions):
            obs, all_rewards, done, info = env.step(a)
            steps += 1

            if done['__all__']:
                break

        while not done['__all__']:
            obs, all_rewards, done, info = env.step({})
            steps += 1

        pc = np.sum(np.array([1 for a in env.agents if is_done(a)])) / env.get_num_agents()
        print("EPISODE PC:", pc)
        n_episodes += 1


def main(n_episodes):
    n_cpu = 4
    res_queue = multiprocessing.Queue()
    with Pool(processes=n_cpu) as pool:
        for _ in range(n_cpu):
            pool.apply_async(evaluate, args=(int(n_episodes / n_cpu), res_queue,))

    res = []
    for _ in tqdm(range(n_episodes)):
        res.append(res_queue.get())

    print(f"NUMBER EPISODES: {len(res)}, MEAN_PERCENTAGE_COMPLETE {np.mean(res)}")


if __name__ == "__main__":
    evaluate(10, )
