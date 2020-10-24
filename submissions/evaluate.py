import multiprocessing
import os
from multiprocessing import Pool

import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.malfunction_generators import NoMalfunctionGen, ParamMalfunctionGen
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from tqdm import tqdm

from flatlander.agents.heuristic_agent import HeuristicPriorityAgent
from flatlander.envs.observations.conflict_piority_shortest_path_obs import ConflictPriorityShortestPathObservation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
seed = 934775


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

    stochastic_data = {
        'malfunction_rate': 250,
        'min_duration': 20,
        'max_duration': 50
    }

    malfunction_generator = ParamMalfunctionGen(stochastic_data)

    env = RailEnv(
        width=25,
        height=25,
        rail_generator=rail_generator,
        schedule_generator=schedule_generator,
        number_of_agents=5,
        malfunction_generator=NoMalfunctionGen(),
        obs_builder_object=obs_builder,
        remove_agents_at_target=True,
        random_seed=seed,
    )

    return env


def is_done(agent):
    return agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED


def evaluate(n_episodes, res_queue: multiprocessing.Queue):
    env = get_env()

    for _ in range(n_episodes):

        obs, _ = env.reset(regenerate_schedule=True, regenerate_rail=True)
        # env_renderer = RenderTool(env)
        # env_renderer.render_env(show=True, frames=True, show_observations=False)

        if not obs:
            break

        steps = 0

        while True:
            action_batch = HeuristicPriorityAgent().compute_actions(obs, env=env)
            obs, all_rewards, done, info = env.step(action_batch)
            # env_renderer.render_env(show=True, frames=True, show_observations=False)
            steps += 1

            if done['__all__']:
                pc = np.sum(np.array([1 for a in env.agents if is_done(a)])) / env.get_num_agents()
                res_queue.put(pc)
                n_episodes += 1
                break


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
    main(100)
