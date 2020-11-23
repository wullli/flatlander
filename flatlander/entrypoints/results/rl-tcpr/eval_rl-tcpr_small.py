import os
from collections import defaultdict

import numpy as np
import pandas as pd
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from tqdm import tqdm

from flatlander.agents.shortest_path_agent import ShortestPathAgent
from flatlander.agents.shortest_path_rllib_agent import ShortestPathRllibAgent
from flatlander.envs.observations import make_obs
from flatlander.envs.observations.dummy_obs import DummyObs
from flatlander.envs.utils.cpr_gym_env import CprFlatlandGymEnv
from flatlander.envs.utils.priorization.priorizer import DistToTargetPriorizer
from flatlander.submission.helper import is_done, init_run, get_agent
from flatlander.submission.submissions import SUBMISSIONS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
seed = 0
RENDER = False

EVAL_NAME = "RL-TCPR-small"


def get_env(config=None, rl=False):
    n_agents = 16
    schedule_generator = sparse_schedule_generator(None)

    rail_generator = sparse_rail_generator(
        seed=seed,
        max_num_cities=3,
        grid_mode=False,
        max_rails_between_cities=2,
        max_rails_in_city=4,
    )

    if rl:
        obs_builder = make_obs("combined", {"path": None, "simple_meta": None}).builder()
    else:
        obs_builder = DummyObs()

    params = MalfunctionParameters(malfunction_rate=1 / 1000,
                                   max_duration=50,
                                   min_duration=20)
    malfunction_generator = ParamMalfunctionGen(params)

    env = RailEnv(
        width=28,
        height=28,
        rail_generator=rail_generator,
        schedule_generator=schedule_generator,
        number_of_agents=n_agents,
        malfunction_generator=malfunction_generator,
        obs_builder_object=obs_builder,
        remove_agents_at_target=True,
        random_seed=seed,
    )

    return env


def evaluate(n_episodes):
    run = SUBMISSIONS["rlpr-tcpr"]
    config, run = init_run(run)
    prio_agent = get_agent(config, run)

    run = SUBMISSIONS["rlps-tcpr"]
    config, run = init_run(run)
    step_agent = ShortestPathRllibAgent(get_agent(config, run), explore=False)

    env = get_env(None, rl=True)
    env_renderer = RenderTool(env, screen_width=8800)
    returns = []
    pcs = []
    malfs = []

    for _ in tqdm(range(n_episodes)):

        obs, _ = env.reset(regenerate_schedule=True, regenerate_rail=True)
        if RENDER:
            env_renderer.reset()
            env_renderer.render_env(show=True, frames=True, show_observations=False)

        if not obs:
            break

        steps = 0
        ep_return = 0
        done = defaultdict(lambda: False)
        robust_env = CprFlatlandGymEnv(rail_env=env,
                                       max_nr_active_agents=200,
                                       observation_space=None,
                                       priorizer=DistToTargetPriorizer(),
                                       allow_noop=True)
        meta_obs = {h: o[1] for h, o in obs.items()}
        priorities = prio_agent.compute_actions(meta_obs, explore=False)
        sorted_actions = {k: v for k, v in sorted(priorities.items(), key=lambda item: item[1], reverse=True)}
        sorted_handles = list(sorted_actions.keys())

        while not done['__all__']:
            agent_obs = {h: o[0] for h, o in obs.items()}
            actions = step_agent.compute_actions(agent_obs, env)
            robust_actions = robust_env.get_robust_actions(actions, sorted_handles)
            obs, all_rewards, done, info = env.step(robust_actions)
            if RENDER:
                env_renderer.render_env(show=True, frames=True, show_observations=False)
            print('.', end='', flush=True)
            steps += 1
            ep_return += np.sum(list(all_rewards.values()))

        pc = np.sum(np.array([1 for a in env.agents if is_done(a)])) / env.get_num_agents()
        print("EPISODE PC:", pc)
        n_episodes += 1
        pcs.append(pc)
        returns.append(ep_return / (env._max_episode_steps * env.get_num_agents()))
        malfs.append(np.sum([a.malfunction_data['nr_malfunctions'] for a in env.agents]))
    return pcs, returns, malfs


if __name__ == "__main__":
    episodes = 1000
    pcs, returns, malfs = evaluate(episodes)
    df = pd.DataFrame(data={"pc": pcs, "returns": returns, 'malfs': malfs})
    df.to_csv(os.path.join('..', f'{EVAL_NAME}_{episodes}-episodes.csv'))
    print(f'Mean PC: {np.mean(pcs)}')
    print(f'Mean Episode return: {np.mean(returns)}')
