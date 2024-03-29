import logging

import gym
import numpy as np

from flatland.envs.malfunction_generators import no_malfunction_generator, malfunction_from_params
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatlander.envs import get_generator_config
from flatlander.envs.observations import make_obs
from flatlander.envs.utils.global_gym_env import GlobalFlatlandGymEnv
from flatlander.envs.utils.gym_env import FlatlandGymEnv, StepOutput
from flatlander.envs.utils.gym_env_wrappers import SkipNoChoiceCellsWrapper, AvailableActionsWrapper, \
    ShortestPathActionWrapper, SparseRewardWrapper, DeadlockWrapper, DeadlockResolutionWrapper
from flatlander.envs.utils.single_gym_env import SingleFlatlandGymEnv


class FlatlandSingle(gym.Env):
    _gym_envs = {"default": FlatlandGymEnv,
                 "global": GlobalFlatlandGymEnv,
                 "single": SingleFlatlandGymEnv}

    def render(self, mode='human'):
        pass

    def __init__(self, env_config):
        self._observation = make_obs(env_config['observation'], env_config.get('observation_config'))
        self._config = get_generator_config(env_config['generator_config'])
        self._global_obs = env_config.get('observation_config', {}).get('global_obs', False)

        # Overwrites with env_config seed if it exists
        if env_config.get('seed'):
            self._config['seed'] = env_config.get('seed')

        if env_config.get('gym_env', None) is not None:
            self._gym_env_class = self._gym_envs[env_config["gym_env"]]
        else:
            self._gym_env_class = self._gym_envs["default"] if not self._global_obs else self._gym_envs["single"]

        self._env = self._gym_env_class(
            rail_env=self._launch(),
            observation_space=self._observation.observation_space(),
            regenerate_rail_on_reset=self._config['regenerate_rail_on_reset'],
            regenerate_schedule_on_reset=self._config['regenerate_schedule_on_reset'],
            config=env_config
        )
        if env_config['observation'] == 'shortest_path':
            self._env = ShortestPathActionWrapper(self._env)
        if env_config.get('sparse_reward', False):
            self._env = SparseRewardWrapper(self._env, finished_reward=env_config.get('done_reward', 1),
                                            not_finished_reward=env_config.get('not_finished_reward', -1))
        if env_config.get('deadlock_reward', 0) != 0:
            self._env = DeadlockWrapper(self._env, deadlock_reward=env_config['deadlock_reward'])
        if env_config.get('resolve_deadlocks', False):
            deadlock_reward = env_config.get('deadlock_reward', 0)
            self._env = DeadlockResolutionWrapper(self._env, deadlock_reward)
        if env_config.get('skip_no_choice_cells', False):
            self._env = SkipNoChoiceCellsWrapper(self._env, env_config.get('accumulate_skipped_rewards', False))
        if env_config.get('available_actions_obs', False):
            self._env = AvailableActionsWrapper(self._env, allow_noop=False)

    def get_rail_generator(self):
        rail_generator = sparse_rail_generator(
            seed=self._config['seed'],
            max_num_cities=self._config['max_num_cities'],
            grid_mode=self._config['grid_mode'],
            max_rails_between_cities=self._config['max_rails_between_cities'],
            max_rails_in_city=self._config['max_rails_in_city']
        )
        return rail_generator

    def _launch(self):
        rail_generator = self.get_rail_generator()

        malfunction_generator = no_malfunction_generator()
        if {'malfunction_rate', 'min_duration', 'max_duration'} <= self._config.keys():
            stochastic_data = {
                'malfunction_rate': self._config['malfunction_rate'],
                'min_duration': self._config['malfunction_min_duration'],
                'max_duration': self._config['malfunction_max_duration']
            }
            malfunction_generator = malfunction_from_params(stochastic_data)

        speed_ratio_map = None
        if 'speed_ratio_map' in self._config:
            speed_ratio_map = {
                float(k): float(v) for k, v in self._config['speed_ratio_map'].items()
            }
        schedule_generator = sparse_schedule_generator(speed_ratio_map)

        env = None
        try:
            env = RailEnv(
                width=self._config['width'],
                height=self._config['height'],
                rail_generator=rail_generator,
                schedule_generator=schedule_generator,
                number_of_agents=self._config['number_of_agents'],
                malfunction_generator_and_process_data=malfunction_generator,
                obs_builder_object=self._observation.builder(),
                remove_agents_at_target=False,
                random_seed=self._config['seed']
            )

            env.reset()
        except ValueError as e:
            logging.error("=" * 50)
            logging.error(f"Error while creating env: {e}")
            logging.error("=" * 50)

        return env

    def step(self, action_list):

        action_dict = {}
        for i, action in enumerate(action_list):
            action_dict[i] = action

        step_r = self._env.step(action_dict)

        if not self._global_obs:
            return StepOutput(
                obs=[step for step in step_r.obs.values()],
                reward=np.sum([r for r in step_r.reward.values()]),
                done=all(step_r.done.values()),
                info=step_r.info[0]
            )
        else:
            return step_r

    def reset(self):
        if not self._global_obs:
            foo = self._env.reset()

            # print("="*50)
            # print(foo)
            # print("="*50)

            return [step for step in foo.values()]
            # return foo
        else:
            obs = self._env.reset()
            return obs

    @property
    def observation_space(self) -> gym.spaces.Space:
        observation_space = self._env.observation_space

        if not self._global_obs:
            if isinstance(observation_space, gym.spaces.Box):
                return gym.spaces.Box(low=-np.inf, high=np.inf,
                                      shape=(self._config['number_of_agents'], *observation_space.shape,))
            elif isinstance(observation_space, gym.spaces.Tuple):
                spaces = observation_space.spaces * self._config['number_of_agents']
                return gym.spaces.Tuple(spaces)
            else:
                raise ValueError("Unhandled space:", observation_space.__class__)
        else:
            return observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.MultiDiscrete([5] * self._config['number_of_agents'])
