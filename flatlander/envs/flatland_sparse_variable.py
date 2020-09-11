import logging

from flatland.envs.malfunction_generators import NoMalfunctionGen, \
    ParamMalfunctionGen
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatlander.envs.flatland_sparse import FlatlandSparse
from flatlander.envs.utils.env_config_generator import get_round_2_env
from flatlander.envs.utils.gym_env import FlatlandGymEnv
from flatlander.envs.utils.gym_env_wrappers import FlatlandRenderWrapper as RailEnv


class FlatlandSparseVariable(FlatlandSparse):

    def __init__(self, env_config) -> None:
        super().__init__(env_config)

    def get_env(self):
        return FlatlandGymEnv(
            rail_env=self._launch(),
            observation_space=self._observation.observation_space(),
            render=self._env_config.get('render'),
            regenerate_rail_on_reset=self._config['regenerate_rail_on_reset'],
            regenerate_schedule_on_reset=self._config['regenerate_schedule_on_reset']
        )

    def _launch(self):
        print("NEW ENV LAUNCHED")
        n_agents, n_cities, dim = get_round_2_env()

        rail_generator = sparse_rail_generator(
            seed=self._config['seed'],
            max_num_cities=n_cities,
            grid_mode=self._config['grid_mode'],
            max_rails_between_cities=self._config['max_rails_between_cities'],
            max_rails_in_city=self._config['max_rails_in_city']
        )

        malfunction_generator = NoMalfunctionGen()
        if {'malfunction_rate', 'malfunction_min_duration', 'malfunction_max_duration'} <= self._config.keys():
            stochastic_data = {
                'malfunction_rate': self._config['malfunction_rate'],
                'min_duration': self._config['malfunction_min_duration'],
                'max_duration': self._config['malfunction_max_duration']
            }
            malfunction_generator = ParamMalfunctionGen(stochastic_data)

        speed_ratio_map = None
        if 'speed_ratio_map' in self._config:
            speed_ratio_map = {
                float(k): float(v) for k, v in self._config['speed_ratio_map'].items()
            }
        schedule_generator = sparse_schedule_generator(speed_ratio_map)

        env = None
        try:
            env = RailEnv(
                width=dim,
                height=dim,
                rail_generator=rail_generator,
                schedule_generator=schedule_generator,
                number_of_agents=n_agents,
                malfunction_generator=malfunction_generator,
                obs_builder_object=self._observation.builder(),
                remove_agents_at_target=False,
                random_seed=self._config['seed'],
                use_renderer=self._env_config.get('render')
            )

            env.reset()
        except ValueError as e:
            logging.error("=" * 50)
            logging.error(f"Error while creating env: {e}")
            logging.error("=" * 50)

        return env

    def reset(self, *args, **kwargs):
        self._env = self.get_env()
        self._env.reset(*args, **kwargs)
        return self._env.reset(*args, **kwargs)
