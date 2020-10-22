import numpy as np
from gym.wrappers import monitor
from ray.rllib import MultiAgentEnv


class FlatlandBase(MultiAgentEnv):
    reward_range = (-float('inf'), float('inf'))
    spec = None
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10,
        'semantics.autoreset': True
    }

    def __init__(self, actions_are_logits=False, max_steps=None):
        self._actions_are_logits = actions_are_logits
        self._max_steps = max_steps

    def step(self, action_dict):
        if self._actions_are_logits:
            action_dict = {
                k: np.random.choice([0, 1, 2, 3, 4], p=v)
                for k, v in action_dict.items()
            }

        obs, all_rewards, done, info = self._env.step(action_dict)
        if done['__all__']:
            self.close()
        return obs, all_rewards, done, info

    def reset(self, *args, **kwargs):
        if self._env_config.get('render', None):
            env_name = "flatland"
            monitor.FILE_PREFIX = env_name
            folder = self._env_config.get('video_dir', env_name)
            monitor.Monitor._after_step = self._after_step
            self._env = monitor.Monitor(self._env, folder, resume=True)
        obs = self._env.reset(*args, **kwargs)
        if self._max_steps is not None:
            self._env.rail_env._max_episode_steps = self._max_steps
        return obs

    def render(self, mode='human'):
        return self._env.render(self._env_config.get('render'))

    def close(self):
        self._env.close()

    def _after_step(self, observation, reward, done, info):
        if not self.enabled: return done

        if type(done) == dict:
            _done_check = done['__all__']
        else:
            _done_check = done
        if _done_check and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
            self.reset_video_recorder()
            self.episode_id += 1
            self._flush()

        # Record stats - Disabled as it causes error in multi-agent set up
        # self.stats_recorder.after_step(observation, reward, done, info)
        # Record video
        self.video_recorder.capture_frame()

        return done
