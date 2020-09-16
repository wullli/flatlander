import fnmatch
import multiprocessing
import numbers
import os
import shutil

import wandb
from ray import tune
from ray.tune.utils import flatten_dict


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


ignored_keys = ["callbacks", "policies", "policy_mapping_fn"]


class WandbLogger(tune.logger.Logger):
    """Pass WandbLogger to the loggers argument of tune.run

       tune.run("PG", loggers=[WandbLogger], config={
           "monitor": True, "env_config": {
               "wandb": {"project": "my-project-name"}}})
    """

    def _init(self):
        self._config = None
        self.metrics_queue_dict = {}
        self.reset_state()
        if self.config.get("env_config", {}).get("render", None):
            '''
            Cleans or Resume folder based on resume
            To avoid overwriting current video folder
            Add `resume: True` under `env_config`
            '''
            resume = self.config.get("env_config", {}).get("resume", False)
            self.clear_folders(resume)

    def clear_folders(self, resume):
        env_name = "flatland"
        self._save_folder = self.config.get("env_config", {}).get("video_dir", env_name)
        if not resume:
            if os.path.exists(self._save_folder):
                try:
                    shutil.rmtree(self._save_folder)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

    def reset_state(self):
        # Holds list of uploaded/moved files so that we dont upload them again
        self._upload_files = {}
        # Holds information of envs state and put them in an unique file name
        # and maps it to the original video file
        self._file_map = {}
        self._save_folder = None

    def on_result(self, result):
        experiment_tag = result.get('experiment_tag', 'no_experiment_tag')
        experiment_id = result.get('experiment_id', 'no_experiment_id')
        if experiment_tag not in self.metrics_queue_dict:
            print("=" * 50)
            print("Setting up new w&b logger")
            print("Experiment tag:", experiment_tag)
            print("Experiment id:", experiment_id)
            config = result.get("config")
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=wandb_process, args=(queue, config,))
            p.start()
            self.metrics_queue_dict[experiment_tag] = queue
            print("=" * 50)

        queue = self.metrics_queue_dict[experiment_tag]

        tmp = result.copy()
        for k in ["done", "config", "pid", "timestamp"]:
            if k in tmp:
                del tmp[k]

        metrics = {}
        for key, value in flatten_dict(tmp, delimiter="/").items():
            if not isinstance(value, numbers.Number):
                continue
            metrics[key] = value

        if self.config.get("env_config", {}).get("render", None):
            # uploading relevant videos to wandb
            # we do this step to ensure any config changes done during training is incorporated
            # resume is set to True as we don't want to delete any older videos
            self.clear_folders(resume=True)

            if self._save_folder:
                metrics = self.update_video_metrics(result, metrics)

        queue.put(metrics)

    def update_video_metrics(self, result, metrics):
        iterations = result['training_iteration']
        steps = result['timesteps_total']
        perc_comp_mean = result['custom_metrics'].get('percentage_complete_mean', 0) * 100
        # We ignore *1.mp4 videos which just has the last frame
        _video_file = f'*0.mp4'
        _found_videos = find(_video_file, self._save_folder)
        # _found_videos = list(set(_found_videos) - set(self._upload_files))
        # Sort by create time for uploading to wandb
        _found_videos.sort(key=os.path.getctime)
        for _found_video in _found_videos:
            _splits = _found_video.split(os.sep)
            _check_file = os.stat(_found_video)
            _video_file = _splits[-1]
            _file_split = _video_file.split('.')
            _video_file_name = _file_split[0] + "-" + str(iterations)
            _original_name = ".".join(_file_split[2:-1])
            _video_file_name = ".".join([str(_video_file_name), str(steps), str(int(perc_comp_mean)), _original_name,
                                         str(_check_file.st_ctime)])
            _key = _found_video  # Use the video file path as key to identify the video_name
            if not self._file_map.get(_key):
                # Allocate steps, iteration, completion rate to the earliest case
                # when the video file was first created. Discard recent file names
                # TODO: Cannot match exact envs details on which video was created
                # and hence defaulting to the envs details from when the video was first created
                # To help identify we must record the video file with the envs iteration or/and steps etc.
                # Using the envs details when the video was created may be useful when recording video during evaluation
                # where we are more interested in the current training state
                self._file_map[_key] = _video_file_name

            # We only move videos that have been flushed out.
            # This is done by checking against a threshold size of 1000 bytes
            # Also check if file has changed from last time
            if _check_file.st_size > 1000 and _check_file.st_ctime > self._upload_files.get(_found_video, True):
                _video_file_name = self._file_map.get(_key, "Unknown")
                # wandb.log({_video_file_name: wandb.Video(_found_video, format="mp4")})
                metrics[_video_file_name] = wandb.Video(_found_video, format="mp4")

                self._upload_files[_found_video] = _check_file.st_size

        return metrics

    def close(self):
        # kills logger processes
        for queue in self.metrics_queue_dict.values():
            metrics = {"KILL": True}
            queue.put(metrics)
        wandb.join()

        all_uploaded_videos = self._upload_files.keys()

        for _found_video in all_uploaded_videos:
            try:
                # Copy upload videos and their meta data once done to the logdir
                src = _found_video
                _video_file = _found_video.split(os.sep)[-1]
                dst = os.path.join(self.logdir, _video_file)
                shutil.copy2(src, dst)
                shutil.copy2(src.replace("mp4", "meta.json"), dst.replace("mp4", "meta.json"))
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
        self.reset_state()


# each logger has to run in a separate process
def wandb_process(queue, config) -> None:
    run = wandb.init(reinit=True, **config.get("env_config", {}).get("wandb", {}))

    if config:
        for k in config.keys():
            if k not in ignored_keys:
                if wandb.config.get(k) is None:
                    wandb.config[k] = config[k]

        if 'yaml_config' in config['env_config']:
            yaml_config = config['env_config']['yaml_config']
            print("Saving full experiment config:", yaml_config)
            wandb.save(yaml_config)

    while True:
        metrics = queue.get()

        if "KILL" in metrics:
            break

        run.log(metrics)
