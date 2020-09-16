#!/usr/bin/envs python
import glob
import importlib.machinery
import os
import types

import gym
import humps
from ray.rllib import MultiAgentEnv
from ray.rllib.models import Model
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.tune import registry, register_env

"""
Helper functions
"""


def load_class_from_file(_file_path):
    """
    A loader utility, which takes an experiment directory
    path, and loads necessary things into the ModelRegistry.

    This imposes an opinionated directory structure on the
    users, which looks something like :

    - envs/
        - my_env_1.py
        - my_env_2.py
        ....
        - my_env_N.py
    - models/
        - my_model_1.py
        - my_model_2.py
        .....
        - my_model_N.py
    """

    basename = os.path.basename(_file_path)
    filename = basename.replace(".py", "")
    class_name = humps.pascalize(filename)

    # Load the module
    loader = importlib.machinery.SourceFileLoader(filename, _file_path)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    try:
        _class = getattr(mod, class_name)
    except KeyError:
        raise Exception(
            "Looking for a class named {} in the file {}."
            "Did you name the class correctly ?".format(
                filename, class_name
            ))
    return filename, class_name, _class


def load_envs(local_dir="."):
    """
    This function takes a path to a local directory
    and looks for an `envs` folder, and imports
    all the available files in there.

    Determine the filename, env_name and class_name

    # Convention :
        - filename : snake_case
        - classname : PascalCase

        the class implementation, should be an inheritance
        of gym.Env
    """
    load_count = 0
    for _file_path in glob.glob(os.path.join(
            local_dir, "..", "envs", "*.py")):
        if "__init__" in _file_path:
            continue
        env_name, class_name, _class = load_class_from_file(_file_path)
        env = _class

        if not issubclass(env, gym.Env) and not issubclass(env, MultiAgentEnv):
            raise Exception("We expected the class named {} to be "
                            "a subclass of either gym.Env or ray.rllib.MultiAgentEnv. "
                            "Please read more here : https://ray.readthedocs.io/en/latest/rllib-env.html"
                            .format(class_name))

        registry.register_env(env_name, lambda config: env(config))
        load_count += 1

    print("- Successfully loaded", load_count, "environment classes")


def load_models(local_dir="."):
    """
    This function takes a path to a local directory
    and looks for a `models` folder, and imports
    all the available files in there.

    Determine the filename, env_name and class_name

    # Convention :
        - filename : snake_case
        - classname : PascalCase

        the class implementation, should be an inheritance
        of TFModelV2 (ModelV2 : Added PyTorch Model support too,
                      Model: Added Custom loss Model support)
    """
    load_count = 0
    for _file_path in glob.glob(os.path.join(
            local_dir, "..", "models", "*.py")):
        if "__init__" in _file_path:
            continue
        model_name, class_name, _class = load_class_from_file(_file_path)
        custom_model = _class

        if not issubclass(custom_model, ModelV2) and not \
                issubclass(custom_model, TFModelV2) and not \
                issubclass(custom_model, Model):
            raise Exception("We expected the class named {} to be "
                            "a subclass of TFModelV2. "
                            "Please read more here : <insert-link>".format(class_name))

        ModelCatalog.register_custom_model(model_name, custom_model)
        load_count += 1

    print("- Successfully loaded", load_count, "model classes")
