import os

import yaml
import inspect
src_file_path = inspect.getfile(lambda: None)

GENERATOR_CONFIG_REGISTRY = {}
EVAL_CONFIG_REGISTRY = {}


def get_eval_config(name: str = None):
    return EVAL_CONFIG_REGISTRY[name]


def get_generator_config(name: str):
    return GENERATOR_CONFIG_REGISTRY[name]


config_folder = os.path.join(os.path.dirname(src_file_path), "..", "resources", "generator_configs")
load_count = 0
for file in os.listdir(config_folder):
    if file.endswith('.yaml') and not file.startswith('_'):
        basename = os.path.basename(file)
        filename = basename.replace(".yaml", "")

        with open(os.path.join(config_folder, file)) as f:
            GENERATOR_CONFIG_REGISTRY[filename] = yaml.safe_load(f)

        load_count += 1

print("- Successfully loaded", load_count, "generator configs")


load_count = 0
eval_config_folder = os.path.join(os.path.dirname(src_file_path), "..", "resources", "eval_configs")
for file in os.listdir(eval_config_folder):
    if file.endswith('.yaml') and not file.startswith('_'):
        basename = os.path.basename(file)
        filename = basename.replace(".yaml", "")

        with open(os.path.join(eval_config_folder, file)) as f:
            EVAL_CONFIG_REGISTRY[filename] = yaml.safe_load(f)

        load_count += 1

print("- Successfully loaded", load_count, "evaluation configs")

