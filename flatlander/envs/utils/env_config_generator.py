import os

import numpy as np
import pandas as pd

envs = pd.read_csv(os.path.join(os.path.dirname(__file__), "round_2_envs.csv"))
envs = envs[["n_agents", "x_dim", "n_cities"]].iloc[:20]
env_vals = envs.values
indexes = np.array(list(range(len(env_vals))))


def get_round_2_env():
    row = np.random.choice(indexes)
    n_agents = env_vals[row, 0]
    n_cities = env_vals[row, 1]
    dim = env_vals[row, 2]
    return n_agents, n_cities, dim
