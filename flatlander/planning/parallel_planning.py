from collections import Callable
from multiprocessing import cpu_count
import concurrent.futures
import numpy as np
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv

from flatlander.submission.submissions import CURRENT_ENV_PATH


def parallel_plan(planning_function: Callable, env: RailEnv, **kwargs):
    RailEnvPersister.save(env, CURRENT_ENV_PATH)
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as worker_pool:
        best_action_results = []
        best_pc_results = []
        best_return_results = []

        futures = [worker_pool.submit(planning_function, **kwargs) for _ in range(cpu_count())]

        for future in concurrent.futures.as_completed(futures):
            best_actions, best_pc, best_return = future.result()
            if best_pc == 1.0:
                print(f'MAX PC: {best_pc}, MAX RETURN: {best_return}\n')
                for f in futures:
                    f.cancel()
                return best_actions

            best_action_results.append(best_actions)
            best_pc_results.append(best_pc)
            best_return_results.append(best_return)

        for f in futures:
            f.cancel()

        return best_action_results[int(np.argmax(best_pc))]
