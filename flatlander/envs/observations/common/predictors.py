from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from flatlander.envs.observations.common.malf_shortest_path_predictor import MalfShortestPathPredictorForRailEnv

PREDICTORS = {"custom": MalfShortestPathPredictorForRailEnv,
              "default": ShortestPathPredictorForRailEnv}


def get_predictor(config):
    return PREDICTORS[config.get('predictor', 'default')](config['shortest_path_max_depth'])
