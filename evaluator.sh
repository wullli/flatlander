#!/usr/bin/envs bash
conda activate flatland-rl
redis-cli -c "flushall";
export AICROWD_TESTS_FOLDER=./scratch/test-envs
# export FLATLAND_OVERALL_TIMEOUT=60
flatland-evaluator --tests ./scratch/test-envs/ --shuffle False
