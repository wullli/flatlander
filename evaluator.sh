#!/usr/bin/envs bash
conda activate flatland-rl
redis-cli -c "flushall";
export AICROWD_TESTS_FOLDER=./scratch/test-envs
flatland-evaluator --tests ./scratch/test-envs/
