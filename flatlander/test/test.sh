#!/bin/bash

# broken!
#echo "===================="
#echo "MARWIL"
#echo "===================="
#time python ./trainImitate.py -f experiments/tests/MARWIL.yaml

eval "$(conda shell.bash hook)"
conda activate flatland-rl
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "===================="
echo "TEST GLOBAL OBS"
echo "===================="
time python3 ../scripts/experiment.py -f $DIR/../resources/experiments/tests/global_obs_ppo.yaml

echo "===================="
echo "TEST GLOBAL DENSITY OBS"
echo "===================="
time python3 ../scripts/experiment.py -f $DIR/../resources/experiments/tests/global_density_obs_apex.yaml

echo "===================="
echo "TEST LOCAL CONFLICT OBS"
echo "===================="
time python3 ../scripts/experiment.py -f $DIR/../resources/experiments/tests/local_conflict_obs_apex.yaml

echo "===================="
echo "TREE OBS"
echo "===================="
time python3 ../scripts/experiment.py -f $DIR/../resources/experiments/tests/tree_obs_apex.yaml

echo "===================="
echo "TEST COMBINED OBS (TREE + LOCAL CONFLICT)"
echo "===================="
time python3 ../scripts/experiment.py -f $DIR/../resources/experiments/tests/combined_obs_apex.yaml
