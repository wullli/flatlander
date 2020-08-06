#!/usr/bin/env bash

# broken!
#echo "===================="
#echo "MARWIL"
#echo "===================="
#time python ./trainImitate.py -f experiments/tests/MARWIL.yaml

echo "===================="
echo "GLOBAL OBS"
echo "===================="
time python ./train.py -f experiments/tests/global_obs_ppo.yaml

echo "===================="
echo "GLOBAL DENSITY OBS"
echo "===================="
time python ./train.py -f experiments/tests/global_density_obs_apex.yaml

echo "===================="
echo "LOCAL CONFLICT OBS"
echo "===================="
time python ./train.py -f experiments/tests/local_conflict_obs_apex.yaml

echo "===================="
echo "TREE OBS"
echo "===================="
time python ./train.py -f experiments/tests/tree_obs_apex.yaml

echo "===================="
echo "COMBINED OBS (TREE + LOCAL CONFLICT)"
echo "===================="
time python ./train.py -f experiments/tests/combined_obs_apex.yaml
