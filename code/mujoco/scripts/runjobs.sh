#!/bin/bash
# usage: runjobs.sh <jobs.txt> <parallel> <done_marker>   each line of jobs.txt is one command without quotes
cd <PROJECT_ROOT>/penspin-mujoco-20261107
xargs -a "$1" -P "$2" -L 1 -I CMD bash -c CMD
touch "$3"
