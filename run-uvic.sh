#!/bin/bash

abort() {
  if [ $1 -ne 0 ]; then
    echo "Error $1, aborting."
    exit $1
  fi
}

SC_SPREAD_SLP='{
  "title": "Test",
  "seed": 0,
  "sample_size": 10,
  "t_pre": [2010, 3, 25],
  "t_start": [2010, 4, 1],
  "t_block_start": [2010, 4, 2],
  "t_block_end": [2010, 4, 3],
  "t_end": [2010, 4, 4],
  "objective": "spreadreduce-slp",
  "device_templates": [
    ["Vaillant EcoPower 1.0", 0],
    ["Vaillant EcoPower 3.0", 0],
    ["Vaillant EcoPower 4.7", 0],
    ["Vaillant EcoPower 20.0", 0],
    ["Stiebel Eltron WPF 5", 1],
    ["Stiebel Eltron WPF 7", 1],
    ["Stiebel Eltron WPF 10", 1],
    ["Stiebel Eltron WPF 13", 0],
    ["Weishaupt WWP S 24", 0],
    ["Weishaupt WWP S 30", 0],
    ["Weishaupt WWP S 37", 0],
    ["RedoxFlow 100 kWh", 0]
  ],
  "state_files": [],
  "state_files_ctrl": [],
  "sched_file": null,
  "svsm": false,
  "slp_file": "/home/chh/data/crystal-chp/slp/2010_slp_profile_eon_mitte_ag/H0 - Haushalt.csv"
}'

REV=$(python revision.py)
abort $?

source /home/chh/.virtualenv/appsim/bin/activate
abort $?
SC_FILE=$(python prepare_scenario.py "$SC_SPREAD_SLP" "$REV")
abort $?
python run_unctrl.py "$SC_FILE"
abort $?
python run_pre.py "$SC_FILE"
abort $?
deactivate

echo "--- Running COHDA for [block_start, block_end]"
OLD_PWD=$(pwd)
cd ../crystal/src
python appsim.py "$SC_FILE"
abort $?
cd $OLD_PWD

source /home/chh/.virtualenv/appsim/bin/activate
abort $?
python run_schedule.py "$SC_FILE"
abort $?
python run_post.py "$SC_FILE"
abort $?
deactivate

echo "Simulation done, see $(dirname $SC_FILE)"

source /home/chh/.virtualenv/appsim/bin/activate
python analyze-headless.py "$SC_FILE"
abort $?
deactivate
