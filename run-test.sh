#!/bin/bash

abort() {
  if [ $1 -ne 0 ]; then
    echo "Error $1, aborting."
    exit $1
  fi
}

SC='{
  "title": "Peakload-20-CHP-large-flex",
  "seed": 0,
  "sample_size": 200,
  "t_pre": [2010, 3, 25],
  "t_start": [2010, 4, 1],
  "t_block_start": [2010, 4, 2, 9],
  "t_block_end": [2010, 4, 2, 20],
  "t_end": [2010, 4, 4],
  "objective": "epex",
  "block": [100000],
  "device_templates": [
    ["Vaillant EcoPower 1.0", 0],
    ["Vaillant EcoPower 3.0", 0],
    ["Vaillant EcoPower 4.7", 15],
    ["Vaillant EcoPower 20.0", 5],
    ["Stiebel Eltron WPF 5", 0],
    ["Stiebel Eltron WPF 7", 0],
    ["Stiebel Eltron WPF 10", 0],
    ["Stiebel Eltron WPF 13", 0],
    ["Weishaupt WWP S 24", 0],
    ["Weishaupt WWP S 30", 0],
    ["Weishaupt WWP S 37", 0],
    ["RedoxFlow 100 kWh", 0]
  ],
  "svsm": false
}'

REV=$(python revision.py)
abort $?

source /home/chh/.virtualenv/appsim/bin/activate
abort $?
SC_FILE=$(python prepare_scenario.py "$SC" "$REV")
abort $?
python run_unctrl.py "$SC_FILE"
abort $?
python run_pre.py "$SC_FILE"
abort $?

echo "--- Running COHDA for [block_start, block_end]"
OLD_PWD=$(pwd)
deactivate
cd ../cohda-fast/src
python appsim.py "$SC_FILE"
abort $?
cd $OLD_PWD

source /home/chh/.virtualenv/appsim/bin/activate
abort $?
python run_state.py "$SC_FILE"
abort $?
python run_post.py "$SC_FILE"
abort $?
python run_cleanup.py "$SC_FILE"
abort $?

echo "Simulation done, see $(dirname $SC_FILE)"

python analyze.py "$SC_FILE"
abort $?
