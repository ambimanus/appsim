#!/bin/bash

abort() {
  if [ $1 -eq 1 ]; then
    echo "Error, aborting."
    exit 1
  fi
}

SC='{
  "title": "Test",
  "seed": 0,
  "sample_size": 100,
  "t_pre": [2010, 4, 2, 0, 0, 0, 4, 92, -1],
  "t_start": [2010, 4, 3, 0, 0, 0, 5, 93, -1],
  "t_block_start": [2010, 4, 3, 11, 0, 0, 5, 93, -1],
  "t_block_end": [2010, 4, 3, 14, 0, 0, 5, 93, -1],
  "t_end": [2010, 4, 4, 0, 0, 0, 6, 94, -1],
  "device_templates": [
    ["Vaillant EcoPower 1.0", 1],
    ["Stiebel Eltron WPF 10", 1]
  ],
  "state_files": [],
  "sched_file": null
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

echo "Simulation done, see $(dirname $SC_FILE)"
