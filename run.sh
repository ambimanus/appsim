#!/bin/bash

abort() {
  if [ $1 -ne 0 ]; then
    echo "Error, aborting."
    exit 1
  fi
}

# Tradable Contracts:
#     Hour 01: the period between midnight and 1.00 am
#     Hour 02: the period between 1.00 am and 2.00 am
#     [...]
#     Hour 24: the period between 11.00 pm and midnight
# The following Block Orders are pre-defined in the system:
#     Block Baseload covering hours 1 to 24
#     Block Peakload covering hours 9 to 20
#     Block Night covering hours 1 to 6
#     Block Morning covering hours 7 to 10
#     Block High Noon covering hours 11 to 14
#     Block Sun-Peak covering hours 11 to 16
#     Block Afternoon covering hours 15 to 18
#     Block Evening covering hours 19 to 24
#     Block Rush Hour covering hours 17 to 20
#     Block Off-Peak 1 covering hours 1 to 8
#     Block Off-Peak 2 covering hours 21 to 24
#     Block Business covering hours 9 to 16
#     Block Middle-Night covering hours 1 to 4
#     Block Early Morning covering hours 5 to 8
#     Block Late Morning covering hours 9 to 12
#     Block Early Afternoon covering hours 13 to 1

SC_BLOCK='{
  "title": "Test",
  "seed": 0,
  "sample_size": 100,
  "t_pre": [2010, 4, 2],
  "t_start": [2010, 4, 3],
  "t_block_start": [2010, 4, 3, 11],
  "t_block_end": [2010, 4, 3, 14],
  "t_end": [2010, 4, 4],
  "objective": "epex",
  "block": [
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1000],
  "device_templates": [
    ["Vaillant EcoPower 1.0", 10],
    ["Vaillant EcoPower 3.0", 2],
    ["Vaillant EcoPower 4.7", 2],
    ["Vaillant EcoPower 20.0", 0],
    ["Stiebel Eltron WPF 5", 2],
    ["Stiebel Eltron WPF 7", 2],
    ["Stiebel Eltron WPF 10", 2],
    ["Stiebel Eltron WPF 13", 2],
    ["Weishaupt WWP S 24", 0],
    ["Weishaupt WWP S 30", 0],
    ["Weishaupt WWP S 37", 0]
  ],
  "state_files": [],
  "state_files_ctrl": [],
  "sched_file": null
}'

SC_PEAKSHAVING='{
  "title": "Test",
  "seed": 0,
  "sample_size": 100,
  "t_pre": [2010, 4, 2],
  "t_start": [2010, 4, 3],
  "t_block_start": [2010, 4, 3, 10],
  "t_block_end": [2010, 4, 3, 16],
  "t_end": [2010, 4, 4],
  "objective": "peakshaving",
  "device_templates": [
    ["Stiebel Eltron WPF 10", 20]
  ],
  "state_files": [],
  "state_files_ctrl": [],
  "sched_file": null
}'

REV=$(python revision.py)
abort $?

source /home/chh/.virtualenv/appsim/bin/activate
abort $?
SC_FILE=$(python prepare_scenario.py "$SC_BLOCK" "$REV")
abort $?
python run_unctrl.py "$SC_FILE"
abort $?
python run_pre.py "$SC_FILE"
abort $?

echo "--- Running COHDA for [block_start, block_end]"
OLD_PWD=$(pwd)
source /home/chh/.virtualenv/jpype/bin/activate
cd ../crystal-jpype/src
python appsim.py "$SC_FILE"
abort $?
cd $OLD_PWD

source /home/chh/.virtualenv/appsim/bin/activate
abort $?
python run_schedule.py "$SC_FILE"
abort $?
python run_post.py "$SC_FILE"
abort $?

echo "Simulation done, see $(dirname $SC_FILE)"
