#!/bin/bash

abort() {
  if [ $1 -ne 0 ]; then
    echo "Error $1, aborting."
    exit $1
  fi
}

SCENARIO='{
  "title": "Test",
  "seed": 0,
  "sample_size": 10,
  "sample_noise": true,
  "t_pre": [2010, 3, 25],
  "t_start": [2010, 4, 1],
  "t_end": [2010, 4, 4],
  "device_templates": [
    ["Vaillant EcoPower 1.0", 1],
    ["Vaillant EcoPower 3.0", 1],
    ["Vaillant EcoPower 4.7", 1],
    ["Vaillant EcoPower 20.0", 1],
    ["Stiebel Eltron WPF 5", 1],
    ["Stiebel Eltron WPF 7", 1],
    ["Stiebel Eltron WPF 10", 1],
    ["Stiebel Eltron WPF 13", 1],
    ["Weishaupt WWP S 24", 1],
    ["Weishaupt WWP S 30", 1],
    ["Weishaupt WWP S 37", 1],
    ["RedoxFlow 100 kWh", 0]
  ]
}'

SC_FILE=$(python prepare.py "$SCENARIO")
abort $?
python simulator.py "$SC_FILE"
abort $?
echo "Simulation done, see $(dirname $SC_FILE)"

python plot.py "$SC_FILE"
abort $?
