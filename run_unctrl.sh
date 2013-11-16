#!/bin/bash

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

source /home/chh/.virtualenv/appsim/bin/activate
python run_unctrl.py "$SC" "$REV"
