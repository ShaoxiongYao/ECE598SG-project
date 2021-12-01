#!/bin/bash

echo "Evaluate kitchen environment:" > kitchen_stats.txt
# eval mppi
python kitchen_mppi.py --model_mode PlainNet --model_type q_no >> kitchen_stats.txt
