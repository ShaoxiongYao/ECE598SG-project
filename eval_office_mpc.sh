#!/bin/bash

echo "Evaluate office environment:" > office_stats.txt
echo "evaluate mppi" >> office_stats.txt
echo "MLP q_no" >> office_stats.txt
python office_mppi.py --model_mode PlainNet --model_type q_no >> office_stats.txt

