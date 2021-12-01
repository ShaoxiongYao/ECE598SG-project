#!/bin/bash

echo "Evaluate office environment:" > office_stats.txt
echo "evaluate mppi" >> office_stats.txt
echo "MLP q_no" >> office_stats.txt
python office_mppi.py --model_mode PlainNet --model_type q_no >> office_stats.txt
echo "MLP q_true" >> office_stats.txt
python office_mppi.py --model_mode PlainNet --model_type q_true >> office_stats.txt
echo "MLP q_only" >> office_stats.txt
python office_mppi.py --model_mode PlainNet --model_type q_only >> office_stats.txt
echo "MLP qhat_no" >> office_stats.txt 
python office_mppi.py --model_mode PlainNet --model_type qhat_no >> office_stats.txt
echo "MLP qhat_true" >> office_stats.txt
python office_mppi.py --model_mode PlainNet --model_type qhat_true >> office_stats.txt
echo "MLP qhat_only" >> office_stats.txt
python office_mppi.py --model_mode PlainNet --model_type qhat_only >> office_stats.txt
echo "LSTM q_no" >> office_stats.txt
python office_mppi.py --model_mode PlainNet --model_type q_no >> office_stats.txt
echo "LSTM qhat_no" >> office_stats.txt
python office_mppi.py --model_mode PlainNet --model_type qhat_no >> office_stats.txt

echo "evaluate cem" >> office_stats.txt
echo "MLP q_no" >> office_stats.txt 
python office_cem.py --model_mode PlainNet --model_type q_no >> office_stats.txt
echo "MLP q_true" >> office_stats.txt
python office_cem.py --model_mode PlainNet --model_type q_true >> office_stats.txt
echo "MLP q_only" >> office_stats.txt
python office_cem.py --model_mode PlainNet --model_type q_only >> office_stats.txt
echo "MLP qhat_no" >> office_stats.txt 
python office_cem.py --model_mode PlainNet --model_type qhat_no >> office_stats.txt
echo "MLP qhat_true" >> office_stats.txt
python office_cem.py --model_mode PlainNet --model_type qhat_true >> office_stats.txt
echo "MLP qhat_only" >> office_stats.txt
python office_cem.py --model_mode PlainNet --model_type qhat_only >> office_stats.txt
echo "LSTM q_no" >> office_stats.txt
python office_cem.py --model_mode PlainNet --model_type q_no >> office_stats.txt
echo "LSTM qhat_no" >> office_stats.txt
python office_cem.py --model_mode PlainNet --model_type qhat_no >> office_stats.txt
