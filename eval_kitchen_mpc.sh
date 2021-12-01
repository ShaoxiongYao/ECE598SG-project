#!/bin/bash

echo "Evaluate kitchen environment:" > kitchen_stats.txt
echo "evaluate mppi" >> kitchen_stats.txt
echo "MLP q_no" >> kitchen_stats.txt 
python kitchen_mppi.py --model_mode PlainNet --model_type q_no >> kitchen_stats.txt
echo "MLP q_true" >> kitchen_stats.txt
python kitchen_mppi.py --model_mode PlainNet --model_type q_true >> kitchen_stats.txt
echo "MLP q_only" >> kitchen_stats.txt
python kitchen_mppi.py --model_mode PlainNet --model_type q_only >> kitchen_stats.txt
echo "MLP qhat_no" >> kitchen_stats.txt 
python kitchen_mppi.py --model_mode PlainNet --model_type qhat_no >> kitchen_stats.txt
echo "MLP qhat_true" >> kitchen_stats.txt
python kitchen_mppi.py --model_mode PlainNet --model_type qhat_true >> kitchen_stats.txt
echo "MLP qhat_only" >> kitchen_stats.txt
python kitchen_mppi.py --model_mode PlainNet --model_type qhat_only >> kitchen_stats.txt
echo "LSTM q_no" >> kitchen_stats.txt
python kitchen_mppi.py --model_mode PlainNet --model_type q_no >> kitchen_stats.txt
echo "LSTM qhat_no" >> kitchen_stats.txt
python kitchen_mppi.py --model_mode PlainNet --model_type qhat_no >> kitchen_stats.txt

echo "evaluate cem" >> kitchen_stats.txt
echo "MLP q_no" >> kitchen_stats.txt 
python kitchen_cem.py --model_mode PlainNet --model_type q_no >> kitchen_stats.txt
echo "MLP q_true" >> kitchen_stats.txt
python kitchen_cem.py --model_mode PlainNet --model_type q_true >> kitchen_stats.txt
echo "MLP q_only" >> kitchen_stats.txt
python kitchen_cem.py --model_mode PlainNet --model_type q_only >> kitchen_stats.txt
echo "MLP qhat_no" >> kitchen_stats.txt 
python kitchen_cem.py --model_mode PlainNet --model_type qhat_no >> kitchen_stats.txt
echo "MLP qhat_true" >> kitchen_stats.txt
python kitchen_cem.py --model_mode PlainNet --model_type qhat_true >> kitchen_stats.txt
echo "MLP qhat_only" >> kitchen_stats.txt
python kitchen_cem.py --model_mode PlainNet --model_type qhat_only >> kitchen_stats.txt
echo "LSTM q_no" >> kitchen_stats.txt
python kitchen_cem.py --model_mode PlainNet --model_type q_no >> kitchen_stats.txt
echo "LSTM qhat_no" >> kitchen_stats.txt
python kitchen_cem.py --model_mode PlainNet --model_type qhat_no >> kitchen_stats.txt
