#!/bin/bash

echo "Evaluate office environment:" > office_stats.txt
echo "evaluate mppi" >> office_stats.txt
for MODEL_MODE in PlainNet LSTM
do
	for MODEL_TYPE in q_no q_only q_true qhat_no qhat_true 
    do
    echo "Model mode: $MODEL_MODE, model type: $MODEL_TYPE" >> office_stats.txt 
    python office_mppi.py --model_mode $MODEL_MODE --model_type $MODEL_TYPE >> office_stats.txt
    done
done

echo "evaluate cem" >> office_stats.txt
for MODEL_MODE in PlainNet LSTM
do
	for MODEL_TYPE in q_no q_only q_true qhat_no qhat_true 
    do
    echo "Model mode: $MODEL_MODE, model type: $MODEL_TYPE" >> office_stats.txt 
    python office_cem.py --model_mode $MODEL_MODE --model_type $MODEL_TYPE >> office_stats.txt
    done
done
