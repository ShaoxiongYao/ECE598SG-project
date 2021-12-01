#!/bin/bash

echo "Evaluate kitchen environment:" > kitchen_stats.txt
echo "evaluate mppi" >> kitchen_stats.txt
for MODEL_MODE in PlainNet LSTM
do
	for MODEL_TYPE in q_no q_only q_true qhat_no qhat_true 
    do
    echo "Model mode: $MODEL_MODE, model type: $MODEL_TYPE" >> kitchen_stats.txt 
    python kitchen_mppi.py --model_mode $MODEL_MODE --model_type $MODEL_TYPE >> kitchen_stats.txt
    done
done

echo "evaluate cem" >> kitchen_stats.txt
for MODEL_MODE in PlainNet LSTM
do
	for MODEL_TYPE in q_no q_only q_true qhat_no qhat_true 
    do
    echo "Model mode: $MODEL_MODE, model type: $MODEL_TYPE" >> kitchen_stats.txt 
    python kitchen_cem.py --model_mode $MODEL_MODE --model_type $MODEL_TYPE >> kitchen_stats.txt
    done
done

