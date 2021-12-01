
kitchen_mppi_stats = """Model mode: PlainNet, model type: q_no
episode reward, mean: 0.07 std: 0.25514701644346144
steps for first subtask, mean: 14.285714285714286 std: 7.610411772495545

Model mode: PlainNet, model type: q_only
episode reward, mean: 0.02 std: 0.13999999999999999
steps for first subtask, mean: 12.0 std: 1.0

Model mode: PlainNet, model type: q_true
episode reward, mean: 0.08 std: 0.2712931993250107
steps for first subtask, mean: 9.875 std: 5.688530126491377

Model mode: PlainNet, model type: qhat_no
episode reward, mean: 0.06 std: 0.23748684174075835
steps for first subtask, mean: 9.0 std: 4.163331998932265

Model mode: PlainNet, model type: qhat_true
episode reward, mean: 0.01 std: 0.09949874371066199
steps for first subtask, mean: 7.0 std: 0.0

Model mode: LSTM, model type: q_no
episode reward, mean: 0.01 std: 0.09949874371066199
steps for first subtask, mean: 18.0 std: 0.0

Model mode: LSTM, model type: q_only
episode reward, mean: 0.02 std: 0.13999999999999999
steps for first subtask, mean: 6.5 std: 0.5

Model mode: LSTM, model type: q_true
episode reward, mean: 0.04 std: 0.2416609194718914
steps for first subtask, mean: 7.0 std: 2.160246899469287

Model mode: LSTM, model type: qhat_no
episode reward, mean: 0.06 std: 0.3104834939252005
steps for first subtask, mean: 10.75 std: 5.402545696243577

Model mode: LSTM, model type: qhat_true
episode reward, mean: 0.04 std: 0.19595917942265426
steps for first subtask, mean: 14.5 std: 9.86154146165801"""

office_mppi_stats="""Model mode: PlainNet, model type: q_no
Final success rate: 0.74
skill steps: [3, 4, 6, 8, 7, 3, 1, 8, 1, 4, 0, 6, 4, 5, 6, 1, 2, 3, 6, 2, 3, 7, 5, 9, 6, 4, 9, 0, 9, 4, 3, 2, 3, 6, 3, 6, 6, 9, 2, 7, 9, 4, 1, 4, 4, 0, 9, 9, 7, 7, 9, 9, 1, 3, 9, 9, 3, 0, 4, 0, 1, 2, 3, 2, 9, 0, 2, 3, 1, 6, 6, 5, 6, 7]

Model mode: PlainNet, model type: q_only
Final success rate: 0.78
skill steps: [2, 3, 6, 8, 6, 6, 6, 5, 9, 6, 0, 7, 5, 1, 3, 3, 0, 4, 6, 6, 3, 9, 2, 2, 7, 7, 0, 2, 8, 7, 7, 3, 1, 9, 0, 1, 2, 6, 8, 8, 8, 8, 9, 0, 9, 8, 6, 0, 0, 3, 9, 6, 2, 8, 9, 4, 8, 1, 8, 4, 5, 8, 0, 3, 9, 6, 1, 8, 9, 9, 1, 8, 7, 2, 2, 9, 3, 5]

Model mode: PlainNet, model type: q_true
Final success rate: 0.74
skill steps: [5, 4, 2, 0, 6, 7, 6, 2, 6, 2, 3, 0, 4, 4, 2, 9, 3, 9, 5, 3, 9, 3, 4, 6, 3, 4, 7, 2, 9, 1, 3, 0, 3, 1, 8, 5, 7, 9, 8, 0, 1, 2, 4, 8, 0, 7, 0, 6, 0, 8, 1, 4, 5, 0, 8, 8, 6, 0, 4, 0, 7, 1, 8, 8, 7, 1, 3, 5, 8, 8, 7, 5, 6, 7]

Model mode: PlainNet, model type: qhat_true
Final success rate: 0.76
skill steps: [9, 6, 2, 6, 5, 9, 4, 2, 3, 3, 8, 0, 1, 6, 9, 8, 3, 7, 4, 9, 1, 7, 6, 6, 3, 0, 6, 0, 0, 6, 0, 8, 7, 8, 7, 3, 9, 7, 1, 6, 9, 7, 1, 7, 6, 7, 7, 9, 0, 2, 5, 4, 8, 0, 7, 2, 1, 0, 2, 9, 1, 8, 2, 4, 6, 2, 2, 6, 9, 0, 8, 8, 9, 4, 7, 8]

Model mode: LSTM, model type: q_no
Final success rate: 0.74
skill steps: [8, 2, 3, 3, 5, 1, 8, 1, 1, 8, 2, 3, 3, 1, 7, 9, 2, 2, 4, 1, 6, 7, 7, 9, 1, 5, 3, 3, 6, 1, 7, 7, 7, 0, 7, 1, 3, 4, 9, 0, 6, 0, 0, 6, 9, 4, 0, 8, 7, 7, 6, 4, 1, 0, 1, 7, 5, 4, 5, 3, 5, 8, 1, 0, 1, 0, 0, 1, 1, 0, 2, 8, 4, 2]

Model mode: LSTM, model type: q_only
Final success rate: 0.68
skill steps: [2, 4, 2, 6, 9, 9, 0, 3, 7, 1, 5, 2, 3, 1, 1, 4, 7, 3, 2, 0, 6, 6, 1, 3, 9, 3, 9, 4, 7, 4, 0, 1, 5, 3, 4, 2, 5, 3, 9, 2, 5, 6, 7, 8, 6, 5, 2, 8, 9, 2, 6, 7, 6, 6, 3, 8, 9, 0, 8, 7, 7, 1, 6, 3, 4, 7, 9, 0]

Model mode: LSTM, model type: q_true
Final success rate: 0.7
skill steps: [7, 7, 7, 5, 9, 1, 4, 3, 2, 6, 5, 5, 5, 6, 6, 4, 0, 7, 2, 6, 5, 1, 4, 0, 9, 4, 6, 8, 7, 0, 1, 4, 7, 3, 1, 8, 1, 7, 7, 1, 7, 6, 8, 8, 5, 6, 7, 4, 7, 4, 6, 2, 8, 6, 7, 7, 6, 1, 6, 9, 1, 3, 1, 3, 7, 7, 6, 0, 1, 6]

Model mode: LSTM, model type: qhat_no
Final success rate: 0.75
skill steps: [9, 3, 4, 6, 4, 1, 9, 0, 5, 8, 5, 7, 2, 3, 0, 1, 5, 9, 6, 0, 6, 2, 6, 7, 1, 9, 8, 4, 4, 7, 7, 4, 4, 1, 2, 8, 4, 8, 2, 8, 0, 0, 5, 4, 0, 8, 7, 0, 3, 9, 3, 2, 5, 9, 4, 6, 5, 9, 6, 3, 3, 7, 6, 0, 2, 2, 6, 5, 3, 2, 7, 1, 2, 7, 2]

Model mode: LSTM, model type: qhat_true
Final success rate: 0.66
skill steps: [7, 6, 0, 7, 7, 1, 6, 8, 1, 9, 4, 0, 7, 8, 1, 2, 1, 4, 6, 4, 2, 6, 7, 7, 8, 8, 6, 8, 1, 1, 9, 1, 4, 5, 3, 2, 7, 6, 7, 1, 4, 3, 1, 5, 9, 0, 1, 6, 3, 6, 7, 6, 6, 6, 4, 1, 4, 0, 6, 4, 3, 1, 0, 4, 9, 7]"""

office_cem_stats="""Model mode: PlainNet, model type: q_no
Final success rate: 0.34
skill steps: [4, 7, 9, 6, 9, 1, 7, 3, 3, 5, 9, 7, 6, 1, 7, 2, 0, 9, 1, 7, 3, 4, 5, 8, 5, 4, 4, 5, 7, 1, 0, 2, 6, 7]

Model mode: PlainNet, model type: q_only
Final success rate: 0.69
skill steps: [8, 8, 5, 2, 8, 7, 2, 5, 2, 7, 7, 6, 6, 2, 9, 1, 2, 2, 2, 8, 8, 3, 6, 0, 9, 3, 9, 1, 5, 8, 4, 1, 7, 7, 0, 2, 9, 6, 8, 7, 6, 9, 6, 9, 0, 4, 3, 5, 8, 6, 5, 9, 6, 7, 7, 9, 7, 6, 6, 7, 0, 8, 6, 6, 9, 7, 5, 6, 2]

Model mode: PlainNet, model type: q_true
Final success rate: 0.39
skill steps: [9, 5, 8, 7, 8, 8, 6, 9, 5, 3, 6, 3, 0, 7, 2, 6, 5, 1, 9, 4, 1, 8, 4, 8, 0, 9, 6, 8, 2, 4, 6, 6, 8, 7, 6, 8, 4, 8, 7]

Model mode: PlainNet, model type: qhat_no
Final success rate: 0.64
skill steps: [9, 1, 6, 0, 6, 1, 6, 1, 8, 6, 8, 1, 2, 1, 7, 3, 5, 8, 1, 3, 7, 1, 7, 9, 7, 2, 9, 7, 8, 6, 0, 4, 6, 9, 7, 0, 0, 6, 1, 1, 4, 3, 3, 1, 1, 2, 6, 6, 7, 9, 8, 7, 1, 3, 8, 1, 9, 1, 0, 3, 1, 5, 3, 7]

Model mode: PlainNet, model type: qhat_true
Final success rate: 0.49
skill steps: [8, 7, 8, 7, 7, 1, 0, 4, 7, 0, 5, 8, 6, 8, 8, 8, 4, 7, 1, 7, 7, 0, 7, 9, 3, 3, 2, 8, 1, 5, 8, 2, 5, 4, 0, 8, 1, 6, 9, 0, 0, 6, 1, 5, 3, 9, 3, 3, 6]

Model mode: LSTM, model type: q_no
Final success rate: 0.78
skill steps: [4, 2, 4, 0, 7, 9, 6, 5, 2, 7, 7, 3, 4, 5, 0, 1, 8, 9, 8, 2, 6, 8, 3, 6, 6, 7, 8, 9, 2, 4, 2, 0, 8, 3, 7, 8, 3, 7, 0, 0, 4, 7, 4, 6, 0, 5, 2, 6, 9, 7, 1, 5, 4, 1, 4, 7, 7, 3, 4, 0, 0, 9, 3, 4, 7, 5, 6, 5, 5, 2, 4, 7, 6, 8, 2, 3, 4, 1]

Model mode: LSTM, model type: q_only
Final success rate: 0.84
skill steps: [2, 1, 9, 6, 5, 5, 7, 2, 6, 0, 2, 0, 5, 1, 2, 8, 1, 9, 7, 6, 0, 1, 3, 4, 9, 0, 7, 1, 9, 7, 0, 0, 7, 9, 2, 9, 8, 1, 9, 8, 7, 9, 1, 6, 3, 9, 1, 9, 6, 7, 8, 8, 3, 5, 1, 2, 8, 0, 9, 1, 0, 1, 0, 6, 3, 0, 7, 7, 2, 9, 1, 0, 0, 7, 9, 2, 4, 9, 0, 3, 5, 1, 3, 7]

Model mode: LSTM, model type: qhat_no
Final success rate: 0.89
skill steps: [8, 4, 8, 6, 7, 4, 2, 3, 6, 0, 1, 2, 8, 3, 2, 8, 7, 3, 7, 9, 9, 3, 9, 6, 8, 2, 6, 3, 2, 7, 2, 4, 8, 6, 4, 5, 9, 6, 1, 1, 8, 7, 7, 8, 7, 9, 3, 0, 9, 7, 7, 8, 0, 0, 5, 2, 2, 8, 7, 1, 6, 0, 2, 5, 9, 4, 6, 7, 1, 8, 2, 1, 0, 1, 6, 5, 3, 4, 3, 9, 4, 2, 5, 2, 9, 7, 8, 5, 6]

Model mode: LSTM, model type: qhat_true
Final success rate: 0.77
skill steps: [6, 1, 8, 0, 2, 7, 7, 6, 1, 8, 1, 8, 5, 4, 5, 3, 7, 0, 1, 5, 7, 7, 2, 5, 8, 6, 8, 7, 2, 6, 8, 9, 0, 0, 3, 2, 4, 6, 7, 3, 8, 3, 2, 2, 6, 2, 2, 3, 0, 4, 7, 5, 2, 4, 2, 3, 5, 6, 4, 1, 5, 7, 5, 2, 6, 2, 0, 0, 8, 2, 4, 7, 8, 9, 6, 1, 5]"""