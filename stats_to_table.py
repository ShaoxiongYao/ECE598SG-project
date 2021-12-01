import numpy as np

from stats_str import kitchen_mppi_stats, office_cem_stats, office_mppi_stats

model_type_to_str = {
    'q_no': r'$q$,demo', 
    'q_only': r'$q$,augment',
    'q_true': r'$q$,all',
    'qhat_no': r'$\hat{q}$,demo',
    'qhat_true': r'$\hat{q}$,all'
}

def process_kitchen_state(kitchen_stats, mpc_name):
    # split by an empty line
    stats_lst = kitchen_stats.split('\n\n')
    for stats in stats_lst:
        model_str, reward_str, steps_str = stats.split('\n')

        # parse stats
        _, _, model_mode, _, _, model_type = model_str.split()
        _, _, _, r_mean, _, r_std = reward_str.split()
        _, _, _, _, _, steps_mean, _, steps_std = steps_str.split()

        model_mode = model_mode[:-1]
        if model_mode == 'PlainNet':
            model_mode = 'MLP'

        exp_head     = f"{model_mode}({model_type_to_str[model_type]})+"+mpc_name
        reward_stats = f"${float(r_mean):.2f}\pm{float(r_std):.2f}$"
        steps_stats  = f"${float(steps_mean):.2f}\pm{float(steps_std):.2f}$"

        print(exp_head + " & " + reward_stats + " & " + steps_stats + r" \\")

def process_office_stats(office_stats, mpc_name):
    # split by an empty line
    stats_lst = office_stats.split('\n\n')
    for stats in stats_lst:
        model_str, reward_str, steps_str = stats.split('\n')

        # parse stats
        _, _, model_mode, _, _, model_type = model_str.split()
        _, _, _, succ_rate = reward_str.split()
        steps_lst = [int(x) for x in steps_str[14:-1].split(', ')]

        steps_mean = np.mean(steps_lst)
        steps_std  = np.std(steps_lst)

        model_mode = model_mode[:-1]
        if model_mode == 'PlainNet':
            model_mode = 'MLP'

        exp_head     = f"{model_mode}({model_type_to_str[model_type]})+"+mpc_name
        reward_stats = f"${float(succ_rate):.2f}$"
        steps_stats  = f"${float(steps_mean):.2f}\pm{float(steps_std):.2f}$"

        print(exp_head + " & " + reward_stats + " & " + steps_stats + r" \\")

if __name__ == "__main__":
    print("kitchen MPPI table:")
    process_kitchen_state(kitchen_mppi_stats, 'MPPI')
    print("office CEM table:")
    process_office_stats(office_cem_stats, 'CEM')
    print("office MPPI table:")
    process_office_stats(office_mppi_stats, 'MPPI')
