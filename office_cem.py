import torch
import numpy as np
import cv2
from torch._C import parse_ir

from spirl.rl.envs.office import OfficeEnv
from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from tqdm import tqdm
from spirl.utils.general_utils import AttrDict
from pytorch_cem import cem
from office_utils import office_dims, create_dynamics_model
from data.example_dynamics_data.reader import PlainNet, LSTM


# kitchen is a recurrent decoder; office is not.

"""
model_config = {'state_dim': 97, 'action_dim': 8, 'n_rollout_steps': 10, 'kl_div_weight': 0.0005, 'nz_mid': 128, 'n_processing_layers': 5, 'cond_decode': True, 'batch_size': 271, 'dataset_class': <class 'spirl.components.data_loader.GlobalSplitVideoDataset'>, 'n_actions': 8, 'split': {'train': 0.9, 'val': 0.1, 'test': 0.0}, 'env_name': 'Widow250OfficeFixed-v0', 'res': 64, 'crop_rand_subseq': True, 'max_seq_len': 350, 'subseq_len': 11, 'device': 'cuda'}
"""
model_config = AttrDict(state_dim=97, 
                        action_dim=8, 
                        n_rollout_steps=10, 
                        kl_div_weight=0.0005, 
                        nz_mid=128, 
                        n_processing_layers=5, 
                        cond_decode=True, 
                        batch_size=271, 
                        n_actions=8, 
                        res=64, 
                        max_seq_len=350, 
                        subseq_len=11, 
                        device='cpu')

# state dimenison
nx = office_dims['s']
nu = office_dims['z']

# Setup MPPI parameters
TIMESTEPS = 10  # T
N_SAMPLES = 100  # K
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
N_ELITES = 15
SAMPLE_ITER = 3  # M

d = "cpu"
dtype = torch.double

# skill dim: 10
noise_sigma = 3*torch.eye(10)
lambda_ = 1.


def cost_func(state, action):
    """
    Output:
        cost: tensor shape (K,)
    """
    drawer_idx = 70
    drawer_goal = 0.3

    drawer_state = state[:, [drawer_idx]]
    drawer_state_err = drawer_state - drawer_goal
    cost = 100*drawer_state_err.pow(2).sum(axis=1)
    
    return cost.cpu()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_mode', type=str)
    parser.add_argument('--model_type', type=str)
    args = parser.parse_args()
    env = OfficeEnv({})    
    
    # create skill priori
    skill_model = ClSPiRLMdl(model_config)
    ckp_data = torch.load('data/office_weights/weights_ep2.pth')
    skill_model.load_state_dict(ckp_data['state_dict'])

    skill_model.eval() 

    print("skill prior type:", skill_model._hp.learned_prior_type)

    # different options:
    # PlainNet q_no
    # PlainNet q_only
    # PlainNet q_true
    # PlainNet qhat_no
    # PlainNet qhat_only
    # PlainNet qhat_true
    # LSTM q_no
    # LSTM qhat_no
    skill_dynamics = create_dynamics_model(args.model_mode, args.model_type)

    cem_ctrl = cem.CEM(skill_dynamics, cost_func, nx, nu, num_samples=N_SAMPLES, num_iterations=SAMPLE_ITER,
                       horizon=TIMESTEPS, device=d, num_elite=N_ELITES,
                       u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d), init_cov_diag=1)


    tot_reward = 0
    success_episodes = 0

    for episode_idx in range(100):
        print("episode index:", episode_idx)
        obs = env.reset()
        s_ts, s_tplusNs, zs = [], [], []

        task_done = False
        for _ in range(50):
            s = torch.tensor(obs.reshape(1, -1))
            # sample skill from state dependent prior
            with torch.no_grad():
                z_p = skill_model.compute_learned_prior(s)

            mu = z_p.mu.reshape(-1)
            sigma = torch.diag(z_p.log_sigma.exp().reshape(-1)) 
            cem_ctrl.init_action_distribution(mu, sigma)

            # get skill from MPPI controller
            z = cem_ctrl.command(s) 
            z = z.reshape(1, -1)
            
            s_ts.append(s)
            zs.append(z)
            # execute first action from skill
            with torch.no_grad():
                # execeute actions decode from skill
                for step_idx in range(10):
                    a = skill_model.decoder(torch.cat((s, z), dim=1).float())
                    obs, reward, done, info = env.step(a[0])

                    open_drawer_reward = np.abs(obs[70]-0.3)
                    if open_drawer_reward < 0.02:
                        task_done = True
                    
                    # img = env._render_raw(mode='rgb_array')
                    # cv2.imshow("Office", img)
                    # cv2.waitKey(1)  

                    s = torch.from_numpy(obs).view(1, -1)

                    if task_done:
                        # early stop when task completed
                        break
            
            s_tplusNs.append(obs.reshape(1, -1))     

            if task_done:
                success_episodes += 1
                break  
        
        print("success rate:", success_episodes/float(episode_idx+1))
        
    
    print("reward:", tot_reward)