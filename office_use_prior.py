import torch
import numpy as np

from spirl.rl.envs.office import OfficeEnv
from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from tqdm import tqdm
from spirl.utils.general_utils import AttrDict

# kitchen is a recurrent decoder; office is not.

"""
model_config = {'state_dim': 97, 'action_dim': 8, 'n_rollout_steps': 10, 'kl_div_weight': 0.0005, 'nz_mid': 128, 'n_processing_layers': 5, 'cond_decode': True, 'batch_size': 271, 'dataset_class': <class 'spirl.components.data_loader.GlobalSplitVideoDataset'>, 'n_actions': 8, 'split': {'train': 0.9, 'val': 0.1, 'test': 0.0}, 'env_name': 'Widow250OfficeFixed-v0', 'res': 64, 'crop_rand_subseq': True, 'max_seq_len': 350, 'subseq_len': 11, 'device': 'cuda'}
"""
model_config = AttrDict(state_dim=97, action_dim=8, n_rollout_steps=10, kl_div_weight=0.0005, nz_mid=128, n_processing_layers=5, cond_decode=True, batch_size=271, n_actions=8, res=64, max_seq_len=350, subseq_len=11, device='cpu')

if __name__ == '__main__':
    env = OfficeEnv({})    
    
    skill_model = ClSPiRLMdl(model_config)
    ckp_data = torch.load('weights_ep2.pth')
    skill_model.load_state_dict(ckp_data['state_dict'])

    skill_model.eval() 

    print("skill prior type:", skill_model._hp.learned_prior_type)
    print("number of prior:", len(skill_model.p))

    obs = env.reset()
    tot_reward = 0
    s_ts, s_tplusNs, zs = [], [], []
    for __ in tqdm(range(500)):
        for _ in range(50):
            s = torch.tensor(obs.reshape(1, -1))
            # print("step:", _)
            # sample skill from state dependent prior
            with torch.no_grad():
                z = skill_model.compute_learned_prior(s)
            z = z.sample().reshape(1, -1)
            
            s_ts.append(s)
            zs.append(z)
            # execute first action from skill
            with torch.no_grad():
                 # skill_model.decode(z, s, model_config.n_rollout_steps) # see https://github.com/clvrai/spirl/blob/61e91926d9e06a976b15c733c8cfbb65548097c3/spirl/rl/policies/cl_model_policies.py
                # print("obs:", obs.shape)
                for step_idx in range(10):
                    a = skill_model.decoder(torch.cat((s, z), dim=1))
                    # print("action:", a.shape)
                    obs, reward, done, info = env.step(a[0])
                    s = torch.from_numpy(obs).view(1, -1)
                    tot_reward += reward
            
            s_tplusNs.append(obs.reshape(1, -1))
        if __ % 100 == 99 and __ > 0:
            s_ts2 = torch.cat([torch.tensor(x) for x in s_ts], dim=0)
            zs2 = torch.cat([torch.tensor(x) for x in zs], dim=0)
            s_tplusNs2 = torch.cat([torch.tensor(x) for x in s_tplusNs], dim=0)
            torch.save(s_ts2, "data/example_dynamics_data/office/states_prior_1.pt")            
            torch.save(zs2, "data/example_dynamics_data/office/skill_z_prior_1.pt")
            torch.save(s_tplusNs2, "data/example_dynamics_data/office/states_prior_2.pt")        
    
    # print(zs[0])
    
    print("reward:", tot_reward)
            # env._render_raw(mode='human')    