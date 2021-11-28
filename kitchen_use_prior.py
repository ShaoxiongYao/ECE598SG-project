import torch
import numpy as np
from kitchen_utils import model_config
from spirl.rl.envs.kitchen import KitchenEnv
from spirl.models.skill_prior_mdl import SkillPriorMdl
from tqdm import tqdm
from spirl.utils.general_utils import AttrDict

def render_demos(env):
    actions = torch.load('data/example_dynamics_data/kitchen/actions0.pt', map_location='cpu')

    obs = env.reset()
    # num skills: 12325
    for skill_idx in range(actions.shape[0]):
        # skill length: 10
        for step_idx in range(actions.shape[1]):
            a = actions[skill_idx, step_idx, :].numpy()

            obs, reward, done, info = env.step(a)

            env._render_raw(mode='human')

if __name__ == '__main__':
    env = KitchenEnv({})    
    
    skill_model = SkillPriorMdl(model_config)
    ckp_data = torch.load('weights_ep190.pth')
    skill_model.load_state_dict(ckp_data['state_dict'])

    skill_model.eval()

    print("skill prior type:", skill_model._hp.learned_prior_type)
    print("number of prior:", len(skill_model.p))

    obs = env.reset()
    tot_reward = 0
    s_ts, s_tplusNs, zs = [], [], []
    for __ in tqdm(range(2)):
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
                a_seq = skill_model.decode(z, s, model_config.n_rollout_steps)
            a_seq_np = a_seq[0, :, :].cpu().numpy()
    
            for step_idx in range(a_seq_np.shape[0]):
                obs, reward, done, info = env.step(a_seq_np[0, :])
                tot_reward += reward
            
            s_tplusNs.append(obs.reshape(1, -1))
    # print(zs[0])
    s_ts = torch.cat([torch.tensor(x) for x in s_ts], dim=0)
    zs = torch.cat([torch.tensor(x) for x in zs], dim=0)
    s_tplusNs = torch.cat([torch.tensor(x) for x in s_tplusNs], dim=0)
    torch.save(s_ts, "data/example_dynamics_data/kitchen/states1.pt")            
    torch.save(zs, "data/example_dynamics_data/kitchen/skill_z1.pt")
    torch.save(s_tplusNs, "data/example_dynamics_data/kitchen/states2.pt")
    print("reward:", tot_reward)
            # env._render_raw(mode='human')    