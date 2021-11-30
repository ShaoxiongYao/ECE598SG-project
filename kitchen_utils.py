import numpy as np
import torch
from torch.nn import MSELoss
from spirl.rl.envs.kitchen import KitchenEnv
from spirl.models.skill_prior_mdl import SkillPriorMdl
from data.example_dynamics_data.reader import PlainNet, LSTM
from spirl.utils.general_utils import AttrDict
from spirl.modules.variational_inference import MultivariateGaussian
import matplotlib.pyplot as plt
from tqdm import tqdm
model_config = AttrDict(
    state_dim=60,
    action_dim=9,
    batch_size=128,
    device='cpu',
    n_rollout_steps=10,
    kl_div_weight=5e-4,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
)


# FROM: https://github.com/rail-berkeley/d4rl/blob/master/d4rl/kitchen/kitchen_envs.py
OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }
OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    }
BONUS_THRESH = 0.3

kitchen_dims = {
    'z': 10, 's': 60, 'a': 9
}

model_type = "prior_endless"

def create_dynamics_model(model_mode, model_type): # or 'q'
    
    if model_mode == 'PlainNet':
        dynamics_model = PlainNet(kitchen_dims['s'] + kitchen_dims['z'], kitchen_dims['s'])
        dynamics_model = torch.load('data/example_dynamics_data/kitchen_MLP_'+model_type+'.pth')
    elif model_mode == "LSTM":
        dynamics_model = LSTM(kitchen_dims['s'] + kitchen_dims['z'], kitchen_dims['s'])
        dynamics_model = torch.load('data/example_dynamics_data/kitchen_LSTM_'+model_type+'.pth')
        
    def skill_dynamics(state, skill):
        state = state.to(skill.device)
        state_skill = torch.cat([state, skill], axis=1)
        state_skill = state_skill.to('cuda:0')
        with torch.no_grad():
            next_state = dynamics_model.predict(state_skill)
        return next_state
    
    return skill_dynamics

def create_skill_step(step_mode='single', render_mode=None):

    skill_model = SkillPriorMdl(model_config)
    ckp_data = torch.load('weights_ep190.pth')
    skill_model.load_state_dict(ckp_data['state_dict'])
    
    skill_model.eval() #batchnorm inside!
    
    if step_mode == 'single':
        step_len = 1
    elif step_mode == 'all':
        step_len = model_config.n_rollout_steps

    def env_skill_step(env, z, s):
        z = z.reshape(1, -1)
        s = torch.tensor(s).reshape(1, -1)
        with torch.no_grad():
            a_seq = skill_model.decode(z, s, model_config.n_rollout_steps)
        a_seq_np = a_seq[0, :, :].cpu().numpy()

        step_info_lst = []
        for step_idx in range(step_len):
           
            a_np = a_seq_np[step_idx, :]
            step_info = env.step(a_np)
            step_info_lst.append(step_info)

            if render_mode is not None:
                env._render_raw(mode=render_mode)
        
        return step_info_lst
    
    return env_skill_step

def create_running_cost():

    def cost_func(state, action):
        """
        Output:
            cost: tensor shape (K,)
        """
        batch_size = state.shape[0]
        cost = torch.zeros(batch_size).to('cuda:0')

        for key in OBS_ELEMENT_GOALS.keys():
            key_idx  = OBS_ELEMENT_INDICES[key]
            key_goal = OBS_ELEMENT_GOALS[key]

            key_goal_tsr = torch.tensor(key_goal).to('cuda:0')

            key_state = state[:, key_idx]
            key_state_err = key_state - key_goal_tsr
            key_cost = key_state_err.pow(2).sum(axis=1)

            cost += key_cost
        
        return cost.cpu()

    return cost_func

if __name__ == '__main__':
    env = KitchenEnv({})
    M = 50
    N = 3
    
    skill_dynamics = create_dynamics_model('PlainNet', model_type)
    env_skill_step = create_skill_step(step_mode='all')
    
    skill_model = SkillPriorMdl(model_config)
    ckp_data = torch.load('weights_ep190.pth')
    skill_model.load_state_dict(ckp_data['state_dict'])
    
    
    skill_model.eval()
    
    f = np.zeros((N, M))
    for i in tqdm(range(N)):
        obs = env.reset()
        for step_idx in range(M):
            s = torch.tensor(obs).reshape(1, -1)
            # calculation of z using p
            """
            for param in skill_model.p[0]:
                s = param(s)
            z_prob_distrib = s
            n_dim = z_prob_distrib.shape[1] // 2
            print("n_dim:", n_dim)
            print("sample_prior:", skill_model._sample_prior)
            mu, log_delta = z_prob_distrib[0, :n_dim], z_prob_distrib[0, n_dim:]
            sigma = torch.exp(log_delta)
            print("p_mu:", mu, "p_sigma:", sigma)
            z = (mu + sigma * torch.randn_like(sigma)).view(1, -1)
            print("p_z:", z)
            # z = torch.zeros((1, kitchen_dims['z']))
            
            # calculation of z using q as sanity check
            #z_prob_by_q = skill_model._run_inference(s)
            # q_input = actions # torch.cat((actions, states[:, :-1, :]), dim=-1)
            # print("q:", skill_model.q, "q_input.shape:", q_input.shape)
            # q_output = skill_model.q(q_input)
            """
            inputs = AttrDict()
            inputs.states = s.reshape(1, 1, -1)
            q_hat = skill_model.compute_learned_prior(skill_model._learned_prior_input(inputs))
            # print(q_hat.mu, q_hat.sigma)
            z = q_hat.sample()
            # print("z:", z)
            # print("q_output.shape:", q_output.shape, "q_output:", q_output)
           
            """
            1. the big variance & undersampling of z - train dynamic model must use f(s,z) z as value
               z apparently plays a big role in the state prediction
            2. p(q_hat) is not a good approximator?
            """ 
            
            next_s = skill_dynamics(s, z)
            step_info_lst = env_skill_step(env, z, s)
    
            obs, reward, done, info = step_info_lst[-1]
    
            next_s = next_s.cpu().numpy().reshape(-1)
            # print(f"step {step_idx}, prediction error:", np.linalg.norm(obs - next_s))
            f[i, step_idx] = np.linalg.norm(obs - next_s)
    torch.save(f, model_type+".pt")
    avg, std = f.mean(axis=0), f.std(axis=0)
    plt.plot([i for i in range(M)], avg-std, color='g')
    plt.plot([i for i in range(M)], avg, color='r')
    plt.plot([i for i in range(M)], avg+std, color='g')
    plt.title("MLP with "+model_type+" as z")
    plt.xlabel("steps")
    plt.ylabel("Frobenius norm error")
    plt.yscale('log')
    plt.savefig("MLP_"+model_type+".jpg")
    plt.cla()



