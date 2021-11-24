import numpy as np
import torch
from spirl.rl.envs.kitchen import KitchenEnv
from spirl.models.skill_prior_mdl import SkillPriorMdl
from data.example_dynamics_data.reader import PlainNet, LSTM
from spirl.utils.general_utils import AttrDict

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

if __name__ == '__main__':
    env = KitchenEnv({})

    model_mode = 'PlainNet'
    
    if model_mode == 'PlainNet':
        dynamics_model = PlainNet(kitchen_dims['s'] + kitchen_dims['z'], kitchen_dims['s'])
        dynamics_model = torch.load('data/example_dynamics_data/kitchen_MLP.pth')
    
    skill_model = SkillPriorMdl(model_config)
    ckp_data = torch.load('experiments/skill_prior_learning/kitchen/hierarchical_Oct28/weights/weights_ep190.pth')
    skill_model.load_state_dict(ckp_data['state_dict'])

    obs = env.reset()
    for step_idx in range(100):
        z = torch.zeros((1, kitchen_dims['z']))
        s = torch.tensor(obs).reshape(1, -1)

        with torch.no_grad():
            a_tsr = skill_model.decode(z, s, model_config.n_rollout_steps)
        a_np = a_tsr[0, :, :].cpu().numpy()

        for ii in range(model_config.n_rollout_steps):
            
            a = a_np[ii, :]

            obs, reward, done, info = env.step(a)
            print(f"step {step_idx}, reward: {reward}")

            img = env._render_raw(mode='human')
