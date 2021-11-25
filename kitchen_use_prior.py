import torch
import numpy as np
from kitchen_utils import model_config
from spirl.rl.envs.kitchen import KitchenEnv
from spirl.models.skill_prior_mdl import SkillPriorMdl

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
    ckp_data = torch.load('experiments/skill_prior_learning/kitchen/hierarchical_Oct28/weights/weights_ep190.pth')
    skill_model.load_state_dict(ckp_data['state_dict'])

    skill_model.eval()

    print("skill prior type:", skill_model._hp.learned_prior_type)
    print("number of prior:", len(skill_model.p))

    obs = env.reset()
    for _ in range(1000):
        s = torch.tensor(obs.reshape(1, -1))
        
        # sample skill from state dependent prior
        with torch.no_grad():
            z = skill_model.compute_learned_prior(s)
        z = z.sample().reshape(1, -1)

        # execute first action from skill
        with torch.no_grad():
            a_seq = skill_model.decode(z, s, model_config.n_rollout_steps)
        a_seq_np = a_seq[0, :, :].cpu().numpy()

        for step_idx in range(a_seq_np.shape[0]):
            obs, reward, done, info = env.step(a_seq_np[0, :])

            env._render_raw(mode='human')    
