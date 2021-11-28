import numpy as np
import torch
import logging
from pytorch_mppi import mppi
from gym import logger as gym_log
from data.example_dynamics_data.reader import PlainNet, LSTM, ResMLP
from spirl.rl.envs.kitchen import KitchenEnv
from kitchen_utils import OBS_ELEMENT_GOALS, model_config, OBS_ELEMENT_INDICES
from spirl.models.skill_prior_mdl import SkillPriorMdl

from kitchen_utils import kitchen_dims
from kitchen_utils import create_dynamics_model, create_skill_step, create_running_cost

# MPPI logger
gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    # state dimenison
    nx = kitchen_dims['s']

    # Setup MPPI parameters
    TIMESTEPS = 10  # T
    N_SAMPLES = 100  # K
    ACTION_LOW = -1.0
    ACTION_HIGH = 1.0

    d = "cpu"
    dtype = torch.double

    # skill dim: 10
    noise_sigma = 3*torch.eye(10)
    lambda_ = 1.

    render_mode='human'

    env = KitchenEnv({})

    skill_dynamics = create_dynamics_model(model_mode='ResMLP', model_type="qhat")
    running_cost   = create_running_cost()
    
    # HACK: directly load model
    ckp_data = torch.load('experiments/skill_prior_learning/kitchen/hierarchical_Oct28/weights/weights_ep190.pth')
    skill_model = SkillPriorMdl(model_config)
    skill_model.load_state_dict(ckp_data['state_dict'])
    skill_model.eval()

    print("skill prior type:", skill_model._hp.learned_prior_type)
    print("number of prior:", len(skill_model.p))

    mppi_ctrl = mppi.MPPI(skill_dynamics, running_cost, nx, noise_sigma, 
                          num_samples=N_SAMPLES, horizon=TIMESTEPS,
                          lambda_=lambda_, device=d, 
                          u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
                          u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))


    for rand_seed in range(10):
        torch.manual_seed(rand_seed)
        total_reward = 0.0

        # assuming you have a gym-like env
        obs = env.reset()
        mppi_ctrl.reset()
        for i in range(100):
            s = torch.tensor(obs.reshape(1, -1))
            # sample skill from state dependent prior
            with torch.no_grad():
                z_p = skill_model.compute_learned_prior(s)
            
            # set noise parameters to MPPI
            mu = z_p.mu.reshape(-1)
            sigma = torch.diag(z_p.log_sigma.exp().reshape(-1)) 
            mppi_ctrl.set_noise_dist(mu, sigma)

            # get skill from MPPI controller
            z = mppi_ctrl.command(obs) 
            z = z.reshape(1, -1)

            # print(f"step {i} selected skill:", z.cpu().numpy())

            # execute first action from skill
            with torch.no_grad():
                a_seq = skill_model.decode(z, s, model_config.n_rollout_steps)
            a_seq_np = a_seq[0, :, :].cpu().numpy()

            for step_idx in range(a_seq_np.shape[0]):
            
                a_np = a_seq_np[step_idx, :]
                obs, reward, done, info = env.step(a_np)
                total_reward += reward

                env._render_raw(mode=render_mode)

            # print("light switch state:", obs[OBS_ELEMENT_INDICES['light switch']])
            # print("light switch goal:", OBS_ELEMENT_GOALS['light switch'])

        print(f"seed {rand_seed}, total reward:", total_reward)        

