import logging
import sys
import time

import numpy as np
import torch
from gym import logger as gym_log

from data.example_dynamics_data.reader import LSTM, PlainNet
from kitchen_utils import (OBS_ELEMENT_GOALS, OBS_ELEMENT_INDICES,
                           create_dynamics_model, create_running_cost,
                           create_skill_step, kitchen_dims, model_config)
from pytorch_cem import cem
from spirl.models.skill_prior_mdl import SkillPriorMdl
from spirl.rl.envs.kitchen import KitchenEnv

# MPPI logger
gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_mode', type=str)
    parser.add_argument('--model_type', type=str)
    args = parser.parse_args()

    # state dimenison
    nx = kitchen_dims['s']
    nu = kitchen_dims['z']

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

    render_mode='human'

    env = KitchenEnv({})

    skill_dynamics = create_dynamics_model(model_mode=args.model_mode, 
                                           model_type=args.model_type)
    running_cost   = create_running_cost()
    
    # HACK: directly load model
    ckp_data = torch.load('weights_ep190.pth')
    skill_model = SkillPriorMdl(model_config)
    skill_model.load_state_dict(ckp_data['state_dict'])
    skill_model = skill_model.double()
    skill_model.eval()

    print("skill prior type:", skill_model._hp.learned_prior_type)
    print("number of prior:", len(skill_model.p))

    cem_ctrl = cem.CEM(skill_dynamics, running_cost, nx, nu, num_samples=N_SAMPLES, num_iterations=SAMPLE_ITER,
                       horizon=TIMESTEPS, device=d, num_elite=N_ELITES,
                       u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d), init_cov_diag=1)

    total_reward_lst = []
    first_subtask_steps_lst = []
    for rand_seed in range(100):
        torch.manual_seed(rand_seed)
        total_reward = 0.0

        start_time = time.time()

        # assuming you have a gym-like env
        obs = env.reset()
        cem_ctrl.reset()
        first_subtask_steps = None
        for i in range(50):
            s = torch.tensor(obs.reshape(1, -1), dtype=torch.double)
            # sample skill from state dependent prior
            with torch.no_grad():
                z_p = skill_model.compute_learned_prior(s)
            
            # set noise parameters to MPPI
            mu = z_p.mu.reshape(-1)
            sigma = torch.diag(z_p.log_sigma.exp().reshape(-1)) 
            cem_ctrl.init_action_distribution(mu, sigma)

            # get skill from MPPI controller
            z = cem_ctrl.command(obs) 
            z = z.reshape(1, -1).double()

            # print(f"step {i} selected skill:", z.cpu().numpy())

            # execute first action from skill
            with torch.no_grad():
                a_seq = skill_model.decode(z, s, model_config.n_rollout_steps)
            a_seq_np = a_seq[0, :, :].cpu().numpy()

            for step_idx in range(a_seq_np.shape[0]):
            
                a_np = a_seq_np[step_idx, :]
                obs, reward, done, info = env.step(a_np)
                if first_subtask_steps is None and reward > 0:
                    first_subtask_steps = i
                total_reward += reward

                # env._render_raw(mode=render_mode)

            # print("light switch state:", obs[OBS_ELEMENT_INDICES['light switch']])
            # print("light switch goal:", OBS_ELEMENT_GOALS['light switch'])

        if first_subtask_steps is not None:
            first_subtask_steps_lst.append(first_subtask_steps)
        print(f"seed {rand_seed}, total reward:", total_reward, file=sys.stderr) 
        print("episode time:", time.time() - start_time, file=sys.stderr)
        total_reward_lst.append(total_reward)
    
    print("episode reward, mean:", np.mean(total_reward_lst), 
          "std:", np.std(total_reward_lst) )
    print("steps for first subtask, mean:", np.mean(first_subtask_steps_lst), 
          "std:", np.std(first_subtask_steps_lst))

