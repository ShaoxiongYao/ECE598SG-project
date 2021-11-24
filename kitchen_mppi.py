import numpy as np
import torch
import logging
from pytorch_mppi import mppi
from gym import logger as gym_log
from data.example_dynamics_data.reader import PlainNet, LSTM
from spirl.rl.envs.kitchen import KitchenEnv

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
    TIMESTEPS = 15   # T
    N_SAMPLES = 100  # K
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    d = "cpu"
    dtype = torch.double

    # skill dim: 10
    noise_sigma = 3*torch.eye(10)
    lambda_ = 1.

    env = KitchenEnv({})

    skill_dynamics = create_dynamics_model()
    env_skill_step = create_skill_step(step_mode='all', render_mode='human')
    running_cost   = create_running_cost()

    mppi_ctrl = mppi.MPPI(skill_dynamics, running_cost, nx, noise_sigma, 
                          num_samples=N_SAMPLES, horizon=TIMESTEPS,
                          lambda_=lambda_, device=d, 
                          u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
                          u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))

    # assuming you have a gym-like env
    obs = env.reset()
    for i in range(100):
        z = mppi_ctrl.command(obs)
        step_info_lst = env_skill_step(env, z, obs)
        print(f"step {i} selected skill:", z.cpu().numpy())

        obs, reward, done, _ = step_info_lst[-1]
