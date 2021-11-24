import numpy as np
from spirl.rl.envs.kitchen import KitchenEnv

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

if __name__ == '__main__':
    env = KitchenEnv({})

    for step_idx in range(100):
        obs = env.reset()
        a = np.array([0, 0, 0, 0.01, 0, 0, 0, 0, 0])
        obs, reward, done, info = env.step(a)
        print(f"step {step_idx}, reward: {reward}")
    