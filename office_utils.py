import cv2
import numpy as np
from spirl.rl.envs.office import OfficeEnv

if __name__ == '__main__':
    env = OfficeEnv({})

    obs = env.reset()

    # Office env: 
    #   state space dim 97
    #   action space dim 8
    #       ee position action[:3] 
    #       ee orientation action[3:6] 
    #       gripper_action = action[6]
    #       neutral_action = action[7]

    for step_idx in range(100):
        print("step_idx:", step_idx)
        a = np.random.uniform(low=-1, high=-1, size=(8,))

        obs, reward, done, info = env.step(a)
        
        img = env._render_raw(mode='rgb_array')
        cv2.imshow("Office", img)
        cv2.waitKey(1)