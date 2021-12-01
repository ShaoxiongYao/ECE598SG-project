import cv2
import numpy as np
import torch
import os
from spirl.rl.envs.office import OfficeEnv
from data.example_dynamics_data.reader import PlainNet, LSTM

office_dims = {
    's': 97, 
    'a': 8, 
    'z': 10
}

def create_dynamics_model(model_mode, model_type): # or 'q'

    model_dir = 'data/example_dynamics_data/models/office'
    if model_mode == 'PlainNet':
        dynamics_model = PlainNet(office_dims['s'] + office_dims['z'], office_dims['s'])
        dynamics_model = torch.load(os.path.join(model_dir, 'MLP_'+model_type+'.pth'))
    elif model_mode == "LSTM":
        dynamics_model = LSTM(office_dims['s'] + office_dims['z'], office_dims['s'])
        dynamics_model = torch.load(os.path.join(model_dir, 'LSTM_'+model_type+'.pth'))
        
    def skill_dynamics(state, skill):
        state = state.to(skill.device)
        state_skill = torch.cat([state, skill], axis=1)
        state_skill = state_skill.to('cuda:0').float()
        with torch.no_grad():
            next_state = dynamics_model.predict(state_skill)
        return next_state
    
    return skill_dynamics

if __name__ == '__main__':
    env = OfficeEnv({})

    obs = env.reset()

    # Office env: 
    #   state space dim 97
    # objects shape: (49,)
    # container states: (21,)
    # drawer handle shape: (3,)
    # (3,)
    # (10,)
    # (14,)
    #   action space dim 8
    #       ee position action[:3] 
    #       ee orientation action[3:6] 
    #       gripper_action = action[6]
    #       neutral_action = action[7]

    for step_idx in range(100):
        print("step_idx:", step_idx)
        a = np.random.uniform(low=-1, high=-1, size=(8,))

        obs, reward, done, info = env.step(a)
        print("drawer x:", obs[70])
        
        img = env._render_raw(mode='rgb_array')
        cv2.imshow("Office", img)
        cv2.waitKey(1)