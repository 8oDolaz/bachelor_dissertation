import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robosuite'))

import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

# for i in range(1000):
action = np.random.randn(*env.action_spec[0].shape) * 0.1
obs, reward, done, info = env.step(action)  # take action in the environment

print(list(obs.keys()), end='\n\n')
env.render()  # render on display

while True:
    try:
        pass
    except KeyboardInterrupt:
        env.close()
        break
