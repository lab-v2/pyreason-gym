import pyreason_gym
import gym
import time

env = gym.make('PyReasonGridWorld-v0', render_mode='human')
obs = env.reset()
time.sleep(10)

# Sample actions:
# action = {
#     'red_team': [0],
#     'blue_team': [1]
# }
# obs = env.step(action)
# action = {
#     'red_team': [2],
#     'blue_team': [3]
# }
# obs = env.step(action)

# Randomly sample actions from the action space
for i in range(50):
    action = env.action_space.sample()
    print(action)
    env.step(action)
    time.sleep(1)

env.close()