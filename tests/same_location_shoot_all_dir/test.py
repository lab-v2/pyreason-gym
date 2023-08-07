import pyreason_gym
import gym
import time

env = gym.make('PyReasonGridWorld-v0', render_mode='human', num_agents_per_team=1)
obs = env.reset()
action = {
    'red_team': [3],
    'blue_team': [1]
}
obs = env.step(action)
print(obs)
time.sleep(1)
# action = {
#     'red_team': [5],
#     'blue_team': [1]
# }
# obs = env.step(action)
# print(obs)
# time.sleep(1)
# action = {
#     'red_team': [6],
#     'blue_team': [1]
# }
# obs = env.step(action)
# print(obs)
# time.sleep(1)
action = {
    'red_team': [7],
    'blue_team': [1]
}
obs = env.step(action)
print(obs)
time.sleep(1)
action = {
    'red_team': [3],
    'blue_team': [1]
}
obs = env.step(action)
print(obs)
time.sleep(1)
env.close()