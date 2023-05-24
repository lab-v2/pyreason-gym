import pyreason_gym
import gym
import time

env = gym.make('PyReasonGridWorld-v0', render_mode='human')
obs = env.reset()

# Sample actions:
action = {
    'red_team': [7],
    'blue_team': [6]
}
obs = env.step(action)
time.sleep(1)
action = {
    'red_team': [2],
    'blue_team': [3]
}
obs = env.step(action)
time.sleep(1)
action = {
    'red_team': [2],
    'blue_team': [3]
}
obs = env.step(action)
time.sleep(1)
action = {
    'red_team': [2],
    'blue_team': [3]
}
obs = env.step(action)
print(obs)
time.sleep(1)
action = {
    'red_team': [2],
    'blue_team': [3]
}
obs = env.step(action)
print(obs)
time.sleep(1)
env.close()