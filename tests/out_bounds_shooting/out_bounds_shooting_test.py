import pyreason_gym
import gym
import time

env = gym.make('PyReasonGridWorld-v0', render_mode='human')
obs = env.reset()

# Sample actions:
action = {
    'red_team': [5],
    'blue_team': [4]
}
obs = env.step(action)
print(obs)
time.sleep(1)
for _ in range(10):
    action = {
        'red_team': [2],
        'blue_team': [1]
    }
    obs = env.step(action)
    print(obs)
    time.sleep(1)

env.close()