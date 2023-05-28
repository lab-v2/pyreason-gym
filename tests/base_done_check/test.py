import pyreason_gym
import gym
import time

env = gym.make('PyReasonGridWorld-v0', render_mode='human')
obs = env.reset()

# Sample actions:
action = {
    'red_team': [0],
    'blue_team': [1]
}
obs = env.step(action)
print(obs)
for _ in range(8):
    time.sleep(1)
    action = {
        'red_team': [0],
        'blue_team': [1]
    }
    obs = env.step(action)
    print(obs)
env.close()