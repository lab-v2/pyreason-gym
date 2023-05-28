import pyreason_gym
import gym
import time

env = gym.make('PyReasonGridWorld-v0', render_mode='rgb_array')
obs = env.reset()

# Sample actions:
action = {
    'red_team': [7],
    'blue_team': [1]
}
obs = env.step(action)
print(obs)
print(env.render())
for _ in range(5):
    time.sleep(1)
    action = {
        'red_team': [1],
        'blue_team': [1]
    }
    obs = env.step(action)
    print(obs)
    print(env.render())
env.close()