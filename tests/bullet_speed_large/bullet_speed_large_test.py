import pyreason_gym
import gym
import time

env = gym.make('PyReasonGridWorld-v0', render_mode='human', grid_size=30)
obs = env.reset()
steps = 0
# Sample actions:
action = {
    'red_team': [7],
    'blue_team': [3]
}
obs = env.step(action)
steps +=1
print("step:", steps)
print(obs)
time.sleep(1)
for _ in range(35):
    action = {
        'red_team': [1],
        'blue_team': [3]
    }
    obs = env.step(action)
    steps +=1
    print("step:", steps)
    print(obs)
    time.sleep(1)
env.close()