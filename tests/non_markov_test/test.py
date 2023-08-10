import pyreason_gym
import gym
import time

env = gym.make('PyReasonGridWorld-v0', render_mode='human', num_agents_per_team=2)
obs = env.reset()
print("\n")
print("Starting Experiment with init obs:")
print(obs)
print("\n")
action = {
            'red_team':  [7, 7],
            'blue_team': [2, 2]
        }
obs = env.step(action)
print(obs)
print("\n")
time.sleep(1)
for ss in range(6):
    print("Step", ss)
    if ss % 2 == 0:
        print("Taking both actions")
        action = {
            'red_team':  [2, 2],
            'blue_team': [2, 2]
        }
    else:
        print("Taking first agent actions")
        action = {
            'red_team':  [2,2],
            'blue_team': [8,2]
        }
    obs = env.step(action)
    print(obs)
    print("\n")
    time.sleep(1)

env.close()