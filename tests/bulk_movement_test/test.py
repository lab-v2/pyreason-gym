import pyreason_gym
import gym
import time
import random

env = gym.make('PyReasonGridWorld-v0', render_mode=None)
obs = env.reset()

# print("Reset:", obs)
exp_start = time.time()
for _ in range(10000):
    # time.sleep(1)
    # action = env.action_space.sample()
    # print(action)
    action = {
            'red_team': [random.randint(0, 3)],
            'blue_team': [random.randint(0, 3)]
        }
    start = time.time()
    obs, _, terminated, truncated, _ = env.step(action)
    print("Time for single step:", time.time() - start)
    # print(obs)
    done = terminated or truncated
    if done:
        obs = env.reset()
print("Time for 10000 steps:", time.time() - exp_start)
env.close()