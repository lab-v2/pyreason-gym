import pyreason_gym
import gym
import time

env = gym.make('PyReasonGridWorld-v0', render_mode='human', num_agents_per_team=2)
obs = env.reset()
time.sleep(1)
for _ in range(4):
    action = {
        'red_team': [2,2],
        'blue_team': [2,2]
    }
    obs = env.step(action)
    print(obs)
    time.sleep(1)

env.close()