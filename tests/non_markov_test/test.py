import pyreason_gym
import gym
import time

env = gym.make('PyReasonGridWorld-v0', render_mode='human', num_agents_per_team=2)
obs = env.reset()
time.sleep(5)
for _ in range(3):
    action = {
        'red_team': [0,0],
        'blue_team': [1,3]
    }
    obs = env.step(action)
    print(obs)
    time.sleep(1)

env.close()