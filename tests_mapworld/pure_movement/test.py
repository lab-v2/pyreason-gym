import pyreason_gym
import gym
import time
import sys

env = gym.make('PyReasonMapWorld-v0', start_point='202927092', end_point='1833473102')
obs = env.reset()
print(obs)
print()
print()
sys.stdout.flush()

# Randomly sample actions from the action space
for i in range(5):
    action = env.action_space.sample()
    print(action)
    obs = env.step(action)
    print(obs)
    print()
    print()
    sys.stdout.flush()
    time.sleep(1)

env.close()