import pyreason_gym
import gym

env = gym.make('PyReasonGridWorld-v0')
obs = env.reset()
print('output', obs)