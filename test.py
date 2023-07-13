import pyreason_gym
import gym

env = gym.make('PyReasonMapWorld-v0', start_point='201873272', end_point='202973047')
obs = env.reset()
print(obs)

# Randomly sample actions from the action space
for i in range(10):
    action = env.action_space.sample()
    print('action',action)
    obs, rew, done, truncated, info = env.step(action)
    print(obs)
    print(rew)
    print()

env.close()
