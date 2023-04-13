import pyreason_gym
import gym

env = gym.make('PyReasonGridWorld-v0')
obs = env.reset()
action = {
    'red_team': [0],
    'blue_team': [1]
}
obs = env.step(action)
action = {
    'red_team': [2],
    'blue_team': [3]
}
obs = env.step(action)
print('output', obs)