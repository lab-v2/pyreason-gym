import pyreason_gym
import gym

env = gym.make('PyReasonMapWorld-v0', start_point='202811335', end_point='472254560', graph_path='./pyreason_gym/pyreason_map_world/graph/map_graph_sub.graphml', rules_path='pyreason_gym/pyreason_map_world/rules/rules.txt')
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
