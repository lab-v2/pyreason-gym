import pyreason_gym
import gym
import time

env = gym.make('PyReasonMapWorld-v0', start_point='202811335', end_point='472254560', graph_path='./pyreason_gym/pyreason_map_world/graph/map_graph_sub.graphml', rules_path='pyreason_gym/pyreason_map_world/rules/rules.txt')
# env = gym.make('PyReasonMapWorld-v0', start_point='202899474', end_point='203070550', graph_path='/home/daditya1/Documents/iarpa-haystac/movement-generation/graphs/graphml/reinforcement_learning_graph.graphml', rules_path='pyreason_gym/pyreason_map_world/rules/rules.txt', render_mode=None)
obs = env.reset()
print(obs)

# Actions (pre defined for rule firing)
# actions = [1, 2, 0]

# Randomly sample actions from the action space
# for i in range(20):
for i in actions:
    action = env.action_space.sample()
    # action = i
    print('action', action)
    s = time.time()
    obs, rew, done, truncated, info = env.step(action)
    e = time.time()
    print('TIME,', e-s)
    print(obs)
    print(rew)
    print()

env.close()
