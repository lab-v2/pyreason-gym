import pyreason_gym
import gym
import time

# Local graph
# env = gym.make('PyReasonMapWorld-v0', start_point='202811335', end_point='472254560', graph_path='./pyreason_gym/pyreason_map_world/graph/map_graph_sub.graphml', rules_path='pyreason_gym/pyreason_map_world/rules/rules.txt')
# Remote graph
env = gym.make('PyReasonMapWorld-v0', start_point='113420', end_point='51230', graph_path='bolt://ec2-54-242-206-65.compute-1.amazonaws.com:7687', graph_auth=('neo4j', 'N30n30j44'), rules_path='pyreason_gym/pyreason_map_world/rules/rules.txt', render_mode=None, city='Knoxville')
obs = env.reset()
print(obs)

# Actions (pre defined for rule firing)
# actions = [1, 2, 0]

# Randomly sample actions from the action space
for i in range(50):
    print(i)
# for i in actions:
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
    # time.sleep(3)

env.close()
