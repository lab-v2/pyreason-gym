import pyreason_gym
import gym
import time

env = gym.make('PyReasonGridWorld-v0', render_mode='human')
obs = env.reset()

print("Reset:", obs)
for _ in range(20):
    time.sleep(1)
    action = env.action_space.sample()
    
    obs, _, terminated, truncated, _ = env.step(action)
    print(obs)
    done = terminated or truncated
    if done:
        obs = env.reset()
env.close()