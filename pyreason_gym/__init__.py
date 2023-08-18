from gym.envs.registration import register


register(
    id='PyReasonMapWorld-v0',
    entry_point='pyreason_gym.envs:MapWorldEnv'
)
