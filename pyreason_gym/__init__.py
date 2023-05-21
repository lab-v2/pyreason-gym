from gym.envs.registration import register

register(
    id='PyReasonGridWorld-v0',
    entry_point='pyreason_gym.envs:GridWorldEnv'
)

register(
    id='PyReasonMapWorld-v0',
    entry_point='pyreason_gym.envs:MapWorldEnv'
)
