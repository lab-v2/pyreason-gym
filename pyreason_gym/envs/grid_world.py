import gym
from gym import spaces

from pyreason_gym.pyreason_grid_world.pyreason_grid_world import PyReasonGridWorld


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=8, num_agents_per_team=1, render_mode=None):
        """Initialize grid world

        :param render_mode: whether to render in human viewable format or not, defaults to None
        :param size: size of the grid world square, defaults to 8
        """
        super(GridWorldEnv, self).__init__()

        self.grid_size = grid_size
        self.render_mode = render_mode

        # Initialize the PyReason gridworld
        self.pyreason_grid_world = PyReasonGridWorld(grid_size, num_agents_per_team)

        # Currently the observation space consists of the positions of the agents as well as their state (health etc.)
        # Length of the sequence = num_agents_per_team
        self.observation_space = spaces.Dict(
            {
                'red_team': spaces.Sequence(spaces.Dict({'pos': spaces.Discrete(grid_size*grid_size), 'health': spaces.Box(0,1)})),
                'blue_team': spaces.Sequence(spaces.Dict({'pos': spaces.Discrete(grid_size*grid_size), 'health': spaces.Box(0,1)}))
            }
        )

        # We have 4 actions, corresponding to "up", "down", "left", "right".
        # TODO: Add shoot later
        self.action_space = spaces.Dict(
            {
                'red_team': spaces.MultiDiscrete([4]*num_agents_per_team),
                'blue_team': spaces.MultiDiscrete([4]*num_agents_per_team)
            }
        )
        self.actions = {0:'up', 1:'down', 2:'left', 3:'right'}

        assert render_mode is None or render_mode in self.metadata["render_modes"]


        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        self.window = None
        self.clock = None
        
    def _get_obs(self):
        return self.pyreason_grid_world.get_obs()

    def _get_info(self):
        pass

    def _get_rew(self):
        return 0

    def reset(self, seed=None):
        """Resets the environment to the initial conditions

        :param seed: random seed if there is a random component, defaults to None
        :param options: defaults to None
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.pyreason_grid_world.reset()

        observation = self._get_obs()
        info = self._get_info()

        # Render if necessary
        if self.render_mode == "human":
            pass

        return observation, info
    
    def step(self, action):
        # TODO: Send action to pyreason and get observation
        self.pyreason_grid_world.move(action)

        observation = self._get_obs()
        info = self._get_info()

        # Get reward
        rew = self._get_rew()

        # End of game
        done = False

        # Render if necessary
        if self.render_mode == "human":
            pass

        return observation, rew, done, False, info
    
    def close(self):
        pass