import gym
from gym import spaces
import numpy as np
import pygame

from pyreason_gym.pyreason_grid_world.pyreason_grid_world import PyReasonGridWorld


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid_size=8, num_agents_per_team=1, render_mode=None):
        """Initialize grid world

        :param render_mode: whether to render in human viewable format or not, defaults to None
        :param size: size of the grid world square, defaults to 8
        """
        super(GridWorldEnv, self).__init__()

        self.grid_size = grid_size
        self.render_mode = render_mode
        self.window_size = 512

        # Initialize the PyReason gridworld
        self.pyreason_grid_world = PyReasonGridWorld(grid_size, num_agents_per_team)

        # Get the position of obstacles for the render function
        self.obstacle_positions = self.pyreason_grid_world.get_obstacle_locations()
        self.base_positions = self.pyreason_grid_world.get_base_locations()

        # Currently the observation space consists of the positions of the agents as well as their state (health etc.)
        # Length of the sequence = num_agents_per_team
        self.observation_space = spaces.Dict(
            {
                'red_team': spaces.Sequence(spaces.Dict({'pos': spaces.Box(0, grid_size-1, shape=(2,), dtype=int), 'health': spaces.Box(0,1, dtype=np.float32)})),
                'blue_team': spaces.Sequence(spaces.Dict({'pos': spaces.Box(0, grid_size-1, shape=(2,), dtype=int), 'health': spaces.Box(0,1, dtype=np.float32)}))
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
        return {}

    def _get_rew(self):
        return 0

    def reset(self, seed=None, options=None):
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
            self._render_frame(observation)

        return observation, info
    
    def step(self, action):
        self.pyreason_grid_world.move(action)

        observation = self._get_obs()
        info = self._get_info()

        # Get reward
        rew = self._get_rew()

        # End of game
        done = self.is_done(observation)

        # Render if necessary
        if self.render_mode == "human":
            self._render_frame(observation)

        return observation, rew, done, False, info
    
    def _render_frame(self, observation):
        if self.window is None and self.render_mode=="human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode=="human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        # The size of a single grid square in pixels
        pix_square_size = (
            self.window_size / self.grid_size
        )

        # First draw both bases
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.to_pygame_coords(self.base_positions[0]),
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                pix_square_size * self.to_pygame_coords(self.base_positions[1]),
                (pix_square_size, pix_square_size),
            ),
        )


        # Draw the obstacles
        for i in self.obstacle_positions:
            triangle_coords = [pix_square_size * self.to_pygame_coords(i), pix_square_size * self.to_pygame_coords(i), pix_square_size * self.to_pygame_coords(i)]
            triangle_coords[0][0] += pix_square_size/2
            triangle_coords[1][1] += pix_square_size
            triangle_coords[2][0] += pix_square_size
            triangle_coords[2][1] += pix_square_size
            pygame.draw.polygon(
                canvas,
                (0, 0, 0),
                triangle_coords,
            )

        # Draw the agents according to the observation
        for i in observation['red_team']:
            pos = self.to_pygame_coords(i['pos']) * pix_square_size
            pos += int(pix_square_size/2)
            # Draw circle and border
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                pos,
                pix_square_size/3,
            )
            pygame.draw.circle(
                canvas,
                (0, 0, 0),
                pos,
                pix_square_size/3,
                5
            )
        
        for i in observation['blue_team']:
            pos = self.to_pygame_coords(i['pos']) * pix_square_size
            pos += int(pix_square_size/2)
            # Draw circle and border
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                pos,
                pix_square_size/3,
            )
            pygame.draw.circle(
                canvas,
                (0, 0, 0),
                pos,
                pix_square_size/3,
                5
            )

        # Finally, add some gridlines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode=="human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode=='rgb_array':
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def is_done(self, observation):
        # End the game when the health goes to zero of an entire team
        red_end = True
        blue_end = True
        for i in observation['red_team']:
            if i['health']!=0:
                red_end = False

        for i in observation['blue_team']:
            if i['health']!=0:
                blue_end = False

        return red_end or blue_end

    def to_pygame_coords(self, coords):
        """Convert coordinates into pygame coordinates (lower-left => top left)."""
        return np.array([coords[0], self.grid_size - 1 - coords[1]])