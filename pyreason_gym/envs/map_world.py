import gym
from gym import spaces
import numpy as np
import pygame
import time

from pyreason_gym.pyreason_map_world.pyreason_map_world import PyReasonMapWorld

np.set_printoptions(precision=20)

# Odd order is due to orientation on canvas while displaying
LAT_LONG_SCALE = int(10e14)
LAT_MAX = int(35.64402770996094 * 10e14)
LAT_MIN = int(36.58740997314453 * 10e14)
LONG_MIN = int(-84.71105957031250 * 10e14)
LONG_MAX = int(-83.68660736083984 * 10e14)
PYGAME_MIN = 0
PYGAME_MAX = 1000


def map_lat_long_to_pygame_coords(lat, long):
    # Map from lat long range to pygame range
    lat = int(lat * LAT_LONG_SCALE)
    long = int(long * LAT_LONG_SCALE)
    lat_range = LAT_MAX - LAT_MIN
    long_range = LONG_MAX - LONG_MIN
    pygame_range = PYGAME_MAX - PYGAME_MIN
    new_lat = (((lat - LAT_MIN) * pygame_range) / lat_range) + PYGAME_MIN
    new_long = (((long - LONG_MIN) * pygame_range) / long_range) + PYGAME_MIN
    coord = np.array([new_long, new_lat])

    return coord


class MapWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, start_point, end_point, render_mode=None):
        """Initialize map world

        :param start_point: Point where agent will start
        :type start_point: str
        :param end_point: Point where agent should end
        :type end_point: str
        :param render_mode: how to render the environment, defaults to None
        :type render_mode: str or None
        """
        super(MapWorldEnv, self).__init__()

        self.render_mode = render_mode
        self.window_size = PYGAME_MAX

        # Start End points are required for observations
        self.start_point = start_point
        self.end_point = end_point

        # Rendering info
        self.start_point_lat_long = None
        self.end_point_lat_long = None

        # Initialize the PyReason map-world
        self.pyreason_map_world = PyReasonMapWorld(end_point)

        # Observation space is how close/far it is to the goal point. Coordinates from current point to end point
        # And how many valid actions there are in the state
        self.observation_space = spaces.Tuple((spaces.Text(max_length=20), spaces.Box(-100, 100, shape=(2,), dtype=np.float128), spaces.Box(-100, 100, shape=(2,), dtype=np.float128), spaces.Discrete(100)))

        # The choice of action is limited to the number of outgoing edges from one node. The agent has to pick one edge to go on
        self.action_space = spaces.Discrete(1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. `self.canvas` is a pygame surface where we draw out the game
        # They will remain `None` until human-mode is used for the first time.
        self.window = None
        self.clock = None
        self.canvas = None

    def _get_obs(self):
        return self.pyreason_map_world.get_obs()

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

        self.pyreason_map_world.reset()

        observation = self._get_obs()
        info = self._get_info()

        # Save new action space
        _, _, _, new_action_space = observation
        self.action_space = spaces.Discrete(new_action_space)

        # Render if necessary
        if self.render_mode == "human":
            self._render_frame(observation)

        return observation, info

    def step(self, action):
        self.pyreason_map_world.move(action)

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

    def _render_init(self):
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()
        self.canvas = pygame.Surface((self.window_size, self.window_size))
        self.canvas.fill((255, 255, 255))

        # Draw the nodes and edges on this canvas so that we don't have to keep drawing it every step
        nodes_lat_long, edges_lat_long = self.pyreason_map_world.get_map()

        # Draw points for nodes
        for node in nodes_lat_long:
            pygame.draw.circle(
                self.canvas,
                (69, 69, 69),
                map_lat_long_to_pygame_coords(*node),
                2
            )

        # Draw edges between points
        for edge in edges_lat_long:
            pygame.draw.aaline(
                self.canvas,
                (169, 169, 169),
                map_lat_long_to_pygame_coords(*edge[0]),
                map_lat_long_to_pygame_coords(*edge[1])
            )

        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])
        time.sleep(10)

    def _render_frame(self, observation):
        if self.canvas is None:
            self._render_init()
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = self.canvas.copy()

        current_node, current_lat_long, end_lat_long, new_action_space = observation

        if self.start_point_lat_long is None:
            self.start_point_lat_long = current_lat_long
        if self.end_point_lat_long is None:
            self.end_point_lat_long = end_lat_long

        # Draw start and end nodes
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            map_lat_long_to_pygame_coords(self.start_point_lat_long[0], self.start_point_lat_long[1]),
            5,
        )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            map_lat_long_to_pygame_coords(current_lat_long[0], current_lat_long[1]),
            5,
        )
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            map_lat_long_to_pygame_coords(self.end_point_lat_long[0], self.end_point_lat_long[1]),
            5,
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == 'rgb_array':
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def is_done(self, observation):
        # End the game when the agent reaches the end point

        return
