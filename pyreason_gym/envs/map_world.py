import gym
from gym import spaces
import numpy as np
import pygame

from pyreason_gym.pyreason_map_world.pyreason_map_world import PyReasonMapWorld

np.set_printoptions(precision=20)

LAT_LONG_SCALE = int(10e14)
LAT_MIN = int(35.64402770996094 * 10e14)
LAT_MAX = int(36.58740997314453 * 10e14)
LONG_MIN = int(-84.71105957031250 * 10e14)
LONG_MAX = int(-83.68660736083984 * 10e14)
PYGAME_MIN = 0
PYGAME_MAX = 500


def map_lat_long_to_pygame_coords(lat, long):
    # Map from lat long range to pygame range
    lat = int(lat * LAT_LONG_SCALE)
    long = int(long * LAT_LONG_SCALE)
    lat_range = LAT_MAX - LAT_MIN
    long_range = LONG_MAX - LONG_MIN
    pygame_range = PYGAME_MAX - PYGAME_MIN
    new_lat = (((lat - LAT_MIN) * pygame_range) / lat_range) + PYGAME_MIN
    new_long = (((long - LONG_MIN) * pygame_range) / long_range) + PYGAME_MIN
    coord = np.array([new_lat, new_long])

    # Scale to zoom in
    zoom = 2
    half_x = PYGAME_MAX / 2
    half_y = PYGAME_MAX / 2
    delta_x = coord[0] - half_x
    delta_y = coord[1] - half_y
    x = half_x + delta_x * zoom
    y = half_y + delta_y * zoom
    coord[0] = x
    coord[1] = y
    print(coord)

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
        self.window_size = 512

        # Start End points are required for observations
        self.start_point = start_point
        self.end_point = end_point
        self.current_node = None

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
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        self.window = None
        self.clock = None

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
        self.current_node, _, _, new_action_space = observation
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

    def _render_frame(self, observation):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        current_node, current_lat_long, end_lat_long, new_action_space = observation

        # Draw start and end nodes
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            map_lat_long_to_pygame_coords(current_lat_long[0], current_lat_long[1]),
            10,
        )
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            map_lat_long_to_pygame_coords(end_lat_long[0], end_lat_long[1]),
            10,
        )


        # # The size of a single grid square in pixels
        # pix_square_size = (
        #         self.window_size / self.grid_size
        # )
        #
        # # First draw both bases
        # pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self.to_pygame_coords(self.base_positions[0]),
        #         (pix_square_size, pix_square_size),
        #     ),
        # )
        # pygame.draw.rect(
        #     canvas,
        #     (0, 0, 255),
        #     pygame.Rect(
        #         pix_square_size * self.to_pygame_coords(self.base_positions[1]),
        #         (pix_square_size, pix_square_size),
        #     ),
        # )
        #
        # # Draw the obstacles
        # for i in self.obstacle_positions:
        #     triangle_coords = [pix_square_size * self.to_pygame_coords(i), pix_square_size * self.to_pygame_coords(i),
        #                        pix_square_size * self.to_pygame_coords(i)]
        #     triangle_coords[0][0] += pix_square_size / 2
        #     triangle_coords[1][1] += pix_square_size
        #     triangle_coords[2][0] += pix_square_size
        #     triangle_coords[2][1] += pix_square_size
        #     pygame.draw.polygon(
        #         canvas,
        #         (0, 0, 0),
        #         triangle_coords,
        #     )
        #
        # # Draw the agents according to the observation
        # for i in observation['red_team']:
        #     pos = self.to_pygame_coords(i['pos']) * pix_square_size
        #     pos += int(pix_square_size / 2)
        #     # Draw circle and border
        #     pygame.draw.circle(
        #         canvas,
        #         (255, 0, 0),
        #         pos,
        #         pix_square_size / 3,
        #     )
        #     pygame.draw.circle(
        #         canvas,
        #         (0, 0, 0),
        #         pos,
        #         pix_square_size / 3,
        #         5
        #     )
        #
        # for i in observation['blue_team']:
        #     pos = self.to_pygame_coords(i['pos']) * pix_square_size
        #     pos += int(pix_square_size / 2)
        #     # Draw circle and border
        #     pygame.draw.circle(
        #         canvas,
        #         (0, 0, 255),
        #         pos,
        #         pix_square_size / 3,
        #     )
        #     pygame.draw.circle(
        #         canvas,
        #         (0, 0, 0),
        #         pos,
        #         pix_square_size / 3,
        #         5
        #     )
        #
        # # Add active bullets to the grid (currently we don't display direction)
        # (red_bullet_positions, blue_bullet_positions), (
        # red_bullet_directions, blue_bullet_directions) = self.pyreason_grid_world.get_bullet_locations()
        # for red_pos, red_dir in zip(red_bullet_positions, red_bullet_directions):
        #     # Which dir the bullet should point
        #     if red_dir == 'up' or red_dir == 'down':
        #         idx = 1
        #     elif red_dir == 'left' or red_dir == 'right':
        #         idx = 0
        #     start_pos = self.to_pygame_coords(red_pos) * pix_square_size + int(pix_square_size / 2)
        #     end_pos = self.to_pygame_coords(red_pos) * pix_square_size + int(pix_square_size / 2)
        #     start_pos[idx] -= pix_square_size / 5
        #     end_pos[idx] += pix_square_size / 5
        #     pygame.draw.line(
        #         canvas,
        #         (255, 0, 0),
        #         start_pos,
        #         end_pos,
        #         10
        #     )
        #
        #     # Draw triangles at the end of each bullet
        #     if red_dir == 'up':
        #         tri_1 = [start_pos[0], start_pos[1] - pix_square_size / 8]
        #         tri_2 = [start_pos[0] + pix_square_size / 8, start_pos[1]]
        #         tri_3 = [start_pos[0] - pix_square_size / 8, start_pos[1]]
        #     elif red_dir == 'down':
        #         tri_1 = [end_pos[0], end_pos[1] + pix_square_size / 8]
        #         tri_2 = [end_pos[0] + pix_square_size / 8, end_pos[1]]
        #         tri_3 = [end_pos[0] - pix_square_size / 8, end_pos[1]]
        #     elif red_dir == 'left':
        #         tri_1 = [start_pos[0] - pix_square_size / 8, start_pos[1]]
        #         tri_2 = [start_pos[0], start_pos[1] + pix_square_size / 8]
        #         tri_3 = [start_pos[0], start_pos[1] - pix_square_size / 8]
        #     elif red_dir == 'right':
        #         tri_1 = [end_pos[0] + pix_square_size / 8, end_pos[1]]
        #         tri_2 = [end_pos[0], end_pos[1] + pix_square_size / 8]
        #         tri_3 = [end_pos[0], end_pos[1] - pix_square_size / 8]
        #
        #     pygame.draw.polygon(
        #         canvas,
        #         (255, 0, 0),
        #         (tri_1, tri_2, tri_3),
        #     )
        #
        # for blue_pos, blue_dir in zip(blue_bullet_positions, blue_bullet_directions):
        #     # Which dir the bullet should point
        #     if blue_dir == 'up' or blue_dir == 'down':
        #         idx = 1
        #     elif blue_dir == 'left' or blue_dir == 'right':
        #         idx = 0
        #     start_pos = self.to_pygame_coords(blue_pos) * pix_square_size + int(pix_square_size / 2)
        #     end_pos = self.to_pygame_coords(blue_pos) * pix_square_size + int(pix_square_size / 2)
        #     start_pos[idx] -= pix_square_size / 5
        #     end_pos[idx] += pix_square_size / 5
        #     pygame.draw.line(
        #         canvas,
        #         (0, 0, 255),
        #         start_pos,
        #         end_pos,
        #         10
        #     )
        #
        #     # Draw triangles at the end of each bullet
        #     if blue_dir == 'up':
        #         tri_1 = [start_pos[0], start_pos[1] - pix_square_size / 8]
        #         tri_2 = [start_pos[0] + pix_square_size / 8, start_pos[1]]
        #         tri_3 = [start_pos[0] - pix_square_size / 8, start_pos[1]]
        #     elif blue_dir == 'down':
        #         tri_1 = [end_pos[0], end_pos[1] + pix_square_size / 8]
        #         tri_2 = [end_pos[0] + pix_square_size / 8, end_pos[1]]
        #         tri_3 = [end_pos[0] - pix_square_size / 8, end_pos[1]]
        #     elif blue_dir == 'left':
        #         tri_1 = [start_pos[0] - pix_square_size / 8, start_pos[1]]
        #         tri_2 = [start_pos[0], start_pos[1] + pix_square_size / 8]
        #         tri_3 = [start_pos[0], start_pos[1] - pix_square_size / 8]
        #     elif blue_dir == 'right':
        #         tri_1 = [end_pos[0] + pix_square_size / 8, end_pos[1]]
        #         tri_2 = [end_pos[0], end_pos[1] + pix_square_size / 8]
        #         tri_3 = [end_pos[0], end_pos[1] - pix_square_size / 8]
        #
        #     pygame.draw.polygon(
        #         canvas,
        #         (0, 0, 255),
        #         (tri_1, tri_2, tri_3),
        #     )
        #
        # # Finally, add some gridlines
        # for x in range(self.grid_size + 1):
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (0, pix_square_size * x),
        #         (self.window_size, pix_square_size * x),
        #         width=3,
        #     )
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (pix_square_size * x, 0),
        #         (pix_square_size * x, self.window_size),
        #         width=3,
        #     )

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
