import gym
from gym import spaces
import numpy as np
import pygame

from pyreason_gym.pyreason_map_world.pyreason_map_world import PyReasonMapWorld

np.set_printoptions(precision=20)

# Odd order is due to orientation on canvas while displaying
LAT_LONG_SCALE = int(10e14)
PYGAME_MIN = 30
PYGAME_MAX = 800
RENDER_BUFFER = 50


class MapWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, start_point, end_point, graph_path, rules_path, render_mode=None, graph_auth=None):
        """Initialize map world

        :param start_point: Point where agent will start
        :type start_point: str
        :param end_point: Point where agent should end
        :type end_point: str
        :param render_mode: how to render the environment, defaults to None
        :type render_mode: str or None
        :param graph_auth: Authorization for graph database in the for of a tuple with username and password for neo4j
        :type graph_auth: tuple[str, str]
        """
        super(MapWorldEnv, self).__init__()

        self.render_mode = render_mode
        self.window_size = PYGAME_MAX + RENDER_BUFFER

        # Start End points are required for observations
        self.start_point = str(start_point)
        self.end_point = str(end_point)

        # Rendering info
        self.start_point_lat_long = None
        self.end_point_lat_long = None

        # Set graph type local or remote (graphml or neo4j)
        self.graph_type = 'local' if graph_auth is None else 'remote'

        # Initialize the PyReason map-world
        self.pyreason_map_world = PyReasonMapWorld(self.start_point, self.end_point, graph_path, rules_path, graph_auth)

        # Observation space is how close/far it is to the goal point. Coordinates from current point to end point
        # And how many valid actions there are in the state
        self.observation_space = spaces.Tuple((spaces.Text(max_length=20), spaces.Box(-180, 180, shape=(2,), dtype=np.float128), spaces.Box(-180, 180, shape=(2,), dtype=np.float128), spaces.Discrete(100)))

        # The choice of action is limited to the number of outgoing edges from one node. The agent has to pick one edge to go on
        self.action_space = spaces.Discrete(1)

        # The total distance travelled by the agent
        self.distance_travelled = 0
        self.current_lat_long = None
        self.end_lat_long = None

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
        normal_bnd, abnormal_bnd = self.pyreason_map_world.get_normal_abnormal()
        reward = (normal_bnd.lower / normal_bnd.upper) - (abnormal_bnd.lower / abnormal_bnd.upper)
        return reward

    def _get_a_star_heuristic(self):
        # normal_bnd, abnormal_bnd = self.pyreason_map_world.get_normal_abnormal()
        # reward = (normal_bnd.lower / normal_bnd.upper) - (abnormal_bnd.lower / abnormal_bnd.upper)

        # This function returns the reward for A* search f(n) = g(n) + h(n)
        # Where g(n) is the distance from the start node to the current node and h(n) is the heuristic function
        # For now h(n) will represent the manhattan distance from the current node to the end node.
        g_n = self.distance_travelled
        h_n = self._get_distance_in_between(self.current_lat_long, self.end_lat_long)
        reward = g_n + h_n
        return reward

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

        self.distance_travelled = 0
        self.current_lat_long = observation[1]
        self.end_lat_long = observation[2]

        # Save new action space
        self.set_action_space(observation)

        # Render if necessary
        if self.render_mode == "human":
            self._render_frame(observation)

        return observation, info

    def step(self, action):
        assert action in self.action_space, f'The selected action ({action}) is not in the current action space ({self.action_space})'
        self.pyreason_map_world.move(action)

        observation = self._get_obs()
        info = self._get_info()

        # Get reward
        rew = self._get_rew()

        # End of game
        done, truncated = self.is_done(observation)
        rew = -10 if truncated else rew

        # Save new action space
        self.set_action_space(observation)

        # Update distance travelled and current lat long
        new_lat_long = observation[1]
        self.distance_travelled += self._get_distance_in_between(self.current_lat_long, new_lat_long)
        self.current_lat_long = new_lat_long

        # Render if necessary
        if self.render_mode == "human":
            self._render_frame(observation)

        return observation, rew, done, truncated, info

    def set_action_space(self, observation):
        _, _, _, new_action_space = observation
        self.action_space = spaces.Discrete(new_action_space+1)

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
                self._map_lat_long_to_pygame_coords(*node),
                2
            )

        # Draw edges between points
        for edge in edges_lat_long:
            pygame.draw.aaline(
                self.canvas,
                (169, 169, 169),
                self._map_lat_long_to_pygame_coords(*edge[0]),
                self._map_lat_long_to_pygame_coords(*edge[1])
            )

        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])

    def _render_frame(self, observation):
        # If it's a local graph we want to create a canvas that contains all the node/edge data. If remote we just want a blank canvas
        if self.canvas is None and self.graph_type == 'local':
            self._render_init()
        if self.canvas is None and self.graph_type == 'remote':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.canvas = pygame.Surface((self.window_size, self.window_size))
            self.canvas.fill((255, 255, 255))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = self.canvas.copy()

        # For neo4j graphs we have to re-render everytime
        if self.graph_type == 'remote':
            nodes_lat_long, edges_lat_long = self.pyreason_map_world.get_map()

            # Draw points for nodes
            for node in nodes_lat_long:
                pygame.draw.circle(
                    canvas,
                    (69, 69, 69),
                    self._map_lat_long_to_pygame_coords(*node),
                    2
                )

            # Draw edges between points
            for edge in edges_lat_long:
                pygame.draw.aaline(
                    canvas,
                    (169, 169, 169),
                    self._map_lat_long_to_pygame_coords(*edge[0]),
                    self._map_lat_long_to_pygame_coords(*edge[1])
                )

        current_node, current_lat_long, end_lat_long, new_action_space = observation

        if self.start_point_lat_long is None:
            self.start_point_lat_long = current_lat_long
        if self.end_point_lat_long is None:
            self.end_point_lat_long = end_lat_long

        # Draw start and end nodes
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            self._map_lat_long_to_pygame_coords(self.start_point_lat_long[0], self.start_point_lat_long[1]),
            5,
        )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self._map_lat_long_to_pygame_coords(current_lat_long[0], current_lat_long[1]),
            5,
        )
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            self._map_lat_long_to_pygame_coords(self.end_point_lat_long[0], self.end_point_lat_long[1]),
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
        
    def _map_lat_long_to_pygame_coords(self, lat, long):
        # Map from lat long range to pygame range
        lat = int(lat * LAT_LONG_SCALE)
        long = int(long * LAT_LONG_SCALE)

        max_lat = int(self.pyreason_map_world.max_lat * LAT_LONG_SCALE)
        max_long = int(self.pyreason_map_world.max_long * LAT_LONG_SCALE)
        min_lat = int(self.pyreason_map_world.min_lat * LAT_LONG_SCALE)
        min_long = int(self.pyreason_map_world.min_long * LAT_LONG_SCALE)

        # delta is so that the nodes don't appear on the borders
        delta = RENDER_BUFFER - 20

        lat_range = (max_lat - min_lat) if max_lat != min_lat else RENDER_BUFFER
        long_range = (max_long - min_long) if max_long != min_long else RENDER_BUFFER
        pygame_range = (PYGAME_MAX - PYGAME_MIN)
        new_lat = (((lat - min_lat) * pygame_range) / lat_range) + PYGAME_MIN + delta
        new_long = (((long - min_long) * pygame_range) / long_range) + PYGAME_MIN + delta
        coord = np.array([new_long, new_lat])

        return coord

    @staticmethod
    def _get_distance_in_between(lat_long_1, lat_long_2):
        # This returns the distance in lat long units (we are not converting to metric here)
        return abs(lat_long_1[0] - lat_long_2[0]) + abs(lat_long_1[1] - lat_long_2[1])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        self.pyreason_map_world.graph_db.close()

    def is_done(self, observation):
        # End the game when the agent reaches the end point
        if observation[0] == self.end_point:
            done = True
            if self.pyreason_map_world.next_time > 200:
                truncated = True
            else:
                truncated = False
        else:
            done = False
            truncated = False

        return done, truncated
