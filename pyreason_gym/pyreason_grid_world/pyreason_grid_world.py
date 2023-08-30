import os
import pyreason as pr
import numpy as np


class PyReasonGridWorld:
    def __init__(self, grid_size, num_agents_per_team, graph, rules):
        self.grid_size = grid_size
        self.num_agents_per_team = num_agents_per_team
        self.interpretation = None

        # Keep track of the next timestep to start
        self.next_time = 0
        
        # Pyreason settings
        pr.settings.verbose = False
        pr.settings.atom_trace = False
        pr.settings.canonical = True
        pr.settings.inconsistency_check = False
        pr.settings.static_graph_facts = False
        pr.settings.store_interpretation_changes = False
        current_path = os.path.abspath(os.path.dirname(__file__))

        # Load the graph
        if graph is None:
            pr.load_graph(f'{current_path}/graph/game_graph.graphml')
        else:
            pr.load_graph(graph)

        # Load rules
        if rules is None:
            pr.load_rules(f'{current_path}/yamls/rules.yaml')
        else:
            pr.load_rules(rules)

    def reset(self):
        # Reason for 1 timestep to initialize everything
        # Certain internal variables need to be reset otherwise memory blows up
        pr.reset()
        self.interpretation = pr.reason(0, again=False)
        self.next_time = self.interpretation.time + 1

    def move(self, action):
        # Define facts, then run pyreason
        # action input is a Dict with two keys, one for each team, consisting of a list of actions, one for each agent
        red_team_actions = action['red_team']
        blue_team_actions = action['blue_team']

        facts = []
        red_available_actions = {0:'moveUp', 1:'moveDown', 2:'moveLeft', 3:'moveRight', 4:'shootUpRed', 5:'shootDownRed', 6:'shootLeftRed', 7:'shootRightRed'}
        blue_available_actions = {0:'moveUp', 1:'moveDown', 2:'moveLeft', 3:'moveRight', 4:'shootUpBlue', 5:'shootDownBlue', 6:'shootLeftBlue', 7:'shootRightBlue'}
        for i, a in enumerate(red_team_actions):
            if a != 8:
                fact_on = pr.fact_node.Fact(f'red_action_{i+1}', f'red-soldier-{i+1}', pr.label.Label(red_available_actions[a]), pr.interval.closed(1,1), self.next_time, self.next_time)
                fact_off = pr.fact_node.Fact(f'red_action_{i+1}_off', f'red-soldier-{i+1}', pr.label.Label(red_available_actions[a]), pr.interval.closed(0,0), self.next_time+1, self.next_time+1)
                facts.append(fact_on)
                facts.append(fact_off)
    
        for i, a in enumerate(blue_team_actions):
            if a != 8:
                fact_on = pr.fact_node.Fact(f'blue_action_{i+1}', f'blue-soldier-{i+1}', pr.label.Label(blue_available_actions[a]), pr.interval.closed(1,1), self.next_time, self.next_time)
                fact_off = pr.fact_node.Fact(f'blue_action_{i+1}_off', f'blue-soldier-{i+1}', pr.label.Label(blue_available_actions[a]), pr.interval.closed(0,0), self.next_time+1, self.next_time+1)
                facts.append(fact_on)
                facts.append(fact_off)
        
        self.interpretation = pr.reason(1, again=True, node_facts=facts)
        self.next_time = self.interpretation.time + 1

    def get_obs(self):
        observation = {'red_team': [], 'blue_team': [], 'red_bullets': [], 'blue_bullets': []}
        
        # Gather bullet info for red and blue bullets
        (red_bullet_positions, blue_bullet_positions), (red_bullet_directions, blue_bullet_directions), (red_killed_who, blue_killed_who) = self._get_bullet_info()
        for red_pos, red_dir in zip(red_bullet_positions, red_bullet_directions):
            observation['red_bullets'].append({'pos': red_pos, 'dir': red_dir})

        for blue_pos, blue_dir in zip(blue_bullet_positions, blue_bullet_directions):
            observation['blue_bullets'].append({'pos': blue_pos, 'dir': blue_dir})

        # Filter edges that are of the form (red-soldier-x, y) where x and y are ints
        red_relevant_edges = [edge for edge in self.interpretation.edges if 'red-soldier' in edge[0] and edge[1].isnumeric()]
        blue_relevant_edges = [edge for edge in self.interpretation.edges if 'blue-soldier' in edge[0] and edge[1].isnumeric()]

        # Select edges that have the atLoc predicate set to [1,1]
        red_position_edges = [edge for edge in red_relevant_edges if self.interpretation.interpretations_edge[edge].world[pr.label.Label('atLoc')]==pr.interval.closed(1,1)]
        blue_position_edges = [edge for edge in blue_relevant_edges if self.interpretation.interpretations_edge[edge].world[pr.label.Label('atLoc')]==pr.interval.closed(1,1)]

        # Make sure that the length of these lists are the same as the num agents per team
        assert len(red_position_edges)==self.num_agents_per_team and len(blue_position_edges)==self.num_agents_per_team, 'Number of agents per team does not match info retrieved about agent position from interpretations'

        # Sort the lists according to the last char of the source node (eg. red-soldier-1, 1 is the last char)
        red_position_edges = sorted(red_position_edges, key=lambda x: str(x[0][-1]))
        blue_position_edges = sorted(blue_position_edges, key=lambda x: str(x[0][-1]))

        # Gather info about the agents
        for i in range(1, self.num_agents_per_team+1):
            red_pos = int(red_position_edges[i-1][1])
            blue_pos = int(blue_position_edges[i-1][1])
            red_pos_coords = [red_pos%self.grid_size, red_pos//self.grid_size]
            blue_pos_coords = [blue_pos%self.grid_size, blue_pos//self.grid_size]
            red_health = self.interpretation.interpretations_node[f'red-soldier-{i}'].world[pr.label.Label('health')].lower
            blue_health = self.interpretation.interpretations_node[f'blue-soldier-{i}'].world[pr.label.Label('health')].lower

            observation['red_team'].append({'pos': np.array(red_pos_coords, dtype=np.int32), 'health': np.array([red_health], dtype=np.float32), 'killed': list(red_killed_who[i-1])})
            observation['blue_team'].append({'pos': np.array(blue_pos_coords, dtype=np.int32), 'health': np.array([blue_health], dtype=np.float32), 'killed': list(blue_killed_who[i-1])})

        return observation
    
    def get_obstacle_locations(self):
        # Return the coordinates of all the mountains in the grid to be able to draw them
        relevant_edges = [edge for edge in self.interpretation.edges if edge[1]=='mountain']
        obstacle_positions = [int(edge[0]) for edge in relevant_edges]
        obstacle_positions_coords = np.array([[pos%self.grid_size, pos//self.grid_size] for pos in obstacle_positions])
        return obstacle_positions_coords
    
    def get_base_locations(self):
        # Return the locations of the two bases
        relevant_edges = [edge for edge in self.interpretation.edges if 'base' in edge[0]]
        sorted_relevant_edges = [relevant_edges[0], relevant_edges[1]] if relevant_edges[0][0]=='red-base' else [relevant_edges[1], relevant_edges[0]]
        base_positions = [int(edge[1]) for edge in sorted_relevant_edges]
        base_positions_coords = np.array([[pos%self.grid_size, pos//self.grid_size] for pos in base_positions])
        return base_positions_coords

    def _get_bullet_info(self):
        # Return the location of red and blue bullets to be displayed on the grid
        relevant_edges = [edge for edge in self.interpretation.edges if 'bullet' in edge[1] and edge[0].isdigit()]
        filtered_edges = [edge for edge in relevant_edges if self.interpretation.interpretations_edge[edge].world[pr.label.Label('atLoc')] == pr.interval.closed(1,1)
                          and self.interpretation.interpretations_edge[edge].world[pr.label.Label('life')] == pr.interval.closed(1,1)]
        red_bullet_positions = [int(edge[0]) for edge in filtered_edges if 'red' in edge[1]]
        blue_bullet_positions = [int(edge[0]) for edge in filtered_edges if 'blue' in edge[1]]
        red_bullet_positions_coords = np.array([[pos%self.grid_size, pos//self.grid_size] for pos in red_bullet_positions])
        blue_bullet_positions_coords = np.array([[pos%self.grid_size, pos//self.grid_size] for pos in blue_bullet_positions])
        positions = (red_bullet_positions_coords, blue_bullet_positions_coords)

        # Get info about who killed whom. Stored in the form a list for every agent: (red-killer: [blue-casualties]) or (blue-killer: [red-casualties])
        kill_info_edges = [edge for edge in self.interpretation.edges if pr.label.Label('killed') in self.interpretation.interpretations_edge[edge].world
                           and self.interpretation.interpretations_edge[edge].world[pr.label.Label('killed')] == pr.interval.closed(1, 1)]
        kill_info_edges = sorted(kill_info_edges, key=lambda x: int(x[0][-1]))
        red_killed_who_tuple = [(int(edge[0][-1]), int(edge[1][-1])) for edge in kill_info_edges if 'red' in edge[0]]
        blue_killed_who_tuple = [(int(edge[0][-1]), int(edge[1][-1])) for edge in kill_info_edges if 'blue' in edge[0]]
        red_killed_who = [[] for _ in range(self.num_agents_per_team)]
        blue_killed_who = [[] for _ in range(self.num_agents_per_team)]

        for shooter, casualty in red_killed_who_tuple:
            red_killed_who[shooter-1].append(casualty)
        for shooter, casualty in blue_killed_who_tuple:
            blue_killed_who[shooter-1].append(casualty)

        who_killed_who = (red_killed_who, blue_killed_who)

        # Bullet direction of movement
        direction_map = {0.2: 0, 0.6: 1, 0.4: 2, 0.8: 3}
        red_bullet_directions = [direction_map[self.interpretation.interpretations_edge[edge].world[pr.label.Label('direction')].lower] for edge in filtered_edges if 'red' in edge[1]]
        blue_bullet_directions = [direction_map[self.interpretation.interpretations_edge[edge].world[pr.label.Label('direction')].lower] for edge in filtered_edges if 'blue' in edge[1]]
        directions = (red_bullet_directions, blue_bullet_directions)

        # Make sure the length of positions is the same as directions
        assert len(red_bullet_positions) == len(red_bullet_directions), 'Length of bullet positions does not math length of bullet directions'
        assert len(blue_bullet_positions) == len(blue_bullet_directions), 'Length of bullet positions does not math length of bullet directions'

        return positions, directions, who_killed_who
