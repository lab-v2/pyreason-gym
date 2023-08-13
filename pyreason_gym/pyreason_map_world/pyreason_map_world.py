import os
import pyreason as pr
import numpy as np
import networkx as nx


class PyReasonMapWorld:
    def __init__(self, start_point, end_point, graph_path, rules_path):
        self.graph_path = os.path.abspath(graph_path)
        self.interpretation = None
        self.start_point = start_point
        self.end_point = end_point

        # Store the lat/long of the end point
        self.end_point_lat = None
        self.end_point_long = None

        # Keep track of the next timestep to start
        self.next_time = 0
        self.steps = 0

        # Edges that we add to the graph to represent a trajectory
        self.edges_added = []

        # Pyreason settings
        pr.settings.verbose = False
        pr.settings.atom_trace = False
        pr.settings.canonical = True
        pr.settings.inconsistency_check = False
        pr.settings.static_graph_facts = False
        pr.settings.parallel_computing = True

        # Load the graph
        pr.load_graphml(self.graph_path)

        # Load rules
        pr.add_rules_from_file(os.path.abspath(rules_path), infer_edges=True)

    def reset(self):
        # Reason for 1 timestep to initialize everything
        # Certain internal variables need to be reset otherwise memory blows up
        pr.reset()
        self._reset_graph()

        # Add facts for normal and abnormal to agent
        pr.add_fact(pr.Fact('normal-fact', 'agent', 'normal', [0, 1], 0, 0))
        pr.add_fact(pr.Fact('abnormal-fact', 'agent', 'abnormal', [0, 1], 0, 0))

        self.interpretation = pr.reason(0, again=False)
        self.next_time = self.interpretation.time + 1

        # Set initial position of agent
        self.interpretation.add_edge(('agent', self.start_point), pr.label.Label('atLoc'))
        self.interpretation.interpretations_edge[('agent', self.start_point)].world[pr.label.Label('atLoc')] = pr.interval.closed(1, 1)

        # Store the lat/long of the end point
        self.end_point_lat, self.end_point_long = self._get_lat_long(self.end_point)

    def move(self, action):
        # Define facts, then run pyreason
        # action input is a number corresponding to which path (edge from one node to another) the agent should take
        facts = []

        # Reset normal and abnormal bounds at each timestep
        reset_normal_fact = pr.fact_node.Fact(f'reset_normal_{self.steps}', 'agent', pr.label.Label('normal'), pr.interval.closed(0, 1), self.next_time, self.next_time)
        reset_abnormal_fact = pr.fact_node.Fact(f'reset_normal_{self.steps}', 'agent', pr.label.Label('abnormal'), pr.interval.closed(0, 1), self.next_time, self.next_time)
        facts.append(reset_normal_fact)
        facts.append(reset_abnormal_fact)

        # Do nothing on action 0
        if action != 0:
            fact_on = pr.fact_node.Fact(f'move_{self.steps}', 'agent', pr.label.Label(f'move_{action-1}'), pr.interval.closed(1, 1), self.next_time, self.next_time)
            fact_off = pr.fact_node.Fact(f'move_{self.steps}', 'agent', pr.label.Label(f'move_{action-1}'), pr.interval.closed(0, 0), self.next_time + 1, self.next_time + 1)
            facts.append(fact_on)
            facts.append(fact_off)

        self.interpretation = pr.reason(1, again=True, node_facts=facts)
        self.next_time = self.interpretation.time + 1
        self.steps += 1

    def get_obs(self):
        # Calculate current and end point lat longs
        relevant_edges = [edge for edge in self.interpretation.edges if edge[0] == 'agent' and self.interpretation.interpretations_edge[edge].world[pr.label.Label('atLoc')] == pr.interval.closed(1,1)]
        assert len(relevant_edges) == 1, 'Agent cannot be in multiple places at once--mistake in the interpretation data'
        current_edge = relevant_edges[0]
        loc = current_edge[1]

        lat, long = self._get_lat_long(loc)

        current_lat_long = np.array([lat, long], dtype=np.float128)
        end_lat_long = np.array([self.end_point_lat, self.end_point_long], dtype=np.float128)

        # Get info about current action space
        # Get number of outgoing edges. New action space = num outgoing edges. The outgoing edges should not be connected to timesteps
        outgoing_edges = [edge for edge in self.interpretation.edges if edge[0] == loc and not (edge[1][0] == 't' and edge[1][1:].isdigit())]
        num_outgoing_edges = len(outgoing_edges)

        # Add trajectory to graph based on the loc of observation. This is done everytime get_obs is called
        self._add_trajectory_to_graph(loc)

        observation = (loc, current_lat_long, end_lat_long, num_outgoing_edges)
        return observation

    def _get_lat_long(self, node):
        world = self.interpretation.interpretations_node[node].world
        lat = None
        long = None
        for label, _ in world.items():
            # Represented internally by lat-x and long-y or latitude-x, longitude-y or x-x, y-y
            idx_lat = -1
            idx_long = -1
            if 'latitude-' == label._value[:9]:
                idx_lat = 9
            elif 'lat-' == label._value[:4]:
                idx_lat = 4
            elif 'y-' == label._value[:2]:
                idx_lat = 2
                 
            elif 'longitude-' == label._value[:10]:
                idx_long = 10
            elif 'long-' == label._value[:5]:
                idx_long = 5
            elif 'x-' == label._value[:2]:
                idx_long = 2

            if idx_lat != -1:
                lat = float(label._value[idx_lat:])
            if idx_long != -1:
                long = float(label._value[idx_long:])

        assert lat is not None and long is not None, 'Latitude or Longitude attributes were not found for this location'
        return lat, long

    def get_map(self):
        nodes = [node for node in self.interpretation.nodes if node != 'agent' and not (node[0] == 't' and node[1:].isdigit())]
        edges = [edge for edge in self.interpretation.edges if edge[0] in nodes and edge[1] in nodes]

        # Return list of nodes (landmarks/stops) and list of edges connecting these points
        nodes_lat_long = [(self._get_lat_long(node)) for node in nodes]
        edges_lat_long = [((self._get_lat_long(edge[0])), (self._get_lat_long(edge[1]))) for edge in edges]
        return nodes_lat_long, edges_lat_long
    
    def get_normal_abnormal(self):
        normal_bnd = self.interpretation.interpretations_node['agent'].world[pr.label.Label('normal')]
        abnormal_bnd = self.interpretation.interpretations_node['agent'].world[pr.label.Label('abnormal')]
        return normal_bnd, abnormal_bnd

    def _add_trajectory_to_graph(self, loc):
        # This comes from get_obs and adds a location to the trajectory
        time = 't' + str(self.interpretation.time)
        edge1 = ('agent', loc)
        edge2 = (loc, time)

        # Add edges to location and timestep
        self.interpretation.add_edge(edge1, pr.label.Label('passed_by'))
        self.interpretation.add_edge(edge2, pr.label.Label('timestep'))
        self.interpretation.interpretations_edge[edge1].world[pr.label.Label('passed_by')] = pr.interval.closed(1, 1)
        self.interpretation.interpretations_edge[edge2].world[pr.label.Label('timestep')] = pr.interval.closed(1, 1)
        if edge1 not in self.edges_added:
            self.edges_added.append(edge1)
        if edge2 not in self.edges_added:
            self.edges_added.append(edge2)

    def _reset_graph(self):
        # This function removes any trajectory that was added during step when reset is called
        for edge in self.edges_added:
            self.interpretation.delete_edge(edge)

        self.edges_added.clear()

    def get_max_min_lat_long(self, lat_long_scale):
        g = nx.read_graphml(self.graph_path)
        max_lat = float('-inf')
        max_long = float('-inf')
        min_lat = float('inf')
        min_long = float('inf')

        for node in g.nodes(data=True):
            if 'latitude' in node[1].keys():
                max_lat = max(max_lat, float(node[1]['latitude']))
                min_lat = min(min_lat, float(node[1]['latitude']))
            if 'longitude' in node[1].keys():
                max_long = max(max_long, float(node[1]['longitude']))
                min_long = min(min_long, float(node[1]['longitude']))

        max_lat = int(max_lat * lat_long_scale)
        max_long = int(max_long * lat_long_scale)
        min_lat = int(min_lat * lat_long_scale)
        min_long = int(min_long * lat_long_scale)

        return max_lat, max_long, min_lat, min_long
