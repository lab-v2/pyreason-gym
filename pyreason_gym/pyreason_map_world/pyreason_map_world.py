import os
import pyreason as pr
import numpy as np
import networkx as nx
import neo4j


class PyReasonMapWorld:
    def __init__(self, start_point, end_point, graph_path, rules_path, graph_auth):
        self.graph_path = os.path.abspath(graph_path)
        self.interpretation = None
        self.start_point = start_point
        self.end_point = end_point

        # Store the lat/long of the end point
        self.end_point_lat = None
        self.end_point_long = None

        # Keep track of max lat longs for rendering
        self.max_lat = float('-inf')
        self.max_long = float('-inf')
        self.min_lat = float('inf')
        self.min_long = float('inf')

        # Keep track of the next timestep to start
        self.next_time = 0
        self.steps = 0

        # Edges that we add to the graph to represent a trajectory
        # Nodes that we added using graph queries
        self.edges_added = []
        self.nodes_added = []

        # Pyreason settings
        pr.settings.verbose = False
        pr.settings.atom_trace = False
        pr.settings.canonical = True
        pr.settings.inconsistency_check = False
        pr.settings.static_graph_facts = False
        pr.settings.parallel_computing = False
        # pr.settings.parallel_computing = True

        # Set graph type local or remote (graphml or neo4j)
        self.graph_type = 'local' if graph_auth is None else 'remote'

        # Setup neo4j graph db connection or graphml
        # Load the graph
        if self.graph_type == 'remote':
            self.graph_db = neo4j.GraphDatabase.driver(graph_path, auth=graph_auth)
            initial_graph = self._make_initial_graph(self.start_point, self.end_point)
            pr.load_graph(initial_graph)
        else:
            pr.load_graphml(graph_path)

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

        # Set the initial max min lat longs
        self._set_initial_max_min_lat_long(omit_end_point=False)

    def move(self, action):
        # Define facts, then run pyreason
        # action input is a number corresponding to which path (edge from one node to another) the agent should take
        facts = []

        # Reset normal and abnormal bounds at each timestep
        self.interpretation.interpretations_node['agent'].world[pr.label.Label('normal')].set_lower_upper(0, 1)
        self.interpretation.interpretations_node['agent'].world[pr.label.Label('abnormal')].set_lower_upper(0, 1)

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

        # Update max min lat longs
        self.max_lat = max(self.max_lat, lat)
        self.max_long = max(self.max_long, long)
        self.min_lat = min(self.min_lat, lat)
        self.min_long = min(self.min_long, long)

        current_lat_long = np.array([lat, long], dtype=np.float128)
        end_lat_long = np.array([self.end_point_lat, self.end_point_long], dtype=np.float128)

        # Get info about current action space
        # Get number of outgoing edges. New action space = num outgoing edges. The outgoing edges should not be connected to timesteps
        outgoing_edges = [edge for edge in self.interpretation.edges if edge[0] == loc and not (edge[1][0] == 't' and edge[1][1:].isdigit())]
        num_outgoing_edges = len(outgoing_edges)

        # Add trajectory to graph based on the loc of observation. This is done everytime get_obs is called
        self._add_trajectory_to_graph(loc)

        # Add new neighbors to graph when agent moves to a new location
        if self.graph_type == 'remote':
            self._add_neighbors_to_graph(loc)

        observation = (loc, current_lat_long, end_lat_long, num_outgoing_edges)
        return observation

    def _make_initial_graph(self, start_node, end_node):
        g = nx.DiGraph()

        # Add agent node
        attributes = {f'move_{i}': 0 for i in range(10)}
        g.add_node('agent', **attributes, agent=1)

        # Query for and add start node and end node
        result_start, _, _ = self.graph_db.execute_query(f'MATCH (n) WHERE ID(n) = {start_node} RETURN n')
        result_end, _, _ = self.graph_db.execute_query(f'MATCH (n) WHERE ID(n) = {end_node} RETURN n')
        record_start = result_start[0].data()
        record_end = result_end[0].data()
        g.add_node(start_node, **record_start['n'])
        g.add_node(end_node, **record_end['n'])
        return g

    def _add_neighbors_to_graph(self, node):
        # Query for all in/out neighbor nodes
        nodes_added = []
        result, _, _ = self.graph_db.execute_query(f'MATCH (s)-[*1]-(t) WHERE ID(s) = {node} RETURN t, ID(t)')
        for record in result:
            neighbor_id = str(record['ID(t)'])
            neighbor_attributes = record['t']
            nodes_added.append(neighbor_id)

            # Prepare attribute list
            neighbor_attribute_list = []
            for key, value in neighbor_attributes.items():
                neighbor_attribute_list.append(f'{key}-{value}')

            # Add nodes internally along with their attributes
            self.interpretation.add_node(neighbor_id, neighbor_attribute_list)
            for attrib in neighbor_attribute_list:
                self.interpretation.interpretations_node[neighbor_id].world[pr.label.Label(attrib)] = pr.interval.closed(1, 1)

        # Add path attribute and edges between neighbors and original node
        for i, node_added in enumerate(nodes_added):
            # Skip adding edge for a node added if it has an edge to the main node. This means we've already added path attributes to this edge
            if node_added not in self.interpretation.neighbors[node]:
                # Add bidirectional edge. The path attrib corresponds to how many neighbors the node has
                edge1 = (node, node_added)
                edge2 = (node_added, node)
                path_num_1 = len(self.interpretation.neighbors[node])
                path_num_2 = len(self.interpretation.neighbors[node_added])
                label_1 = pr.label.Label(f'path-{path_num_1}')
                label_2 = pr.label.Label(f'path-{path_num_2}')
                self.interpretation.add_edge(edge1, label_1)
                self.interpretation.add_edge(edge2, label_2)
                self.interpretation.interpretations_edge[edge1].world[label_1] = pr.interval.closed(1, 1)
                self.interpretation.interpretations_edge[edge2].world[label_2] = pr.interval.closed(1, 1)

        # Update the list of all nodes added
        self.nodes_added += nodes_added

    def _get_lat_long(self, node):
        assert node in self.interpretation.interpretations_node, f'node: {node} is not in the graph, change start/end node accordingly'
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
        nodes = [node for node in self.interpretation.nodes if node != 'agent']
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
        time = str(self.interpretation.time)
        edge = ('agent', loc)

        # Add edges to location and add predicates
        self.interpretation.add_edge(edge, pr.label.Label('passed_by'))
        self.interpretation.interpretations_edge[edge].world[pr.label.Label('passed_by')] = pr.interval.closed(1, 1)
        self.interpretation.interpretations_edge[edge].world[pr.label.Label(f'time-{time}')] = pr.interval.closed(1, 1)
        if edge not in self.edges_added:
            self.edges_added.append(edge)

    def _reset_graph(self):
        # This function removes any trajectory that was added during step when reset is called
        # This function will also reset the graph to its initial state during the use of a neo4j graph
        for edge in set(self.edges_added):
            self.interpretation.delete_edge(edge)

        for node in set(self.nodes_added):
            self.interpretation.delete_node(node)

        self.edges_added.clear()
        self.nodes_added.clear()

    def _set_initial_max_min_lat_long(self, omit_end_point=False):
        # Loop through nodes in pyreason graph and find the max min lat long
        for node in self.interpretation.graph.nodes(data=True):
            if omit_end_point and node[0] == self.end_point:
                continue
            print(node[0])
            if 'latitude' in node[1].keys():
                self.max_lat = max(self.max_lat, float(node[1]['latitude']))
                self.min_lat = min(self.min_lat, float(node[1]['latitude']))
            if 'longitude' in node[1].keys():
                self.max_long = max(self.max_long, float(node[1]['longitude']))
                self.min_long = min(self.min_long, float(node[1]['longitude']))
