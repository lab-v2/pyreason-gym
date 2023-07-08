import os
import pyreason as pr
import numpy as np
import time


class PyReasonMapWorld:
    def __init__(self, start_point, end_point):
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
        pr.settings.atom_trace = True
        pr.settings.canonical = True
        pr.settings.inconsistency_check = False
        pr.settings.static_graph_facts = False
        current_path = os.path.abspath(os.path.dirname(__file__))

        # Load the graph
        pr.load_graphml(f'{current_path}/graph/map_graph.graphml')

        # Load rules
        # pr.load_rules(f'{current_path}/yamls/rules.yaml')
        pr.add_rules_from_file(f'{current_path}/yamls/rules.txt')

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
        self.interpretation._add_edge('agent', self.start_point, self.interpretation.neighbors, self.interpretation.reverse_neighbors, self.interpretation.nodes, self.interpretation.edges, pr.label.Label('atLoc'), self.interpretation.interpretations_node, self.interpretation.interpretations_edge)
        self.interpretation.interpretations_edge[('agent', self.start_point)].world[pr.label.Label('atLoc')] = pr.interval.closed(1, 1)

        # Store the lat/long of the end point
        self.end_point_lat, self.end_point_long = self._get_lat_long(self.end_point)

    def move(self, action):
        # Define facts, then run pyreason
        # action input is a number corresponding to which path (edge from one node to another) the agent should take
        facts = []
        fact_on = pr.fact_node.Fact(f'move_{self.steps}', 'agent', pr.label.Label(f'move_{action}'), pr.interval.closed(1, 1), self.next_time, self.next_time)
        fact_off = pr.fact_node.Fact(f'move_{self.steps}', 'agent', pr.label.Label(f'move_{action}'), pr.interval.closed(0, 0), self.next_time + 1, self.next_time + 1)
        facts.append(fact_on)
        facts.append(fact_off)

        self.interpretation = pr.reason(2, again=True, node_facts=facts)
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
        # Get number of outgoing edges. New action space = num outgoing edges
        outgoing_edges = [edge for edge in self.interpretation.edges if edge[1] == current_edge[1] and edge[0] != 'agent']
        num_outgoing_edges = len(outgoing_edges)

        # Add trajectory to graph based on the loc of observation. This is done everytime get_obs is called
        self._add_trajectory_to_graph(loc)

        observation = (loc, current_lat_long, end_lat_long, num_outgoing_edges)
        return observation

    def _get_lat_long(self, node):
        world = self.interpretation.interpretations_node[node].world
        lat = None
        long = None
        for label, interval in world.items():
            # Represented internally by lat-x and long-y
            if 'lat' in label._value:
                lat = float(label._value[4:])
            elif 'long' in label._value:
                long = float(label._value[5:])

        return lat, long

    def get_map(self):
        nodes = [node for node in self.interpretation.nodes if node != 'agent']
        edges = [edge for edge in self.interpretation.edges if edge[0] != 'agent']

        # Return list of nodes (landmarks/stops) and list of edges connecting these points
        nodes_lat_long = [(self._get_lat_long(node)) for node in nodes]
        edges_lat_long = [((self._get_lat_long(edge[0])), (self._get_lat_long(edge[1]))) for edge in edges]
        return nodes_lat_long, edges_lat_long

    def _add_trajectory_to_graph(self, loc):
        # This comes from get_obs and adds a location to the trajectory
        # TODO: Revise this time and change reason to 1 timestep
        time = str(self.interpretation.time)
        print(time)
        edge1 = ('agent', loc)
        edge2 = (loc, time)

        self.interpretation._add_edge(edge1[0], edge1[1], self.interpretation.neighbors, self.interpretation.reverse_neighbors, self.interpretation.nodes, self.interpretation.edges, pr.label.Label('passed_by'), self.interpretation.interpretations_node, self.interpretation.interpretations_edge)
        self.interpretation._add_edge(edge2[0], edge2[1], self.interpretation.neighbors, self.interpretation.reverse_neighbors, self.interpretation.nodes, self.interpretation.edges, pr.label.Label('timestep'), self.interpretation.interpretations_node, self.interpretation.interpretations_edge)
        self.interpretation.interpretations_edge[edge1].world[pr.label.Label('passed_by')] = pr.interval.closed(1, 1)
        self.interpretation.interpretations_edge[edge2].world[pr.label.Label('timestep')] = pr.interval.closed(1, 1)
        self.edges_added.append(edge1)
        self.edges_added.append(edge2)

    def _reset_graph(self):
        # This function removes any trajectory that was added during step when reset is called
        for edge in self.edges_added:
            self.interpretation.edges.remove(edge)
            del self.interpretation.interpretations_edge[edge]

        self.edges_added.clear()
