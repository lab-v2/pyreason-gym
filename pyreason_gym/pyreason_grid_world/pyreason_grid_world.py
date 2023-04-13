import os
import pyreason as pr


class PyReasonGridWorld:
    def __init__(self, size, num_agents_per_team):
        self.size = size
        self.num_agents_per_team = num_agents_per_team
        self.interpretation = None
        
        # Pyreason settings
        pr.settings.verbose = False
        pr.settings.atom_trace = False
        current_path = os.path.abspath(os.path.dirname(__file__))

        # Load the graph
        pr.load_graph(f'{current_path}/graph/game_graph.graphml')

        # Load rules
        pr.load_rules(f'{current_path}/yamls/rules.yaml')


    def reset(self):
        # Reason for 1 timestep to initialize everything
        self.interpretation = pr.reason(0, again=False)

    def move(self, action):
        # Define facts, then run pyreason
        # action input is a Dict with two keys, one for each team, consisting of a list of actions, one for each agent
        red_team_actions = action['red_team']
        blue_team_actions = action['blue_team']

        facts = []
        actions = {0:'moveUp', 1:'moveDown', 2:'moveLeft', 3:'moveRight'}
        for i, a in enumerate(red_team_actions):
            fact = pr.fact_node.Fact(f'red_action_{i+1}', f'red-soldier-{i+1}', pr.label.Label(actions[a]), pr.interval.closed(1,1), 0, 0)
            facts.append(fact)
    
        for i, a in enumerate(blue_team_actions):
            fact = pr.fact_node.Fact(f'blue_action_{i+1}', f'blue-soldier-{i+1}', pr.label.Label(actions[a]), pr.interval.closed(1,1), 0, 0)
            facts.append(fact)

        self.interpretation = pr.reason(1, again=True, node_facts=facts)
    
    def get_obs(self):
        observation = {'red_team': [], 'blue_team': []}

        # Filter edges that are of the form (red-soldier-x, y) where x and y are ints
        red_relevant_edges = [edge for edge in self.interpretation.edges if 'red-soldier' in edge[0] and edge[1].isnumeric()]
        blue_relevant_edges = [edge for edge in self.interpretation.edges if 'blue-soldier' in edge[0] and edge[1].isnumeric()]
        
        # Select edges that have the atLoc predicate set to [1,1]
        red_position_edges = [edge for edge in red_relevant_edges if self.interpretation.interpretations_edge[edge].world[pr.label.Label('atLoc')]==pr.interval.closed(1,1)]
        blue_position_edges = [edge for edge in blue_relevant_edges if self.interpretation.interpretations_edge[edge].world[pr.label.Label('atLoc')]==pr.interval.closed(1,1)]
        print('position edges', red_position_edges, blue_position_edges)

        # Make sure that the length of these lists are the same as the num agents per team
        assert len(red_position_edges)==self.num_agents_per_team and len(blue_position_edges)==self.num_agents_per_team, 'Number of agents per team does not match info retrieved about agent position from interpretations'

        # Sort the lists according to the last char of the source node (eg. red-soldier-1, 1 is the last char)
        red_position_edges = sorted(red_position_edges, key=lambda x: str(x[0][-1]))
        blue_position_edges = sorted(blue_position_edges, key=lambda x: str(x[0][-1]))


        for i in range(1, self.num_agents_per_team+1):
            red_pos = int(red_position_edges[i][1])
            blue_pos = int(blue_position_edges[i][1])
            red_health = self.interpretation.interpretations_node[f'red-soldier-{i}'].world['health'].lower
            blue_health = self.interpretation.interpretations_node[f'blue-soldier-{i}'].world['health'].lower
                    
            observation['red_team'].append({'pos': red_pos, 'health': red_health})
            observation['blue_team'].append({'pos': blue_pos, 'health': blue_health})
            print('obs', observation)
        return observation