import networkx as nx


def generate_graph(grid_dim, num_agents_per_team, base_loc, start_loc, obstacle_loc):
    # Check parameters
    assert len(base_loc) == 2, 'There are only two bases--supply two locations'
    assert len(start_loc) == 2, 'There are only two teams--supply lists of start positions for each team in a nested list'
    assert len(start_loc[0]) == num_agents_per_team and len(start_loc[1]) == num_agents_per_team, 'Supply correct number of start locations'

    g = nx.DiGraph()

    # Game variables
    game_height = grid_dim
    game_width = grid_dim

    # =======================================================================
    # Add a node for each grid location in the game
    nodes = list(range(0, game_height * game_width))
    g.add_nodes_from([str(node) for node in nodes], blocked='0,0')

    # =======================================================================
    # Add edges connecting each of the grid nodes. Add up, down, left, right attributes to correct edges
    # Right edges
    for node in g.nodes:
        if (int(node) + 1) % game_width != 0:
            g.add_edge(node, str(int(node) + 1), right=1)

    # Left edges
    for node in g.nodes:
        if int(node) % game_width != 0:
            g.add_edge(node, str(int(node) - 1), left=1)

    # Up edges
    for node in g.nodes:
        if (int(node) // game_width) + 1 != game_height:
            g.add_edge(node, str(int(node) + game_width), up=1)

    # Down edges
    for node in g.nodes:
        if int(node) // game_width != 0:
            g.add_edge(node, str(int(node) - game_width), down=1)

    # Add edges between border nodes and end nodes to demarcate the end of the grid world. Bullets will disappear after it crosses this end
    g.add_node('end', blocked=1)
    # Bottom border
    for i in range(game_width):
        g.add_edge(f'{i}', 'end', down=1)
    # Top border
    for i in range(game_width):
        g.add_edge(f'{i + game_width * (game_height - 1)}', 'end', up=1)
    # Left border
    for i in range(game_height):
        g.add_edge(f'{i * game_width}', 'end', left=1)
    # Right border
    for i in range(game_height):
        g.add_edge(f'{i * game_width + game_width - 1}', 'end', right=1)

    # =======================================================================
    # Add the bases and connect them to the correct location: bottom right and top left
    g.add_node('red-base')
    g.add_node('blue-base')
    g.add_edge('red-base', str(base_loc[0]), atLoc=1)
    g.add_edge('blue-base', str(base_loc[1]), atLoc=1)

    # =======================================================================
    # Add mountains and obstacle attributes
    mountain_loc = obstacle_loc
    g.add_node('mountain', isMountain=1)
    for i in mountain_loc:
        g.add_edge(str(i), 'mountain', atLoc=1)

    # =======================================================================
    # Initialize players health, action choice and team
    for i in range(1, num_agents_per_team + 1):
        g.add_node(f'red-soldier-{i}', health=1, moveUp=0, moveDown=0, moveLeft=0, moveRight=0, shootUpRed=0,
                   shootDownRed=0, shootLeftRed=0, shootRightRed=0, teamRed=1, justDied='0,0')
        g.add_node(f'blue-soldier-{i}', health=1, moveUp=0, moveDown=0, moveLeft=0, moveRight=0, shootUpBlue=0,
                   shootDownBlue=0, shootLeftBlue=0, shootRightBlue=0, teamBlue=1, justDied='0,0')
        # Teams
        g.add_edge(f'red-soldier-{i}', 'red-base', team=1)
        g.add_edge(f'blue-soldier-{i}', 'blue-base', team=1)
        # Soldier Start Locations (dual edge)
        g.add_edge(f'red-soldier-{i}', str(start_loc[0][i - 1]), atLoc=1)
        g.add_edge(f'blue-soldier-{i}', str(start_loc[1][i - 1]), atLoc=1)
        g.add_edge(str(start_loc[0][i - 1]), f'red-soldier-{i}', atLoc=1)
        g.add_edge(str(start_loc[1][i - 1]), f'blue-soldier-{i}', atLoc=1)
        # Bullets
        g.add_node(f'red-bullet-{i}', teamRed=1, bullet=1)
        g.add_node(f'blue-bullet-{i}', teamBlue=1, bullet=1)
        g.add_edge(f'red-soldier-{i}', f'red-bullet-{i}', bullet=1)
        g.add_edge(f'blue-soldier-{i}', f'blue-bullet-{i}', bullet=1)

    # =======================================================================

    nx.write_graphml_lxml(g, 'pyreason_gym/pyreason_grid_world/graph/game_graph.graphml', named_key_ids=True)


def main():
    ## Red is first then Blue
    generate_graph(grid_dim=30, num_agents_per_team=1, base_loc=[7, 56], start_loc=[[0], [2]],
                   obstacle_loc=[])


if __name__ == '__main__':
    main()
