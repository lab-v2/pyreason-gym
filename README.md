# PyReason Gym 🏋
An OpenAI gym wrapper for PyReason to use in a reinforcement learning Grid World setting.

<!-- Insert Image -->
![Grid World Demo](media/pyreason-gym-demo.gif)


## Table of Contents
  
* [Getting Started](#getting-started)
    * [Conda Environment](#conda-environment)
    * [Requirements](#requirements)
    * [Installation](#installation)
    * [Build](#build)
* [Usage](#usage)
* [Documentation](#documentation)
* [Contributing](#contributing)
* [License](#License)

## Getting Started
This is an OpenAI Gym environment for reinforcement learning in a grid world setting using [PyReason](https://github.com/lab-v2/pyreason) as a simulator.

### The Setting
1. There are two teams: Red and Blue
2. There are two bases: Red Base and Blue Base
3. There are a certain number of agents in each team

### The Actions
There are 5 actions an agent can take:

1. Move Up
2. Move Down
3. Move Left
4. Move Right
5. Shoot a bullet (not implemented yet)

### The Objective
The objecive of the game is to kill all enemy agents or make their `health=0`. The game will terminate (or signal `done=True` when this happens)

### Rewards
**The reward function is currently not defined** A Reward of `0` is given at each step. TODO: Create a reward function.

## Installation
Clone the repository, and install:
```bash
git clone https://github.com/lab-v2/pyreason-gym
pip install -e pyreason-gym
```

## Usage
This Grid World scenario needs a graph in GraphML format to run. A graph file has already been generated in the [graphs folder](./pyreason_gym/pyreason_grid_world/graph/). However if you wish to change certain parameters such as

1. Number of agents per team
2. Start locations of the agents
3. Obstacle locations in the grid
4. The Grid World size (height, width)
5. The locations of the Red and Blue bases

You will need to re-generate the graph file using the [`generate_graph.py`](./generate_graph.py) script by changing the appropriate parameters. This will generate the graph in the appropriate location for PyReason to find.

This is an OpenAI Gym custom environment. More on OpenAI Gym:

1. [Documentation](https://www.gymlibrary.dev/)
2. [GitHub Repo](https://github.com/openai/gym)

The interface is just like a normal Gym environment. To create an environment and start using it, insert the following into your Python script. Make sure you've [Installed](#installation) this package before this.

```python
import gym
import pyreason_gym

env = gym.make('PyReasonGridWorld-v0')

# Reset the environment
obs, _ = env.reset()

# Take a random action and get observation, rewards, done signal etc.
# This will sample a random action from the action space of the environment 
action = env.action_space.sample()
obs, rew, done, _, _ = env.step(action)

# Keep using `env.step(action)` and `env.reset()` to get observations and run the grid world game.
```

### Actions
The action space is currently a list for each team with discrete numbers representing each action:

1. Move Up is represented by `0`
2. Move Down is represented by `1`
3. Move Left is represented by `2`
4. Move Right is represented by `3`
5. Shoot is represented by `4` (**Not implemented yet**)

A sample action with `1` agent per team is of the form:
```python
# Sample action. The list will increase with the number of agents per team
action = {
    'red_team': [0],
    'blue_team': [2]
}

# Send the action to the environment
obs, rew, done, _, _ = env.step(action)
```

### Observations
Observations contain information about each player's position in the grid (`[x,y]`) as well as their `health`. A sample observation with `1` agent per team is a dictionary of the form:

```python
observation = {
    'red_team': [{'pos': [1,3], 'health': [1]}],
    'blue_team': [{'pos': [7,2], 'health': [1]}]
}
```
Information about positions and health can be extracted from this observation space.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Bibtex
If you used this software in your work please cite our paper
```
@inproceedings{aditya_pyreason_2023,
title = {{PyReason}: Software for Open World Temporal Logic},
booktitle = {{AAAI} Spring Symposium},
author = {Aditya, Dyuman and Mukherji, Kaustuv and Balasubramanian, Srikar and Chaudhary, Abhiraj and Shakarian, Paulo},
year = {2023}}
```

## License
This repository is licensed under [BSD-3-Clause](./LICENSE)

## Contact
Dyuman Aditya - dyuman.aditya@asu.edu

Kaustuv Mukherji - kmukher2@asu.edu

Paulo Shakarian - pshak02@asu.edu
