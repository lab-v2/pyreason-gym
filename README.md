# Map World for IARPA-HAYSTAC

## Installation

### PyReason
We use the `update-average` branch of pyreason
```bash
git clone https://github.com/lab-v2/pyreason
cd pyreason
git checkout update-average
pip install .
python
>>> import pyreason
```
The `import pyreason` is to initialize all caches and will take a few minutes to complete

### PyReason Gym
We use the `haystac-map-world` branch of pyreason-gym
```bash
git clone https://github.com/lab-v2/pyreason-gym
cd pyreason-gym
git checkout haystac-map-world
cd ..
pip install -e pyreason-gym
```

### Place the graph inside pyreason-gym
Download the full graph from [this link](https://drive.google.com/file/d/1SyMiVpaWePKgoQfyxQ6IhpAyy37e1WtE/view?usp=drive_link).

Download the sub-graph from [this link](). (If you use the subgraph, you'll have to change the name of the graph being loaded in [pyreason_map_world.py](https://github.com/lab-v2/pyreason-gym/blob/haystac-map-world/pyreason_gym/pyreason_map_world/pyreason_map_world.py#L33))

Then copy it into the directory `pyreason-gym/pyreason_gym/pyreason_map_world/graph/`

## Actions
1. `0` is for no action
2. `1` - `n` (where `n` is the number of outgoing edges in the current state) is for taking a path that corresponds to the selected number

## Termination
The game is terminated when the agent reaches the end point or if the agent has taken more than 200 steps (in which case negative 10 reward will be given)

## Input
The name of the start node and name of the end node has to be passed to `env.make(...)`. See [`test.py`](/test.py) as an example
