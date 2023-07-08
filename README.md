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
Download the graph from [this link](https://drive.google.com/file/d/1SyMiVpaWePKgoQfyxQ6IhpAyy37e1WtE/view?usp=drive_link)
Then copy it into the directory `pyreason-gym/pyreason_gym/pyreason_map_world/graph/`
