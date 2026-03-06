# Refactored SmallVille

A minimal pygame-based viewer/runner for the SmallVille environment from [Generative Agents](https://github.com/joonspk-research/generative_agents). Replaces the original Django + Phaser.js frontend with a standalone pygame application.

## Features

- Renders the original SmallVille Tiled tilemap with all visual layers (ground, buildings, furniture, etc.)
- Loads collision data from the original CSVs
- Agents displayed as colored circles with initials, moving on the tile grid
- Camera follows agents with smooth interpolation, or free camera with arrow keys
- A\* pathfinding over the collision grid
- Gymnasium-compatible environment interface (`SmallvilleEnv`)
- Optional video recording

## Prerequisites

1. Clone the generative\_agents repo at `../generative_agents/`:
   ```bash
   cd .. && git clone https://github.com/joonspk-research/generative_agents.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
# Interactive camera mode (arrow keys / WASD to pan, ESC to quit)
python main.py

# Demo with random-walking agents
python main.py --demo

# Record to video
python main.py --record simulation.mp4 --steps 300

# Custom viewport size
python main.py --demo --width 1920 --height 1080

# Disable camera follow (free camera)
python main.py --demo --no-follow
```

## Architecture

```
refactored_smallville/
├── main.py                 # Entry point + demo/recording modes
├── requirements.txt
├── env/
│   ├── __init__.py
│   ├── constants.py        # Action enum, AgentConfig, collision block ID
│   ├── world_object.py     # SmallvilleAgent dataclass
│   ├── tilemap.py          # Tiled JSON loader + pygame renderer
│   └── smallville_env.py   # Gymnasium environment + A* pathfinding
```

The `TiledMapRenderer` loads the Tiled JSON export (`the_ville_jan7.json`) and all 18 tilesets,
pre-renders background and foreground layers into pygame surfaces. `SmallvilleEnv` wraps this
with a gymnasium step/reset interface and handles agent movement, collision, rendering, and A\*.
