# CS294-286 Final Project — MiniGrid Fire Evacuation Simulator

This project builds a **MiniGrid-based simulation** that loads **floorplan-style JSON maps** and visualizes fire evacuation environments with:
- custom walls, corridors, and goals (safe zones),
- color-coded tiles defined entirely in JSON,
- agent spawn positions, and
- easy Hydra configuration for running experiments.

---

## Project Structure
```
cs294_286_final_project/
├── build_map.py # Builds MiniGrid env from JSON
├── load_map.py # Parses JSON and returns geometry
├── main.py # Hydra-based entry point
├── configs/
│ ├── config.yaml # Hydra config (sim + map)
│ └── maps/
│ └── simple.json



```

###  Create and activate a virtual environment