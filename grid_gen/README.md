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
├── configs
│   ├── config.yaml         # main config file to call personas and map
│   ├── maps
│   │   ├── simple.json
│   │   └── urbanWorld.json
│   └── personas
│       └── persona_spec.json
├── env
│   ├── build_map.py        # build the grid world based on the json file
│   ├── constants.py        # IDX of the different elements in the world
│   ├── load_map.py         # reads the json file
│   └── world_object.py
├── main.py
├── persona
│   ├── cognitive
│   │   ├── converse.py
│   │   ├── execute.py
│   │   ├── perceive.py
│   │   ├── plan.py                    
│   │   ├── reflect.py                 # a bit tricky this one - get memories and conversation and make decisios
│   │   └── retrieve.py                # a bit 
│   ├── memory
│   │   ├── associative_memory.py   # get the core long term memory, the one printed on sim_outputs
│   │   ├── scratch.py              # gets all the different elements about the persona described in the .json file
│   │   └── spatial_memory.py       # this sould get info about what the agent remembers about the environemnt, objects, etc
│   └── promp_templates             # we want to create all the parser files
├── README.md
├── requirements.txt
└── sim_outputs             # we want to output here the memories the person creates


```

###  Create and activate a virtual environment