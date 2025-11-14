from __future__ import annotations
import hydra
import sys
import numpy as np
import time

from omegaconf import DictConfig
from hydra.utils import to_absolute_path



from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml

from env.load_map import load_map
from persona.load_personas import load_agent_configs
from env.grid import MultiHumanGridEnv
from env.constants import Action 

# this might be temporary map until we move this high-level to the other cognitive models
from persona.cognitive.plan import high_level_planner, astar
from env.constants import SemanticMap
from persona.cognitive.perceive import describe_perception



@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):
    """
    Loads map and runs the MiniGrid environment defined in JSON.
    """
    map_path = to_absolute_path(cfg.map.file)
    personas_path = to_absolute_path(cfg.personas.file)
    map_spec = load_map(map_path)
    agent_configs = load_agent_configs(personas_path)

    env = MultiHumanGridEnv(
        map_spec=map_spec,
        agent_configs=agent_configs,
        max_steps=cfg.sim.steps,
        render_mode=cfg.sim.render_mode,
    )

    print("[Env] Resetting...")
    obs, info = env.reset()
    print("Initial state:")
    env.render()



    for t in range(cfg.sim.steps):


        # this will go to plannner eventially
        # FOR NOW WE have two agents only so this hardcoding works but eventually change to env.agent_ids
        start0, goal0 = high_level_planner(env, agent_id=0, command="go to work")
        start1, goal1 = high_level_planner(env, agent_id=1, command="go to park")
        full_path0 = astar(env, start0, goal0)
        full_path1 = astar(env, start1, goal1)

        # in here we are shortening their low level planner path based on their fov
        fov0 = env.agents[0].config.fov.range_cells
        fov1 = env.agents[1].config.fov.range_cells
        path0 = full_path0[:fov0]
        path1 = full_path1[:fov1]

        for step in range(max(len(path0), len(path1))):
            a0 = path0[step] if step < len(path0) else Action.STAY
            a1 = path1[step] if step < len(path1) else Action.STAY

            action = np.array([a0, a1], dtype=np.int64)
            obs, _, terminated, truncated, info = env.step(action)

            print(f"\nStep {t + 1}, Sub-step {step + 1}, action={action}")
            for agent_id in range(env.num_agents):
                desc = describe_perception(env, agent_id)
                print(f"[Agent {agent_id}] {desc}")

            env.render()
            time.sleep(0.5)

            if terminated or truncated:
                break

    env.close()
   

if __name__ == "__main__":
    run()

