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

import os
import datetime


# we can move these logger functions to other folder later
def setup_sim_output_dir():
    """
    Creates a run folder in sim_outputs/ with timestamp.
    Returns the folder path.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "sim_outputs"
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def create_agent_logs(run_dir, env):
    """
    Creates one text file per agent using their name.
    Returns a dict: agent_id -> file_handle
    """
    logs = {}
    for agent_id, agent in enumerate(env.agents):
        safe_name = agent.config.name.replace(" ", "_")
        path = os.path.join(run_dir, f"{safe_name}.txt")
        logs[agent_id] = open(path, "w")
        logs[agent_id].write(f"Log for {agent.config.name}\n")
        logs[agent_id].write("=" * 40 + "\n\n")
    return logs

def log_agent_step(log_file, agent_id, step, substep, agent, action, desc):
    """

    """
    log_file.write(f"Step {step}, Sub-step {substep}\n")
    log_file.write(f" Position: ({agent.x}, {agent.y})\n")
    log_file.write(f" Action: {action}\n")
    log_file.write(f" Observation: {desc}\n")
    log_file.write("-" * 30 + "\n")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):
    """
    Loads map and runs the MiniGrid environment defined in JSON.
    """
    map_path = to_absolute_path(cfg.map.file)
    personas_path = to_absolute_path(cfg.personas.file)
    map_spec = load_map(map_path)
    print("Unique access codes in grid:", np.unique(map_spec.access_grid))
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
    run_dir = setup_sim_output_dir()
    agent_logs = create_agent_logs(run_dir, env)


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
                agent = env.agents[agent_id]
                log_agent_step(
                    log_file=agent_logs[agent_id],
                    agent_id=agent_id,
                    step=t + 1,
                    substep=step + 1,
                    agent=env.agents[agent_id],
                    action=int(action[agent_id]),
                    desc=desc,
                )
                # testing the spatial memoery
                sm = agent.config.spatial_memory
                if not sm:
                    continue

                x, y = int(agent.x), int(agent.y)

                info = sm.elements_at_position(env, x, y)
                contents = info["contents"]

                if contents:
                    rooms_here = list(contents.keys())
                    print(
                        f"[SM] Agent {agent_id} at ({x}, {y}) -> "
                        f"cell='{info['cell_name']}', lookup_key='{info.get('lookup_key')}', rooms={rooms_here}")

                else:
                    print(
                        f"[SM] Agent {agent_id} at ({x}, {y}) -> "
                        f"cell='{info['cell_name']}'")

                print(f"[Agent {agent_id}] {desc}")

                env.render()
                time.sleep(0.5)

            if terminated or truncated:
                break

    
    for f in agent_logs.values():
        f.close()
    env.close()
   

if __name__ == "__main__":
    run()

