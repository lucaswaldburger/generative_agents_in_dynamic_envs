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
        # FOR NOW WE have two agents only so this hardcoding works but eventually change to env.agent_ids
        start0, goal0 = high_level_planner(env, agent_id=0, command="go to work")
        start1, goal1 = high_level_planner(env, agent_id=1, command="go to park")
        path0 = astar(env, start0, goal0)
        path1 = astar(env, start1, goal1)
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
    # print(f"[Env] obs keys: {obs['image']}")clear


    # if cfg.sim.render_mode == "human":
    #     env.render()

    # done = False
    # steps = 0
    # while not done and steps < cfg.sim.steps:
    #     # action = np.random.randint(0, env.action_space.n)

    #     actions = {
    #         agent.index: agent.action_space.sample()
    #         for agent in env.agents
    #     }
    #     obs, rewards, terminations, truncations, infos = env.step(actions)

    #     # perceive -> store in memory -> plant ->reflect -> act -store in memory -> repeat
    #     # (1) we can instantiate and grab the state of the persona 
    #     # these next few lines might need to go into execute, they need to go after observe
    #     # for pid in env.agent_ids:
    #     #     if pid == "human_1":
    #     #         # pick an action from your plan/path
    #     #         # get_next_plan_text = "go to work"
    #     #         # current_location_text = "home_A"
    #     #         # path = get_next_plan_and_waypoint(current_location_text, get_next_plan_text, env)
    #     #         actions[pid] =  env.action_space[pid].sample()
    #     #     else:
    #     #         actions[pid] = env.action_space[pid].sample()

        
    #     # action = 0 # just for testing
    #     ## we should model each step in the grid because we need to record the observations
    #     # (2) perceive
    #     # for action in path: # we want to move this to a per step action but for not just testing
    #     #     obs, reward, term, trunc, info = env.step(action)
    #     #     print(obs.keys())
    #     #     vis = get_obs(env, include_world_coords=True, include_street=False)
    #     # # (3) store observations in the enviornment
    #     #     print(f"Step {steps}, action={action}")
    #     #     print_visible(vis)
    #     # obs_from_env = get_obs(obs, lookup_name=env.region_name)
    #     # print(f"Obs shape: {np.array(obs_from_env).shape}")
    #     # summary = summarize_obs(obs_from_env)
    #     # print(f"Step {steps}:\n{summary}")


    #     ## at some point we need to add the logic for exteral environment 
    #     # Env update
    #     # send signal to agents about evacuation and fire spread
    #     time.sleep(0.5) 
    #     done = terminations or truncations
    #     steps += 1
    # env.close()


if __name__ == "__main__":
    run()

