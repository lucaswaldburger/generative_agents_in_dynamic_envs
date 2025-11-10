import hydra
import sys
import numpy as np
import time

from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from env.load_map import load_map
from env.build_map import MapMiniGrid
from persona.cognitive.perceive import get_obs, print_visible
from persona.cognitive.plan import get_path, get_next_plan_and_waypoint




@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):
    """
    Loads map and runs the MiniGrid environment defined in JSON.
    """
    map_path = to_absolute_path(cfg.map.file)

    # Create environment
    env = MapMiniGrid(
        json_path=map_path,
        agent_view_size=cfg.sim.agent_view_size,
        render_mode=cfg.sim.render_mode,
    )

    # Initialize environment
    print("[Env] Resetting...")
    obs, info = env.reset(seed=cfg.sim.seed)
    # print(f"[Env] obs keys: {obs['image']}")clear


    if cfg.sim.render_mode == "human":
        env.render()

    done = False
    steps = 0
    while not done and steps < cfg.sim.steps:
        # action = np.random.randint(0, env.action_space.n)
        get_next_plan_text = "go to work"
        current_location_text = "home_A"
        path = get_next_plan_and_waypoint(current_location_text, get_next_plan_text, env)
        
        # action = 0 # just for testing
        for action in path: # we want to move this to a per step action but for not just testing
            obs, reward, term, trunc, info = env.step(action)
            vis = get_obs(env, include_world_coords=True, include_street=False)
            print(f"Step {steps}, action={action}")
            print_visible(vis)
        # obs_from_env = get_obs(obs, lookup_name=env.region_name)
        # print(f"Obs shape: {np.array(obs_from_env).shape}")
        # summary = summarize_obs(obs_from_env)
        # print(f"Step {steps}:\n{summary}")
        time.sleep(0.2) 
        done = term or trunc
        steps += 1
    env.close()


if __name__ == "__main__":
    run()

