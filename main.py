import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path, get_original_cwd

from load_map import load_map
from build_map import MapMiniGrid
import numpy as np


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):
    """
    Main entry point for the simulation.
    Loads map and runs the MiniGrid environment defined in JSON.
    """
    # Resolve the absolute path to the map file (relative to the original working dir)
    map_path = to_absolute_path(cfg.map.file)

    print(f"[Hydra] Loading map from: {map_path}")

    # Optional sanity check before building environment
    elements, (W, H) = load_map(map_path)
    print(f"Loaded map size: {W}x{H} with {len(elements)} elements")

    # Create environment
    env = MapMiniGrid(
        json_path=map_path,
        agent_view_size=cfg.sim.agent_view_size,
        render_mode=cfg.sim.render_mode,
    )

    # Initialize environment
    obs, info = env.reset(seed=cfg.sim.seed)

    done = False
    steps = 0
    while not done and steps < cfg.sim.steps:
        action = np.random.randint(0, env.action_space.n)
        obs, reward, term, trunc, info = env.step(action)

        done = term or trunc
        steps += 1

    print(f"Simulation finished after {steps} steps.")
    env.close()


if __name__ == "__main__":
    run()

