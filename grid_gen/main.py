import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from load_map import load_map
from build_map import MapMiniGrid
import numpy as np


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
    obs, info = env.reset(seed=cfg.sim.seed)

    done = False
    steps = 0
    while not done and steps < cfg.sim.steps:
        # action = np.random.randint(0, env.action_space.n)
        action = 0 # just for testing
        obs, reward, term, trunc, info = env.step(action)

        done = term or trunc
        steps += 1
    env.close()


if __name__ == "__main__":
    run()

