import json
import numpy as np
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj
from minigrid.core.mission import MissionSpace
from load_map import load_map

class ColorTile(WorldObj):
    """

    """
    def __init__(self, obj_type: str, color_name: str, color_rgba):
        super().__init__(obj_type, color=color_name) 
        self.color_rgba = tuple(color_rgba)
        self.can_overlap = (obj_type != "wall")
        self.see_through = (obj_type != "wall")

    def render(self, img):

        if len(self.color_rgba) == 4:
            r, g, b, a = self.color_rgba
            density = 2
            for yy in range(img.shape[0]):
                for xx in range(img.shape[1]):
                    if ((xx // density) + (yy // density)) % 2 == 0:
                        img[yy, xx, :] = (r, g, b)
        else:
            img[:, :, :] = self.color_rgba[:3]

class MapMiniGrid(MiniGridEnv):
    def __init__(self, json_path, agent_view_size=7, render_mode=None, max_steps=10_000):
        self.json_path = json_path
        with open(self.json_path, "r") as f:
            data = json.load(f)
        W, H = int(data["width"]), int(data["height"])
        self._agent_start = tuple(data["agent_start"].values()) if "agent_start" in data else None

        mission_space = MissionSpace(mission_func=lambda: "Human evacuation")
        super().__init__(
            width=W, height=H, max_steps=max_steps,
            mission_space=mission_space,
            agent_view_size=agent_view_size,
            see_through_walls=True,
            render_mode=render_mode,
        )

    def _gen_grid(self, width, height):
        elements, (W, H) = load_map(self.json_path)
        self.grid = Grid(W, H)

        for obj_type, rgba, color_name, (x, y) in elements:
            tile = ColorTile(obj_type, color_name, rgba)
            self.grid.set(x, y, tile)

        if self._agent_start is not None:
            ax, ay = self._agent_start
            cell = self.grid.get(ax, ay)
            if cell is None or cell.type in ("floor", "goal", "lava"):
                self.agent_pos = (ax, ay)
                self.agent_dir = 0
            else:
                self.place_agent()
        else:
            self.place_agent()

    def step(self, action):
        return super().step(action)

