import json
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from .load_map import load_map
from env.world_object import make_tile         
from env.constants import SEM_TO_ID            

def heading_deg_to_dir(deg: int) -> int:
    return int(((deg % 360) + 45) // 90) % 4

def _in_bounds(x, y, W, H):
    return 0 <= x < W and 0 <= y < H

class MapMiniGrid(MiniGridEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, json_path, agent_view_size=7, render_mode=None, max_steps=10_000):
        self.json_path = json_path
        elements, (W, H), agents = load_map(self.json_path)
        self._elements = elements
        self._W, self._H = W, H

        with open(self.json_path, "r") as f:
            raw = json.load(f)
        self._agent_spec = None
        if agents:
            a0 = agents[0]
            self._agent_spec = {
                "x": int(a0.get("start", {}).get("x", 0)),
                "y": int(a0.get("start", {}).get("y", 0)),
                "dir": heading_deg_to_dir(int(a0.get("heading_deg", a0.get("heading", 0)))),
            }
        elif "agent_start" in raw:
            sx, sy = tuple(raw["agent_start"].values())
            self._agent_spec = {"x": int(sx), "y": int(sy), "dir": 0}

        mission_space = MissionSpace(mission_func=lambda: "Human evacuation")
        super().__init__(
            width=W, height=H, max_steps=max_steps,
            mission_space=mission_space,
            agent_view_size=agent_view_size,
            see_through_walls=False,
            render_mode=render_mode,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(self._W, self._H)
        for obj_type, rgba, color_name, sem_name, (x, y) in self._elements:
            tile = make_tile(sem_name=sem_name, rgba=rgba, name=None)
            self.grid.set(x, y, tile)

        # perimeter for now - bugs without it
        wall_rgba = (110, 130, 140)
        for x in range(width):
            self.grid.set(x, 0,              make_tile("block", wall_rgba))
            self.grid.set(x, height - 1,     make_tile("block", wall_rgba))
        for y in range(height):
            self.grid.set(0,          y,     make_tile("block", wall_rgba))
            self.grid.set(width - 1,  y,     make_tile("block", wall_rgba))

        self._place_agent_from_json()


    def _nearest_free(self, x0, y0, max_radius=4):
        W, H = self.width, self.height
        for r in range(max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    x, y = x0 + dx, y0 + dy
                    if not _in_bounds(x, y, W, H):
                        continue
                    cell = self.grid.get(x, y)
                    if cell is None or (hasattr(cell, "can_overlap") and cell.can_overlap()):
                        return (x, y)
        return None

    def _place_agent_from_json(self):
        """
        Place a single controllable agent from JSON.
        If blocked, fallback to place_agent(); keep JSON heading.
        """
        if not self._agent_spec:
            self.place_agent()
            return

        ax, ay, d = self._agent_spec["x"], self._agent_spec["y"], self._agent_spec["dir"]
        cell = self.grid.get(ax, ay)
        if cell is None or (hasattr(cell, "can_overlap") and cell.can_overlap()):
            self.agent_pos = (ax, ay)
            self.agent_dir = d
        else:
            spot = self._nearest_free(ax, ay)
            if spot:
                self.agent_pos = spot
                self.agent_dir = d
            else:
                self.place_agent()
                self.agent_dir = d 

    def step(self, action):
        return super().step(action)
