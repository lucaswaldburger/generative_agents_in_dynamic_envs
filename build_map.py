import json
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj
from minigrid.core.mission import MissionSpace
from load_map import load_map

class ColorTile(WorldObj):
    """
    Use MiniGrid's valid base types ("wall", "lava", "goal") and paint custom RGB.
    """
    def __init__(self, obj_type: str, color_rgb):
        super().__init__(obj_type, color="grey")  # valid named color required
        self.color_rgb = color_rgb
        self.can_overlap = True
        self.see_through = True

    def render(self, img):
        img[:, :, :] = self.color_rgb


class MapMiniGrid(MiniGridEnv):
    """MiniGrid environment built from a JSON map file."""
    def __init__(self, json_path, agent_view_size=7, render_mode=None, max_steps=10_000):
        self.json_path = json_path

        # Read size and (optional) agent_start BEFORE calling super().__init__()
        with open(self.json_path, "r") as f:
            data = json.load(f)
        W, H = int(data["width"]), int(data["height"])
        self._agent_start = tuple(data["agent_start"].values()) if "agent_start" in data else None

        mission_space = MissionSpace(mission_func=lambda: "reach the goal")

        super().__init__(
            width=W,
            height=H,
            max_steps=max_steps,
            mission_space=mission_space,
            agent_view_size=agent_view_size,
            see_through_walls=True,
            render_mode=render_mode,
        )

    def _gen_grid(self, width, height):
        """Generate grid and honor agent_start (if given)."""
        elements, (W, H) = load_map(self.json_path)

        self.grid = Grid(W, H)
        self.grid.wall_rect(0, 0, W, H)

        for obj_type, color, (x, y) in elements:
            tile = ColorTile(obj_type, color)
            if obj_type == "wall":
                tile.can_overlap = False
                tile.see_through = False
            # lava/goal remain walkable/see-through
            self.grid.set(x, y, tile)

        # Place agent last (after tiles exist)
        if self._agent_start is not None:
            ax, ay = self._agent_start
            # If the start cell is blocked, fall back to default placement
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

