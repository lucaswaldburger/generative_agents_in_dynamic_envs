# env/grid.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
from .constants import Action, DIR_TO_VEC, DEFAULT_MAX_STEPS, AgentConfig
from .load_map import MapSpec
from .world_object import HumanAgent


class MultiHumanGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 8}

    def __init__(
        self,
        map_spec: MapSpec,
        agent_configs: list[AgentConfig],
        max_steps: int = DEFAULT_MAX_STEPS,
        render_mode: str | None = "human",
    ):
        super().__init__()
        self.map_spec = map_spec
        self.agent_configs = agent_configs
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.num_agents = len(agent_configs)
        self.agents: list[HumanAgent] = []

        # joint action space...
        self.action_space = spaces.MultiDiscrete([len(Action)] * self.num_agents)

        self.observation_space = spaces.Dict(
            {
                "agent_positions": spaces.Box(
                    low=0,
                    high=max(self.map_spec.width, self.map_spec.height),
                    shape=(self.num_agents, 2),
                    dtype=np.int32,
                ),
                "step": spaces.Discrete(self.max_steps + 1),
            }
        )

        self.step_count: int = 0

        self._can_enter = {
            int(k): bool(v)
            for k, v in self.map_spec.semantics.get("can_enter", {}).items()
        }

     
        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.cell_size: int = 40  # pixels per grid cell, this can change how big the window is
    

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0

        self.agents = [HumanAgent.from_config(cfg) for cfg in self.agent_configs]

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        self.step_count += 1
        ACTION_TO_HEADING = {
            Action.RIGHT: 0,     # +x
            Action.DOWN: 90,     # +y
            Action.LEFT: 180,    # -x
            Action.UP: 270,      # -y
        }
        # Move each agent
        for i, agent in enumerate(self.agents):
            act = Action(int(action[i]))
            dx, dy = DIR_TO_VEC[act]
            nx, ny = agent.x + dx, agent.y + dy


            if self._can_move_to(nx, ny):
                agent.x, agent.y = nx, ny
                if act in ACTION_TO_HEADING:
                    agent.heading_deg = ACTION_TO_HEADING[act]



        # TODO: remove, we dont need this
        reward = 0.0
        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            frame = self._render_human(return_array=True)
            return frame
        elif self.render_mode == "ansi":
            return self._render_ansi()
        else:
            return None


    def _init_pygame(self):
        if self.window is not None:
            return
        pygame.init()
        w = self.map_spec.width * self.cell_size
        h = self.map_spec.height * self.cell_size
        self.window = pygame.display.set_mode((w, h))
        pygame.display.set_caption("MultiHumanGridEnv")
        self.clock = pygame.time.Clock()

    def _render_human(self, return_array: bool = False):
        self._init_pygame()
        assert self.window is not None

        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.window = None
                return None

        W, H = self.map_spec.width, self.map_spec.height

        # Background
        self.window.fill((20, 20, 20))

        # Draw base grid (streets)
        for y in range(H):
            for x in range(W):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.window, (50, 50, 50), rect)

        # Draw regions (blocks, park, home, fire, work)
        palette = self.map_spec.raw.get("palette", {})
        for region in self.map_spec.regions:
            x, y = int(region["x"]), int(region["y"])
            w, h = int(region["w"]), int(region["h"])
            rtype = region["type"]

            color_list = palette.get(rtype, [100, 100, 100])
            color = tuple(color_list[:3])  # ignore alpha if present

            rect = pygame.Rect(
                x * self.cell_size,
                y * self.cell_size,
                w * self.cell_size,
                h * self.cell_size,
            )
            pygame.draw.rect(self.window, color, rect)

        # Draw grid lines
        for x in range(W + 1):
            pygame.draw.line(
                self.window,
                (30, 30, 30),
                (x * self.cell_size, 0),
                (x * self.cell_size, H * self.cell_size),
                1,
            )
        for y in range(H + 1):
            pygame.draw.line(
                self.window,
                (30, 30, 30),
                (0, y * self.cell_size),
                (W * self.cell_size, y * self.cell_size),
                1,
            )

        # ---------- FOV overlay (semi-transparent, agent-colored) ----------
        fov_surface = pygame.Surface(self.window.get_size(), pygame.SRCALPHA)

        for agent in self.agents:
            fov_cfg = agent.config.fov
            rng = int(fov_cfg.range_cells)
            angle_deg = float(fov_cfg.angle_deg)

            # Heading convention: 0° = right, 90° = down (screen coordinates)
            heading_rad = math.radians(agent.heading_deg)

            # Base direction unit vector
            hx = math.cos(heading_rad)
            hy = math.sin(heading_rad)

            # Color: same as agent, but transparent
            base_r, base_g, base_b = self._agent_rgb(agent.config.color)
            fov_color = (base_r, base_g, base_b, 70)  # last = alpha (0–255)

            ax, ay = agent.x, agent.y

            # Check cells in a square around the agent
            for gy in range(max(0, ay - rng), min(H, ay + rng + 1)):
                for gx in range(max(0, ax - rng), min(W, ax + rng + 1)):
                    dx = gx - ax
                    dy = gy - ay

                    # distance in cells
                    dist = math.hypot(dx, dy)
                    if dist == 0 or dist > rng:
                        continue

                    # direction to this cell
                    vx = dx / dist
                    vy = dy / dist

                    # angle between heading and cell vector
                    dot = max(min(hx * vx + hy * vy, 1.0), -1.0)
                    cell_angle = math.degrees(math.acos(dot))

                    if cell_angle <= angle_deg / 2.0:
                        # inside FOV cone: shade this cell
                        rect = pygame.Rect(
                            gx * self.cell_size,
                            gy * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        )
                        pygame.draw.rect(fov_surface, fov_color, rect)


        self.window.blit(fov_surface, (0, 0))


        for idx, agent in enumerate(self.agents):
            cx = agent.x * self.cell_size + self.cell_size // 2
            cy = agent.y * self.cell_size + self.cell_size // 2

            agent_color = self._agent_rgb(agent.config.color)

            pygame.draw.circle(
                self.window,
                agent_color,
                (cx, cy),
                self.cell_size // 3,
            )

        assert self.clock is not None
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

        if return_array:
            frame = pygame.surfarray.array3d(self.window)
            frame = np.transpose(frame, (1, 0, 2))
            return frame
        else:
            return None


    def _agent_rgb(self, color_name: str | None) -> tuple[int, int, int]:
        name = (color_name or "white").lower()
        return {
            "red": (255, 80, 80),
            "blue": (80, 80, 255),
            "green": (80, 200, 120),
            "yellow": (230, 230, 90),
            "white": (240, 240, 240),
        }.get(name, (240, 240, 240))

    def _render_ansi(self) -> str:
        W, H = self.map_spec.width, self.map_spec.height
        char_grid = np.full((H, W), ".", dtype="<U1")

        for region in self.map_spec.regions:
            x, y = int(region["x"]), int(region["y"])
            w, h = int(region["w"]), int(region["h"])
            tile_char = self._region_char(region["type"])
            char_grid[y : y + h, x : x + w] = tile_char

        for idx, agent in enumerate(self.agents):
            if 0 <= agent.x < W and 0 <= agent.y < H:
                char_grid[agent.y, agent.x] = str(idx + 1)

        lines = ["".join(row) for row in char_grid]
        txt = "\n".join(lines)
        print(txt)
        return txt



    def close(self):
        pass



    def _can_move_to(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.map_spec.width or y < 0 or y >= self.map_spec.height:
            return False
        code = int(self.map_spec.access_grid[y, x])
        return self._can_enter.get(code, False)

    def _region_char(self, region_type: str) -> str:
        return {
            "block": "#",
            "park": "P",
            "home": "H",
            "fire": "F",
            "work": "W",
        }.get(region_type, "?")

    def _get_obs(self) -> Dict[str, Any]:
        positions = np.array([[a.x, a.y] for a in self.agents], dtype=np.int32)
        return {
            "agent_positions": positions,
            "step": self.step_count,
        }
