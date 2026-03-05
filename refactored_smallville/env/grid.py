"""
SmallVilleEnv – a Gymnasium environment that renders the SmallVille map
through Pygame, replacing the original Django/Phaser frontend.

Modeled after grid_gen/env/grid.py (MultiHumanGridEnv).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from pathlib import Path

from .maze import Maze
from .path_finder import closest_coordinate, path_finder
from .tiled_renderer import TiledRenderer


class Action(IntEnum):
    STAY = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


DIR_TO_VEC: Dict[Action, Tuple[int, int]] = {
    Action.STAY: (0, 0),
    Action.UP: (0, -1),
    Action.RIGHT: (1, 0),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
}


@dataclass
class AgentState:
    name: str
    x: int
    y: int
    color: Tuple[int, int, int] = (255, 80, 80)
    portrait: Optional[pygame.Surface] = None
    description: str = ""
    pronunciatio: str = ""
    path: List[Tuple[int, int]] = field(default_factory=list)


AGENT_COLORS = [
    (220, 60, 60),
    (60, 120, 220),
    (60, 200, 80),
    (230, 180, 40),
    (180, 80, 220),
    (40, 200, 200),
    (255, 140, 60),
    (200, 200, 200),
]

DEFAULT_WINDOW_W = 1280
DEFAULT_WINDOW_H = 800


class SmallVilleEnv(gym.Env):
    """Gymnasium environment wrapping the SmallVille tile map with Pygame rendering."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        assets_dir: str | Path,
        agent_names: Optional[List[str]] = None,
        render_mode: str = "human",
        window_w: int = DEFAULT_WINDOW_W,
        window_h: int = DEFAULT_WINDOW_H,
    ):
        super().__init__()
        self.assets_dir = Path(assets_dir)
        self.render_mode = render_mode
        self.window_w = window_w
        self.window_h = window_h

        self.maze = Maze(self.assets_dir)
        self.renderer = TiledRenderer(self.assets_dir)

        n_agents = len(agent_names) if agent_names else 0
        self.action_space = spaces.MultiDiscrete([len(Action)] * max(n_agents, 1))
        self.observation_space = spaces.Dict(
            {
                "agents": spaces.Box(
                    low=0,
                    high=max(self.maze.maze_width, self.maze.maze_height),
                    shape=(max(n_agents, 1), 2),
                    dtype=np.int32,
                )
            }
        )

        self._agent_names = agent_names or []
        self.agents: List[AgentState] = []
        self._portraits: Dict[str, pygame.Surface] = {}

        self._camera_x: float = 0.0
        self._camera_y: float = 0.0
        self._zoom: float = 1.0
        self._dragging: bool = False
        self._drag_start: Tuple[int, int] = (0, 0)
        self._cam_start: Tuple[float, float] = (0.0, 0.0)

        self._pygame_inited = False
        self._window: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._map_surface: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None
        self._small_font: Optional[pygame.font.Font] = None

    def _ensure_pygame(self):
        if self._pygame_inited:
            return
        import os
        if self.render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.font.init()
        if self.render_mode == "human":
            self._window = pygame.display.set_mode(
                (self.window_w, self.window_h), pygame.RESIZABLE
            )
            pygame.display.set_caption("SmallVille – Pygame")
        else:
            pygame.display.set_mode((1, 1))
            self._window = pygame.Surface((self.window_w, self.window_h))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Arial", 14, bold=True)
        self._small_font = pygame.font.SysFont("Arial", 11)
        self._map_surface = self.renderer.build_map_surface()
        self._load_portraits()
        self._pygame_inited = True

    def _load_portraits(self):
        chars_dir = self.assets_dir / "characters"
        if not chars_dir.exists():
            return
        for agent in self._agent_names:
            fname = agent.replace(" ", "_") + ".png"
            path = chars_dir / fname
            if path.exists():
                img = pygame.image.load(str(path)).convert_alpha()
                self._portraits[agent] = pygame.transform.smoothscale(img, (28, 28))

    def _spawn_agents(self):
        self.agents = []
        spawn_locs = [
            (k, v)
            for k, v in self.maze.address_tiles.items()
            if k.startswith("<spawn_loc>")
        ]
        used: set[Tuple[int, int]] = set()

        for idx, name in enumerate(self._agent_names):
            color = AGENT_COLORS[idx % len(AGENT_COLORS)]

            tile: Optional[Tuple[int, int]] = None
            for key, coords in spawn_locs:
                for c in coords:
                    if c not in used:
                        tile = c
                        break
                if tile:
                    break

            if tile is None:
                tile = (self.maze.maze_width // 2, self.maze.maze_height // 2)
            used.add(tile)

            self.agents.append(
                AgentState(
                    name=name,
                    x=tile[0],
                    y=tile[1],
                    color=color,
                    portrait=self._portraits.get(name),
                )
            )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self._ensure_pygame()
        self._spawn_agents()
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> Dict[str, Any]:
        positions = np.array(
            [[a.x, a.y] for a in self.agents] or [[0, 0]], dtype=np.int32
        )
        return {"agents": positions}

    def step(
        self, actions: np.ndarray | List[int]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        for i, agent in enumerate(self.agents):
            if i >= len(actions):
                break

            if agent.path:
                nx, ny = agent.path.pop(0)
                if not self.maze.access_tile((nx, ny))["collision"]:
                    agent.x, agent.y = nx, ny
                continue

            act = Action(int(actions[i]))
            dx, dy = DIR_TO_VEC[act]
            nx, ny = agent.x + dx, agent.y + dy
            if (
                0 <= nx < self.maze.maze_width
                and 0 <= ny < self.maze.maze_height
                and not self.maze.access_tile((nx, ny))["collision"]
            ):
                agent.x, agent.y = nx, ny

        obs = self._get_obs()
        return obs, 0.0, False, False, {}

    def move_agent_to(self, agent_idx: int, target: Tuple[int, int]):
        """Compute and assign a path for the given agent to the target tile."""
        agent = self.agents[agent_idx]
        path = path_finder(
            self.maze.collision_maze, (agent.x, agent.y), target, "32125"
        )
        if path and len(path) > 1:
            agent.path = path[1:]

    def move_agent_to_address(self, agent_idx: int, address: str):
        """Move agent to a tile matching the given address string."""
        tiles = self.maze.address_tiles.get(address)
        if not tiles:
            return
        agent = self.agents[agent_idx]
        target = closest_coordinate((agent.x, agent.y), tiles)
        if target:
            self.move_agent_to(agent_idx, target)

    def handle_pygame_events(self) -> bool:
        """Process pygame events. Returns False if the user closed the window."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.VIDEORESIZE:
                self.window_w, self.window_h = event.w, event.h
                if self.render_mode == "human":
                    self._window = pygame.display.set_mode(
                        (self.window_w, self.window_h), pygame.RESIZABLE
                    )
            elif event.type == pygame.MOUSEWHEEL:
                old_zoom = self._zoom
                self._zoom *= 1.1 if event.y > 0 else 0.9
                self._zoom = max(0.15, min(4.0, self._zoom))
                mx, my = pygame.mouse.get_pos()
                self._camera_x = mx - (mx - self._camera_x) * (self._zoom / old_zoom)
                self._camera_y = my - (my - self._camera_y) * (self._zoom / old_zoom)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._dragging = True
                self._drag_start = event.pos
                self._cam_start = (self._camera_x, self._camera_y)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._dragging = False
            elif event.type == pygame.MOUSEMOTION and self._dragging:
                dx = event.pos[0] - self._drag_start[0]
                dy = event.pos[1] - self._drag_start[1]
                self._camera_x = self._cam_start[0] + dx
                self._camera_y = self._cam_start[1] + dy
            elif event.type == pygame.KEYDOWN:
                scroll_speed = 40
                if event.key == pygame.K_LEFT:
                    self._camera_x += scroll_speed
                elif event.key == pygame.K_RIGHT:
                    self._camera_x -= scroll_speed
                elif event.key == pygame.K_UP:
                    self._camera_y += scroll_speed
                elif event.key == pygame.K_DOWN:
                    self._camera_y -= scroll_speed
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self._zoom = min(4.0, self._zoom * 1.15)
                elif event.key == pygame.K_MINUS:
                    self._zoom = max(0.15, self._zoom / 1.15)
                elif event.key == pygame.K_HOME:
                    self._zoom = 1.0
                    self._camera_x = 0.0
                    self._camera_y = 0.0
        return True

    def render(self) -> Optional[np.ndarray]:
        self._ensure_pygame()
        if self._map_surface is None:
            return None

        self._window.fill((30, 30, 30))

        scaled_w = int(self.renderer.pixel_width * self._zoom)
        scaled_h = int(self.renderer.pixel_height * self._zoom)

        if self._zoom != 1.0:
            scaled_map = pygame.transform.smoothscale(
                self._map_surface, (scaled_w, scaled_h)
            )
        else:
            scaled_map = self._map_surface

        self._window.blit(scaled_map, (self._camera_x, self._camera_y))

        tile_sz = self.renderer.tile_w * self._zoom
        for agent in self.agents:
            sx = self._camera_x + agent.x * tile_sz + tile_sz / 2
            sy = self._camera_y + agent.y * tile_sz + tile_sz / 2

            if not (-30 < sx < self.window_w + 30 and -30 < sy < self.window_h + 30):
                continue

            radius = max(4, int(tile_sz * 0.4))
            pygame.draw.circle(
                self._window, (0, 0, 0), (int(sx), int(sy)), radius + 2
            )
            pygame.draw.circle(
                self._window, agent.color, (int(sx), int(sy)), radius
            )

            if agent.portrait and self._zoom >= 0.6:
                pw = int(28 * max(0.5, self._zoom))
                portrait = pygame.transform.smoothscale(agent.portrait, (pw, pw))
                self._window.blit(
                    portrait, (int(sx - pw / 2), int(sy - pw / 2 - radius - pw - 2))
                )

            if self._zoom >= 0.35:
                label = self._font.render(agent.name, True, (255, 255, 255))
                shadow = self._font.render(agent.name, True, (0, 0, 0))
                lx = int(sx - label.get_width() / 2)
                ly = int(sy + radius + 4)
                self._window.blit(shadow, (lx + 1, ly + 1))
                self._window.blit(label, (lx, ly))

            if agent.pronunciatio and self._zoom >= 0.5:
                bubble = self._small_font.render(agent.pronunciatio, True, (50, 50, 50))
                bw, bh = bubble.get_width() + 8, bubble.get_height() + 4
                bx = int(sx - bw / 2)
                by = int(sy - radius - bh - 4)
                pygame.draw.rect(
                    self._window, (255, 255, 230), (bx, by, bw, bh), border_radius=4
                )
                pygame.draw.rect(
                    self._window, (180, 180, 150), (bx, by, bw, bh), 1, border_radius=4
                )
                self._window.blit(bubble, (bx + 4, by + 2))

        self._draw_hud()

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        else:
            frame = pygame.surfarray.array3d(self._window)
            return np.transpose(frame, (1, 0, 2))

    def _draw_hud(self):
        y_off = 8
        for agent in self.agents:
            tile = self.maze.access_tile((agent.x, agent.y))
            loc_parts = [tile["sector"], tile["arena"]]
            loc_str = " > ".join(p for p in loc_parts if p)
            text = f"{agent.name}: ({agent.x},{agent.y}) [{loc_str}]"
            if agent.description:
                text += f" – {agent.description}"
            surf = self._small_font.render(text, True, agent.color)
            bg = pygame.Surface((surf.get_width() + 6, surf.get_height() + 2), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 160))
            self._window.blit(bg, (4, y_off - 1))
            self._window.blit(surf, (7, y_off))
            y_off += surf.get_height() + 4

        zoom_text = self._small_font.render(
            f"Zoom: {self._zoom:.1f}x  |  Arrow/drag to pan, scroll to zoom, Home to reset",
            True,
            (180, 180, 180),
        )
        self._window.blit(
            zoom_text, (self.window_w - zoom_text.get_width() - 8, self.window_h - 20)
        )

    def close(self):
        if self._pygame_inited:
            pygame.quit()
            self._pygame_inited = False
