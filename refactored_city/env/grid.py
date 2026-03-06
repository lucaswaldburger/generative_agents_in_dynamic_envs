"""
CityEnv – a Gymnasium environment that renders the procedural city map
through Pygame, adapted from refactored_smallville's SmallVilleEnv.

Adds autonomous car entities that drive along defined lane paths.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from pathlib import Path

from .city_map import CityMap
from .city_renderer import CityRenderer
from .detailed_renderer import DetailedCityRenderer
from .path_finder import closest_coordinate, path_finder


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

SPRITE_FRAME_W = 32
SPRITE_FRAME_H = 32
SPRITE_DIR_ROW = {"down": 0, "left": 1, "right": 2, "up": 3}
SPRITE_WALK_COLS = [0, 1, 2, 1]


@dataclass
class AgentState:
    name: str
    x: int
    y: int
    color: Tuple[int, int, int] = (255, 80, 80)
    description: str = ""
    pronunciatio: str = ""
    path: List[Tuple[int, int]] = field(default_factory=list)
    direction: str = "down"
    anim_tick: int = 0
    prev_x: int = 0
    prev_y: int = 0


@dataclass
class CarState:
    lane_name: str
    color: Tuple[int, int, int]
    x: float
    y: float
    waypoints: List[Tuple[int, int]] = field(default_factory=list)
    wp_idx: int = 0
    speed: float = 0.4
    pause_timer: int = 0
    direction: str = "right"


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


class CityEnv(gym.Env):
    """Gymnasium environment wrapping the procedural city map with Pygame rendering."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

    def __init__(
        self,
        config_path: str | Path,
        agent_names: Optional[List[str]] = None,
        render_mode: str = "human",
        window_w: int = DEFAULT_WINDOW_W,
        window_h: int = DEFAULT_WINDOW_H,
        use_sprites: bool = False,
        use_detailed: bool = False,
        assets_dir: Optional[str | Path] = None,
    ):
        super().__init__()
        self.config_path = Path(config_path)
        self.render_mode = render_mode
        self.window_w = window_w
        self.window_h = window_h
        self.use_sprites = use_sprites
        self.use_detailed = use_detailed
        self.assets_dir = Path(assets_dir) if assets_dir else Path(__file__).parent.parent / "assets"

        self.maze = CityMap(self.config_path)
        if use_detailed:
            self.renderer = DetailedCityRenderer(self.maze)
        else:
            self.renderer = CityRenderer(self.maze)

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
        self.cars: List[CarState] = []
        self._sprite_sheets: Dict[str, pygame.Surface] = {}

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
            pygame.display.set_caption("Urban City – Pygame")
        else:
            pygame.display.set_mode((1, 1))
            self._window = pygame.Surface((self.window_w, self.window_h))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Arial", 14, bold=True)
        self._small_font = pygame.font.SysFont("Arial", 11)
        self._map_surface = self.renderer.build_map_surface()
        if self.use_detailed:
            self._load_sprite_sheets()
        self._pygame_inited = True

    def _spawn_agents(self):
        self.agents = []
        building_tiles: List[Tuple[int, int]] = []
        for y in range(self.maze.maze_height):
            for x in range(self.maze.maze_width):
                if self.maze.tiles[y][x]["tile_type"] == "building":
                    building_tiles.append((x, y))

        used: set[Tuple[int, int]] = set()
        for idx, name in enumerate(self._agent_names):
            color = AGENT_COLORS[idx % len(AGENT_COLORS)]

            tile: Optional[Tuple[int, int]] = None
            if building_tiles:
                candidates = [t for t in building_tiles if t not in used]
                if candidates:
                    tile = random.choice(candidates)
            if tile is None:
                tile = (self.maze.maze_width // 2, self.maze.maze_height // 2)
            used.add(tile)

            self.agents.append(
                AgentState(
                    name=name,
                    x=tile[0],
                    y=tile[1],
                    color=color,
                    prev_x=tile[0],
                    prev_y=tile[1],
                )
            )

    def _spawn_cars(self):
        self.cars = []
        for lane in self.maze.car_lanes:
            wps = lane.get("waypoints", [])
            if len(wps) < 2:
                continue
            color = tuple(lane.get("color", [180, 60, 60]))
            sx, sy = wps[0]

            dx = wps[-1][0] - wps[0][0]
            dy = wps[-1][1] - wps[0][1]
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "down" if dy > 0 else "up"

            offset = random.randint(0, max(1, abs(dx) + abs(dy)) // 2)
            if direction == "right":
                sx += offset
            elif direction == "left":
                sx -= offset
            elif direction == "down":
                sy += offset
            elif direction == "up":
                sy -= offset

            self.cars.append(
                CarState(
                    lane_name=lane["name"],
                    color=color,
                    x=float(sx),
                    y=float(sy),
                    waypoints=[(p[0], p[1]) for p in wps],
                    speed=random.uniform(0.3, 0.6),
                    direction=direction,
                )
            )

    def _step_cars(self):
        for car in self.cars:
            if car.pause_timer > 0:
                car.pause_timer -= 1
                continue

            target = car.waypoints[-1]
            dx = target[0] - car.x
            dy = target[1] - car.y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < car.speed:
                car.x = float(car.waypoints[0][0])
                car.y = float(car.waypoints[0][1])
                car.pause_timer = random.randint(2, 8)
                continue

            if dist > 0:
                car.x += (dx / dist) * car.speed
                car.y += (dy / dist) * car.speed

            ix, iy = int(round(car.x)), int(round(car.y))
            for inter in self.maze.intersections_data:
                if (
                    inter["x"] <= ix < inter["x"] + inter["w"]
                    and inter["y"] <= iy < inter["y"] + inter["h"]
                ):
                    if random.random() < 0.05:
                        car.pause_timer = random.randint(3, 10)
                    break

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self._ensure_pygame()
        self._spawn_agents()
        self._spawn_cars()
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> Dict[str, Any]:
        positions = np.array(
            [[a.x, a.y] for a in self.agents] or [[0, 0]], dtype=np.int32
        )
        return {"agents": positions}

    def _update_agent_direction(self, agent: AgentState):
        dx = agent.x - agent.prev_x
        dy = agent.y - agent.prev_y
        if dx > 0:
            agent.direction = "right"
        elif dx < 0:
            agent.direction = "left"
        elif dy > 0:
            agent.direction = "down"
        elif dy < 0:
            agent.direction = "up"
        moving = dx != 0 or dy != 0
        if moving:
            agent.anim_tick += 1
        agent.prev_x = agent.x
        agent.prev_y = agent.y

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
                self._update_agent_direction(agent)
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
            self._update_agent_direction(agent)

        self._step_cars()

        obs = self._get_obs()
        return obs, 0.0, False, False, {}

    def move_agent_to(self, agent_idx: int, target: Tuple[int, int]):
        agent = self.agents[agent_idx]
        path = path_finder(
            self.maze.collision_maze, (agent.x, agent.y), target, "1"
        )
        if path and len(path) > 1:
            agent.path = path[1:]

    def move_agent_to_address(self, agent_idx: int, address: str):
        tiles = self.maze.address_tiles.get(address)
        if not tiles:
            return
        agent = self.agents[agent_idx]
        target = closest_coordinate((agent.x, agent.y), tiles)
        if target:
            self.move_agent_to(agent_idx, target)

    def handle_pygame_events(self) -> bool:
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
                self._zoom = max(0.15, min(6.0, self._zoom))
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
                    self._zoom = min(6.0, self._zoom * 1.15)
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

        self._window.fill((25, 25, 30))

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

        for car in self.cars:
            if self.use_detailed:
                self._draw_car_detailed(car, tile_sz)
            else:
                self._draw_car(car, tile_sz)

        for agent in self.agents:
            sx = self._camera_x + agent.x * tile_sz
            sy = self._camera_y + agent.y * tile_sz

            if not (
                -tile_sz * 2 < sx < self.window_w + tile_sz
                and -tile_sz * 2 < sy < self.window_h + tile_sz
            ):
                continue

            if self.use_detailed and agent.name in self._sprite_sheets:
                self._draw_agent_sprite(agent, sx, sy, tile_sz)
            elif self.use_detailed:
                self._draw_agent_character(agent, sx, sy, tile_sz)
            else:
                self._draw_agent_dot(agent, sx, sy, tile_sz)

        self._draw_hud()

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        else:
            frame = pygame.surfarray.array3d(self._window)
            return np.transpose(frame, (1, 0, 2))

    def _draw_car(self, car: CarState, tile_sz: float):
        cx = self._camera_x + car.x * tile_sz
        cy = self._camera_y + car.y * tile_sz

        if not (
            -tile_sz * 4 < cx < self.window_w + tile_sz * 2
            and -tile_sz * 4 < cy < self.window_h + tile_sz * 2
        ):
            return

        if car.direction in ("left", "right"):
            w = max(4, int(tile_sz * 1.6))
            h = max(3, int(tile_sz * 0.8))
        else:
            w = max(3, int(tile_sz * 0.8))
            h = max(4, int(tile_sz * 1.6))

        rx = int(cx + tile_sz / 2 - w / 2)
        ry = int(cy + tile_sz / 2 - h / 2)

        pygame.draw.rect(self._window, car.color, (rx, ry, w, h), border_radius=2)
        darker = tuple(max(0, c - 40) for c in car.color)
        pygame.draw.rect(self._window, darker, (rx, ry, w, h), 1, border_radius=2)

        if car.direction in ("left", "right"):
            ww = max(2, w // 3)
            wh = max(1, h // 2)
            wx = rx + w // 2 - ww // 2
            wy = ry + h // 2 - wh // 2
            pygame.draw.rect(self._window, (180, 200, 220), (wx, wy, ww, wh))
        else:
            ww = max(1, w // 2)
            wh = max(2, h // 3)
            wx = rx + w // 2 - ww // 2
            wy = ry + h // 2 - wh // 2
            pygame.draw.rect(self._window, (180, 200, 220), (wx, wy, ww, wh))

    def _draw_agent_dot(
        self, agent: AgentState, sx: float, sy: float, tile_sz: float
    ):
        cx = sx + tile_sz / 2
        cy = sy + tile_sz / 2
        radius = max(4, int(tile_sz * 0.4))
        pygame.draw.circle(
            self._window, (0, 0, 0), (int(cx), int(cy)), radius + 2
        )
        pygame.draw.circle(
            self._window, agent.color, (int(cx), int(cy)), radius
        )

        if self._zoom >= 0.35:
            label = self._font.render(agent.name, True, (255, 255, 255))
            shadow = self._font.render(agent.name, True, (0, 0, 0))
            lx = int(cx - label.get_width() / 2)
            ly = int(cy + radius + 4)
            self._window.blit(shadow, (lx + 1, ly + 1))
            self._window.blit(label, (lx, ly))

        if agent.pronunciatio and self._zoom >= 0.5:
            bubble = self._small_font.render(agent.pronunciatio, True, (50, 50, 50))
            bw, bh = bubble.get_width() + 8, bubble.get_height() + 4
            bx = int(cx - bw / 2)
            by = int(cy - radius - bh - 4)
            pygame.draw.rect(
                self._window, (255, 255, 230), (bx, by, bw, bh), border_radius=4
            )
            pygame.draw.rect(
                self._window, (180, 180, 150), (bx, by, bw, bh), 1, border_radius=4
            )
            self._window.blit(bubble, (bx + 4, by + 2))

    def _load_sprite_sheets(self):
        """Load SmallVille character sprites and assign them to city agents round-robin."""
        chars_dir = self.assets_dir / "characters"
        if not chars_dir.exists():
            return
        available = sorted(p for p in chars_dir.iterdir()
                           if p.suffix == ".png" and p.stem != "atlas")
        if not available:
            return
        for idx, name in enumerate(self._agent_names):
            sprite_path = available[idx % len(available)]
            sheet = pygame.image.load(str(sprite_path)).convert_alpha()
            self._sprite_sheets[name] = sheet

    def _get_sprite_frame(
        self, agent_name: str, direction: str, anim_tick: int, moving: bool
    ) -> Optional[pygame.Surface]:
        sheet = self._sprite_sheets.get(agent_name)
        if sheet is None:
            return None
        row = SPRITE_DIR_ROW.get(direction, 0)
        col = SPRITE_WALK_COLS[anim_tick % len(SPRITE_WALK_COLS)] if moving else 1
        src = pygame.Rect(
            col * SPRITE_FRAME_W, row * SPRITE_FRAME_H,
            SPRITE_FRAME_W, SPRITE_FRAME_H,
        )
        return sheet.subsurface(src).copy()

    def _draw_agent_sprite(
        self, agent: AgentState, sx: float, sy: float, tile_sz: float
    ):
        moving = agent.x != agent.prev_x or agent.y != agent.prev_y or bool(agent.path)
        frame = self._get_sprite_frame(
            agent.name, agent.direction, agent.anim_tick, moving
        )
        if frame is None:
            self._draw_agent_character(agent, sx, sy, tile_sz)
            return

        sprite_px = max(8, int(tile_sz * 1.2))
        scaled_frame = pygame.transform.smoothscale(frame, (sprite_px, sprite_px))
        draw_x = int(sx + tile_sz / 2 - sprite_px / 2)
        draw_y = int(sy + tile_sz - sprite_px)
        self._window.blit(scaled_frame, (draw_x, draw_y))

        cx = sx + tile_sz / 2

        if self._zoom >= 0.35:
            label = self._font.render(agent.name, True, (255, 255, 255))
            shadow = self._font.render(agent.name, True, (0, 0, 0))
            lx = int(cx - label.get_width() / 2)
            ly = int(sy + tile_sz + 2)
            self._window.blit(shadow, (lx + 1, ly + 1))
            self._window.blit(label, (lx, ly))

        if agent.pronunciatio and self._zoom >= 0.5:
            bubble = self._small_font.render(agent.pronunciatio, True, (50, 50, 50))
            bw, bh = bubble.get_width() + 10, bubble.get_height() + 6
            bx = int(cx - bw / 2)
            by = int(draw_y - bh - 3)
            pygame.draw.rect(
                self._window, (255, 255, 235), (bx, by, bw, bh), border_radius=5
            )
            pygame.draw.rect(
                self._window, (170, 170, 140), (bx, by, bw, bh), 1, border_radius=5
            )
            tri_cx = int(cx)
            tri_top = by + bh
            pygame.draw.polygon(self._window, (255, 255, 235),
                                [(tri_cx - 3, tri_top),
                                 (tri_cx + 3, tri_top),
                                 (tri_cx, tri_top + 4)])
            self._window.blit(bubble, (bx + 5, by + 3))

    def _draw_car_detailed(self, car: CarState, tile_sz: float):
        cx = self._camera_x + car.x * tile_sz
        cy = self._camera_y + car.y * tile_sz

        if not (
            -tile_sz * 4 < cx < self.window_w + tile_sz * 2
            and -tile_sz * 4 < cy < self.window_h + tile_sz * 2
        ):
            return

        horiz = car.direction in ("left", "right")
        if horiz:
            w = max(6, int(tile_sz * 1.8))
            h = max(4, int(tile_sz * 0.85))
        else:
            w = max(4, int(tile_sz * 0.85))
            h = max(6, int(tile_sz * 1.8))

        rx = int(cx + tile_sz / 2 - w / 2)
        ry = int(cy + tile_sz / 2 - h / 2)

        shadow_c = (30, 30, 35)
        pygame.draw.rect(self._window, shadow_c,
                         (rx + 1, ry + 2, w, h), border_radius=2)

        body = car.color
        pygame.draw.rect(self._window, body, (rx, ry, w, h), border_radius=2)

        darker = tuple(max(0, c - 50) for c in body)
        pygame.draw.rect(self._window, darker, (rx, ry, w, h), 1, border_radius=2)

        glass = (160, 190, 220)
        if horiz:
            gw = max(2, w // 3)
            gh = max(2, h - 4)
            gx = rx + w // 2 - gw // 2
            gy = ry + 2
            pygame.draw.rect(self._window, glass, (gx, gy, gw, gh), border_radius=1)
        else:
            gw = max(2, w - 4)
            gh = max(2, h // 3)
            gx = rx + 2
            gy = ry + h // 2 - gh // 2
            pygame.draw.rect(self._window, glass, (gx, gy, gw, gh), border_radius=1)

        hl_col = (255, 240, 180)
        tl_col = (220, 50, 40)
        hl_r = max(1, int(tile_sz * 0.08))
        if car.direction == "right":
            pygame.draw.circle(self._window, hl_col, (rx + w - 1, ry + 2), hl_r)
            pygame.draw.circle(self._window, hl_col, (rx + w - 1, ry + h - 3), hl_r)
            pygame.draw.circle(self._window, tl_col, (rx + 1, ry + 2), hl_r)
            pygame.draw.circle(self._window, tl_col, (rx + 1, ry + h - 3), hl_r)
        elif car.direction == "left":
            pygame.draw.circle(self._window, hl_col, (rx + 1, ry + 2), hl_r)
            pygame.draw.circle(self._window, hl_col, (rx + 1, ry + h - 3), hl_r)
            pygame.draw.circle(self._window, tl_col, (rx + w - 1, ry + 2), hl_r)
            pygame.draw.circle(self._window, tl_col, (rx + w - 1, ry + h - 3), hl_r)
        elif car.direction == "down":
            pygame.draw.circle(self._window, hl_col, (rx + 2, ry + h - 1), hl_r)
            pygame.draw.circle(self._window, hl_col, (rx + w - 3, ry + h - 1), hl_r)
            pygame.draw.circle(self._window, tl_col, (rx + 2, ry + 1), hl_r)
            pygame.draw.circle(self._window, tl_col, (rx + w - 3, ry + 1), hl_r)
        else:
            pygame.draw.circle(self._window, hl_col, (rx + 2, ry + 1), hl_r)
            pygame.draw.circle(self._window, hl_col, (rx + w - 3, ry + 1), hl_r)
            pygame.draw.circle(self._window, tl_col, (rx + 2, ry + h - 1), hl_r)
            pygame.draw.circle(self._window, tl_col, (rx + w - 3, ry + h - 1), hl_r)

    def _draw_agent_character(
        self, agent: AgentState, sx: float, sy: float, tile_sz: float
    ):
        cx = sx + tile_sz / 2
        cy = sy + tile_sz / 2

        head_r = max(3, int(tile_sz * 0.28))
        body_w = max(4, int(tile_sz * 0.4))
        body_h = max(3, int(tile_sz * 0.3))
        shadow_r = max(3, int(tile_sz * 0.32))

        pygame.draw.ellipse(
            self._window, (30, 30, 35),
            (int(cx - shadow_r), int(cy + body_h),
             shadow_r * 2, max(2, shadow_r // 2)),
        )

        body_x = int(cx - body_w / 2)
        body_y = int(cy - body_h // 2 + head_r // 2)
        darker_body = tuple(max(0, c - 30) for c in agent.color)
        pygame.draw.rect(
            self._window, darker_body,
            (body_x, body_y, body_w, body_h),
            border_radius=2,
        )
        pygame.draw.rect(
            self._window, agent.color,
            (body_x + 1, body_y, body_w - 2, body_h - 1),
            border_radius=2,
        )

        head_y = int(cy - body_h // 2 - head_r + head_r // 2)
        skin = (230, 195, 160)
        pygame.draw.circle(self._window, (0, 0, 0), (int(cx), head_y), head_r + 1)
        pygame.draw.circle(self._window, skin, (int(cx), head_y), head_r)

        hair_col = tuple(max(0, c - 100) for c in agent.color)
        pygame.draw.arc(
            self._window, hair_col,
            (int(cx - head_r), head_y - head_r, head_r * 2, head_r * 2),
            0.3, 2.8, max(1, head_r // 2),
        )

        eye_r = max(1, head_r // 3)
        eye_off = max(1, head_r // 3)
        pygame.draw.circle(self._window, (30, 30, 40),
                           (int(cx - eye_off), head_y), eye_r)
        pygame.draw.circle(self._window, (30, 30, 40),
                           (int(cx + eye_off), head_y), eye_r)

        if self._zoom >= 0.35:
            label = self._font.render(agent.name, True, (255, 255, 255))
            shadow = self._font.render(agent.name, True, (0, 0, 0))
            lx = int(cx - label.get_width() / 2)
            ly = int(cy + body_h + max(2, shadow_r // 2) + 2)
            self._window.blit(shadow, (lx + 1, ly + 1))
            self._window.blit(label, (lx, ly))

        if agent.pronunciatio and self._zoom >= 0.5:
            bubble = self._small_font.render(agent.pronunciatio, True, (50, 50, 50))
            bw, bh = bubble.get_width() + 10, bubble.get_height() + 6
            bx = int(cx - bw / 2)
            by = int(head_y - head_r - bh - 4)
            pygame.draw.rect(
                self._window, (255, 255, 235), (bx, by, bw, bh), border_radius=5
            )
            pygame.draw.rect(
                self._window, (170, 170, 140), (bx, by, bw, bh), 1, border_radius=5
            )
            tri_cx = int(cx)
            tri_top = by + bh
            pygame.draw.polygon(self._window, (255, 255, 235),
                                [(tri_cx - 3, tri_top),
                                 (tri_cx + 3, tri_top),
                                 (tri_cx, tri_top + 4)])
            self._window.blit(bubble, (bx + 5, by + 3))

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
            bg = pygame.Surface(
                (surf.get_width() + 6, surf.get_height() + 2), pygame.SRCALPHA
            )
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
            zoom_text,
            (self.window_w - zoom_text.get_width() - 8, self.window_h - 20),
        )

    def close(self):
        if self._pygame_inited:
            pygame.quit()
            self._pygame_inited = False
