"""
PiedmontEnv – a Gymnasium environment that renders an OSM-derived map of
Piedmont, CA through Pygame.  Adapted from refactored_city's CityEnv.
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
from PIL import Image, ImageDraw, ImageFont

from .osm_map import OSMMap
from .osm_renderer import OSMRenderer, PALETTE, BUILDING_TYPE_COLORS
from .path_finder import closest_coordinate, path_finder

LEGEND_ENTRIES = [
    ("Background", PALETTE["background"]),
    ("Street (narrow)", PALETTE["street_narrow"]),
    ("Street", PALETTE["street"]),
    ("Street (wide)", PALETTE["street_wide"]),
    ("Traffic signal", PALETTE["traffic_signals"]),
    ("Stop sign", PALETTE["stop_sign"]),
    ("Safe zone (park)", PALETTE["safe_zone"]),
    ("Water", PALETTE["water"]),
    ("Railway", PALETTE["railway"]),
    ("Residential", BUILDING_TYPE_COLORS["residential"]),
    ("Commercial", BUILDING_TYPE_COLORS["commercial"]),
    ("Office", BUILDING_TYPE_COLORS["office"]),
    ("School", BUILDING_TYPE_COLORS["school"]),
    ("Hospital", BUILDING_TYPE_COLORS["hospital"]),
    ("Restaurant", BUILDING_TYPE_COLORS["restaurant"]),
    ("Cafe", BUILDING_TYPE_COLORS["cafe"]),
    ("Church", BUILDING_TYPE_COLORS["church"]),
    ("Library", BUILDING_TYPE_COLORS["library"]),
    ("Pharmacy", BUILDING_TYPE_COLORS["pharmacy"]),
    ("Park (bldg)", BUILDING_TYPE_COLORS["park"]),
    ("Other building", PALETTE["building_default"]),
]

_EMOJI_FONT_PATH = "/usr/share/fonts/noto/NotoColorEmoji.ttf"
_EMOJI_NATIVE_SIZE = 109


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
    description: str = ""
    pronunciatio: str = ""
    path: List[Tuple[int, int]] = field(default_factory=list)
    direction: str = "down"
    anim_tick: int = 0
    prev_x: int = 0
    prev_y: int = 0


def _generate_agent_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate *n* visually distinct, saturated colors via HSV hue rotation."""
    import colorsys
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0  # golden-ratio spacing
        sat = 0.7 + (i % 3) * 0.1
        val = 0.85 + (i % 2) * 0.1
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


AGENT_COLORS = _generate_agent_colors(100)

DEFAULT_WINDOW_W = 1280
DEFAULT_WINDOW_H = 800


class PiedmontEnv(gym.Env):
    """Gymnasium environment wrapping the OSM-derived Piedmont map."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

    def __init__(
        self,
        place_name: str = "Piedmont, California, USA",
        grid_size: int = 250,
        agent_names: Optional[List[str]] = None,
        render_mode: str = "human",
        window_w: int = DEFAULT_WINDOW_W,
        window_h: int = DEFAULT_WINDOW_H,
        cache_dir: Optional[str | Path] = None,
        osm_map: Optional[OSMMap] = None,
        spawn_mode: str = "street",
    ):
        super().__init__()
        self.render_mode = render_mode
        self.window_w = window_w
        self.window_h = window_h

        if osm_map is not None:
            self.maze = osm_map
        else:
            self.maze = OSMMap(
                place_name=place_name,
                grid_size=grid_size,
                cache_dir=cache_dir,
            )
        self.renderer = OSMRenderer(self.maze)

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
        self._spawn_mode = spawn_mode
        self.agents: List[AgentState] = []

        self._camera_x: float = 0.0
        self._camera_y: float = 0.0
        self._zoom: float = 1.0
        self._base_scale: float = 1.0
        self._dragging: bool = False
        self._drag_start: Tuple[int, int] = (0, 0)
        self._cam_start: Tuple[float, float] = (0.0, 0.0)

        self._pygame_inited = False
        self._window: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._map_surface: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None
        self._small_font: Optional[pygame.font.Font] = None
        self._hud_font: Optional[pygame.font.Font] = None

        self._emoji_font: Optional[ImageFont.FreeTypeFont] = None
        self._emoji_cache: Dict[Tuple[str, int], pygame.Surface] = {}

        self.sim_step: int = 0
        self.sim_time: str = ""

    @property
    def _effective_zoom(self) -> float:
        """Actual pixel scale: base_scale makes zoom=1.0 fit the grid."""
        return self._base_scale * self._zoom

    def _compute_base_scale(self):
        pw = self.renderer.pixel_width
        ph = self.renderer.pixel_height
        if pw > 0 and ph > 0:
            self._base_scale = min(self.window_w / pw, self.window_h / ph)
        else:
            self._base_scale = 1.0

    def center_camera(self):
        """Center the map in the window at the current zoom level."""
        ez = self._effective_zoom
        map_w = self.renderer.pixel_width * ez
        map_h = self.renderer.pixel_height * ez
        self._camera_x = (self.window_w - map_w) / 2
        self._camera_y = (self.window_h - map_h) / 2

    # ------------------------------------------------------------------
    # Pygame init
    # ------------------------------------------------------------------

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
            pygame.display.set_caption("Piedmont, CA – Agent Simulation")
        else:
            pygame.display.set_mode((1, 1))
            self._window = pygame.Surface((self.window_w, self.window_h))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Arial", 14, bold=True)
        self._small_font = pygame.font.SysFont("Arial", 11)
        self._hud_font = pygame.font.SysFont("Arial", 18, bold=True)
        try:
            self._emoji_font = ImageFont.truetype(
                _EMOJI_FONT_PATH, _EMOJI_NATIVE_SIZE
            )
        except (OSError, IOError):
            self._emoji_font = None
        self._map_surface = self.renderer.build_map_surface()
        self._compute_base_scale()
        self.center_camera()
        self._pygame_inited = True

    # ------------------------------------------------------------------
    # Emoji rendering (via Pillow -> pygame Surface)
    # ------------------------------------------------------------------

    def _render_emoji(self, text: str, size: int) -> pygame.Surface:
        key = (text, size)
        cached = self._emoji_cache.get(key)
        if cached is not None:
            return cached

        if self._emoji_font is not None:
            canvas = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((0, 0), text, font=self._emoji_font, embedded_color=True)
            bbox = canvas.getbbox()
            if bbox:
                cropped = canvas.crop(bbox)
                w, h = cropped.size
                scale = size / max(w, h)
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                scaled = cropped.resize((new_w, new_h), Image.LANCZOS)
                raw = scaled.tobytes()
                surf = pygame.image.frombuffer(raw, scaled.size, "RGBA").convert_alpha()
                self._emoji_cache[key] = surf
                return surf

        surf = self._small_font.render(text, True, (50, 50, 50))
        self._emoji_cache[key] = surf
        return surf

    def _draw_speech_bubble(
        self, text: str, cx: float, anchor_y: float, tile_sz: float
    ):
        emoji_px = max(16, int(tile_sz * 0.7))
        emoji_surf = self._render_emoji(text, emoji_px)
        ew, eh = emoji_surf.get_size()

        pad_x, pad_y = 6, 4
        bw = ew + pad_x * 2
        bh = eh + pad_y * 2
        tail_h = 5

        bx = int(cx - bw / 2)
        by = int(anchor_y - bh - tail_h - 2)

        pygame.draw.rect(
            self._window, (255, 255, 235), (bx, by, bw, bh), border_radius=5
        )
        pygame.draw.rect(
            self._window, (170, 170, 140), (bx, by, bw, bh), 1, border_radius=5
        )

        tri_cx = int(cx)
        tri_top = by + bh
        pygame.draw.polygon(
            self._window,
            (255, 255, 235),
            [(tri_cx - 4, tri_top), (tri_cx + 4, tri_top), (tri_cx, tri_top + tail_h)],
        )
        pygame.draw.line(
            self._window, (170, 170, 140),
            (tri_cx - 4, tri_top), (tri_cx, tri_top + tail_h), 1,
        )
        pygame.draw.line(
            self._window, (170, 170, 140),
            (tri_cx + 4, tri_top), (tri_cx, tri_top + tail_h), 1,
        )

        self._window.blit(emoji_surf, (bx + pad_x, by + pad_y))

    # ------------------------------------------------------------------
    # Agent spawning
    # ------------------------------------------------------------------

    def _spawn_agents(self):
        self.agents = []

        if self._spawn_mode == "building":
            self._spawn_agents_building()
        else:
            self._spawn_agents_street()

    def _spawn_agents_street(self):
        street_tiles: List[Tuple[int, int]] = []
        for y in range(self.maze.maze_height):
            for x in range(self.maze.maze_width):
                if self.maze.tiles[y][x]["tile_type"] == "street":
                    street_tiles.append((x, y))

        used: set[Tuple[int, int]] = set()
        for idx, name in enumerate(self._agent_names):
            color = AGENT_COLORS[idx % len(AGENT_COLORS)]

            tile: Optional[Tuple[int, int]] = None
            if street_tiles:
                candidates = [t for t in street_tiles if t not in used]
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

    def _spawn_agents_building(self):
        """Spawn agents at building tiles, weighted by each building's population."""
        building_tiles: List[Tuple[int, int, int]] = []
        for y in range(self.maze.maze_height):
            for x in range(self.maze.maze_width):
                t = self.maze.tiles[y][x]
                if t["tile_type"] == "building":
                    pop = t.get("population", 1)
                    building_tiles.append((x, y, pop))

        if not building_tiles:
            self._spawn_agents_street()
            return

        coords = [(x, y) for x, y, _ in building_tiles]
        weights = [p for _, _, p in building_tiles]

        used: set[Tuple[int, int]] = set()
        for idx, name in enumerate(self._agent_names):
            color = AGENT_COLORS[idx % len(AGENT_COLORS)]
            available = [(c, w) for c, w in zip(coords, weights) if c not in used]
            if not available:
                available = list(zip(coords, weights))

            av_coords, av_weights = zip(*available)
            tile = random.choices(av_coords, weights=av_weights, k=1)[0]
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

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Pygame event handling
    # ------------------------------------------------------------------

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
                old_ez = self._effective_zoom
                self._zoom *= 1.1 if event.y > 0 else 0.9
                self._zoom = max(0.15, min(6.0, self._zoom))
                new_ez = self._effective_zoom
                mx, my = pygame.mouse.get_pos()
                self._camera_x = mx - (mx - self._camera_x) * (new_ez / old_ez)
                self._camera_y = my - (my - self._camera_y) * (new_ez / old_ez)
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
                    self.center_camera()
        return True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> Optional[np.ndarray]:
        self._ensure_pygame()
        if self._map_surface is None:
            return None

        self._window.fill((25, 25, 30))

        ez = self._effective_zoom
        scaled_w = int(self.renderer.pixel_width * ez)
        scaled_h = int(self.renderer.pixel_height * ez)

        scaled_map = pygame.transform.smoothscale(
            self._map_surface, (scaled_w, scaled_h)
        )

        self._window.blit(scaled_map, (self._camera_x, self._camera_y))

        tile_sz = self.renderer.tile_w * ez
        for agent in self.agents:
            sx = self._camera_x + agent.x * tile_sz
            sy = self._camera_y + agent.y * tile_sz

            if not (
                -tile_sz * 2 < sx < self.window_w + tile_sz
                and -tile_sz * 2 < sy < self.window_h + tile_sz
            ):
                continue

            self._draw_agent_dot(agent, sx, sy, tile_sz)

        self._draw_hud()

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        else:
            frame = pygame.surfarray.array3d(self._window)
            return np.transpose(frame, (1, 0, 2))

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

        ez = self._effective_zoom
        if ez >= 0.35:
            label = self._font.render(agent.name, True, (255, 255, 255))
            shadow = self._font.render(agent.name, True, (0, 0, 0))
            lx = int(cx - label.get_width() / 2)
            ly = int(cy + radius + 4)
            self._window.blit(shadow, (lx + 1, ly + 1))
            self._window.blit(label, (lx, ly))

        if agent.pronunciatio and ez >= 0.2:
            self._draw_speech_bubble(
                agent.pronunciatio, cx, cy - radius, tile_sz
            )

    def _draw_legend(self, top_y: int):
        """Draw a color legend in the top-right corner starting at top_y."""
        pad = 8
        swatch = 10
        gap = 4
        row_h = swatch + gap

        widths = []
        for label, _ in LEGEND_ENTRIES:
            surf = self._small_font.render(label, True, (200, 200, 200))
            widths.append(swatch + 6 + surf.get_width())
        max_w = max(widths) if widths else 80

        panel_w = max_w + pad * 2
        panel_h = len(LEGEND_ENTRIES) * row_h + pad * 2 - gap
        px = self.window_w - panel_w - 8
        py = top_y + 4

        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 170))
        self._window.blit(bg, (px, py))

        cy = py + pad
        for label, color in LEGEND_ENTRIES:
            sx = px + pad
            pygame.draw.rect(self._window, color, (sx, cy, swatch, swatch))
            pygame.draw.rect(self._window, (180, 180, 180), (sx, cy, swatch, swatch), 1)
            txt = self._small_font.render(label, True, (200, 200, 200))
            self._window.blit(txt, (sx + swatch + 6, cy - 1))
            cy += row_h

    def _draw_hud(self):
        legend_top = 6
        if self.sim_time:
            time_label = f"Step {self.sim_step}  |  {self.sim_time}"
            time_surf = self._hud_font.render(time_label, True, (255, 255, 255))
            tw, th = time_surf.get_size()
            pad = 8
            bg = pygame.Surface((tw + pad * 2, th + pad * 2), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 180))
            self._window.blit(bg, (self.window_w - tw - pad * 2 - 8, 6))
            self._window.blit(time_surf, (self.window_w - tw - pad - 8, 6 + pad))
            legend_top = 6 + th + pad * 2

        self._draw_legend(legend_top)

        y_off = 8
        for agent in self.agents:
            tile = self.maze.access_tile((agent.x, agent.y))
            loc_parts = [tile["sector"], tile["arena"]]
            loc_str = " > ".join(p for p in loc_parts if p)
            text = f"{agent.name}: ({agent.x},{agent.y}) [{loc_str}]"
            if agent.description:
                text += f" \u2013 {agent.description}"
            surf = self._small_font.render(text, True, agent.color)
            bg = pygame.Surface(
                (surf.get_width() + 6, surf.get_height() + 2), pygame.SRCALPHA
            )
            bg.fill((0, 0, 0, 160))
            self._window.blit(bg, (4, y_off - 1))
            self._window.blit(surf, (7, y_off))
            y_off += surf.get_height() + 4

        zoom_text = self._small_font.render(
            f"Zoom: {self._effective_zoom:.2f}x  |  Arrow/drag to pan, scroll to zoom, Home to reset",
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
