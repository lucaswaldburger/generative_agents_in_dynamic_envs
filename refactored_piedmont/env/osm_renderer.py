"""
OSMRenderer – Pre-renders the rasterised OSM map to a pygame Surface.
Provides the same interface as CityRenderer / TiledRenderer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame

from .osm_map import (
    OSMMap,
    GRID_BG, GRID_STREET, GRID_BUILDING, GRID_INTERSECTION,
    GRID_SAFE_ZONE, GRID_WATER, GRID_RAILWAY,
)

BUILDING_TYPE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "residential": (139, 111, 94),
    "house": (139, 111, 94),
    "apartments": (150, 120, 100),
    "detached": (139, 111, 94),
    "commercial": (90, 122, 154),
    "retail": (140, 120, 80),
    "office": (70, 100, 150),
    "industrial": (120, 120, 130),
    "school": (90, 150, 90),
    "church": (160, 140, 120),
    "hospital": (200, 80, 80),
    "restaurant": (180, 80, 60),
    "cafe": (140, 90, 50),
    "fast_food": (190, 100, 50),
    "bar": (120, 50, 80),
    "pub": (120, 50, 80),
    "library": (80, 80, 130),
    "pharmacy": (50, 160, 100),
    "supermarket": (160, 140, 60),
    "convenience": (160, 140, 60),
    "parking": (100, 100, 105),
    "garage": (100, 100, 105),
    "park": (58, 138, 58),
    "building": (110, 105, 120),
}

PALETTE = {
    "background": (42, 42, 42),
    "street": (200, 200, 200),
    "street_narrow": (165, 165, 165),
    "street_wide": (220, 220, 220),
    "building_default": (110, 105, 120),
    "traffic_signals": (220, 50, 50),
    "stop_sign": (200, 80, 40),
    "safe_zone": (75, 160, 75),
    "water": (50, 100, 180),
    "railway": (80, 70, 65),
    "safe_building_border": (40, 180, 40),
}


class OSMRenderer:
    """Pre-renders the OSM-based city map to a single pygame Surface."""

    def __init__(self, osm_map: OSMMap):
        self.osm_map = osm_map
        self.tile_w: int = osm_map.sq_tile_size
        self.tile_h: int = osm_map.sq_tile_size
        self.map_w: int = osm_map.maze_width
        self.map_h: int = osm_map.maze_height
        self.map_surface: Optional[pygame.Surface] = None
        self._built = False

    def _street_color(self, row: int, col: int) -> Tuple[int, int, int]:
        w = int(self.osm_map._street_width_grid[row, col])
        if w >= 5:
            return PALETTE["street_wide"]
        elif w >= 3:
            return PALETTE["street"]
        else:
            return PALETTE["street_narrow"]

    def build_map_surface(self) -> pygame.Surface:
        if self._built and self.map_surface is not None:
            return self.map_surface

        tw, th = self.tile_w, self.tile_h
        pw, ph = self.pixel_width, self.pixel_height

        surf = pygame.Surface((pw, ph))
        surf.fill(PALETTE["background"])

        grid = self.osm_map._grid

        for row in range(self.map_h):
            for col in range(self.map_w):
                val = int(grid[row, col])
                if val == GRID_BG:
                    continue
                elif val == GRID_STREET:
                    color = self._street_color(row, col)
                elif val == GRID_INTERSECTION:
                    tile = self.osm_map.tiles[row][col]
                    sig = tile.get("game_object", "traffic_signals")
                    color = PALETTE["traffic_signals"] if sig == "traffic_signals" else PALETTE["stop_sign"]
                elif val == GRID_SAFE_ZONE:
                    color = PALETTE["safe_zone"]
                elif val == GRID_WATER:
                    color = PALETTE["water"]
                elif val == GRID_RAILWAY:
                    color = PALETTE["railway"]
                elif val == GRID_BUILDING:
                    color = self._building_color(row, col)
                else:
                    continue

                pygame.draw.rect(surf, color, (col * tw, row * th, tw, th))

        self._draw_building_outlines(surf)
        self._draw_safe_building_borders(surf)
        self._draw_building_labels(surf)
        self._draw_railway_dashes(surf)

        self.map_surface = surf
        self._built = True
        return surf

    def _building_color(self, row: int, col: int) -> Tuple[int, int, int]:
        tile = self.osm_map.tiles[row][col]
        btype = tile.get("game_object", "").lower()
        return BUILDING_TYPE_COLORS.get(btype, PALETTE["building_default"])

    def _draw_building_outlines(self, surf: pygame.Surface):
        tw, th = self.tile_w, self.tile_h
        grid = self.osm_map._grid
        h, w = grid.shape

        for row in range(h):
            for col in range(w):
                if grid[row, col] != GRID_BUILDING:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != GRID_BUILDING:
                        base = self._building_color(row, col)
                        darker = tuple(max(0, c - 40) for c in base)
                        px, py = col * tw, row * th
                        if dr == -1:
                            pygame.draw.line(surf, darker, (px, py), (px + tw - 1, py))
                        elif dr == 1:
                            pygame.draw.line(surf, darker, (px, py + th - 1), (px + tw - 1, py + th - 1))
                        elif dc == -1:
                            pygame.draw.line(surf, darker, (px, py), (px, py + th - 1))
                        elif dc == 1:
                            pygame.draw.line(surf, darker, (px + tw - 1, py), (px + tw - 1, py + th - 1))
                        break

    def _draw_safe_building_borders(self, surf: pygame.Surface):
        """Draw a bright green border around buildings flagged as safe zones (hospital/school)."""
        tw, th = self.tile_w, self.tile_h
        grid = self.osm_map._grid
        h, w = grid.shape
        border_color = PALETTE["safe_building_border"]

        for row in range(h):
            for col in range(w):
                if grid[row, col] != GRID_BUILDING:
                    continue
                tile = self.osm_map.tiles[row][col]
                if not tile.get("safe_zone", False):
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != GRID_BUILDING:
                        px, py = col * tw, row * th
                        if dr == -1:
                            pygame.draw.line(surf, border_color, (px, py), (px + tw - 1, py), 2)
                        elif dr == 1:
                            pygame.draw.line(surf, border_color, (px, py + th - 1), (px + tw - 1, py + th - 1), 2)
                        elif dc == -1:
                            pygame.draw.line(surf, border_color, (px, py), (px, py + th - 1), 2)
                        elif dc == 1:
                            pygame.draw.line(surf, border_color, (px + tw - 1, py), (px + tw - 1, py + th - 1), 2)

    def _draw_railway_dashes(self, surf: pygame.Surface):
        """Draw white dashes over railway tiles to visually distinguish them from background."""
        tw, th = self.tile_w, self.tile_h
        grid = self.osm_map._grid
        h, w = grid.shape
        dash_color = (140, 130, 120)

        for row in range(h):
            for col in range(w):
                if grid[row, col] != GRID_RAILWAY:
                    continue
                if (row + col) % 3 == 0:
                    px, py = col * tw + tw // 2, row * th + th // 2
                    pygame.draw.circle(surf, dash_color, (px, py), max(1, tw // 4))

    def _draw_building_labels(self, surf: pygame.Surface):
        tw, th = self.tile_w, self.tile_h
        try:
            font = pygame.font.SysFont("Arial", max(7, tw * 2))
        except Exception:
            return

        for bm in self.osm_map.named_buildings:
            name = bm["name"]
            cr = bm["centroid_row"]
            cc = bm["centroid_col"]

            label = font.render(name, True, (230, 230, 230))
            shadow = font.render(name, True, (0, 0, 0))
            lx = cc * tw - label.get_width() // 2
            ly = cr * th - label.get_height() // 2

            surf.blit(shadow, (lx + 1, ly + 1))
            surf.blit(label, (lx, ly))

    @property
    def pixel_width(self) -> int:
        return self.map_w * self.tile_w

    @property
    def pixel_height(self) -> int:
        return self.map_h * self.tile_h
