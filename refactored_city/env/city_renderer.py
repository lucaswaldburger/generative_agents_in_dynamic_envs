"""
Procedural Pygame renderer for the city map.
Replaces tiled_renderer.py – draws roads, buildings, park, crosswalks,
lane markings, and building labels using pygame primitives.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pygame


class CityRenderer:
    """Pre-renders the urban city map to a pygame Surface."""

    def __init__(self, city_map: Any):
        self.city_map = city_map
        self.tile_w: int = city_map.sq_tile_size
        self.tile_h: int = city_map.sq_tile_size
        self.map_w: int = city_map.maze_width
        self.map_h: int = city_map.maze_height
        self.map_surface: Optional[pygame.Surface] = None
        self._built = False

    def build_map_surface(self) -> pygame.Surface:
        """Render the city to a single surface. Call once after pygame.init()."""
        if self._built and self.map_surface is not None:
            return self.map_surface

        cfg = self.city_map.config
        pal = cfg["palette"]
        tw, th = self.tile_w, self.tile_h
        pw, ph = self.pixel_width, self.pixel_height

        surf = pygame.Surface((pw, ph))
        surf.fill(pal["road"])

        self._draw_sidewalks(surf, pal)
        self._draw_blocks(surf, cfg, pal)
        self._draw_corridors(surf, pal)
        self._draw_park(surf, cfg, pal)
        self._draw_buildings(surf, cfg, pal)
        self._draw_intersections(surf, cfg, pal)
        self._draw_lane_markings(surf, cfg, pal)
        self._draw_building_labels(surf, cfg)

        self.map_surface = surf
        self._built = True
        return surf

    def _draw_sidewalks(self, surf: pygame.Surface, pal: Dict):
        tw, th = self.tile_w, self.tile_h
        col = pal["sidewalk"]
        for y in range(self.map_h):
            for x in range(self.map_w):
                if self.city_map.tiles[y][x]["tile_type"] == "sidewalk":
                    pygame.draw.rect(surf, col, (x * tw, y * th, tw, th))

    def _draw_corridors(self, surf: pygame.Surface, pal: Dict):
        tw, th = self.tile_w, self.tile_h
        col = (120, 115, 110)
        for y in range(self.map_h):
            for x in range(self.map_w):
                if self.city_map.tiles[y][x]["tile_type"] == "corridor":
                    pygame.draw.rect(surf, col, (x * tw, y * th, tw, th))

    def _draw_blocks(self, surf: pygame.Surface, cfg: Dict, pal: Dict):
        tw, th = self.tile_w, self.tile_h
        for block in cfg.get("blocks", []):
            r = pygame.Rect(
                block["x"] * tw, block["y"] * th,
                block["w"] * tw, block["h"] * th,
            )
            pygame.draw.rect(surf, pal["block"], r)
            pygame.draw.rect(surf, _darken(pal["block"], 0.6), r, 1)

    def _draw_park(self, surf: pygame.Surface, cfg: Dict, pal: Dict):
        park = cfg.get("park")
        if not park:
            return
        tw, th = self.tile_w, self.tile_h

        pr = pygame.Rect(
            park["x"] * tw, park["y"] * th,
            park["w"] * tw, park["h"] * th,
        )
        pygame.draw.rect(surf, pal["park"], pr)

        for path_rect in park.get("paths", []):
            r = pygame.Rect(
                path_rect["x"] * tw, path_rect["y"] * th,
                path_rect["w"] * tw, path_rect["h"] * th,
            )
            pygame.draw.rect(surf, pal["park_path"], r)

        trunk_col = (90, 60, 30)
        canopy_dark = (25, 95, 25)
        canopy_light = (40, 130, 40)
        for tx, ty in park.get("trees", []):
            cx = tx * tw + tw // 2
            cy = ty * th + th // 2
            trunk_w, trunk_h = max(2, tw // 5), max(3, th // 3)
            pygame.draw.rect(
                surf, trunk_col,
                (cx - trunk_w // 2, cy, trunk_w, trunk_h),
            )
            r = tw // 2 + 2
            pygame.draw.circle(surf, canopy_dark, (cx, cy - 1), r)
            pygame.draw.circle(surf, canopy_light, (cx, cy - 1), r - 2)

        font = pygame.font.SysFont("Arial", max(9, tw - 2), bold=True)
        label = font.render(park["name"], True, (220, 240, 220))
        lx = park["x"] * tw + (park["w"] * tw - label.get_width()) // 2
        ly = park["y"] * th + park["h"] * th - label.get_height() - 2
        surf.blit(label, (lx, ly))

    def _draw_buildings(self, surf: pygame.Surface, cfg: Dict, pal: Dict):
        tw, th = self.tile_w, self.tile_h
        for bld in cfg.get("buildings", []):
            color = bld.get("color", pal["building"])
            r = pygame.Rect(
                bld["x"] * tw, bld["y"] * th,
                bld["w"] * tw, bld["h"] * th,
            )
            pygame.draw.rect(surf, color, r)
            pygame.draw.rect(surf, _darken(color, 0.55), r, 1)

            roof_h = max(2, th // 4)
            roof_col = _darken(color, 0.75)
            pygame.draw.rect(surf, roof_col, (r.x, r.y, r.w, roof_h))

    def _draw_intersections(self, surf: pygame.Surface, cfg: Dict, pal: Dict):
        tw, th = self.tile_w, self.tile_h
        stripe_col = pal["crosswalk"]
        for inter in cfg.get("intersections", []):
            ix, iy = inter["x"] * tw, inter["y"] * th
            iw, ih = inter["w"] * tw, inter["h"] * th

            stripe_w = max(2, tw // 3)
            gap = max(2, tw // 3)
            x = ix
            while x < ix + iw:
                pygame.draw.rect(
                    surf, stripe_col,
                    (x, iy, stripe_w, ih),
                )
                x += stripe_w + gap

    def _draw_lane_markings(self, surf: pygame.Surface, cfg: Dict, pal: Dict):
        tw, th = self.tile_w, self.tile_h
        mark_col = pal.get("lane_mark", (200, 200, 60))

        for lane in cfg.get("car_lanes", []):
            wps = lane.get("waypoints", [])
            if len(wps) < 2:
                continue
            sx, sy = wps[0]
            ex, ey = wps[-1]

            if sy == ey:
                y_px = sy * th + th // 2
                x_start = min(sx, ex) * tw
                x_end = max(sx, ex) * tw + tw
                dash_len = tw
                gap_len = tw
                x = x_start
                while x < x_end:
                    pygame.draw.line(
                        surf, mark_col,
                        (x, y_px), (min(x + dash_len, x_end), y_px), 1,
                    )
                    x += dash_len + gap_len
            elif sx == ex:
                x_px = sx * tw + tw // 2
                y_start = min(sy, ey) * th
                y_end = max(sy, ey) * th + th
                dash_len = th
                gap_len = th
                y = y_start
                while y < y_end:
                    pygame.draw.line(
                        surf, mark_col,
                        (x_px, y), (x_px, min(y + dash_len, y_end)), 1,
                    )
                    y += dash_len + gap_len

    def _draw_building_labels(self, surf: pygame.Surface, cfg: Dict):
        tw, th = self.tile_w, self.tile_h
        try:
            font = pygame.font.SysFont("Arial", max(7, tw * 2 // 3))
        except Exception:
            return

        for bld in cfg.get("buildings", []):
            bw_px = bld["w"] * tw
            bh_px = bld["h"] * th

            name = bld["name"]
            label = font.render(name, True, (230, 230, 230))
            if label.get_width() > bw_px - 2:
                words = name.split()
                name = words[0][:6] if words else name[:6]
                label = font.render(name, True, (230, 230, 230))

            bx = bld["x"] * tw + (bw_px - label.get_width()) // 2
            by = bld["y"] * th + (bh_px - label.get_height()) // 2
            surf.blit(label, (bx, by))

    @property
    def pixel_width(self) -> int:
        return self.map_w * self.tile_w

    @property
    def pixel_height(self) -> int:
        return self.map_h * self.tile_h


def _darken(color: list | tuple, factor: float = 0.6) -> Tuple[int, int, int]:
    return tuple(max(0, int(c * factor)) for c in color[:3])
