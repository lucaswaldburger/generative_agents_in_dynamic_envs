"""
Detailed procedural renderer for the city map, inspired by SmallVille's
RPG pixel-art aesthetic.  Draws textured tiles – cobblestone sidewalks,
brick walls, grass with flowers, patterned building interiors, etc. –
entirely with pygame primitives (no external tileset images needed).
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import pygame

_RNG_SEED_BASE = 42


class DetailedCityRenderer:
    """Pre-renders the urban city map with rich pixel-art-style tiles."""

    def __init__(self, city_map: Any):
        self.city_map = city_map
        self.tile_w: int = city_map.sq_tile_size
        self.tile_h: int = city_map.sq_tile_size
        self.map_w: int = city_map.maze_width
        self.map_h: int = city_map.maze_height
        self.map_surface: Optional[pygame.Surface] = None
        self._built = False

    def build_map_surface(self) -> pygame.Surface:
        if self._built and self.map_surface is not None:
            return self.map_surface

        pw, ph = self.pixel_width, self.pixel_height
        surf = pygame.Surface((pw, ph))

        tw, th = self.tile_w, self.tile_h
        for y in range(self.map_h):
            for x in range(self.map_w):
                tt = self.city_map.tiles[y][x]["tile_type"]
                rng = random.Random(_RNG_SEED_BASE + y * self.map_w + x)
                px, py = x * tw, y * th
                _TILE_DRAW[tt](surf, px, py, tw, th, rng)

        cfg = self.city_map.config
        self._draw_buildings(surf, cfg)
        self._draw_park_overlay(surf, cfg)
        self._draw_intersections(surf, cfg)
        self._draw_lane_markings(surf, cfg)
        self._draw_labels(surf, cfg)

        self.map_surface = surf
        self._built = True
        return surf

    # ── building overlay ────────────────────────────────────────────────

    def _draw_buildings(self, surf: pygame.Surface, cfg: Dict):
        tw, th = self.tile_w, self.tile_h
        for bld in cfg.get("buildings", []):
            color = tuple(bld.get("color", [90, 85, 100]))
            bx = bld["x"] * tw
            by = bld["y"] * th
            bw = bld["w"] * tw
            bh = bld["h"] * th
            btype = bld.get("type", "")

            pygame.draw.rect(surf, color, (bx, by, bw, bh))

            roof_h = max(3, th // 3)
            roof_col = _darken(color, 0.65)
            pygame.draw.rect(surf, roof_col, (bx, by, bw, roof_h))
            highlight = _lighten(roof_col, 1.2)
            pygame.draw.line(surf, highlight, (bx, by), (bx + bw - 1, by))

            outline = _darken(color, 0.5)
            pygame.draw.rect(surf, outline, (bx, by, bw, bh), 1)

            win_col = _WINDOW_COLORS.get(btype, (200, 210, 230))
            frame_col = _darken(color, 0.45)
            win_w, win_h = max(2, tw // 4), max(2, th // 4)
            gap_x = max(win_w + 2, tw // 2)
            gap_y = max(win_h + 2, th // 2)

            wy_start = by + roof_h + 2
            wx_start = bx + 2
            wy = wy_start
            while wy + win_h < by + bh - 1:
                wx = wx_start
                while wx + win_w < bx + bw - 1:
                    rng = random.Random(wx * 777 + wy)
                    lit = rng.random() < 0.6
                    wc = win_col if lit else _darken(win_col, 0.4)
                    pygame.draw.rect(surf, frame_col, (wx - 1, wy - 1, win_w + 2, win_h + 2))
                    pygame.draw.rect(surf, wc, (wx, wy, win_w, win_h))
                    if lit and win_w >= 3:
                        pygame.draw.line(surf, (255, 255, 240, 80),
                                         (wx, wy), (wx + win_w - 1, wy))
                    wx += gap_x
                wy += gap_y

            if btype in _TYPE_AWNING:
                aw_col = _TYPE_AWNING[btype]
                aw_h = max(2, th // 4)
                aw_y = by + bh - aw_h - 1
                pygame.draw.rect(surf, aw_col, (bx + 1, aw_y, bw - 2, aw_h))
                pygame.draw.line(surf, _darken(aw_col, 0.6),
                                 (bx + 1, aw_y + aw_h - 1), (bx + bw - 2, aw_y + aw_h - 1))

            if btype in ("hospital", "government", "transit"):
                icon_col = (255, 255, 255) if btype == "hospital" else (200, 200, 180)
                icx = bx + bw // 2
                icy = by + roof_h // 2
                if btype == "hospital":
                    cr = max(2, roof_h // 2 - 1)
                    pygame.draw.line(surf, (255, 60, 60),
                                     (icx - cr, icy), (icx + cr, icy), 2)
                    pygame.draw.line(surf, (255, 60, 60),
                                     (icx, icy - cr), (icx, icy + cr), 2)

    def _draw_park_overlay(self, surf: pygame.Surface, cfg: Dict):
        park = cfg.get("park")
        if not park:
            return
        tw, th = self.tile_w, self.tile_h

        for tx, ty in park.get("trees", []):
            cx = tx * tw + tw // 2
            cy = ty * th + th // 2

            trunk_col = (100, 65, 30)
            trunk_w = max(2, tw // 5)
            trunk_h = max(4, th // 2)
            pygame.draw.rect(surf, trunk_col,
                             (cx - trunk_w // 2, cy + 1, trunk_w, trunk_h))
            pygame.draw.rect(surf, _darken(trunk_col, 0.7),
                             (cx - trunk_w // 2, cy + 1, 1, trunk_h))

            r = tw // 2 + 3
            rng = random.Random(tx * 100 + ty)
            for ox, oy, dr in [(-2, -2, r - 1), (2, -1, r - 2), (0, -3, r)]:
                shade = (25 + rng.randint(0, 20),
                         90 + rng.randint(0, 30),
                         20 + rng.randint(0, 15))
                pygame.draw.circle(surf, shade, (cx + ox, cy + oy), dr)
            canopy = (45 + rng.randint(0, 15),
                      130 + rng.randint(0, 25),
                      35 + rng.randint(0, 15))
            pygame.draw.circle(surf, canopy, (cx, cy - 2), r - 2)

            for _ in range(3):
                hx = cx + rng.randint(-r + 2, r - 2)
                hy = cy - 2 + rng.randint(-r + 2, r - 2)
                pygame.draw.circle(surf, _lighten(canopy, 1.15), (hx, hy), 1)

        font = pygame.font.SysFont("Arial", max(10, tw), bold=True)
        lbl = font.render(park["name"], True, (235, 250, 235))
        shadow = font.render(park["name"], True, (20, 60, 20))
        lx = park["x"] * tw + (park["w"] * tw - lbl.get_width()) // 2
        ly = park["y"] * th + park["h"] * th - lbl.get_height() - 3
        surf.blit(shadow, (lx + 1, ly + 1))
        surf.blit(lbl, (lx, ly))

    def _draw_intersections(self, surf: pygame.Surface, cfg: Dict):
        tw, th = self.tile_w, self.tile_h
        stripe_w = max(2, tw // 3)
        stripe_gap = max(2, tw // 4)
        for inter in cfg.get("intersections", []):
            ix, iy = inter["x"] * tw, inter["y"] * th
            iw, ih = inter["w"] * tw, inter["h"] * th

            x = ix
            while x < ix + iw:
                sw = min(stripe_w, ix + iw - x)
                pygame.draw.rect(surf, (230, 230, 225), (x, iy, sw, ih))
                x += stripe_w + stripe_gap

            y = iy
            while y < iy + ih:
                sh = min(stripe_w, iy + ih - y)
                pygame.draw.rect(surf, (230, 230, 225), (ix, y, iw, sh))
                y += stripe_w + stripe_gap

    def _draw_lane_markings(self, surf: pygame.Surface, cfg: Dict):
        tw, th = self.tile_w, self.tile_h
        mark_col = (230, 220, 80)
        edge_col = (200, 200, 200)

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
                dash, gap = tw, tw - 2
                x = x_start
                while x < x_end:
                    end = min(x + dash, x_end)
                    pygame.draw.line(surf, mark_col, (x, y_px), (end, y_px), 1)
                    x += dash + gap
                pygame.draw.line(surf, edge_col,
                                 (x_start, sy * th), (x_end, sy * th), 1)
                pygame.draw.line(surf, edge_col,
                                 (x_start, sy * th + th - 1), (x_end, sy * th + th - 1), 1)
            elif sx == ex:
                x_px = sx * tw + tw // 2
                y_start = min(sy, ey) * th
                y_end = max(sy, ey) * th + th
                dash, gap = th, th - 2
                y = y_start
                while y < y_end:
                    end = min(y + dash, y_end)
                    pygame.draw.line(surf, mark_col, (x_px, y), (x_px, end), 1)
                    y += dash + gap
                pygame.draw.line(surf, edge_col,
                                 (sx * tw, y_start), (sx * tw, y_end), 1)
                pygame.draw.line(surf, edge_col,
                                 (sx * tw + tw - 1, y_start), (sx * tw + tw - 1, y_end), 1)

    def _draw_labels(self, surf: pygame.Surface, cfg: Dict):
        tw, th = self.tile_w, self.tile_h
        try:
            font = pygame.font.SysFont("Arial", max(7, tw * 2 // 3))
            shadow_font = font
        except Exception:
            return

        for bld in cfg.get("buildings", []):
            bw_px = bld["w"] * tw
            bh_px = bld["h"] * th

            name = bld["name"]
            label = font.render(name, True, (245, 245, 240))
            if label.get_width() > bw_px - 4:
                words = name.split()
                name = words[0][:7] if words else name[:7]
                label = font.render(name, True, (245, 245, 240))

            shadow = shadow_font.render(name, True, (20, 20, 20))
            bx = bld["x"] * tw + (bw_px - label.get_width()) // 2
            by = bld["y"] * th + (bh_px - label.get_height()) // 2
            surf.blit(shadow, (bx + 1, by + 1))
            surf.blit(label, (bx, by))

    @property
    def pixel_width(self) -> int:
        return self.map_w * self.tile_w

    @property
    def pixel_height(self) -> int:
        return self.map_h * self.tile_h


# ── per-tile drawing functions ──────────────────────────────────────────

def _draw_road(surf: pygame.Surface, px: int, py: int,
               tw: int, th: int, rng: random.Random):
    base_r = 58 + rng.randint(-4, 4)
    base_g = 58 + rng.randint(-4, 4)
    base_b = 62 + rng.randint(-4, 4)
    pygame.draw.rect(surf, (base_r, base_g, base_b), (px, py, tw, th))
    for _ in range(rng.randint(2, 5)):
        dx = rng.randint(0, tw - 1)
        dy = rng.randint(0, th - 1)
        v = rng.randint(-12, 6)
        c = (max(0, min(255, base_r + v)),
             max(0, min(255, base_g + v)),
             max(0, min(255, base_b + v)))
        surf.set_at((px + dx, py + dy), c)
    if rng.random() < 0.03:
        mx = px + tw // 2
        my = py + th // 2
        r = max(1, tw // 5)
        pygame.draw.circle(surf, (45, 45, 50), (mx, my), r)
        pygame.draw.circle(surf, (55, 55, 60), (mx, my), max(1, r - 1))


def _draw_sidewalk(surf: pygame.Surface, px: int, py: int,
                   tw: int, th: int, rng: random.Random):
    base = (152 + rng.randint(-6, 6),
            150 + rng.randint(-6, 6),
            145 + rng.randint(-6, 6))
    pygame.draw.rect(surf, base, (px, py, tw, th))

    gap_col = _darken(base, 0.78)
    step = max(3, tw // 4)
    for gx in range(step, tw, step):
        pygame.draw.line(surf, gap_col, (px + gx, py), (px + gx, py + th - 1))
    for gy in range(step, th, step):
        offset = step // 2 if (gy // step) % 2 else 0
        pygame.draw.line(surf, gap_col, (px + offset, py + gy),
                         (px + tw - 1, py + gy))

    for _ in range(rng.randint(0, 2)):
        dx = rng.randint(1, tw - 2)
        dy = rng.randint(1, th - 2)
        surf.set_at((px + dx, py + dy), _darken(base, 0.85))


def _draw_block(surf: pygame.Surface, px: int, py: int,
                tw: int, th: int, rng: random.Random):
    base = (75 + rng.randint(-3, 3),
            55 + rng.randint(-3, 3),
            50 + rng.randint(-3, 3))
    pygame.draw.rect(surf, base, (px, py, tw, th))

    mortar = _darken(base, 0.6)
    brick_h = max(2, th // 4)
    brick_w = max(3, tw // 2)

    row = 0
    y = py
    while y < py + th:
        bh = min(brick_h, py + th - y)
        offset = (brick_w // 2) if row % 2 else 0
        x = px - offset
        while x < px + tw:
            bw = min(brick_w, px + tw - max(x, px))
            rx = max(x, px)
            if bw > 0 and bh > 0:
                v = rng.randint(-8, 8)
                bc = (max(0, min(255, base[0] + v)),
                      max(0, min(255, base[1] + v)),
                      max(0, min(255, base[2] + v)))
                pygame.draw.rect(surf, bc, (rx, y, bw, bh))
                pygame.draw.rect(surf, mortar, (rx, y, bw, bh), 1)
            x += brick_w
        y += brick_h
        row += 1


def _draw_building_base(surf: pygame.Surface, px: int, py: int,
                        tw: int, th: int, rng: random.Random):
    base = (115 + rng.randint(-5, 5),
            108 + rng.randint(-5, 5),
            100 + rng.randint(-5, 5))
    pygame.draw.rect(surf, base, (px, py, tw, th))
    step = max(3, tw // 3)
    col_a = _darken(base, 0.92)
    col_b = _lighten(base, 1.05)
    for ty in range(0, th, step):
        for tx in range(0, tw, step):
            c = col_a if ((tx // step) + (ty // step)) % 2 == 0 else col_b
            sw = min(step, tw - tx)
            sh = min(step, th - ty)
            pygame.draw.rect(surf, c, (px + tx, py + ty, sw, sh))


def _draw_grass(surf: pygame.Surface, px: int, py: int,
                tw: int, th: int, rng: random.Random):
    base_g = 128 + rng.randint(-15, 15)
    base = (35 + rng.randint(-5, 5), base_g, 40 + rng.randint(-5, 5))
    pygame.draw.rect(surf, base, (px, py, tw, th))

    for _ in range(rng.randint(6, 12)):
        dx = rng.randint(0, tw - 1)
        dy = rng.randint(0, th - 1)
        v = rng.randint(-15, 20)
        blade = (max(0, base[0] + v // 2),
                 max(0, min(255, base[1] + v)),
                 max(0, base[2] + v // 2))
        surf.set_at((px + dx, py + dy), blade)

    if rng.random() < 0.12:
        fx = px + rng.randint(2, tw - 3)
        fy = py + rng.randint(2, th - 3)
        fc = rng.choice([(240, 220, 60), (220, 80, 120), (200, 60, 200),
                         (255, 160, 40), (255, 255, 255)])
        surf.set_at((fx, fy), fc)
        surf.set_at((fx + 1, fy), fc)
        surf.set_at((fx, fy + 1), _darken(fc, 0.7))


def _draw_park_path(surf: pygame.Surface, px: int, py: int,
                    tw: int, th: int, rng: random.Random):
    base = (175 + rng.randint(-8, 8),
            160 + rng.randint(-8, 8),
            125 + rng.randint(-8, 8))
    pygame.draw.rect(surf, base, (px, py, tw, th))

    edge_col = _darken(base, 0.8)
    pygame.draw.line(surf, edge_col, (px, py), (px + tw - 1, py))
    pygame.draw.line(surf, edge_col, (px, py + th - 1), (px + tw - 1, py + th - 1))

    for _ in range(rng.randint(3, 7)):
        dx = rng.randint(1, tw - 2)
        dy = rng.randint(1, th - 2)
        v = rng.randint(-15, 5)
        pebble = (max(0, base[0] + v), max(0, base[1] + v), max(0, base[2] + v))
        surf.set_at((px + dx, py + dy), pebble)


def _draw_corridor(surf: pygame.Surface, px: int, py: int,
                   tw: int, th: int, rng: random.Random):
    base = (135 + rng.randint(-4, 4),
            128 + rng.randint(-4, 4),
            118 + rng.randint(-4, 4))
    pygame.draw.rect(surf, base, (px, py, tw, th))

    plank_h = max(2, th // 3)
    plank_col = _darken(base, 0.88)
    for gy in range(0, th, plank_h):
        ph = min(plank_h, th - gy)
        pygame.draw.line(surf, plank_col, (px, py + gy), (px + tw - 1, py + gy))
        if rng.random() < 0.4:
            kx = px + rng.randint(tw // 4, 3 * tw // 4)
            pygame.draw.line(surf, plank_col, (kx, py + gy), (kx, py + gy + ph - 1))


def _draw_crosswalk(surf: pygame.Surface, px: int, py: int,
                    tw: int, th: int, rng: random.Random):
    base = (55 + rng.randint(-3, 3), 55 + rng.randint(-3, 3), 60 + rng.randint(-3, 3))
    pygame.draw.rect(surf, base, (px, py, tw, th))

    stripe_col = (225, 225, 220)
    stripe_w = max(2, tw // 3)
    stripe_gap = max(1, tw // 4)
    x = px
    while x < px + tw:
        sw = min(stripe_w, px + tw - x)
        pygame.draw.rect(surf, stripe_col, (x, py, sw, th))
        x += stripe_w + stripe_gap


_TILE_DRAW = {
    "road":       _draw_road,
    "sidewalk":   _draw_sidewalk,
    "block":      _draw_block,
    "building":   _draw_building_base,
    "park":       _draw_grass,
    "park_path":  _draw_park_path,
    "corridor":   _draw_corridor,
    "crosswalk":  _draw_crosswalk,
}

_TYPE_AWNING = {
    "cafe":        (160, 60, 40),
    "restaurant":  (180, 70, 50),
    "bar":         (80, 40, 100),
    "shop":        (60, 120, 80),
    "pharmacy":    (40, 140, 100),
}

_WINDOW_COLORS = {
    "residential": (200, 210, 180),
    "office":      (180, 200, 230),
    "cafe":        (240, 220, 160),
    "restaurant":  (240, 200, 140),
    "bar":         (160, 120, 180),
    "shop":        (210, 220, 200),
    "hospital":    (200, 230, 230),
    "school":      (200, 220, 190),
    "gym":         (220, 220, 200),
    "library":     (200, 200, 220),
    "hotel":       (230, 220, 180),
    "government":  (200, 200, 210),
    "transit":     (190, 210, 230),
    "studio":      (220, 190, 220),
    "entertainment": (200, 160, 220),
    "community":   (210, 200, 180),
}


def _darken(color: tuple | list, factor: float = 0.6) -> Tuple[int, int, int]:
    return tuple(max(0, int(c * factor)) for c in color[:3])


def _lighten(color: tuple | list, factor: float = 1.3) -> Tuple[int, int, int]:
    return tuple(min(255, int(c * factor)) for c in color[:3])
