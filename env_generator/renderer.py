"""
ProceduralRenderer – renders a GeneratedMap using the SmallVille CuteRPG
tileset sprites so generated environments share the same pixel-art aesthetic.

Falls back to flat-colour rendering when tileset assets are unavailable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pygame

from .generated_map import GeneratedMap

# ──────────────────────────────────────────────────────────────────────
# Tileset GID constants (relative to each tileset's firstgid)
# ──────────────────────────────────────────────────────────────────────

# CuteRPG_Field_B  (firstgid = 1, 16 cols, 32x32)
_FB_GRASS       = 2      # local 1  – base grass
# Dirt-path autotile 3x3 (rows 5-7, cols 0-2)
_FB_PATH = {
    "tl": 81, "t": 82, "tr": 83,
    "l": 97, "c": 98, "r": 99,
    "bl": 113, "b": 114, "br": 115,
}
# Inner corners (row 4, cols 3-4 and row 5, cols 3-4)
_FB_PATH_IC = {
    "ic_tl": 68, "ic_tr": 69,
    "ic_bl": 84, "ic_br": 85,
}
# Secondary path / stone border (rows 9-11, cols 0-2)
_FB_PATH2 = {
    "tl": 145, "t": 146, "tr": 147,
    "l": 161, "c": 162, "r": 163,
    "bl": 177, "b": 178, "br": 179,
}
# Flowers / small decorations
_FB_FLOWER1 = 18   # local 17
_FB_FLOWER2 = 50   # local 49
_FB_FLOWER3 = 33   # local 32

# CuteRPG_Field_C  (firstgid = 257, 16 cols, 32x32)
_FC_FLOOR = 490    # local 233 – interior floor (most common)

# CuteRPG_Forest_C (firstgid = 10333, 16 cols, 32x32)
# 2x3 tree pattern (using the two most common tree tile combos)
_TREE_PATTERNS = [
    # Each entry: ((top-left L2, top-right L2), (mid-left L1, mid-right L1), (bot-left L1, bot-right L1))
    ((10458, 10459), (10500, 10501), (10372, 10373)),
    ((10441, 10442), (10356, 10357), (10372, 10373)),
    ((10474, 10475), (10500, 10501), (10356, 10357)),
]

# CuteRPG_Village_B (firstgid = 9053, 16 cols, 32x32)
_VB_ROOF1 = 9080   # local 27 – roof tiles
_VB_FENCE_H = 9279  # local 226 – horizontal fence
_VB_FENCE_V = 9277  # local 224 – vertical fence

# Room_Builder_32x32 (firstgid = 769, 76 cols, 32x32)
# Wall style 1 (Hobbs Cafe - beige/light walls)
_WALL1 = {
    "tl": 6515, "t": 6668, "tr": 6518,
    "tl2": 6591, "t2": 6744, "tr2": 6594,
    "l": 6667, "r": 6670,
}
# Wall style 2 (Lin house - wooden walls)
_WALL2 = {
    "tl": 5011, "t": 5164, "tr": 5014,
    "tl2": 5087, "t2": 5165, "tr2": 5090,
    "l": 5087, "r": 5090,
}
# Wall style 3 (darker brick)
_WALL3 = {
    "tl": 8803, "t": 8804, "tr": 8803,
    "tl2": 8880, "t2": 8804, "tr2": 8880,
    "l": 5832, "r": 5832,
}
_WALL_STYLES = [_WALL1, _WALL2, _WALL3]

# Floor variants from Room_Builder
_RB_FLOORS = [490, 5256, 4951, 4874, 5105, 4881]

# CuteRPG_Harbor_C (firstgid = 513, 16 cols, 32x32)
_HC_WATER = 532    # local 19 – water center

# CuteRPG_Desert_C (firstgid = 9565, 16 cols, 32x32)
_DC_PLAZA = 9598   # local 33 – stone/plaza tile


# ──────────────────────────────────────────────────────────────────────
# Tileset loader
# ──────────────────────────────────────────────────────────────────────

class _TilesetInfo:
    __slots__ = ("name", "firstgid", "tilecount", "columns", "tw", "th", "surface")
    def __init__(self, name, firstgid, tilecount, columns, tw, th, surface):
        self.name = name
        self.firstgid = firstgid
        self.tilecount = tilecount
        self.columns = columns
        self.tw = tw
        self.th = th
        self.surface = surface


class TilesetAtlas:
    """Loads SmallVille tileset PNGs and extracts individual 32x32 tile surfaces."""

    TILESETS = [
        ("CuteRPG_Field_B",     1,     256, 16, "map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Field_B.png"),
        ("CuteRPG_Field_C",     257,   256, 16, "map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Field_C.png"),
        ("CuteRPG_Harbor_C",    513,   256, 16, "map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Harbor_C.png"),
        ("Room_Builder_32x32",  769,  8284, 76, "map_assets/v1/Room_Builder_32x32.png"),
        ("CuteRPG_Village_B",   9053,  256, 16, "map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Village_B.png"),
        ("CuteRPG_Forest_B",    9309,  256, 16, "map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Forest_B.png"),
        ("CuteRPG_Desert_C",    9565,  256, 16, "map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Desert_C.png"),
        ("CuteRPG_Forest_C",    10333, 256, 16, "map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Forest_C.png"),
    ]
    TRANSPARENT = pygame.Color("#ff00ff")

    def __init__(self, visuals_dir: Path):
        self._visuals_dir = visuals_dir
        self._tilesets: List[_TilesetInfo] = []
        self._cache: Dict[int, pygame.Surface] = {}
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        for name, fgid, count, cols, rel_path in self.TILESETS:
            img_path = self._visuals_dir / rel_path
            if not img_path.exists():
                continue
            surf = pygame.image.load(str(img_path)).convert_alpha()
            surf.set_colorkey(self.TRANSPARENT)
            self._tilesets.append(_TilesetInfo(name, fgid, count, cols, 32, 32, surf))
        self._tilesets.sort(key=lambda t: t.firstgid)
        self._loaded = True

    def get_tile(self, gid: int) -> Optional[pygame.Surface]:
        """Return a 32x32 surface for the given global tile ID, or None."""
        self._load()
        cached = self._cache.get(gid)
        if cached is not None:
            return cached
        ts = self._find_ts(gid)
        if ts is None:
            return None
        local = gid - ts.firstgid
        col = local % ts.columns
        row = local // ts.columns
        src = pygame.Rect(col * ts.tw, row * ts.th, ts.tw, ts.th)
        tile_surf = ts.surface.subsurface(src).copy()
        self._cache[gid] = tile_surf
        return tile_surf

    def _find_ts(self, gid: int) -> Optional[_TilesetInfo]:
        result = None
        for ts in self._tilesets:
            if ts.firstgid <= gid:
                result = ts
            else:
                break
        if result and gid < result.firstgid + result.tilecount:
            return result
        return None

    @property
    def available(self) -> bool:
        self._load()
        return len(self._tilesets) > 0


# ──────────────────────────────────────────────────────────────────────
# Renderer
# ──────────────────────────────────────────────────────────────────────

class ProceduralRenderer:
    """Renders a GeneratedMap using SmallVille tileset sprites."""

    TILE_SIZE = 32

    def __init__(self, gen_map: GeneratedMap, assets_dir: Optional[str | Path] = None):
        self.gen_map = gen_map
        self.tile_w = self.TILE_SIZE
        self.tile_h = self.TILE_SIZE
        self.map_w: int = gen_map.maze_width
        self.map_h: int = gen_map.maze_height
        self.map_surface: Optional[pygame.Surface] = None
        self._built = False

        self._assets_dir = Path(assets_dir) if assets_dir else None
        self._atlas: Optional[TilesetAtlas] = None

    def _ensure_atlas(self):
        if self._atlas is not None:
            return
        visuals = None
        if self._assets_dir:
            visuals = self._assets_dir / "the_ville" / "visuals"
        if visuals is None or not visuals.exists():
            candidates = [
                Path(__file__).parent.parent / "refactored_smallville" / "assets" / "the_ville" / "visuals",
                Path(__file__).parent / "assets" / "the_ville" / "visuals",
            ]
            for c in candidates:
                if c.exists():
                    visuals = c
                    break
        if visuals and visuals.exists():
            self._atlas = TilesetAtlas(visuals)
        else:
            self._atlas = None

    def build_map_surface(self) -> pygame.Surface:
        if self._built and self.map_surface is not None:
            return self.map_surface

        self._ensure_atlas()

        tw, th = self.tile_w, self.tile_h
        surf = pygame.Surface((self.pixel_width, self.pixel_height), pygame.SRCALPHA)

        if self._atlas and self._atlas.available:
            self._render_with_tilesets(surf)
        else:
            self._render_fallback(surf)

        self._draw_building_labels(surf)

        self.map_surface = surf
        self._built = True
        return surf

    # ── tileset-based rendering ───────────────────────────────────────

    def _render_with_tilesets(self, surf: pygame.Surface):
        tw, th = self.tile_w, self.tile_h
        tiles = self.gen_map.tiles
        atlas = self._atlas

        # Layer 1: grass base everywhere with subtle variation
        grass = atlas.get_tile(_FB_GRASS)
        grass_alt = atlas.get_tile(226)  # Field_B local 225 – grass variant
        if grass:
            import random
            rng = random.Random(hash(self.gen_map.config.world_name))
            for y in range(self.map_h):
                for x in range(self.map_w):
                    if grass_alt and rng.random() < 0.08:
                        surf.blit(grass_alt, (x * tw, y * th))
                    else:
                        surf.blit(grass, (x * tw, y * th))

        # Layer 2: paths / roads / sidewalks / plazas / parking
        self._render_paths(surf, atlas)

        # Layer 3: building interiors (floors)
        self._render_building_floors(surf, atlas)

        # Layer 4: building walls
        self._render_building_walls(surf, atlas)

        # Layer 5: trees, fences, and decorations
        self._render_trees_and_decorations(surf, atlas)

        # Layer 6: park name labels
        self._draw_park_labels(surf)

    def _is_path_type(self, x: int, y: int) -> bool:
        if 0 <= x < self.map_w and 0 <= y < self.map_h:
            tt = self.gen_map.tiles[y][x]["tile_type"]
            return tt in ("road", "dirt_path", "sidewalk", "corridor",
                          "park_path", "plaza", "crosswalk", "parking")
        return False

    def _render_paths(self, surf: pygame.Surface, atlas: TilesetAtlas):
        tw, th = self.tile_w, self.tile_h
        tiles = self.gen_map.tiles

        for y in range(self.map_h):
            for x in range(self.map_w):
                tt = tiles[y][x]["tile_type"]
                if tt in ("road", "dirt_path", "corridor", "park_path"):
                    if not self._has_any_path_neighbor(x, y):
                        continue
                    gid = self._path_autotile(x, y, _FB_PATH)
                    tile_surf = atlas.get_tile(gid)
                    if tile_surf:
                        surf.blit(tile_surf, (x * tw, y * th))
                elif tt == "sidewalk":
                    gid = self._path_autotile(x, y, _FB_PATH2)
                    tile_surf = atlas.get_tile(gid)
                    if tile_surf:
                        surf.blit(tile_surf, (x * tw, y * th))
                elif tt == "plaza":
                    tile_surf = atlas.get_tile(_DC_PLAZA)
                    if tile_surf:
                        surf.blit(tile_surf, (x * tw, y * th))
                elif tt == "crosswalk":
                    tile_surf = atlas.get_tile(_FB_PATH2["c"])
                    if tile_surf:
                        surf.blit(tile_surf, (x * tw, y * th))
                elif tt == "parking":
                    tile_surf = atlas.get_tile(_DC_PLAZA)
                    if tile_surf:
                        surf.blit(tile_surf, (x * tw, y * th))

    def _has_any_path_neighbor(self, x: int, y: int) -> bool:
        """True if at least one cardinal neighbor is also a path-type tile."""
        return (self._is_path_type(x, y - 1) or self._is_path_type(x, y + 1) or
                self._is_path_type(x - 1, y) or self._is_path_type(x + 1, y))

    def _path_autotile(self, x: int, y: int,
                       palette: Dict[str, int]) -> int:
        """Pick the right autotile variant based on cardinal neighbors."""
        up = self._is_path_type(x, y - 1)
        down = self._is_path_type(x, y + 1)
        left = self._is_path_type(x - 1, y)
        right = self._is_path_type(x + 1, y)

        if up and down and left and right:
            return palette["c"]
        if not up and down and left and right:
            return palette["t"]
        if up and not down and left and right:
            return palette["b"]
        if up and down and not left and right:
            return palette["l"]
        if up and down and left and not right:
            return palette["r"]
        if not up and not down and left and right:
            return palette["c"]
        if up and down and not left and not right:
            return palette["c"]
        if not up and down and not left and right:
            return palette["tl"]
        if not up and down and left and not right:
            return palette["tr"]
        if up and not down and not left and right:
            return palette["bl"]
        if up and not down and left and not right:
            return palette["br"]
        return palette["c"]

    def _render_building_floors(self, surf: pygame.Surface, atlas: TilesetAtlas):
        tw, th = self.tile_w, self.tile_h
        floor_tile = atlas.get_tile(_FC_FLOOR)
        if not floor_tile:
            return

        for bld in self.gen_map.config.buildings:
            for dy in range(bld.h):
                for dx in range(bld.w):
                    x, y = bld.x + dx, bld.y + dy
                    if 0 <= x < self.map_w and 0 <= y < self.map_h:
                        surf.blit(floor_tile, (x * tw, y * th))

    def _render_building_walls(self, surf: pygame.Surface, atlas: TilesetAtlas):
        tw, th = self.tile_w, self.tile_h

        for bi, bld in enumerate(self.gen_map.config.buildings):
            style = _WALL_STYLES[bi % len(_WALL_STYLES)]
            bx, by, bw, bh = bld.x, bld.y, bld.w, bld.h

            # Top wall (2 rows)
            for dx in range(bw):
                x = bx + dx
                if not (0 <= x < self.map_w):
                    continue

                if by >= 0 and by < self.map_h:
                    if dx == 0:
                        gid = style["tl"]
                    elif dx == bw - 1:
                        gid = style["tr"]
                    else:
                        gid = style["t"]
                    t = atlas.get_tile(gid)
                    if t:
                        surf.blit(t, (x * tw, by * th))

                y2 = by + 1
                if 0 <= y2 < self.map_h and bh > 2:
                    if dx == 0:
                        gid = style["tl2"]
                    elif dx == bw - 1:
                        gid = style["tr2"]
                    else:
                        gid = style["t2"]
                    t = atlas.get_tile(gid)
                    if t:
                        surf.blit(t, (x * tw, y2 * th))

            # Side walls (left and right columns, below the top 2 rows)
            start_row = by + 2 if bh > 2 else by + 1
            for dy in range(start_row, by + bh):
                if not (0 <= dy < self.map_h):
                    continue
                lx = bx
                rx = bx + bw - 1
                if 0 <= lx < self.map_w:
                    t = atlas.get_tile(style["l"])
                    if t:
                        surf.blit(t, (lx * tw, dy * th))
                if 0 <= rx < self.map_w:
                    t = atlas.get_tile(style["r"])
                    if t:
                        surf.blit(t, (rx * tw, dy * th))

    def _render_trees_and_decorations(self, surf: pygame.Surface,
                                       atlas: TilesetAtlas):
        tw, th = self.tile_w, self.tile_h

        for park in self.gen_map.config.parks:
            # Fence border around park perimeter
            for dx in range(park.w):
                x = park.x + dx
                if 0 <= x < self.map_w:
                    for y_pos in [park.y - 1, park.y + park.h]:
                        if 0 <= y_pos < self.map_h:
                            t = atlas.get_tile(_VB_FENCE_H)
                            if t:
                                surf.blit(t, (x * tw, y_pos * th))
            for dy in range(-1, park.h + 1):
                y = park.y + dy
                if 0 <= y < self.map_h:
                    for x_pos in [park.x - 1, park.x + park.w]:
                        if 0 <= x_pos < self.map_w:
                            t = atlas.get_tile(_VB_FENCE_V)
                            if t:
                                surf.blit(t, (x_pos * tw, y * th))

            for i, (tx, ty) in enumerate(park.trees):
                pattern = _TREE_PATTERNS[i % len(_TREE_PATTERNS)]
                positions = [
                    (tx, ty - 1, pattern[0]),
                    (tx, ty,     pattern[1]),
                    (tx, ty + 1, pattern[2]),
                ]
                for px, py, (gid_l, gid_r) in positions:
                    if 0 <= py < self.map_h:
                        if 0 <= px < self.map_w:
                            t = atlas.get_tile(gid_l)
                            if t:
                                surf.blit(t, (px * tw, py * th))
                        if 0 <= px + 1 < self.map_w:
                            t = atlas.get_tile(gid_r)
                            if t:
                                surf.blit(t, ((px + 1) * tw, py * th))

        # Sparse flower decorations near buildings (mimics SmallVille style)
        import random
        rng = random.Random(42)
        building_rects = set()
        for bld in self.gen_map.config.buildings:
            for dy in range(-2, bld.h + 2):
                for dx in range(-2, bld.w + 2):
                    building_rects.add((bld.x + dx, bld.y + dy))

        for y in range(self.map_h):
            for x in range(self.map_w):
                if self.gen_map.tiles[y][x]["tile_type"] != "grass":
                    continue
                near_building = (x, y) in building_rects
                chance = 0.06 if near_building else 0.003
                if rng.random() < chance:
                    gid = rng.choice([_FB_FLOWER1, _FB_FLOWER2, _FB_FLOWER3])
                    t = atlas.get_tile(gid)
                    if t:
                        surf.blit(t, (x * tw, y * th))

    def _draw_park_labels(self, surf: pygame.Surface):
        tw, th = self.tile_w, self.tile_h
        try:
            font = pygame.font.SysFont("Arial", max(10, tw // 2), bold=True)
        except Exception:
            return
        for park in self.gen_map.config.parks:
            label = font.render(park.name, True, (255, 255, 240))
            shadow = font.render(park.name, True, (30, 60, 20))
            lx = park.x * tw + (park.w * tw - label.get_width()) // 2
            ly = park.y * th + park.h * th - label.get_height() - 4
            surf.blit(shadow, (lx + 1, ly + 1))
            surf.blit(label, (lx, ly))

    def _draw_building_labels(self, surf: pygame.Surface):
        tw, th = self.tile_w, self.tile_h
        try:
            font = pygame.font.SysFont("Arial", max(8, tw * 2 // 5))
        except Exception:
            return
        for bld in self.gen_map.config.buildings:
            bw_px = bld.w * tw
            bh_px = bld.h * th
            name = bld.name
            label = font.render(name, True, (255, 255, 255))
            shadow = font.render(name, True, (0, 0, 0))
            if label.get_width() > bw_px - 4:
                words = name.split()
                name = words[0][:8] if words else name[:8]
                label = font.render(name, True, (255, 255, 255))
                shadow = font.render(name, True, (0, 0, 0))
            lx = bld.x * tw + (bw_px - label.get_width()) // 2
            ly = bld.y * th + (bh_px - label.get_height()) // 2
            surf.blit(shadow, (lx + 1, ly + 1))
            surf.blit(label, (lx, ly))

    # ── fallback (flat colour) rendering ──────────────────────────────

    def _render_fallback(self, surf: pygame.Surface):
        """Flat-colour fallback when tileset assets aren't available."""
        p = self.gen_map.config.palette
        pal = {
            "road": p.road, "sidewalk": p.sidewalk, "block": p.block,
            "building": p.building, "grass": p.grass, "park": p.park,
            "park_path": p.park_path, "crosswalk": p.crosswalk,
            "dirt_path": p.dirt_path, "parking": p.parking,
            "plaza": (180, 175, 165), "corridor": (120, 115, 110),
        }
        tw, th = self.tile_w, self.tile_h
        surf.fill(pal.get("grass", (75, 150, 65)))

        for y in range(self.map_h):
            for x in range(self.map_w):
                tt = self.gen_map.tiles[y][x]["tile_type"]
                col = pal.get(tt)
                if col:
                    pygame.draw.rect(surf, col, (x * tw, y * th, tw, th))

        for bld in self.gen_map.config.buildings:
            color = bld.color or pal["building"]
            r = pygame.Rect(bld.x * tw, bld.y * th, bld.w * tw, bld.h * th)
            pygame.draw.rect(surf, color, r)
            pygame.draw.rect(surf, _darken(color, 0.55), r, 1)

        for park in self.gen_map.config.parks:
            pr = pygame.Rect(park.x * tw, park.y * th, park.w * tw, park.h * th)
            pygame.draw.rect(surf, pal["park"], pr)

    # ── properties ────────────────────────────────────────────────────

    @property
    def pixel_width(self) -> int:
        return self.map_w * self.tile_w

    @property
    def pixel_height(self) -> int:
        return self.map_h * self.tile_h


def _darken(color: tuple | list, factor: float = 0.6) -> Tuple[int, int, int]:
    return tuple(max(0, int(c * factor)) for c in color[:3])
