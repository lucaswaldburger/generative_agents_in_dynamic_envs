"""
Pygame-based renderer for Tiled JSON maps.
Loads the SmallVille Tiled map (the_ville_jan7.json) and its tileset images,
then pre-renders all visual layers to a single large pygame Surface.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame


@dataclass
class _Tileset:
    name: str
    firstgid: int
    tilecount: int
    columns: int
    tile_w: int
    tile_h: int
    surface: pygame.Surface


VISUAL_LAYERS = [
    "Bottom Ground",
    "Exterior Ground",
    "Exterior Decoration L1",
    "Exterior Decoration L2",
    "Interior Ground",
    "Wall",
    "Interior Furniture L1",
    "Interior Furniture L2 ",
    "Foreground L1",
    "Foreground L2",
]


class TiledRenderer:
    """Pre-renders the SmallVille Tiled map to a pygame Surface."""

    def __init__(self, assets_dir: str | Path):
        self._assets_dir = Path(assets_dir)
        self._visuals_dir = self._assets_dir / "the_ville" / "visuals"
        map_path = self._visuals_dir / "the_ville_jan7.json"

        with open(map_path) as f:
            self.map_data = json.load(f)

        self.map_w: int = self.map_data["width"]
        self.map_h: int = self.map_data["height"]
        self.tile_w: int = self.map_data["tilewidth"]
        self.tile_h: int = self.map_data["tileheight"]

        self._tilesets: List[_Tileset] = []
        self._tilesets_loaded = False

        self._layers: Dict[str, List[int]] = {}
        for layer in self.map_data["layers"]:
            if layer["type"] == "tilelayer":
                self._layers[layer["name"]] = layer["data"]

        self.map_surface: Optional[pygame.Surface] = None

    def _load_tilesets(self):
        """Load tileset images. Must be called after pygame.display is initialized."""
        if self._tilesets_loaded:
            return
        for ts_data in self.map_data["tilesets"]:
            img_path = self._visuals_dir / ts_data["image"]
            if not img_path.exists():
                continue
            surface = pygame.image.load(str(img_path)).convert_alpha()
            transparent = ts_data.get("transparentcolor")
            if transparent:
                surface.set_colorkey(pygame.Color(transparent))
            self._tilesets.append(
                _Tileset(
                    name=ts_data["name"],
                    firstgid=ts_data["firstgid"],
                    tilecount=ts_data["tilecount"],
                    columns=ts_data["columns"],
                    tile_w=ts_data.get("tilewidth", self.tile_w),
                    tile_h=ts_data.get("tileheight", self.tile_h),
                    surface=surface,
                )
            )
        self._tilesets.sort(key=lambda ts: ts.firstgid)
        self._tilesets_loaded = True

    def _find_tileset(self, gid: int) -> Optional[_Tileset]:
        result = None
        for ts in self._tilesets:
            if ts.firstgid <= gid:
                result = ts
            else:
                break
        if result and gid < result.firstgid + result.tilecount:
            return result
        return None

    def _get_tile_rect(self, gid: int, ts: _Tileset) -> pygame.Rect:
        local_id = gid - ts.firstgid
        col = local_id % ts.columns
        row = local_id // ts.columns
        return pygame.Rect(col * ts.tile_w, row * ts.tile_h, ts.tile_w, ts.tile_h)

    def build_map_surface(self) -> pygame.Surface:
        """Render all visual layers to a single surface. Call once after pygame.init()."""
        self._load_tilesets()
        surf = pygame.Surface(
            (self.map_w * self.tile_w, self.map_h * self.tile_h), pygame.SRCALPHA
        )
        surf.fill((0, 0, 0, 255))

        for layer_name in VISUAL_LAYERS:
            data = self._layers.get(layer_name)
            if data is None:
                continue
            for idx, gid in enumerate(data):
                if gid == 0:
                    continue
                ts = self._find_tileset(gid)
                if ts is None:
                    continue
                src_rect = self._get_tile_rect(gid, ts)
                tx = (idx % self.map_w) * self.tile_w
                ty = (idx // self.map_w) * self.tile_h
                surf.blit(ts.surface, (tx, ty), src_rect)

        self.map_surface = surf
        return surf

    @property
    def pixel_width(self) -> int:
        return self.map_w * self.tile_w

    @property
    def pixel_height(self) -> int:
        return self.map_h * self.tile_h
