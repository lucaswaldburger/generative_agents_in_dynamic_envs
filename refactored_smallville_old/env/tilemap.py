"""
Loads a Tiled JSON tilemap and renders it with pygame, replacing the
Phaser.js / Django frontend from the original generative_agents repo.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame


class TilesetInfo:
    """Parsed tileset metadata + loaded pygame surface."""

    def __init__(self, firstgid: int, name: str, image_path: str,
                 tile_width: int, tile_height: int,
                 columns: int, tilecount: int, image_width: int, image_height: int):
        self.firstgid = firstgid
        self.name = name
        self.image_path = image_path
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.columns = columns
        self.tilecount = tilecount
        self.image_width = image_width
        self.image_height = image_height
        self.surface: Optional[pygame.Surface] = None

    def load(self, base_dir: Path) -> None:
        full_path = base_dir / self.image_path
        if not full_path.exists():
            print(f"[WARN] Tileset image not found: {full_path}")
            self.surface = None
            return
        self.surface = pygame.image.load(str(full_path)).convert_alpha()

    def get_tile_rect(self, local_id: int) -> pygame.Rect:
        col = local_id % self.columns
        row = local_id // self.columns
        return pygame.Rect(
            col * self.tile_width,
            row * self.tile_height,
            self.tile_width,
            self.tile_height,
        )


class TiledMapRenderer:
    """
    Loads a Tiled-exported JSON tilemap and its tilesets, then pre-renders
    the visible background layers into a single pygame surface.
    Also parses the collision maze CSV to build a walkability grid.
    """

    def __init__(self, map_json_path: str, collision_csv_path: str,
                 collision_block_id: int = 32125):
        self.map_json_path = Path(map_json_path)
        self.collision_csv_path = Path(collision_csv_path)
        self.collision_block_id = collision_block_id

        with self.map_json_path.open() as f:
            self.map_data: Dict[str, Any] = json.load(f)

        self.map_width: int = self.map_data["width"]
        self.map_height: int = self.map_data["height"]
        self.tile_width: int = self.map_data["tilewidth"]
        self.tile_height: int = self.map_data["tileheight"]

        self.pixel_width = self.map_width * self.tile_width
        self.pixel_height = self.map_height * self.tile_height

        self.tilesets: List[TilesetInfo] = []
        self.layers: List[Dict[str, Any]] = self.map_data["layers"]

        self.collision_grid: np.ndarray = np.zeros(
            (self.map_height, self.map_width), dtype=np.int32
        )
        self._background_surface: Optional[pygame.Surface] = None
        self._foreground_surface: Optional[pygame.Surface] = None

        self._parse_tilesets()
        self._parse_collision()

    def _parse_tilesets(self) -> None:
        visuals_dir = self.map_json_path.parent
        for ts in self.map_data["tilesets"]:
            info = TilesetInfo(
                firstgid=ts["firstgid"],
                name=ts.get("name", ""),
                image_path=ts.get("image", ""),
                tile_width=ts.get("tilewidth", self.tile_width),
                tile_height=ts.get("tileheight", self.tile_height),
                columns=ts.get("columns", 1),
                tilecount=ts.get("tilecount", 0),
                image_width=ts.get("imagewidth", 0),
                image_height=ts.get("imageheight", 0),
            )
            self.tilesets.append(info)
        self.tilesets.sort(key=lambda t: t.firstgid)

    def _parse_collision(self) -> None:
        with self.collision_csv_path.open() as f:
            raw = f.read()
        values = [v.strip() for v in raw.split(",")]
        for idx, v in enumerate(values):
            if not v:
                continue
            if int(v) == self.collision_block_id:
                row = idx // self.map_width
                col = idx % self.map_width
                if row < self.map_height and col < self.map_width:
                    self.collision_grid[row, col] = 1

    def is_blocked(self, tile_x: int, tile_y: int) -> bool:
        if tile_x < 0 or tile_x >= self.map_width:
            return True
        if tile_y < 0 or tile_y >= self.map_height:
            return True
        return bool(self.collision_grid[tile_y, tile_x])

    def load_assets(self) -> None:
        """Load all tileset PNG images into pygame surfaces."""
        visuals_dir = self.map_json_path.parent
        for ts in self.tilesets:
            ts.load(visuals_dir)
        self._render_background()

    def _find_tileset(self, gid: int) -> Optional[TilesetInfo]:
        result = None
        for ts in self.tilesets:
            if ts.firstgid <= gid:
                result = ts
            else:
                break
        return result

    def _render_layer(self, target: pygame.Surface, layer: Dict[str, Any]) -> None:
        data = layer.get("data")
        if data is None:
            return
        for idx, gid in enumerate(data):
            if gid == 0:
                continue
            ts = self._find_tileset(gid)
            if ts is None or ts.surface is None:
                continue
            local_id = gid - ts.firstgid
            if local_id < 0 or local_id >= ts.tilecount:
                continue
            src_rect = ts.get_tile_rect(local_id)
            tile_x = idx % self.map_width
            tile_y = idx // self.map_width
            dest = (tile_x * self.tile_width, tile_y * self.tile_height)
            target.blit(ts.surface, dest, src_rect)

    def _render_background(self) -> None:
        """Pre-render all visible layers below the foreground into one surface."""
        bg_layer_names = {
            "Bottom Ground", "Exterior Ground",
            "Exterior Decoration L1", "Exterior Decoration L2",
            "Interior Ground", "Wall",
            "Interior Furniture L1", "Interior Furniture L2 ",
        }
        fg_layer_names = {"Foreground L1", "Foreground L2"}

        self._background_surface = pygame.Surface(
            (self.pixel_width, self.pixel_height), pygame.SRCALPHA
        )
        self._foreground_surface = pygame.Surface(
            (self.pixel_width, self.pixel_height), pygame.SRCALPHA
        )

        for layer in self.layers:
            name = layer.get("name", "")
            if not layer.get("visible", True) and name not in bg_layer_names | fg_layer_names:
                continue
            if name in bg_layer_names:
                self._render_layer(self._background_surface, layer)
            elif name in fg_layer_names:
                self._render_layer(self._foreground_surface, layer)

    @property
    def background(self) -> pygame.Surface:
        assert self._background_surface is not None, "Call load_assets() first"
        return self._background_surface

    @property
    def foreground(self) -> pygame.Surface:
        assert self._foreground_surface is not None, "Call load_assets() first"
        return self._foreground_surface
