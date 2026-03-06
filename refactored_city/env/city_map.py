"""
CityMap – loads a JSON city layout and builds tile / address / collision
structures with the same interface as refactored_smallville's Maze.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class CityMap:
    """2-D tile map loaded from a JSON city layout definition."""

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        with open(config_path) as f:
            self.config: Dict[str, Any] = json.load(f)

        self.maze_width: int = self.config["width"]
        self.maze_height: int = self.config["height"]
        self.sq_tile_size: int = self.config["tile_size"]

        self._init_tiles()
        self._place_blocks()
        self._place_park()
        self._place_buildings()
        self._add_entrances()
        self._place_intersections()
        self._mark_sidewalks()
        self._build_address_tiles()

        self.car_lanes: List[Dict[str, Any]] = self.config.get("car_lanes", [])
        self.intersections_data: List[Dict[str, Any]] = self.config.get("intersections", [])

    def _init_tiles(self):
        self.tiles: List[List[Dict[str, Any]]] = []
        self.collision_maze: List[List[str]] = []
        for _y in range(self.maze_height):
            tile_row: List[Dict[str, Any]] = []
            col_row: List[str] = []
            for _x in range(self.maze_width):
                tile_row.append({
                    "world": "the City",
                    "sector": "",
                    "arena": "",
                    "game_object": "",
                    "spawning_location": "",
                    "collision": False,
                    "events": set(),
                    "tile_type": "road",
                })
                col_row.append("0")
            self.tiles.append(tile_row)
            self.collision_maze.append(col_row)

    def _fill_rect(
        self,
        x0: int, y0: int, w: int, h: int,
        *,
        sector: str = "",
        arena: str = "",
        game_object: str = "",
        collision: bool = False,
        tile_type: str = "road",
    ):
        for dy in range(h):
            for dx in range(w):
                x, y = x0 + dx, y0 + dy
                if 0 <= x < self.maze_width and 0 <= y < self.maze_height:
                    t = self.tiles[y][x]
                    if sector:
                        t["sector"] = sector
                    if arena:
                        t["arena"] = arena
                    if game_object:
                        t["game_object"] = game_object
                    t["collision"] = collision
                    t["tile_type"] = tile_type
                    self.collision_maze[y][x] = "1" if collision else "0"

    def _place_blocks(self):
        for block in self.config.get("blocks", []):
            self._fill_rect(
                block["x"], block["y"], block["w"], block["h"],
                sector=block["name"],
                collision=True,
                tile_type="block",
            )

    def _place_park(self):
        park = self.config.get("park")
        if not park:
            return
        self._fill_rect(
            park["x"], park["y"], park["w"], park["h"],
            sector=park["name"],
            arena="green space",
            collision=False,
            tile_type="park",
        )
        for path_rect in park.get("paths", []):
            self._fill_rect(
                path_rect["x"], path_rect["y"], path_rect["w"], path_rect["h"],
                sector=park["name"],
                arena="walking path",
                collision=False,
                tile_type="park_path",
            )

    def _place_buildings(self):
        for bld in self.config.get("buildings", []):
            self._fill_rect(
                bld["x"], bld["y"], bld["w"], bld["h"],
                sector=bld.get("block", ""),
                arena=bld["name"],
                game_object=bld.get("type", ""),
                collision=False,
                tile_type="building",
            )

    def _add_entrances(self):
        """Carve a 1-tile walkable corridor from each building to the nearest block edge."""
        blocks_by_name = {b["name"]: b for b in self.config.get("blocks", [])}
        for bld in self.config.get("buildings", []):
            block_name = bld.get("block", "")
            block = blocks_by_name.get(block_name)
            if not block:
                continue

            bx, by, bw, bh = bld["x"], bld["y"], bld["w"], bld["h"]
            blk_x, blk_y = block["x"], block["y"]
            blk_w, blk_h = block["w"], block["h"]

            dist_left = bx - blk_x
            dist_right = (blk_x + blk_w) - (bx + bw)
            dist_top = by - blk_y
            dist_bottom = (blk_y + blk_h) - (by + bh)

            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

            sector = block_name
            arena = bld["name"]

            if min_dist == dist_left:
                mid_y = by + bh // 2
                for x in range(blk_x, bx):
                    self._set_corridor(x, mid_y, sector, arena)
            elif min_dist == dist_right:
                mid_y = by + bh // 2
                for x in range(bx + bw, blk_x + blk_w):
                    self._set_corridor(x, mid_y, sector, arena)
            elif min_dist == dist_top:
                mid_x = bx + bw // 2
                for y in range(blk_y, by):
                    self._set_corridor(mid_x, y, sector, arena)
            else:
                mid_x = bx + bw // 2
                for y in range(by + bh, blk_y + blk_h):
                    self._set_corridor(mid_x, y, sector, arena)

    def _set_corridor(self, x: int, y: int, sector: str, arena: str):
        if 0 <= x < self.maze_width and 0 <= y < self.maze_height:
            t = self.tiles[y][x]
            t["collision"] = False
            t["tile_type"] = "corridor"
            if not t["sector"]:
                t["sector"] = sector
            if not t["arena"]:
                t["arena"] = arena
            self.collision_maze[y][x] = "0"

    def _place_intersections(self):
        for inter in self.config.get("intersections", []):
            for dy in range(inter["h"]):
                for dx in range(inter["w"]):
                    x, y = inter["x"] + dx, inter["y"] + dy
                    if 0 <= x < self.maze_width and 0 <= y < self.maze_height:
                        t = self.tiles[y][x]
                        if t["tile_type"] == "road":
                            t["sector"] = inter["name"]
                            t["tile_type"] = "crosswalk"

    def _mark_sidewalks(self):
        block_rects = [
            (b["x"], b["y"], b["w"], b["h"])
            for b in self.config.get("blocks", [])
        ]
        park = self.config.get("park")
        if park:
            block_rects.append((park["x"], park["y"], park["w"], park["h"]))

        for bx, by, bw, bh in block_rects:
            for x in range(bx - 1, bx + bw + 1):
                for y in range(by - 1, by + bh + 1):
                    if 0 <= x < self.maze_width and 0 <= y < self.maze_height:
                        if not (bx <= x < bx + bw and by <= y < by + bh):
                            if self.tiles[y][x]["tile_type"] == "road":
                                self.tiles[y][x]["tile_type"] = "sidewalk"

    def _build_address_tiles(self):
        self.address_tiles: Dict[str, Set[Tuple[int, int]]] = {}
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                t = self.tiles[y][x]
                addresses: List[str] = []
                if t["sector"]:
                    addresses.append(f'{t["world"]}:{t["sector"]}')
                if t["arena"]:
                    addresses.append(f'{t["world"]}:{t["sector"]}:{t["arena"]}')
                if t["game_object"]:
                    addresses.append(
                        f'{t["world"]}:{t["sector"]}:{t["arena"]}:{t["game_object"]}'
                    )
                for addr in addresses:
                    self.address_tiles.setdefault(addr, set()).add((x, y))

    def access_tile(self, tile: Tuple[int, int]) -> Dict[str, Any]:
        return self.tiles[tile[1]][tile[0]]

    def get_tile_path(self, tile: Tuple[int, int], level: str) -> str:
        t = self.tiles[tile[1]][tile[0]]
        path = t["world"]
        if level == "world":
            return path
        path += f":{t['sector']}"
        if level == "sector":
            return path
        path += f":{t['arena']}"
        if level == "arena":
            return path
        path += f":{t['game_object']}"
        return path

    def get_nearby_tiles(
        self, tile: Tuple[int, int], vision_r: int
    ) -> List[Tuple[int, int]]:
        left = max(0, tile[0] - vision_r)
        right = min(self.maze_width - 1, tile[0] + vision_r + 1)
        top = max(0, tile[1] - vision_r)
        bottom = min(self.maze_height - 1, tile[1] + vision_r + 1)
        return [(x, y) for x in range(left, right) for y in range(top, bottom)]

    def add_event_from_tile(self, curr_event: Any, tile: Tuple[int, int]):
        self.tiles[tile[1]][tile[0]]["events"].add(curr_event)

    def remove_event_from_tile(self, curr_event: Any, tile: Tuple[int, int]):
        evts = self.tiles[tile[1]][tile[0]]["events"].copy()
        for ev in evts:
            if ev == curr_event:
                self.tiles[tile[1]][tile[0]]["events"].remove(ev)

    def remove_subject_events_from_tile(self, subject: str, tile: Tuple[int, int]):
        evts = self.tiles[tile[1]][tile[0]]["events"].copy()
        for ev in evts:
            if ev[0] == subject:
                self.tiles[tile[1]][tile[0]]["events"].remove(ev)

    def turn_coordinate_to_tile(self, px_coordinate: Tuple[int, int]) -> Tuple[int, int]:
        x = math.ceil(px_coordinate[0] / self.sq_tile_size)
        y = math.ceil(px_coordinate[1] / self.sq_tile_size)
        return (x, y)
