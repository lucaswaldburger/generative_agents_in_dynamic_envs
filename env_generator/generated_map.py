"""
GeneratedMap – tile map built from a WorldConfig.

Provides the same interface as ``refactored_smallville.env.maze.Maze`` and
``refactored_city.env.city_map.CityMap`` so all existing agent / pathfinding
code works unchanged.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import WorldConfig


class GeneratedMap:
    """2-D tile map constructed from a :class:`WorldConfig`."""

    def __init__(self, config: WorldConfig):
        self.config = config
        self.maze_width: int = config.width
        self.maze_height: int = config.height
        self.sq_tile_size: int = config.tile_size

        self._init_tiles()
        self._lay_roads()
        self._place_zones()
        self._place_parks()
        self._place_buildings()
        self._carve_entrances()
        self._mark_sidewalks()
        self._place_spawns()
        self._assign_game_objects()
        self._build_address_tiles()

        self.car_lanes = [
            {"name": cl.name,
             "waypoints": list(cl.waypoints),
             "color": list(cl.color)}
            for cl in config.car_lanes
        ]
        self.intersections_data = [
            {"name": it.name, "x": it.x, "y": it.y, "w": it.w, "h": it.h}
            for it in config.intersections
        ]

    # ── initialisation helpers ────────────────────────────────────────

    def _init_tiles(self):
        w, h = self.maze_width, self.maze_height
        bg_type = "grass" if self.config.layout_type in ("village", "campus") else "road"
        bg_collision = False

        self.tiles: List[List[Dict[str, Any]]] = []
        self.collision_maze: List[List[str]] = []
        for _y in range(h):
            tile_row: List[Dict[str, Any]] = []
            col_row: List[str] = []
            for _x in range(w):
                tile_row.append({
                    "world": self.config.world_name,
                    "sector": "",
                    "arena": "",
                    "game_object": "",
                    "spawning_location": "",
                    "collision": bg_collision,
                    "events": set(),
                    "tile_type": bg_type,
                })
                col_row.append("0")
            self.tiles.append(tile_row)
            self.collision_maze.append(col_row)

    def _fill_rect(self, x0: int, y0: int, w: int, h: int, *,
                   sector: str = "", arena: str = "", game_object: str = "",
                   collision: bool = False, tile_type: str = "road"):
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

    def _lay_roads(self):
        for road in self.config.roads:
            self._fill_rect(road.x, road.y, road.w, road.h,
                            tile_type="road" if self.config.layout_type == "urban" else "dirt_path")

    def _place_zones(self):
        for zone in self.config.zones:
            tt = {
                "block": "block",
                "plaza": "plaza",
                "district": "grass",
                "parking": "parking",
            }.get(zone.zone_type, "block")
            self._fill_rect(zone.x, zone.y, zone.w, zone.h,
                            sector=zone.name, collision=zone.collision, tile_type=tt)

    def _place_parks(self):
        for park in self.config.parks:
            self._fill_rect(park.x, park.y, park.w, park.h,
                            sector=park.name, arena="green space",
                            collision=False, tile_type="park")
            for pp in park.paths:
                self._fill_rect(pp["x"], pp["y"], pp["w"], pp["h"],
                                sector=park.name, arena="walking path",
                                collision=False, tile_type="park_path")

    def _place_buildings(self):
        for bld in self.config.buildings:
            self._fill_rect(bld.x, bld.y, bld.w, bld.h,
                            sector=bld.zone or bld.name,
                            arena=bld.name,
                            game_object=bld.building_type,
                            collision=False, tile_type="building")

    def _carve_entrances(self):
        """Carve a walkable corridor from each building to the nearest zone edge."""
        zones_by_name = {z.name: z for z in self.config.zones}
        for bld in self.config.buildings:
            zone = zones_by_name.get(bld.zone)
            if not zone or not zone.collision:
                continue

            bx, by, bw, bh = bld.x, bld.y, bld.w, bld.h
            zx, zy, zw, zh = zone.x, zone.y, zone.w, zone.h

            dist_left = bx - zx
            dist_right = (zx + zw) - (bx + bw)
            dist_top = by - zy
            dist_bottom = (zy + zh) - (by + bh)
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

            sector = zone.name
            arena = bld.name

            if min_dist == dist_left:
                mid_y = by + bh // 2
                for x in range(zx, bx):
                    self._set_corridor(x, mid_y, sector, arena)
            elif min_dist == dist_right:
                mid_y = by + bh // 2
                for x in range(bx + bw, zx + zw):
                    self._set_corridor(x, mid_y, sector, arena)
            elif min_dist == dist_top:
                mid_x = bx + bw // 2
                for y in range(zy, by):
                    self._set_corridor(mid_x, y, sector, arena)
            else:
                mid_x = bx + bw // 2
                for y in range(by + bh, zy + zh):
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

    def _mark_sidewalks(self):
        if self.config.layout_type != "urban":
            return
        zones_and_parks = [
            (z.x, z.y, z.w, z.h) for z in self.config.zones
        ] + [
            (p.x, p.y, p.w, p.h) for p in self.config.parks
        ]
        for bx, by, bw, bh in zones_and_parks:
            for x in range(bx - 1, bx + bw + 1):
                for y in range(by - 1, by + bh + 1):
                    if 0 <= x < self.maze_width and 0 <= y < self.maze_height:
                        if not (bx <= x < bx + bw and by <= y < by + bh):
                            if self.tiles[y][x]["tile_type"] == "road":
                                self.tiles[y][x]["tile_type"] = "sidewalk"

    def _place_spawns(self):
        for sp in self.config.spawns:
            if 0 <= sp.x < self.maze_width and 0 <= sp.y < self.maze_height:
                self.tiles[sp.y][sp.x]["spawning_location"] = sp.name

    def _assign_game_objects(self):
        for bld in self.config.buildings:
            if not bld.game_objects:
                continue
            interior = []
            for dy in range(bld.h):
                for dx in range(bld.w):
                    x, y = bld.x + dx, bld.y + dy
                    if 0 <= x < self.maze_width and 0 <= y < self.maze_height:
                        interior.append((x, y))

            for idx, obj_name in enumerate(bld.game_objects):
                if idx < len(interior):
                    tx, ty = interior[idx]
                    self.tiles[ty][tx]["game_object"] = obj_name
                    full_name = ":".join([
                        self.tiles[ty][tx]["world"],
                        self.tiles[ty][tx]["sector"],
                        self.tiles[ty][tx]["arena"],
                        obj_name,
                    ])
                    self.tiles[ty][tx]["events"].add((full_name, None, None, None))

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
                if t["spawning_location"]:
                    addresses.append(f'<spawn_loc>{t["spawning_location"]}')
                for addr in addresses:
                    self.address_tiles.setdefault(addr, set()).add((x, y))

    # ── public interface (matches Maze / CityMap) ─────────────────────

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
