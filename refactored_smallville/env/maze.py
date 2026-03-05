"""
Maze class for the SmallVille environment.
Adapted from generative_agents/reverie/backend_server/maze.py to be
self-contained (no Django dependency).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _read_csv(path: str, strip: bool = True) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path) as f:
        for row in csv.reader(f, delimiter=","):
            if strip:
                row = [c.strip() for c in row]
            rows.append(row)
    return rows


class Maze:
    """2-D tile map loaded from the SmallVille CSV exports."""

    def __init__(self, assets_dir: str | Path):
        assets_dir = Path(assets_dir)
        matrix_dir = assets_dir / "the_ville" / "matrix"

        meta = json.loads((matrix_dir / "maze_meta_info.json").read_text())
        self.maze_width: int = int(meta["maze_width"])
        self.maze_height: int = int(meta["maze_height"])
        self.sq_tile_size: int = int(meta["sq_tile_size"])
        self.special_constraint: str = meta.get("special_constraint", "")

        blocks = matrix_dir / "special_blocks"
        wb = _read_csv(str(blocks / "world_blocks.csv"))[0][-1]

        sb_dict = {r[0]: r[-1] for r in _read_csv(str(blocks / "sector_blocks.csv"))}
        ab_dict = {r[0]: r[-1] for r in _read_csv(str(blocks / "arena_blocks.csv"))}
        gob_dict = {r[0]: r[-1] for r in _read_csv(str(blocks / "game_object_blocks.csv"))}
        slb_dict = {r[0]: r[-1] for r in _read_csv(str(blocks / "spawning_location_blocks.csv"))}

        maze_dir = matrix_dir / "maze"
        collision_raw = _read_csv(str(maze_dir / "collision_maze.csv"))[0]
        sector_raw = _read_csv(str(maze_dir / "sector_maze.csv"))[0]
        arena_raw = _read_csv(str(maze_dir / "arena_maze.csv"))[0]
        game_obj_raw = _read_csv(str(maze_dir / "game_object_maze.csv"))[0]
        spawn_raw = _read_csv(str(maze_dir / "spawning_location_maze.csv"))[0]

        w = self.maze_width
        self.collision_maze: List[List[str]] = []
        sector_maze: List[List[str]] = []
        arena_maze: List[List[str]] = []
        game_object_maze: List[List[str]] = []
        spawning_location_maze: List[List[str]] = []

        for i in range(0, len(collision_raw), w):
            self.collision_maze.append(collision_raw[i : i + w])
            sector_maze.append(sector_raw[i : i + w])
            arena_maze.append(arena_raw[i : i + w])
            game_object_maze.append(game_obj_raw[i : i + w])
            spawning_location_maze.append(spawn_raw[i : i + w])

        self.tiles: List[List[Dict]] = []
        for row_i in range(self.maze_height):
            row = []
            for col_j in range(self.maze_width):
                td: Dict = {
                    "world": wb,
                    "sector": sb_dict.get(sector_maze[row_i][col_j], ""),
                    "arena": ab_dict.get(arena_maze[row_i][col_j], ""),
                    "game_object": gob_dict.get(game_object_maze[row_i][col_j], ""),
                    "spawning_location": slb_dict.get(
                        spawning_location_maze[row_i][col_j], ""
                    ),
                    "collision": self.collision_maze[row_i][col_j] != "0",
                    "events": set(),
                }
                row.append(td)
            self.tiles.append(row)

        for row_i in range(self.maze_height):
            for col_j in range(self.maze_width):
                go = self.tiles[row_i][col_j]["game_object"]
                if go:
                    name = ":".join(
                        [
                            self.tiles[row_i][col_j]["world"],
                            self.tiles[row_i][col_j]["sector"],
                            self.tiles[row_i][col_j]["arena"],
                            go,
                        ]
                    )
                    self.tiles[row_i][col_j]["events"].add((name, None, None, None))

        self.address_tiles: Dict[str, Set[Tuple[int, int]]] = {}
        for row_i in range(self.maze_height):
            for col_j in range(self.maze_width):
                t = self.tiles[row_i][col_j]
                addresses = []
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
                    self.address_tiles.setdefault(addr, set()).add((col_j, row_i))

    def turn_coordinate_to_tile(self, px_coordinate: Tuple[int, int]) -> Tuple[int, int]:
        x = math.ceil(px_coordinate[0] / self.sq_tile_size)
        y = math.ceil(px_coordinate[1] / self.sq_tile_size)
        return (x, y)

    def access_tile(self, tile: Tuple[int, int]) -> Dict:
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

    def add_event_from_tile(self, curr_event, tile: Tuple[int, int]):
        self.tiles[tile[1]][tile[0]]["events"].add(curr_event)

    def remove_event_from_tile(self, curr_event, tile: Tuple[int, int]):
        evts = self.tiles[tile[1]][tile[0]]["events"].copy()
        for ev in evts:
            if ev == curr_event:
                self.tiles[tile[1]][tile[0]]["events"].remove(ev)

    def remove_subject_events_from_tile(self, subject: str, tile: Tuple[int, int]):
        evts = self.tiles[tile[1]][tile[0]]["events"].copy()
        for ev in evts:
            if ev[0] == subject:
                self.tiles[tile[1]][tile[0]]["events"].remove(ev)
