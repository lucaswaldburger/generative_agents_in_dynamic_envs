"""
CSVExporter – writes a GeneratedMap to the SmallVille CSV format so it can be
loaded by ``refactored_smallville.env.maze.Maze``.

Output structure (mirrors ``assets/the_ville/matrix/``):
    output_dir/
        maze_meta_info.json
        special_blocks/
            world_blocks.csv
            sector_blocks.csv
            arena_blocks.csv
            game_object_blocks.csv
            spawning_location_blocks.csv
        maze/
            collision_maze.csv
            sector_maze.csv
            arena_maze.csv
            game_object_maze.csv
            spawning_location_maze.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .generated_map import GeneratedMap


class CSVExporter:
    """Export a :class:`GeneratedMap` to SmallVille-compatible CSV files."""

    def __init__(self, gen_map: GeneratedMap):
        self.gen_map = gen_map

    def export(self, output_dir: str | Path) -> Path:
        """Write all CSV / JSON files and return the output directory path."""
        out = Path(output_dir)
        maze_dir = out / "maze"
        blocks_dir = out / "special_blocks"
        maze_dir.mkdir(parents=True, exist_ok=True)
        blocks_dir.mkdir(parents=True, exist_ok=True)

        w = self.gen_map.maze_width
        h = self.gen_map.maze_height
        tiles = self.gen_map.tiles

        meta = {
            "world_name": self.gen_map.config.world_name,
            "maze_width": w,
            "maze_height": h,
            "sq_tile_size": self.gen_map.sq_tile_size,
            "special_constraint": "",
        }
        (out / "maze_meta_info.json").write_text(json.dumps(meta, indent=2))

        # Collect unique names and assign numeric IDs (starting at 32130 to
        # avoid collisions with SmallVille's own IDs).
        sectors: Dict[str, str] = {}
        arenas: Dict[str, str] = {}
        game_objects: Dict[str, str] = {}
        spawns: Dict[str, str] = {}
        _next_id = 32130

        def _get_id(mapping: Dict[str, str], name: str) -> str:
            nonlocal _next_id
            if name not in mapping:
                mapping[name] = str(_next_id)
                _next_id += 1
            return mapping[name]

        collision_flat: List[str] = []
        sector_flat: List[str] = []
        arena_flat: List[str] = []
        go_flat: List[str] = []
        spawn_flat: List[str] = []

        for y in range(h):
            for x in range(w):
                t = tiles[y][x]
                collision_flat.append("32125" if t["collision"] else "0")

                if t["sector"]:
                    sector_flat.append(_get_id(sectors, t["sector"]))
                else:
                    sector_flat.append("0")

                if t["arena"]:
                    arena_flat.append(_get_id(arenas, t["arena"]))
                else:
                    arena_flat.append("0")

                if t["game_object"]:
                    arena_flat[-1]  # ensure arena exists
                    go_flat.append(_get_id(game_objects, t["game_object"]))
                else:
                    go_flat.append("0")

                if t["spawning_location"]:
                    spawn_flat.append(_get_id(spawns, t["spawning_location"]))
                else:
                    spawn_flat.append("0")

        # Write flat CSVs (single-row, matching SmallVille format)
        self._write_flat_csv(maze_dir / "collision_maze.csv", collision_flat)
        self._write_flat_csv(maze_dir / "sector_maze.csv", sector_flat)
        self._write_flat_csv(maze_dir / "arena_maze.csv", arena_flat)
        self._write_flat_csv(maze_dir / "game_object_maze.csv", go_flat)
        self._write_flat_csv(maze_dir / "spawning_location_maze.csv", spawn_flat)

        # Write special_blocks CSVs (format: id, name — Maze reads r[0] as key, r[-1] as value)
        world_name = self.gen_map.config.world_name
        world_id = "32134"
        self._write_block_csv(blocks_dir / "world_blocks.csv",
                              {world_name: world_id})
        self._write_block_csv(blocks_dir / "sector_blocks.csv", sectors)
        self._write_block_csv(blocks_dir / "arena_blocks.csv", arenas)
        self._write_block_csv(blocks_dir / "game_object_blocks.csv", game_objects)
        self._write_block_csv(blocks_dir / "spawning_location_blocks.csv", spawns)

        return out

    @staticmethod
    def _write_flat_csv(path: Path, values: List[str]):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(values)

    @staticmethod
    def _write_block_csv(path: Path, mapping: Dict[str, str]):
        """Write name → id mapping as CSV rows ``[id, name]``.
        Maze reads ``{r[0]: r[-1]}`` i.e. id → name."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for name, uid in mapping.items():
                writer.writerow([uid, name])
