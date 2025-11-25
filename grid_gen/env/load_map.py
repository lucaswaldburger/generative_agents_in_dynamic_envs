from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

@dataclass
class MapSpec:
    width: int
    height: int
    access_grid: np.ndarray
    semantics: Dict[str, Any]
    regions: List[Dict[str, Any]]
    raw: Dict[str, Any]
    cell_names: np.ndarray

def load_map(map_path: str | Path) -> MapSpec:
    path = Path(map_path)
    with path.open("r") as f:
        cfg = json.load(f)

    width = int(cfg["width"])
    height = int(cfg["height"])
    regions = cfg["regions"]
    access_codes = cfg["access_codes"]
    semantics = cfg.get("semantics", {})

    bg_code = int(cfg.get("background_access", 1))

    # 1) start with uniform background
    access_grid = np.full((height, width), bg_code, dtype=np.int32)

    # 2) initialize cell_names from the access codes
    cell_names = np.empty_like(access_grid, dtype=object)
    code_to_name = {v: k for k, v in access_codes.items()}
    for y in range(height):
        for x in range(width):
            code = int(access_grid[y, x])
            cell_names[y, x] = code_to_name.get(code, "unknown")

    # 3) apply each region: set BOTH access_grid and cell_names
    for r in regions:
        name = r["name"]
        x0, y0 = r["x"], r["y"]
        w, h = r["w"], r["h"]

        # if a region has an access_code, write it into the grid
        code = int(r.get("access_code", bg_code))
        access_grid[y0:y0 + h, x0:x0 + w] = code

        # give these cells the region name (for spatial memory)
        cell_names[y0:y0 + h, x0:x0 + w] = name

    return MapSpec(
        width=width,
        height=height,
        access_grid=access_grid,
        semantics=semantics,
        regions=regions,
        raw=cfg,
        cell_names=cell_names,
    )