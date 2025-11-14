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


def load_map(map_path: str | Path) -> MapSpec:
    path = Path(map_path)
    with path.open("r") as f:
        cfg = json.load(f)

    width = int(cfg["width"])
    height = int(cfg["height"])

    bg_code = int(cfg.get("background_access", 1))
    grid = np.full((height, width), bg_code, dtype=np.int32)

    for region in cfg.get("regions", []):
        x, y = int(region["x"]), int(region["y"])
        w, h = int(region["w"]), int(region["h"])
        code = int(region["access_code"])
        grid[y : y + h, x : x + w] = code

    semantics = cfg.get("semantics", {})

    return MapSpec(
        width=width,
        height=height,
        access_grid=grid,
        semantics=semantics,
        regions=cfg.get("regions", []),
        raw=cfg,
    )