from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, Dict, Optional
from persona.memory.spatial_memory import SpatialMemory


class Action(IntEnum):
    STAY = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


DIR_TO_VEC: Dict[Action, Tuple[int, int]] = {
    Action.STAY: (0, 0),
    Action.UP: (0, -1),
    Action.RIGHT: (1, 0),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
}


@dataclass
class FOVConfig:
    range_cells: int
    angle_deg: int
    shade_color: str


@dataclass
class AgentConfig:
    id: str
    kind: str
    start_x: int
    start_y: int
    heading_deg: int
    color: str
    fov: FOVConfig
    name: str
    spatial_memory: Optional[SpatialMemory] = None


DEFAULT_MAX_STEPS = 200


Coord = Tuple[int, int]

@dataclass
class SemanticMap:
    places: Dict[str, Coord]
    home_tile: Dict[int, Coord]
    tile_to_region: Dict[Coord, str]

# SEM_TO_ID = {
#     "street":  1,
#     "block":   2,   
#     "park":    3,
#     "home_A":  4,
#     "home_B":  5,
#     "work":    6,
#     "fire":    7,
#     "goal":    8,
# }
# ID_TO_SEM = {v: k for k, v in SEM_TO_ID.items()}


# SEM_TO_MG_COLOR = {
#     "street":  "grey",
#     "block":   "grey",
#     "park":    "green",
#     "home_A":  "purple",
#     "home_B":  "purple",
#     "work":    "blue",
#     "fire":    "red",
#     "goal":    "yellow",
# }