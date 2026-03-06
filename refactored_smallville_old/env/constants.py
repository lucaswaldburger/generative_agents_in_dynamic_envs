from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Tuple, Optional, List


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

COLLISION_BLOCK_ID = 32125

DEFAULT_MAX_STEPS = 1000


@dataclass
class AgentConfig:
    name: str
    start_x: int
    start_y: int
    color: Tuple[int, int, int] = (255, 80, 80)
    sector: str = ""
    arena: str = ""

    @property
    def id(self) -> str:
        return self.name.replace(" ", "_")
