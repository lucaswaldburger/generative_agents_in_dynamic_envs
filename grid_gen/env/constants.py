from __future__ import annotations
import numpy as np


TILE_PIXELS = 32
COLLISION_CHECK_INTERVAL = 0.1

COLORS = {
    "red":    np.array([204, 0,   0]),
    "green":  np.array([69,  192, 69]),
    "blue":   np.array([0,   0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([204, 204, 0]),
    "grey":   np.array([100, 100, 100]),
    "orange": np.array([204, 102, 0]),
}
COLOR_NAMES   = sorted(COLORS.keys())
COLOR_TO_IDX  = {name: i for i, name in enumerate(COLOR_NAMES)}
IDX_TO_COLOR  = {v: k for k, v in COLOR_TO_IDX.items()}


DIR_TO_VEC = [
    np.array(( 1,  0)),  # 0: right / east
    np.array(( 0,  1)),  # 1: down  / south
    np.array((-1,  0)),  # 2: left  / west
    np.array(( 0, -1)),  # 3: up    / north
]


SEM_TO_ID = {
    "street":  1,
    "block":   2,   
    "park":    3,
    "home_A":  4,
    "home_B":  5,
    "work":    6,
    "fire":    7,
    "goal":    8,
}
ID_TO_SEM = {v: k for k, v in SEM_TO_ID.items()}


SEM_TO_MG_COLOR = {
    "street":  "grey",
    "block":   "grey",
    "park":    "green",
    "home_A":  "purple",
    "home_B":  "purple",
    "work":    "blue",
    "fire":    "red",
    "goal":    "yellow",
}