# env/world_object.py
from __future__ import annotations
from typing import Optional, Tuple

from minigrid.core.world_object import WorldObj
from env.constants import (
    SEM_TO_ID, ID_TO_SEM, SEM_TO_MG_COLOR,
)

RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]

def _physics_for(sem_name: str) -> tuple[bool, bool]:
    """
    Returns (can_overlap, see_through) for this semantic.
    """
    s = sem_name.lower()
    if s in ("block",):
        return (False, False)           
    if s in ("fire",):                  # later change to false
        return (True, True)
    # default floors
    return (True, True)

def _obj_type_for(sem_name: str) -> str:
    """
    Map semantic -> MiniGrid 'type' string.
    """
    s = sem_name.lower()
    if s == "block":
        return "wall"
    if s == "fire":
        return "lava"
    if s == "goal":
        return "goal"
    return "floor"

def _as_rgb(rgba_like) -> RGB:
    t = tuple(rgba_like)
    if len(t) >= 3:
        return (int(t[0]), int(t[1]), int(t[2]))
    raise ValueError("RGBA must have at least 3 elements")


class SemanticTile(WorldObj):
    """
    MiniGrid-compatible world object that carries:
      - sem_name / sem_id  (your semantics)
      - rgba               (true render color from JSON)
      - MiniGrid obj_type  ("wall"/"floor"/"lava"/"goal") and color token (enum only)
      - flexible physics   (can_overlap / see_through) with room to depend on agent kind
    """
    def __init__(self, sem_name: str, rgba, name: Optional[str] = None):
        sem = sem_name.lower()
        obj_type = _obj_type_for(sem)
        mg_color = SEM_TO_MG_COLOR.get(sem, "grey")

        super().__init__(obj_type, color=mg_color)


        self.sem_name: str = sem
        self.sem_id: int = int(SEM_TO_ID.get(sem, 0))
        self.label: str = name or sem
        self.rgba: Tuple[int, ...] = tuple(rgba)

        self._can_overlap, self._see_through = _physics_for(sem)

    def can_overlap(self, agent_kind: str = "human") -> bool:
        return self._can_overlap
    
    def see_through(self, agent_kind: str = "human") -> bool:
        """
        Return False for opaque tiles (e.g. blocks/walls), True otherwise.
        """
        s = self.sem_name
        # make walls opaque
        if s in ("block", "home_A", "home_B", "fire"):
            return False

        return True


    def encode_sem(self) -> int:
        return self.sem_id

    def decode_sem(self) -> str:
        return ID_TO_SEM.get(self.sem_id, "unknown")


    def render(self, img):
        r, g, b = _as_rgb(self.rgba)
        img[:, :, :] = (r, g, b)


def make_tile(sem_name: str, rgba, name: Optional[str] = None) -> SemanticTile:
    return SemanticTile(sem_name=sem_name, rgba=rgba, name=name)
