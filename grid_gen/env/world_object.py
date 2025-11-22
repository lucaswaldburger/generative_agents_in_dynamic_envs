# env/world_object.py
from __future__ import annotations

from dataclasses import dataclass

from .constants import AgentConfig


@dataclass
class WorldObject:
    x: int
    y: int
    obj_type: str


@dataclass
class HumanAgent(WorldObject):
    config: AgentConfig
    heading_deg: int

    @classmethod
    def from_config(cls, cfg: AgentConfig) -> "HumanAgent":
        return cls(
            x=cfg.start_x,
            y=cfg.start_y,
            obj_type=cfg.kind,
            config=cfg,
            heading_deg=cfg.heading_deg,  # start heading from persona
        )