from __future__ import annotations

from dataclasses import dataclass
from .constants import AgentConfig


@dataclass
class SmallvilleAgent:
    x: int
    y: int
    config: AgentConfig
    heading_deg: int = 270  # facing up by default
    description: str = ""
    pronunciatio: str = ""

    @classmethod
    def from_config(cls, cfg: AgentConfig) -> SmallvilleAgent:
        return cls(x=cfg.start_x, y=cfg.start_y, config=cfg)
