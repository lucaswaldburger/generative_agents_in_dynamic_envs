# env/load_persona.py
from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

from env.constants import AgentConfig, FOVConfig


def load_agent_configs(persona_path: str | Path) -> List[AgentConfig]:
    path = Path(persona_path)
    with path.open("r") as f:
        data = yaml.safe_load(f)

    agents: List[AgentConfig] = []
    # data is something like {"human_1": {...}, "human_2": {...}}
    for key, cfg in data.items():
        start = cfg["start"]
        fov_cfg = cfg["fov"]
        agent = AgentConfig(
            id=cfg["id"],
            kind=cfg["kind"],
            start_x=int(start["x"]),
            start_y=int(start["y"]),
            heading_deg=int(cfg.get("heading_deg", 0)),
            color=cfg.get("color", "white"),
            fov=FOVConfig(
                range_cells=int(fov_cfg["range_cells"]),
                angle_deg=int(fov_cfg["angle_deg"]),
                shade_color=str(fov_cfg["shade_color"]),
            ),
            name=cfg.get("name", key),
        )
        agents.append(agent)

    return agents