# env/load_persona.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml

from env.constants import AgentConfig, FOVConfig
from persona.memory.spatial_memory import SpatialMemory

def load_agent_configs(persona_path: str | Path) -> List[AgentConfig]:
    path = Path(persona_path)
    with path.open("r") as f:
        data = yaml.safe_load(f)

    agents: List[AgentConfig] = []
    # data is something like {"human_1": {...}, "human_2": {...}}
    for key, cfg in data.items():
        start = cfg["start"]
        fov_cfg = cfg["fov"]

        spatial_mem = None
        sm_file = cfg.get("spatial_memory_file")
        if sm_file:
            sm_path = Path(sm_file)
            if not sm_path.is_absolute():
                sm_path = path.parent / sm_file
            spatial_mem = SpatialMemory.from_file(sm_path)

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
            spatial_memory=spatial_mem
        )
        agents.append(agent)

    return agents