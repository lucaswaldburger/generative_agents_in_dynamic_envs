# persona/memory/spatial_memory.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPATIAL_MEM_DIR = PROJECT_ROOT / "configs" / "personas" / "spatial_memory"


class SpatialMemory:
    def __init__(self, data: Dict[str, Any]):
        """
        data: parsed JSON dict for this agent's spatial memory.
        """
        self.data = data

    @classmethod
    def from_file(cls, path: Path) -> "SpatialMemory":
        with path.open("r") as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def from_agent_id(cls, agent_id: str) -> "SpatialMemory":
        """
        Convention: spatial memory file is <agent_id>.json
        e.g., human_1 -> human_1.json or human_1_isabella.json
        """
        # pick ONE naming convention and stick with it:
        filename = f"{agent_id}_isabella.json" if agent_id == "human_1" else f"{agent_id}.json"
        path = SPATIAL_MEM_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"No spatial memory file found at {path}")
        return cls.from_file(path)


    def worlds(self) -> List[str]:
        return list(self.data.keys())

    def buildings_in_world(self, world: str) -> List[str]:
        return list(self.data.get(world, {}).keys())

    def rooms_in_building(self, world: str, building: str) -> List[str]:
        return list(self.data.get(world, {}).get(building, {}).keys())

    def objects_in_room(
        self, world: str, building: str, room: str
    ) -> List[str]:
        return list(self.data.get(world, {}).get(building, {}).get(room, []))

    def find_building_for_label(self, label_substring: str) -> Optional[str]:
        """
        e.g. 'Isabella' -> 'Home_A (Isabella Rodriguez's apartment)'
        """
        for world, buildings in self.data.items():
            for bname in buildings.keys():
                if label_substring.lower() in bname.lower():
                    return bname
        return None

    def elements_at_position(
        self,
        env,
        x: int,
        y: int,
        world_key: str = "urbanWorld",
    ) -> Dict[str, Any]:
        """
        Purely data-driven:
        - env.map_spec.cell_names[y, x] gives the cell's element name (e.g. 'Home A', 'Park', 'B9')
        - spatial memory JSON decides what that name maps to.
        """

        cell_name = env.map_spec.cell_names[y, x]

        world = self.data.get(world_key)
        if world is None:
            # world not even defined in this JSON
            return {
                "cell_name": cell_name,
                "lookup_key": None,
                "contents": {},
                "warning": f"world_key '{world_key}' not found in spatial memory",
            }

        # optional alias table
        aliases = world.get("__cell_aliases__", {})
        key = cell_name
        if key not in world and key in aliases:
            key = aliases[key]

        contents = world.get(key, {})  # <- DEFAULT TO {} INSTEAD OF None

        return {
            "cell_name": cell_name,
            "lookup_key": key,
            "contents": contents,
        }