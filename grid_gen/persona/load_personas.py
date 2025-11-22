# persona/load_personas.py

from dataclasses import dataclass
from pathlib import Path
import json
import yaml
from typing import List, Dict, Any, Optional

from persona.memory.spatial_memory import SpatialMemory
from env.constants import FOVConfig  # adjust import to your code


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

    # persona fields
    first_name: str
    last_name: str
    age: int
    gender: str
    innate: str
    learned: str
    lifestyle: str
    living_area: str
    likelihood_to_help_others: str
    chatting_likelihood: str

    friends_with: list[str]
    dependents: list[dict]
    daily_plan: list[dict]

    # memory
    spatial_memory: SpatialMemory

    # other cognitive params
    att_bandwidth: int
    retention: int
    concept_forget: int
    daily_reflection_time: int
    daily_reflection_size: int
    overlap_reflect_th: int
    kw_strg_event_reflect_th: int
    kw_strg_thought_reflect_th: int
    recency_w: float
    relevance_w: float
    importance_w: float
    recency_decay: float
    importance_trigger_max: int
    importance_trigger_curr: int
    importance_ele_n: int
    thought_count: int


def load_agent_configs(
    personas_yaml_path: str,
    personas_in_sim: Optional[List[str]] = None,
) -> List[AgentConfig]:
    """
    Load persona spec YAML and construct AgentConfig objects,
    including spatial memory from JSON.
    """
    personas_yaml_path = Path(personas_yaml_path)
    base_dir = personas_yaml_path.parent

    with personas_yaml_path.open("r") as f:
        spec: Dict[str, Any] = yaml.safe_load(f)

    # If no filter list is provided, use all personas in the file
    if personas_in_sim is None:
        personas_in_sim = list(spec.keys())

    agent_configs: List[AgentConfig] = []

    for persona_key in personas_in_sim:
        p = spec[persona_key]

        # spatial memory json path is relative to the YAML file directory
        sm_rel_path = p["spatial_memory_file"]  # e.g. "spatial_memory/human_1_isabella.json"
        sm_path = base_dir / sm_rel_path
        with sm_path.open("r") as f:
            sm_data = json.load(f)

        spatial_memory = SpatialMemory(sm_data)  # whatever your constructor expects

        fov_cfg = FOVConfig(
            range_cells=p["fov"]["range_cells"],
            angle_deg=p["fov"]["angle_deg"],
            shade_color=p["fov"]["shade_color"],
        )

        start = p["start"]

        cfg = AgentConfig(
            id=p["id"],
            kind=p["kind"],
            start_x=start["x"],
            start_y=start["y"],
            heading_deg=p["heading_deg"],
            color=p["color"],
            fov=fov_cfg,
            gender=p["gender"],
            name=p["name"],
            first_name=p["first_name"],
            last_name=p["last_name"],
            age=p["age"],
            innate=p["innate"],
            learned=p["learned"],
            lifestyle=p["lifestyle"],
            living_area=p["living_area"],
            likelihood_to_help_others=p["likelihood_to_help_others"],
            chatting_likelihood=p["chatting_likelihood"],
            spatial_memory=spatial_memory,
            att_bandwidth=p["att_bandwidth"],
            retention=p["retention"],
            concept_forget=p["concept_forget"],
            daily_reflection_time=p["daily_reflection_time"],
            daily_reflection_size=p["daily_reflection_size"],
            overlap_reflect_th=p["overlap_reflect_th"],
            kw_strg_event_reflect_th=p["kw_strg_event_reflect_th"],
            kw_strg_thought_reflect_th=p["kw_strg_thought_reflect_th"],
            recency_w=p["recency_w"],
            relevance_w=p["relevance_w"],
            importance_w=p["importance_w"],
            recency_decay=p["recency_decay"],
            importance_trigger_max=p["importance_trigger_max"],
            importance_trigger_curr=p["importance_trigger_curr"],
            importance_ele_n=p["importance_ele_n"],
            thought_count=p["thought_count"],
            friends_with = p.get("friends_with", []),
            dependents = p.get("dependents", []),
            daily_plan = p.get("daily_plan", [])
        )

        agent_configs.append(cfg)

    return agent_configs
