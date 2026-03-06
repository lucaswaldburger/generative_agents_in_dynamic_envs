"""
env_generator – procedural environment generation for agent simulations.

Supports multiple layout types (village, urban, campus), ships a Gymnasium
environment with pygame rendering, and can export to SmallVille CSV format.

Quick start::

    from env_generator import WorldGenerator

    gen = WorldGenerator()
    config = gen.generate(layout="village", width=80, height=60, seed=42)
    env = gen.build_env(config, agent_names=["Alice", "Bob"])
    obs, info = env.reset()
"""
from .config import (
    BuildingConfig,
    CarLaneConfig,
    IntersectionConfig,
    PaletteConfig,
    ParkConfig,
    RoadConfig,
    SpawnConfig,
    WorldConfig,
    ZoneConfig,
)
from .generated_map import GeneratedMap
from .renderer import ProceduralRenderer
from .env import GeneratedEnv, Action, AgentState
from .exporter import CSVExporter
from .generator import WorldGenerator
from .layouts import (
    LayoutStrategy,
    VillageLayout,
    UrbanLayout,
    CampusLayout,
    LAYOUT_REGISTRY,
)

__all__ = [
    "WorldGenerator",
    "WorldConfig",
    "BuildingConfig",
    "CarLaneConfig",
    "IntersectionConfig",
    "PaletteConfig",
    "ParkConfig",
    "RoadConfig",
    "SpawnConfig",
    "ZoneConfig",
    "GeneratedMap",
    "ProceduralRenderer",
    "GeneratedEnv",
    "Action",
    "AgentState",
    "CSVExporter",
    "LayoutStrategy",
    "VillageLayout",
    "UrbanLayout",
    "CampusLayout",
    "LAYOUT_REGISTRY",
]
