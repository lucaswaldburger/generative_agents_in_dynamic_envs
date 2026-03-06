"""
Data classes describing a procedurally-generated world.

WorldConfig is the pure-data intermediate representation that layout strategies
produce and that GeneratedMap / ProceduralRenderer / CSVExporter consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BuildingConfig:
    name: str
    x: int
    y: int
    w: int
    h: int
    building_type: str = ""
    color: Tuple[int, int, int] = (90, 85, 100)
    zone: str = ""
    game_objects: List[str] = field(default_factory=list)


@dataclass
class ZoneConfig:
    """A named rectangular region (city block, district, campus quad, …)."""
    name: str
    x: int
    y: int
    w: int
    h: int
    zone_type: str = "block"
    collision: bool = True


@dataclass
class ParkConfig:
    name: str
    x: int
    y: int
    w: int
    h: int
    trees: List[Tuple[int, int]] = field(default_factory=list)
    paths: List[Dict[str, int]] = field(default_factory=list)


@dataclass
class RoadConfig:
    """A straight road segment (horizontal or vertical)."""
    name: str
    x: int
    y: int
    w: int
    h: int
    direction: str = "horizontal"


@dataclass
class IntersectionConfig:
    name: str
    x: int
    y: int
    w: int
    h: int


@dataclass
class CarLaneConfig:
    name: str
    waypoints: List[Tuple[int, int]] = field(default_factory=list)
    color: Tuple[int, int, int] = (180, 60, 60)


@dataclass
class SpawnConfig:
    name: str
    x: int
    y: int


@dataclass
class PaletteConfig:
    road: Tuple[int, int, int] = (70, 70, 75)
    sidewalk: Tuple[int, int, int] = (160, 160, 155)
    block: Tuple[int, int, int] = (55, 55, 65)
    building: Tuple[int, int, int] = (90, 85, 100)
    grass: Tuple[int, int, int] = (75, 150, 65)
    park: Tuple[int, int, int] = (45, 140, 55)
    park_path: Tuple[int, int, int] = (180, 170, 140)
    crosswalk: Tuple[int, int, int] = (220, 220, 215)
    lane_mark: Tuple[int, int, int] = (200, 200, 60)
    water: Tuple[int, int, int] = (60, 120, 180)
    dirt_path: Tuple[int, int, int] = (165, 140, 100)
    fence: Tuple[int, int, int] = (130, 100, 60)
    parking: Tuple[int, int, int] = (95, 95, 100)


@dataclass
class WorldConfig:
    """Complete description of a generated world, ready for map / render / export."""
    world_name: str = "Generated World"
    width: int = 80
    height: int = 60
    tile_size: int = 16
    layout_type: str = "village"

    palette: PaletteConfig = field(default_factory=PaletteConfig)
    zones: List[ZoneConfig] = field(default_factory=list)
    buildings: List[BuildingConfig] = field(default_factory=list)
    parks: List[ParkConfig] = field(default_factory=list)
    roads: List[RoadConfig] = field(default_factory=list)
    intersections: List[IntersectionConfig] = field(default_factory=list)
    car_lanes: List[CarLaneConfig] = field(default_factory=list)
    spawns: List[SpawnConfig] = field(default_factory=list)

    extra: Dict[str, Any] = field(default_factory=dict)
