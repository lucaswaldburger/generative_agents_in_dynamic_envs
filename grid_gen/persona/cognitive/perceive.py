from __future__ import annotations
from typing import Tuple, List, Dict, Any
import math

from env.constants import Coord  


def get_agent_pose(env, agent_id: int) -> Tuple[int, int, int, int, int]:
    """
    Returns (x, y, heading_deg, fov_range, fov_angle_deg) for the given agent.
    """
    agent = env.agents[agent_id]
    x, y = int(agent.x), int(agent.y)
    heading_deg = int(agent.heading_deg)
    fov_range = int(agent.config.fov.range_cells)
    fov_angle_deg = int(agent.config.fov.angle_deg)
    return x, y, heading_deg, fov_range, fov_angle_deg


def point_in_region(x: int, y: int, r: Dict[str, Any]) -> bool:
    rx, ry = int(r["x"]), int(r["y"])
    rw, rh = int(r["w"]), int(r["h"])
    return (rx <= x < rx + rw) and (ry <= y < ry + rh)


def regions_containing_point(env, x: int, y: int) -> List[Dict[str, Any]]:
    regions = getattr(env.map_spec, "regions", []) or []
    return [r for r in regions if point_in_region(x, y, r)]


def tiles_in_fov(env, agent_id: int) -> List[Coord]:
    """
    Approximate FOV as a circular sector:
      - radius = fov_range
      - angle   = fov_angle_deg centered at heading_deg
    Returns a list of grid coords (x, y) within that cone that are inside the map.
    No occlusion handling yet.
    """
    x, y, heading_deg, fov_range, fov_angle_deg = get_agent_pose(env, agent_id)

    width, height = env.map_spec.width, env.map_spec.height

    heading_rad = math.radians(heading_deg)
    half_angle = math.radians(fov_angle_deg / 2.0)

    visible: List[Coord] = []

    # Scan a square around the agent, then filter by circle + angle
    for dx in range(-fov_range, fov_range + 1):
        for dy in range(-fov_range, fov_range + 1):
            tx, ty = x + dx, y + dy

            if tx < 0 or tx >= width or ty < 0 or ty >= height:
                continue

            # Distance check (circle)
            dist = math.sqrt(dx * dx + dy * dy)
            if dist == 0 or dist > fov_range:
                continue

            # Angle check (cone)
            # NOTE: screen coords: x right, y down
            # atan2(dy, dx): 0 rad = +x (east), pi/2 = down, pi = west, -pi/2 = up
            angle_to_tile = math.atan2(dy, dx)

            # Smallest angle difference between heading and tile
            diff = (angle_to_tile - heading_rad + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) <= half_angle:
                visible.append((tx, ty))

    return visible


def regions_in_fov(env, agent_id: int) -> List[Dict[str, Any]]:
    """
    Regions that have at least one tile inside the agent's FOV cone.
    """
    regions = getattr(env.map_spec, "regions", []) or []
    if not regions:
        return []

    visible_tiles = tiles_in_fov(env, agent_id)
    if not visible_tiles:
        return []

    visible_regions: List[Dict[str, Any]] = []
    for r in regions:
        rx, ry = int(r["x"]), int(r["y"])
        rw, rh = int(r["w"]), int(r["h"])
        # Check if any visible tile lies inside region rect
        for (tx, ty) in visible_tiles:
            if rx <= tx < rx + rw and ry <= ty < ry + rh:
                visible_regions.append(r)
                break

    return visible_regions


def classify_location(env, x: int, y: int) -> str:
    """
    Rough semantic location: 'Home A', 'Workplace A', 'Park', 'B10', or 'street' if none.
    Prefers non-block regions (home/park/work/fire) over generic blocks.
    """
    here = regions_containing_point(env, x, y)
    if not here:
        code = int(env.map_spec.access_grid[y, x])
        access_codes = env.map_spec.raw.get("access_codes", {})
        inv = {v: k for k, v in access_codes.items()}
        label = inv.get(code, "walkable")
        if "hazard" in label:
            return "a hazard area"
        return "the street"

    preferred_types = ["home", "work", "park", "fire"]
    best = None
    for r in here:
        if r.get("type") in preferred_types:
            best = r
            break
    if best is None:
        best = here[0]

    return best.get("name", "unknown place")


def describe_perception(env, agent_id: int) -> str:
    """
    High-level natural-language description of what the agent perceives.
    """
    x, y, heading_deg, fov_range, fov_angle_deg = get_agent_pose(env, agent_id)
    location_label = classify_location(env, x, y)

    visible_regions = regions_in_fov(env, agent_id)

    visible_names: List[str] = []
    for r in visible_regions:
        name = r.get("name")
        if not name:
            continue
        visible_names.append(name)

    visible_names = sorted(set(visible_names))

    if visible_names:
        visible_str = ", ".join(visible_names)
        return f"I am at {location_label}. I see {visible_str}."
    else:
        return f"I am at {location_label}. I don't see any labeled regions."
