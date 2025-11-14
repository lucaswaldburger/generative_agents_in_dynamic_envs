from __future__ import annotations
from typing import Tuple, List, Dict, Any
import math

from env.constants import Coord  


def blocks_vision(env, x: int, y: int, agent_id: int) -> bool:
    """
    Decide if the tile at (x, y) blocks line of sight for a *specific agent*.

    Base rule: use semantics['blocks_vision'][access_code].

    Override for regions, e.g. homes:
      - homes block vision for everyone,
      - except if the region lists this agent as a resident.
    """
    ms = env.map_spec
    if x < 0 or x >= ms.width or y < 0 or y >= ms.height:
        return True  # outside map is opaque

    code = int(ms.access_grid[y, x])

    semantics = ms.semantics or {}
    block_table = semantics.get("blocks_vision", {})
    # default: if not specified, be conservative
    default_block = bool(block_table.get(str(code), True))

    # Look at any regions covering this tile
    regs = regions_containing_point(env, x, y)

    for r in regs:
        r_type = r.get("type")
        # Example policy: homes are opaque to non-residents
        if r_type == "home":
            residents = r.get("residents", [])
            agent_name = env.agent_configs[agent_id].id  # 'human_1', 'human_2', etc.

            if agent_name in residents:
                # Resident: can see through their own home walls
                # You can choose: fully transparent or only from inside – this is full.
                return False
            else:
                # Non-resident: home blocks vision
                return True

        # later: add custom rules for 'work', 'park', 'fire', etc.
        # e.g., offices: only opaque from outside, transparent inside.

    # No special region rule → use default from blocks_vision table
    return default_block


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
    x, y, heading_deg, fov_range, fov_angle_deg = get_agent_pose(env, agent_id)

    width, height = env.map_spec.width, env.map_spec.height

    heading_rad = math.radians(heading_deg)
    half_angle = math.radians(fov_angle_deg / 2.0)

    visible: List[Coord] = []

    for dx in range(-fov_range, fov_range + 1):
        for dy in range(-fov_range, fov_range + 1):
            tx, ty = x + dx, y + dy

            if tx < 0 or tx >= width or ty < 0 or ty >= height:
                continue

            dist = math.sqrt(dx * dx + dy * dy)
            if dist == 0 or dist > fov_range:
                continue

            angle_to_tile = math.atan2(dy, dx)
            diff = (angle_to_tile - heading_rad + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) > half_angle:
                continue

            # agent-specific occlusion
            if not los_clear(env, x, y, tx, ty, agent_id):
                continue

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


def los_clear(env, x0: int, y0: int, x1: int, y1: int, agent_id: int) -> bool:
    # line of sight
    dx = x1 - x0
    dy = y1 - y0

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    sx = 1 if dx > 0 else -1 if dx < 0 else 0
    sy = 1 if dy > 0 else -1 if dy < 0 else 0

    x, y = x0, y0

    if abs_dx > abs_dy:
        err = abs_dx / 2.0
        while x != x1:
            x += sx
            err -= abs_dy
            if err < 0:
                y += sy
                err += abs_dx

            if (x, y) == (x1, y1):
                break

            if blocks_vision(env, x, y, agent_id):
                return False
    else:
        err = abs_dy / 2.0
        while y != y1:
            y += sy
            err -= abs_dx
            if err < 0:
                x += sx
                err += abs_dy

            if (x, y) == (x1, y1):
                break

            if blocks_vision(env, x, y, agent_id):
                return False

    return True



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
