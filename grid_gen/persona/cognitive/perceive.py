from __future__ import annotations
from typing import Tuple, List, Dict, Any
import math

from env.constants import Coord, Action, DIR_TO_VEC
from persona.cognitive.plan import neighbors, get_agent_tile


def is_intersection(env, agent_id: int) -> bool:
    """
    Intersection-ish if agent has 3+ possible moves,
    OR 2 moves that are not a straight corridor (a turn choice).
    """
    acts = valid_move_actions(env, agent_id)  # excludes STAY
    k = len(acts)
    if k >= 3:
        return True
    if k <= 1:
        return False

    a1, a2 = acts
    v1, v2 = DIR_TO_VEC[a1], DIR_TO_VEC[a2]
    opposite = (v1[0] == -v2[0] and v1[1] == -v2[1])
    return not opposite


def action_names(actions):
    # stable string list for LLM
    name_map = {
        Action.UP: "UP",
        Action.DOWN: "DOWN",
        Action.LEFT: "LEFT",
        Action.RIGHT: "RIGHT",
    }
    return [name_map[a] for a in actions]


def neighbor_semantics(env, agent_id):
    start = get_agent_tile(env, agent_id)
    neigh = neighbors(env, start)
    result = {}
    for (coord, action) in neigh:
        x, y = coord
        loc = classify_location(env, x, y)
        code = int(env.map_spec.access_grid[y, x])
        access_codes = env.map_spec.raw.get("access_codes", {})
        inv = {v: k for k, v in access_codes.items()}
        label = inv.get(code, "")
        hazard = ("hazard" in label.lower()) or ("fire" in label.lower()) or ("smoke" in label.lower())
        result[action] = {
            "coord": coord,
            "location": loc,
            "hazard": hazard,
        }
    return result

def valid_move_actions(env, agent_id):
    start = get_agent_tile(env, agent_id)
    return [act for (_, act) in neighbors(env, start)]

def blocks_vision(env, x: int, y: int, agent_id: int) -> bool:
    ms = env.map_spec
    if x < 0 or x >= ms.width or y < 0 or y >= ms.height:
        return True
    code = int(ms.access_grid[y, x])
    semantics = ms.semantics or {}
    block_table = semantics.get("blocks_vision", {})
    default_block = bool(block_table.get(str(code), True))
    regs = regions_containing_point(env, x, y)
    for r in regs:
        r_type = r.get("type")
        if r_type == "home":
            residents = r.get("residents", [])
            agent_name = env.agent_configs[agent_id].id
            if agent_name in residents:
                return False
            else:
                return True
    return default_block


def get_agent_pose(env, agent_id: int) -> Tuple[int, int, int, int, int]:
    agent = env.agents[agent_id]
    x, y = int(agent.x), int(agent.y)
    heading_deg = int(agent.heading_deg)
    base_fov_range = int(agent.config.fov.range_cells)
    if hasattr(env, '_get_dynamic_fov_range'):
        fov_range = env._get_dynamic_fov_range(x, y, base_fov_range)
    else:
        fov_range = base_fov_range
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
            if not los_clear(env, x, y, tx, ty, agent_id):
                continue
            visible.append((tx, ty))
    return visible



def regions_in_fov(env, agent_id: int) -> List[Dict[str, Any]]:
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
        for (tx, ty) in visible_tiles:
            if rx <= tx < rx + rw and ry <= ty < ry + rh:
                visible_regions.append(r)
                break
    return visible_regions


def classify_location(env, x: int, y: int) -> str:
    here = regions_containing_point(env, x, y)
    if not here:
        code = int(env.map_spec.access_grid[y, x])
        access_codes = env.map_spec.raw.get("access_codes", {})
        inv = {v: k for k, v in access_codes.items()}
        label = inv.get(code, "walkable")
        if "hazard" in label:
            return "hazard area"
        return "street"
    preferred_types = ["home", "work", "park", "fire"]
    best = None
    for r in here:
        if r.get("type") in preferred_types:
            best = r
            break
    if best is None:
        best = here[0]
    return best.get("name", "unknown")


def los_clear(env, x0: int, y0: int, x1: int, y1: int, agent_id: int) -> bool:
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

#------------------------------------------------------------------------
# perceive hazards (Data from Lucas's code)
# those functions are to perceive hazard, aligning with the local environment conditions.

def _cell_name_at(env, x: int, y: int) -> str:
    cell_names = getattr(env.map_spec, "cell_names", None)
    if cell_names is None:
        return ""
    try:
        h, w = cell_names.shape
    except Exception:
        return ""
    if 0 <= x < w and 0 <= y < h:
        name = cell_names[y, x]
        if isinstance(name, str) and name != "":
            return name
    return ""


def _nearest_named_place(env, x: int, y: int, max_radius: int = 2) -> str:
    cell_names = getattr(env.map_spec, "cell_names", None)
    if cell_names is None:
        return ""
    try:
        h, w = cell_names.shape
    except Exception:
        return ""
    for r in range(1, max_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    name = cell_names[ny, nx]
                    if isinstance(name, str) and name != "":
                        return name
    return ""


def get_local_hazards(env, agent_id, radius: int = 3):
    """
    Returns compact hazard descriptions.
    Old format: "There is a fire on the street at (8,12)." (~12 tokens)
    New format: "fire@(8,12):street" (~5 tokens)
    """
    ax, ay, _, _, _ = get_agent_pose(env, agent_id)
    hazards: list[str] = []
    W, H = env.map_spec.width, env.map_spec.height
    fire_locs = getattr(env, "fire_locations", set())
    smoke_locs = getattr(env, "smoke_locations", set())
    traffic_locs = getattr(env, "traffic_locations", set())
 
    reported_fire = set()
    reported_smoke = set()
    reported_traffic = set()
 
    agent = env.agents[agent_id]
    if not hasattr(agent, "known_hazard_cells"):
        agent.known_hazard_cells = set()
 
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x, y = ax + dx, ay + dy
            if not (0 <= x < W and 0 <= y < H):
                continue
            pos = (x, y)
            place = None  # lazy-evaluate only if needed
 
            if pos in fire_locs:
                place = _cell_name_at(env, x, y) or _nearest_named_place(env, x, y) or "street"
                key = ("fire", place)
                if key not in reported_fire:
                    reported_fire.add(key)
                    agent.known_hazard_cells.add(pos)
                    hazards.append(f"fire@({x},{y}):{place}")
 
            elif pos in smoke_locs:
                place = _cell_name_at(env, x, y) or _nearest_named_place(env, x, y) or "nearby"
                key = ("smoke", place)
                if key not in reported_smoke:
                    reported_smoke.add(key)
                    agent.known_hazard_cells.add(pos)
                    hazards.append(f"smoke@({x},{y}):{place}")
 
            elif pos in traffic_locs:
                place = _cell_name_at(env, x, y) or _nearest_named_place(env, x, y) or "street"
                key = ("traffic", place)
                if key not in reported_traffic:
                    reported_traffic.add(key)
                    hazards.append(f"traffic@({x},{y}):{place}")
 
    return hazards



def describe_perception(env, agent_id: int, include_decision_info: bool = False) -> str:
    """
    Compressed perception string for LLM prompts.
 
    Old: "I am at the street. I see Home_A, Park. There is a fire on the
         street at (8,12). There is heavy traffic near Workplace_A at (8,12).
         I see heavy traffic near Workplace_A right now."
         (~50 tokens, with duplicate traffic info)
 
    New: "At:street See:Home_A,Park Hazards:fire@(8,12):street,traffic@(9,12):Workplace_A"
         (~20 tokens, no duplication)
    """
    x, y, heading_deg, fov_range, fov_angle_deg = get_agent_pose(env, agent_id)
    location_label = classify_location(env, x, y)
    visible_regions = regions_in_fov(env, agent_id)
    visible_names = sorted(
        set(r.get("name") for r in visible_regions if r.get("name"))
    )
 
    parts: list[str] = []
 
    # Location + visible landmarks (compact)
    see_str = ",".join(visible_names) if visible_names else "nothing"
    parts.append(f"At:{location_label} See:{see_str}")
 
    # Hazards (already compact from get_local_hazards)
    # NOTE: get_local_hazards already includes traffic, so we do NOT call
    # get_local_traffic_cells separately (this was duplicating traffic info)
    hazards = get_local_hazards(env, agent_id)
    if hazards:
        parts.append(f"Hazards:{','.join(sorted(set(hazards)))}")
 
    # Navigation info for mid-level decisions
    if include_decision_info:
        valid_moves = valid_move_actions(env, agent_id)
        valid_dirs = action_names(valid_moves)
        if valid_dirs:
            parts.append(f"Dirs:{','.join(valid_dirs)}")
        if is_intersection(env, agent_id):
            parts.append("Intersection:yes")
 
    return " ".join(parts)
 
 
def get_local_traffic_cells(env, agent_id: int, radius: int = 3):
    """
    Return a list of (x, y, place_label) where traffic is observed.
    Kept for backward compatibility but no longer called by describe_perception.
    """
    ax, ay, _, _, _ = get_agent_pose(env, agent_id)
    W, H = env.map_spec.width, env.map_spec.height
    traffic_locs = getattr(env, "traffic_locations", set())
    seen = set()
    results: list[tuple[int, int, str]] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x, y = ax + dx, ay + dy
            if not (0 <= x < W and 0 <= y < H):
                continue
            pos = (x, y)
            if pos not in traffic_locs or pos in seen:
                continue
            seen.add(pos)
            place = _cell_name_at(env, x, y) or _nearest_named_place(env, x, y) or "nearby street"
            results.append((x, y, place))
    return results
 
 
def agents_in_fov(env, agent_id: int):
    me = env.agents[agent_id]
    fx, fy = me.x, me.y
    base_r = me.config.fov.range_cells
    if hasattr(env, '_get_dynamic_fov_range'):
        r = env._get_dynamic_fov_range(fx, fy, base_r)
    else:
        r = base_r
    nearby = []
    for other_id, other in enumerate(env.agents):
        if other_id == agent_id:
            continue
        if abs(other.x - fx) + abs(other.y - fy) <= r:
            nearby.append(other_id)
    return nearby