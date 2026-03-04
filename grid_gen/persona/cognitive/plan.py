from __future__ import annotations

import heapq
import numpy as np
import math
from typing import Tuple, List
from env.constants import SemanticMap, Coord, Action, DIR_TO_VEC
# from minigrid.core.actions import Actions 





## daily planner

def get_plan_for_time(agent_cfg, time_str: str):
    """
    Return the plan item active at time_str.

    Rules:
    - If an item has a range "HH:MM-HH:MM", use it directly.
    - If an item has a single time "HH:MM", treat it as starting then
      and lasting until the next plan item's start time.
    - If time_str is before the first item, return None.
    """
    def to_minutes(hhmm: str) -> int:
        h, m = hhmm.strip().split(":")
        return int(h) * 60 + int(m)

    plan = agent_cfg.daily_plan or []
    if not plan:
        return None

    t_now = to_minutes(time_str)

    # Preprocess items into intervals
    intervals = []
    for i, item in enumerate(plan):
        t_field = item["time"].strip()

        if "-" in t_field:
            start_s, end_s = t_field.split("-")
            start_m = to_minutes(start_s)
            end_m = to_minutes(end_s)
            intervals.append((start_m, end_m, item))
        else:
            start_m = to_minutes(t_field)
            # end at next item's start if exists, else end of day
            if i + 1 < len(plan):
                next_field = plan[i + 1]["time"].strip()
                next_start_s = next_field.split("-")[0]  # if next is a range, take its start
                end_m = to_minutes(next_start_s)
            else:
                end_m = 24 * 60  # until midnight
            intervals.append((start_m, end_m, item))

    # Find matching interval
    for start_m, end_m, item in intervals:
        if start_m <= t_now < end_m:
            return item

    return None



def get_accessible_locations(env, agent_id):
    start = get_agent_tile(env, agent_id)
    accessible = []

    for r in env.map_spec.regions:
        name = str(r.get("name"))
        typ  = str(r.get("type", ""))

        gx, gy = int(r["x"]), int(r["y"])

        # quick traversability check at goal
        if not is_traversable(env, gx, gy):
            continue

        path = astar(env, start, (gx, gy))
        if path is not None and len(path) > 0:
            accessible.append(name)
            if typ:
                accessible.append(typ)

        # also allow staying if already there
        if (gx, gy) == start:
            accessible.append(name)
            if typ:
                accessible.append(typ)

    return sorted(set(accessible))


### High level planner

def find_object_from_command(env, place_token: str) -> Coord:
    place_key = place_token.lower().replace("_", " ").strip()
    regions = getattr(env.map_spec, "regions", None)
    if not regions:
        raise RuntimeError("[Planner] env.map_spec.regions is empty or missing")

    for r in regions:
        name = str(r.get("name", "")).lower().replace("_", " ").strip()
        typ  = str(r.get("type", "")).lower()

        if place_key == typ or place_key in name:
            x0, y0 = int(r["x"]), int(r["y"])
            w, h = int(r.get("w", 1)), int(r.get("h", 1))

            # search for first traversable cell inside region
            for yy in range(y0, y0 + h):
                for xx in range(x0, x0 + w):
                    if env._can_move_to(xx, yy):
                        return (xx, yy)

            # If no traversable cell found in region, search for nearest traversable cell
            # Start from center and expand outward
            center_x = x0 + w // 2
            center_y = y0 + h // 2
            
            # Try center first
            if env._can_move_to(center_x, center_y):
                return (center_x, center_y)
            
            # Expand outward in a spiral pattern
            max_radius = max(w, h) + 5  # Search a bit beyond the region
            for radius in range(1, max_radius + 1):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        # Only check cells on the perimeter of current radius
                        if abs(dx) == radius or abs(dy) == radius:
                            nx, ny = center_x + dx, center_y + dy
                            if env._can_move_to(nx, ny):
                                return (nx, ny)
            
            # If still no traversable cell found, raise an error with details
            raise RuntimeError(
                f"[Planner] No traversable cell found for region '{name}' "
                f"at ({x0}, {y0}) size ({w}, {h}). "
                f"Region may be blocked or inaccessible."
            )
        

    raise KeyError(f"[Planner] No object matched place token: '{place_token}'")

def high_level_planner(env, agent_id: int, command: str) -> Tuple[Coord, Coord]:
    """
    Returns (start_cell, goal_cell) in grid coordinates for this agent.
    """
    start = get_agent_tile(env, agent_id)
    goal = resolve_goal_tile(env, agent_id, command)
    return start, goal

def resolve_goal_tile(env, agent_id: int, command: str) -> Coord:
    """
    Parse a high-level command string and resolve it to a goal tile (x, y).
    For now we assume the LAST token is the place token, e.g.:
      'go to home_A' -> 'home_A'
      'go park'      -> 'park'
    """
    cmd = command.strip()
    tokens = cmd.split()
    if not tokens:
        raise ValueError(f"[Planner] Empty command for agent {agent_id}")

    place_token = tokens[-1]
    return find_object_from_command(env, place_token)

def get_agent_tile(env, agent_id: int) -> Coord:
    """
    Return the agent's current grid cell as (x, y).
    Adapt this to however MultiHumanGridEnv stores positions.
    """
    agent = env.agents[agent_id]
    return int(agent.x), int(agent.y)



def reconstruct_actions(came_from: dict, end: Coord) -> List[Action]:
    """
    Reconstruct list of actions from came_from dict:
      came_from[node] = (prev_node, action_taken)
    """
    actions = []
    current = end
    while current in came_from:
        prev, act = came_from[current]
        actions.append(act)
        current = prev
    actions.reverse()
    return actions




## Low level planner
def is_traversable(env, x: int, y: int) -> bool:
    ms = env.map_spec
    if x < 0 or x >= ms.width or y < 0 or y >= ms.height:
        return False

    code = int(ms.access_grid[y, x])
    can_enter = ms.semantics.get("can_enter", {})
    return bool(can_enter.get(str(code), False))

def neighbors(env, node: Coord) -> List[Tuple[Coord, Action]]:
    """
    Return list of (neighbor_coord, action_taken)
    """
    x, y = node
    result = []
    for action, (dx, dy) in DIR_TO_VEC.items():
        if action == Action.STAY:
            continue
        nx, ny = x + dx, y + dy
        # if is_traversable(env, nx, ny):
        #     result.append(((nx, ny), action))
        if env._can_move_to(nx, ny):
            result.append(((nx, ny), action))
    return result

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(env, start: Coord, goal: Coord) -> List[Action]:
    if start == goal:
        return []

    # Validate start and goal are traversable
    original_start = start
    if not env._can_move_to(*start):
        print(f"[Astar error] Start position {start} is not traversable")
        # Try to find nearest traversable cell to start
        new_start = _find_nearest_traversable(env, start)
        if new_start is None:
            print(f"[Astar error] Could not find traversable cell near start {original_start}")
            return []
        start = new_start
        print(f"[Astar] Using alternative start {start} instead of original {original_start}")
    
    original_goal = goal
    if not env._can_move_to(*goal):
        print(f"[Astar warning] Goal position {goal} is not traversable, searching for nearest traversable cell")
        # Try to find nearest traversable cell to goal
        new_goal = _find_nearest_traversable(env, goal)
        if new_goal is None:
            print(f"[Astar error] Could not find traversable cell near goal {goal}")
            return []
        goal = new_goal
        print(f"[Astar] Using alternative goal {goal} instead of original {original_goal}")

    pq = []
    heapq.heappush(pq, (manhattan(start, goal), 0, start))

    came_from = {}   # node -> (prev_node, action)
    g_score = {start: 0}
    closed = set()
    max_iterations = env.map_spec.width * env.map_spec.height * 2  # Prevent infinite loops
    iterations = 0

    while pq and iterations < max_iterations:
        iterations += 1
        f, g, node = heapq.heappop(pq)

        if node in closed:
            continue
        closed.add(node)

        if node == goal:
            return reconstruct_actions(came_from, node)

        for (nb, action) in neighbors(env, node):
            new_g = g + 1
            if new_g < g_score.get(nb, float("inf")):
                came_from[nb] = (node, action)
                g_score[nb] = new_g
                f_nb = new_g + manhattan(nb, goal)
                heapq.heappush(pq, (f_nb, new_g, nb))

    # If we exhausted the search space, try to find nearest reachable cell to goal
    if not pq or iterations >= max_iterations:
        print(f"[Astar warning] No direct path from {start} to {goal}, searching for nearest reachable cell to goal")
        # Find the closest node we reached to the goal
        if closed:
            closest_node = min(closed, key=lambda n: manhattan(n, goal))
            closest_dist = manhattan(closest_node, goal)
            if closest_dist <= 3:  # If we got close, use that path
                print(f"[Astar] Using path to nearest reachable cell {closest_node} (distance {closest_dist} from goal)")
                return reconstruct_actions(came_from, closest_node)
    
    print(f"[Astar error] No path from {start} to {goal} (searched {iterations} iterations, explored {len(closed)} nodes)")
    return []


def _find_nearest_traversable(env, pos: Coord, max_radius: int = 10) -> Coord | None:
    """
    Find the nearest traversable cell to the given position.
    Returns None if no traversable cell is found within max_radius.
    """
    x, y = pos
    
    # Check the position itself first
    if env._can_move_to(x, y):
        return (x, y)
    
    # Search in expanding radius
    for radius in range(1, max_radius + 1):
        # Check all cells at this radius
        candidates = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) == radius or abs(dy) == radius:  # Only perimeter
                    nx, ny = x + dx, y + dy
                    if env._can_move_to(nx, ny):
                        dist = manhattan((x, y), (nx, ny))
                        candidates.append(((nx, ny), dist))
        
        if candidates:
            # Return the closest one
            candidates.sort(key=lambda c: c[1])
            return candidates[0][0]
    
    return None



def path_to_actions(path: List[Coord]) -> List[Action]:
    """
    Convert a list of coords [(x0,y0), (x1,y1), ...] into Actions.
    Assumes y increases *downwards*, so:
      up    = (0, -1)
      down  = (0, 1)
      right = (1, 0)
      left  = (-1, 0)
    """
    actions: List[Action] = []
    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        dx, dy = x2 - x1, y2 - y1
        if   (dx, dy) == (1, 0):
            actions.append(Action.RIGHT)
        elif (dx, dy) == (-1, 0):
            actions.append(Action.LEFT)
        elif (dx, dy) == (0, -1):
            actions.append(Action.UP)
        elif (dx, dy) == (0, 1):
            actions.append(Action.DOWN)
        else:
            raise ValueError(f"[A*] Non-adjacent step in path: {(x1, y1)} -> {(x2, y2)}")
    return actions


def normalize_command_for_planner(decision, agent_cfg, env):
    """
    decision: dict returned by llm_decide_intent
    Always returns a safe planner command.
    """
    raw_intent = (decision.get("intent") or "").lower()
    # collapse things like "evacuate: ..." back to "evacuate"
    intent = "evacuate" if "evacuate" in raw_intent else raw_intent

    target = decision.get("target_location")

    # stay = no movement
    if intent == "stay" or (decision.get("command", "").lower().strip() == "stay"):
        return "stay"

    # if model gave a target, trust it
    if target:
        return f"go to {target}"

    # dependent / family intent -> go home
    if intent in {"check_dependent", "help_other"}:
        home = getattr(agent_cfg, "living_area", None)
        if home:
            return f"go to {home}"
        return "stay"

    # evacuate intent -> go to a safe region you define
    if intent == "evacuate":
        # if it forgot to set target but we know a home, prefer home
        home = getattr(agent_cfg, "living_area", None)
        if home:
            return f"go to {home}"
        return "go to streets"   # default safe zone

    # fallback
    return "stay"


