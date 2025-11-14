import heapq
import numpy as np
import math
from typing import Tuple, List
from env.constants import SemanticMap, Coord, Action, DIR_TO_VEC
# from minigrid.core.actions import Actions 



### High level planner

def find_object_from_command(env, place_token: str) -> Coord:
    """
    Given a symbolic place token like 'home_A' or 'park', find the corresponding
    region in env.map_spec.regions and return a representative grid cell (x, y).
    """
    # Normalize the token: 'Home_A' -> 'home a'
    place_key = place_token.lower().replace("_", " ").strip()

    regions = getattr(env.map_spec, "regions", None)
    if not regions:
        raise RuntimeError("[Planner] env.map_spec.regions is empty or missing")

    for r in regions:
        name = str(r.get("name", "")).lower()          # e.g. "home a"
        typ  = str(r.get("type", "")).lower()          # e.g. "home"

        name_norm = name.replace("_", " ").strip()

        # Match by type or by (normalized) name
        # Examples:
        #   place_key = "park"   -> typ == "park"  or "park" in "park"
        #   place_key = "home a" -> "home a" in "home a"
        if place_key == typ or place_key in name_norm:
            x = int(r["x"])
            y = int(r["y"])
            return x, y

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
        if is_traversable(env, nx, ny):
            result.append(((nx, ny), action))
    return result

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(env, start: Coord, goal: Coord) -> List[Action]:
    if start == goal:
        return []

    pq = []
    heapq.heappush(pq, (manhattan(start, goal), 0, start))

    came_from = {}   # node -> (prev_node, action)
    g_score = {start: 0}
    closed = set()

    while pq:
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

    print(f"[A*] No path from {start} to {goal}")
    return []



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


