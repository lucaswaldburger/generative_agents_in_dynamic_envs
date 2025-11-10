import heapq
import numpy as np
import math
from env.constants import SEM_TO_ID
from minigrid.core.actions import Actions 

# Map grid deltas to Minigrid direction indices (0:E, 1:S, 2:W, 3:N)
DIR_IDX = {
    ( 1,  0): 0,  # east
    ( 0,  1): 1,  # south
    (-1,  0): 2,  # west
    ( 0, -1): 3,  # north
}

def get_neighbors(node, env):
    x, y = node
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < env.width and 0 <= ny < env.height:
            cell = env.grid.get(nx, ny)
            if cell is None or (hasattr(cell, "can_overlap") and cell.can_overlap()):
                neighbors.append((nx, ny))
    return neighbors

def get_path(start, goal, env):
    """A* shortest path on discrete grid."""
    open_list = []
    closed_list = set()
    came_from = {}
    g_costs = {start: 0}

    heapq.heappush(open_list, (0, start))

    while open_list:
        current_cost, current_node = heapq.heappop(open_list)
        if current_node == goal:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            return path[::-1]

        closed_list.add(current_node)

        for neighbor in get_neighbors(current_node, env):
            if neighbor in closed_list:
                continue
            tentative_g = g_costs[current_node] + 1
            if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g
                came_from[neighbor] = current_node
                heapq.heappush(open_list, (tentative_g, neighbor))
    return None


def path_to_minigrid_actions(path, start_dir):
    """Convert path [(x,y)] to Minigrid Actions list based on orientation."""
    actions = []
    cur_dir = start_dir

    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        dx, dy = x1 - x0, y1 - y0
        if (dx, dy) not in DIR_IDX:
            raise ValueError(f"Non-cardinal step: {(dx, dy)}")

        target_dir = DIR_IDX[(dx, dy)]
        delta = (target_dir - cur_dir) % 4

        if delta == 1:
            actions.append(Actions.right)
        elif delta == 3:
            actions.append(Actions.left)
        elif delta == 2:
            actions.extend([Actions.right, Actions.right])

        actions.append(Actions.forward)
        cur_dir = target_dir
    return actions

def parse_current(text):
    text = text.lower()
    if "home" in text:
        return "home"
    elif "work" in text:
        return "work"
    return None


def parse_next(text):
    text = text.lower()
    if "go to" in text:
        goal = text.split("go to")[-1].strip()
        if goal in ["home", "work"]:
            return goal
    return None

def get_goal_coordinates(goal, grid):
    goal_id = SEM_TO_ID.get(goal)
    if goal_id is None:
        return None
    for x in range(grid.width):
        for y in range(grid.height):
            tile = grid.get(x, y)
            if tile and hasattr(tile, "sem_id") and tile.sem_id == goal_id:
                return (x, y)
    return None


def get_next_plan_and_waypoint(current_loc_text, get_next_plan_text, env):
    current_location = parse_current(current_loc_text)
    next_location = parse_next(get_next_plan_text)

    if not current_location:
        return "Current location could not be determined."
    if not next_location:
        return "Next location could not be determined."

    current_coordinates = tuple(env.agent_pos)
    start_dir = int(env.agent_dir)
    next_coordinates = get_goal_coordinates(next_location, env.grid)

    path = get_path(current_coordinates, next_coordinates, env)
    if path is None or len(path) < 2:
        return "No valid path."

    actions = path_to_minigrid_actions(path, start_dir)
    if not actions:
        return "No valid actions could be generated from the path."

    return actions