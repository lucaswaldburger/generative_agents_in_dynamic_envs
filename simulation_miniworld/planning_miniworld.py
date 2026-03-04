"""Navigation planning for MiniWorld 3D agents.

Unlike the grid-based environments, MiniWorld uses continuous coordinates on
the X-Z plane.  Planning here works by:

1. *High-level*: LLM picks a destination room/location name.
2. *Waypoint path*: A* over room-center graph finds a sequence of room centers.
3. *Low-level*: Translate the next waypoint into turn/move actions for the agent.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class PlannerLevel(IntEnum):
    HIGH = 0
    MID = 1
    LOW = 2


@dataclass
class ControlStateMW:
    """Tracks an agent's planning state."""
    high_intent: str = ""
    high_command: str = ""
    last_mid_direction: str = ""
    waypoints: list = field(default_factory=list)
    current_waypoint_idx: int = 0
    next_level: PlannerLevel = PlannerLevel.HIGH

    def clear(self):
        self.high_intent = ""
        self.high_command = ""
        self.last_mid_direction = ""
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.next_level = PlannerLevel.HIGH


# -----------------------------------------------------------------------
# Room-graph A*
# -----------------------------------------------------------------------

def build_room_graph(env) -> dict:
    """Build adjacency graph from room centers and connectivity.

    Returns a dict: room_name -> list of (neighbor_name, distance).
    The graph uses Euclidean distance between room centers as edge weight.
    """
    centers = env.get_room_centers()
    rooms = list(centers.keys())

    graph = {r: [] for r in rooms}

    inner = env._inner if hasattr(env, '_inner') else env

    for i, r1 in enumerate(rooms):
        c1 = np.array(centers[r1])
        for j, r2 in enumerate(rooms):
            if i >= j:
                continue
            c2 = np.array(centers[r2])
            dist = np.linalg.norm(c1 - c2)
            if dist < 15.0:
                graph[r1].append((r2, dist))
                graph[r2].append((r1, dist))

    return graph


def astar_rooms(graph: dict, start_room: str, goal_room: str) -> list:
    """A* search over the room graph, returning a list of room names."""
    if start_room == goal_room:
        return [start_room]
    if start_room not in graph or goal_room not in graph:
        return []

    open_set = [(0.0, start_room)]
    came_from = {}
    g_score = {start_room: 0.0}
    closed = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_room:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        if current in closed:
            continue
        closed.add(current)

        for neighbor, dist in graph.get(current, []):
            if neighbor in closed:
                continue
            tentative_g = g_score[current] + dist
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                heapq.heappush(open_set, (tentative_g, neighbor))

    return []


def plan_waypoints(env, agent_id: int, goal_room: str) -> list:
    """Compute a list of (x, z) waypoints from the agent to the goal room center.

    Uses A* over room centers, returning the sequence of room center coordinates.
    """
    agent = env.agents[agent_id]
    current_room = env.get_room_for_pos(agent.pos)
    if current_room is None:
        current_room = _nearest_room(agent.pos, env.get_room_centers())

    graph = build_room_graph(env)
    room_path = astar_rooms(graph, current_room, goal_room)

    if not room_path:
        centers = env.get_room_centers()
        if goal_room in centers:
            return [centers[goal_room]]
        return []

    centers = env.get_room_centers()
    waypoints = [centers[r] for r in room_path[1:]]
    return waypoints


# -----------------------------------------------------------------------
# Low-level action generation
# -----------------------------------------------------------------------

def compute_action_to_waypoint(
    agent_pos: np.ndarray,
    agent_dir: float,
    target_xz: tuple,
    turn_threshold: float = 0.25,
) -> int:
    """Determine the MiniWorld action (0-3) to move toward a target (x,z).

    Returns
    -------
    int : 0=turn_left, 1=turn_right, 2=move_forward, 3=move_back
    """
    tx, tz = target_xz
    dx = tx - agent_pos[0]
    dz = tz - agent_pos[2]
    desired_angle = math.atan2(-dz, dx)

    diff = (desired_angle - agent_dir + math.pi) % (2 * math.pi) - math.pi

    if abs(diff) < turn_threshold:
        return 2  # move_forward
    elif diff > 0:
        return 0  # turn_left
    else:
        return 1  # turn_right


def at_waypoint(agent_pos: np.ndarray, waypoint_xz: tuple, threshold: float = 1.5) -> bool:
    """Check if the agent has reached the waypoint."""
    tx, tz = waypoint_xz
    dist = math.sqrt((agent_pos[0] - tx) ** 2 + (agent_pos[2] - tz) ** 2)
    return dist < threshold


def normalize_command(decision: dict, agent_config, env) -> str:
    """Convert an LLM intent decision into a planner-friendly command."""
    intent = decision.get("intent", "stay")
    target = decision.get("target_location") or decision.get("action", "")

    if intent == "ignore" or intent == "stay":
        return "stay"

    centers = env.get_room_centers()
    target_lower = target.lower().strip()

    if "exit" in target_lower or "lobby" in target_lower:
        return "go to Lobby"

    for room_name in centers:
        if room_name.lower() in target_lower or target_lower in room_name.lower():
            return f"go to {room_name}"

    if intent == "evacuate":
        return "go to Lobby"

    return f"go to {target}" if target else "stay"


def resolve_goal_room(command: str, env) -> Optional[str]:
    """Extract the goal room name from a command string."""
    if command == "stay" or not command:
        return None

    parts = command.replace("go to ", "").strip()
    centers = env.get_room_centers()

    if parts in centers:
        return parts

    parts_lower = parts.lower()
    for room_name in centers:
        if room_name.lower() == parts_lower:
            return room_name

    for room_name in centers:
        if parts_lower in room_name.lower() or room_name.lower() in parts_lower:
            return room_name

    return None


def get_valid_directions() -> list:
    """Return the list of valid direction names for LLM prompts."""
    return ["FORWARD", "LEFT", "RIGHT", "BACK"]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _nearest_room(pos: np.ndarray, centers: dict) -> str:
    """Find the room whose center is closest to pos."""
    best_name = None
    best_dist = float("inf")
    for name, (cx, cz) in centers.items():
        d = math.sqrt((pos[0] - cx) ** 2 + (pos[2] - cz) ** 2)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name
