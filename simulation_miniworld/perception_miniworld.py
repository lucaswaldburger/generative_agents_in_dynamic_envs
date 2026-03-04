"""Perception and urgency assessment for MiniWorld 3D agents.

Translates the continuous 3D environment state into natural language
descriptions consumed by the LLM decision layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class UrgencyAssessment:
    urgency_score: float
    urgency_level: str          # "low", "medium", "high", "critical"
    nearest_fire_dist: float
    nearest_smoke_dist: float
    fires_in_room: bool
    smoke_in_room: bool


def describe_perception(env, agent_id: int, include_decision_info: bool = False) -> str:
    """Build a natural-language description of what the agent perceives.

    Parameters
    ----------
    env : MultiAgentMiniWorldEnv
    agent_id : int
    include_decision_info : bool
        If True, append valid movement directions for LLM decision prompts.
    """
    agent = env.agents[agent_id]
    pos = agent.pos
    direction_deg = math.degrees(agent.direction) % 360

    current_room = env.get_room_for_pos(pos)
    if current_room is None:
        current_room = "Unknown"

    parts = [
        f"I am in {current_room} "
        f"(position x={pos[0]:.1f}, z={pos[2]:.1f}, facing {_compass(direction_deg)})."
    ]

    fire_rooms, smoke_rooms = env.get_hazard_rooms()

    nearby_fires = _nearby_hazards(pos, env.fire_positions, radius=8.0)
    nearby_smokes = _nearby_hazards(pos, env.smoke_positions, radius=10.0)

    if nearby_fires:
        fire_descs = [f"fire at ({fx:.1f},{fz:.1f}), {d:.1f}m away" for fx, fz, d in nearby_fires[:3]]
        parts.append("Nearby fire: " + "; ".join(fire_descs) + ".")

    if nearby_smokes:
        smoke_descs = [f"smoke at ({sx:.1f},{sz:.1f}), {d:.1f}m away" for sx, sz, d in nearby_smokes[:3]]
        parts.append("Nearby smoke: " + "; ".join(smoke_descs) + ".")

    if fire_rooms:
        parts.append(f"Rooms with fire: {', '.join(fire_rooms)}.")
    if smoke_rooms:
        parts.append(f"Rooms with smoke: {', '.join(smoke_rooms)}.")

    exit_pos = env.get_exit_pos()
    exit_dist = np.linalg.norm(pos - exit_pos)
    exit_dir = _relative_direction(pos, agent.direction, exit_pos)
    parts.append(f"Exit is {exit_dist:.1f}m away, to the {exit_dir}.")

    visible_agents = agents_in_fov(env, agent_id, radius=6.0)
    if visible_agents:
        for oid, oname, odist in visible_agents:
            parts.append(f"{oname} is {odist:.1f}m away.")

    room_centers = env.get_room_centers()
    nearby_rooms = []
    for rname, (cx, cz) in room_centers.items():
        if rname == current_room:
            continue
        d = math.sqrt((pos[0] - cx) ** 2 + (pos[2] - cz) ** 2)
        if d < 12.0:
            rel_dir = _relative_direction(pos, agent.direction, np.array([cx, 0, cz]))
            nearby_rooms.append((rname, d, rel_dir))
    nearby_rooms.sort(key=lambda x: x[1])
    if nearby_rooms:
        room_strs = [f"{n} ({d:.0f}m, {rd})" for n, d, rd in nearby_rooms[:4]]
        parts.append("Nearby rooms: " + ", ".join(room_strs) + ".")

    if include_decision_info:
        parts.append("You can: turn_left, turn_right, move_forward, move_back.")

    return " ".join(parts)


def get_local_hazards(env, agent_id: int, radius: float = 6.0) -> dict:
    """Identify fire and smoke near the agent, update known hazard list."""
    agent = env.agents[agent_id]
    pos = agent.pos

    nearby_fires = _nearby_hazards(pos, env.fire_positions, radius)
    nearby_smokes = _nearby_hazards(pos, env.smoke_positions, radius)

    for fx, fz, _ in nearby_fires:
        h = ("fire", round(fx, 1), round(fz, 1))
        if h not in agent.known_hazard_positions:
            agent.known_hazard_positions.append(h)
    for sx, sz, _ in nearby_smokes:
        h = ("smoke", round(sx, 1), round(sz, 1))
        if h not in agent.known_hazard_positions:
            agent.known_hazard_positions.append(h)

    return {
        "fires": [(fx, fz, d) for fx, fz, d in nearby_fires],
        "smokes": [(sx, sz, d) for sx, sz, d in nearby_smokes],
        "has_local_fire": len(nearby_fires) > 0,
        "has_local_smoke": len(nearby_smokes) > 0,
    }


def agents_in_fov(env, agent_id: int, radius: float = 6.0) -> list:
    """Return list of (other_id, name, distance) for nearby agents."""
    agent = env.agents[agent_id]
    result = []
    for oid, other in enumerate(env.agents):
        if oid == agent_id:
            continue
        if other.reached_exit or other.dead:
            continue
        dist = np.linalg.norm(agent.pos - other.pos)
        if dist <= radius:
            result.append((oid, other.name, dist))
    return result


def assess_urgency(
    env,
    agent_id: int,
    agent,
    fire_replan_count: int = 0,
    smoke_replan_count: int = 0,
) -> UrgencyAssessment:
    """Compute an urgency score based on proximity to hazards."""
    pos = agent.pos

    nearest_fire = float("inf")
    for fx, fz in env.fire_positions:
        d = math.sqrt((pos[0] - fx) ** 2 + (pos[2] - fz) ** 2)
        nearest_fire = min(nearest_fire, d)

    nearest_smoke = float("inf")
    for sx, sz in env.smoke_positions:
        d = math.sqrt((pos[0] - sx) ** 2 + (pos[2] - sz) ** 2)
        nearest_smoke = min(nearest_smoke, d)

    current_room = env.get_room_for_pos(pos)
    fire_rooms, smoke_rooms = env.get_hazard_rooms()
    fires_in_room = current_room in fire_rooms if current_room else False
    smoke_in_room = current_room in smoke_rooms if current_room else False

    score = 0.0
    if nearest_fire < 2.0:
        score += 0.5
    elif nearest_fire < 5.0:
        score += 0.3
    elif nearest_fire < 10.0:
        score += 0.15

    if nearest_smoke < 3.0:
        score += 0.2
    elif nearest_smoke < 8.0:
        score += 0.1

    if fires_in_room:
        score += 0.2
    if smoke_in_room:
        score += 0.1

    score += fire_replan_count * 0.05
    score += smoke_replan_count * 0.02

    score = min(score, 1.0)

    if score >= 0.7:
        level = "critical"
    elif score >= 0.45:
        level = "high"
    elif score >= 0.25:
        level = "medium"
    else:
        level = "low"

    return UrgencyAssessment(
        urgency_score=round(score, 2),
        urgency_level=level,
        nearest_fire_dist=round(nearest_fire, 1),
        nearest_smoke_dist=round(nearest_smoke, 1),
        fires_in_room=fires_in_room,
        smoke_in_room=smoke_in_room,
    )


def format_urgency_for_llm(ua: UrgencyAssessment) -> str:
    """Format urgency assessment as a string for LLM prompts."""
    parts = [f"Urgency: {ua.urgency_level.upper()} ({ua.urgency_score:.2f})."]
    if ua.nearest_fire_dist < 20:
        parts.append(f"Nearest fire: {ua.nearest_fire_dist:.1f}m.")
    if ua.nearest_smoke_dist < 20:
        parts.append(f"Nearest smoke: {ua.nearest_smoke_dist:.1f}m.")
    if ua.fires_in_room:
        parts.append("WARNING: Fire in your current room!")
    if ua.smoke_in_room:
        parts.append("Smoke detected in your current room.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nearby_hazards(pos: np.ndarray, hazard_list: list, radius: float) -> list:
    """Return list of (x, z, distance) for hazards within radius."""
    result = []
    for hx, hz in hazard_list:
        d = math.sqrt((pos[0] - hx) ** 2 + (pos[2] - hz) ** 2)
        if d <= radius:
            result.append((hx, hz, d))
    result.sort(key=lambda x: x[2])
    return result


def _compass(degrees: float) -> str:
    """Convert heading degrees to compass direction."""
    d = degrees % 360
    if d < 22.5 or d >= 337.5:
        return "East"
    elif d < 67.5:
        return "Northeast"
    elif d < 112.5:
        return "North"
    elif d < 157.5:
        return "Northwest"
    elif d < 202.5:
        return "West"
    elif d < 247.5:
        return "Southwest"
    elif d < 292.5:
        return "South"
    else:
        return "Southeast"


def _relative_direction(agent_pos, agent_dir, target_pos) -> str:
    """Describe target position relative to agent's facing direction."""
    dx = target_pos[0] - agent_pos[0]
    dz = target_pos[2] - agent_pos[2]
    angle_to_target = math.atan2(-dz, dx)
    relative = (angle_to_target - agent_dir) % (2 * math.pi)

    if relative < math.pi / 4 or relative > 7 * math.pi / 4:
        return "ahead"
    elif relative < 3 * math.pi / 4:
        return "left"
    elif relative < 5 * math.pi / 4:
        return "behind"
    else:
        return "right"
