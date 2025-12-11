"""
Urgency and Reflection Module

This module evaluates the current situation and provides urgency assessment
and reflection that feeds into high-level planning decisions.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math

from persona.cognitive.perceive import get_local_hazards, get_agent_pose
from env.grid import MultiHumanGridEnv


@dataclass
class UrgencyAssessment:
    """Structured urgency assessment of the current situation."""
    urgency_level: str  # "low", "medium", "high", "critical"
    urgency_score: float  # 0.0 to 1.0
    primary_factors: list[str]  # List of factors contributing to urgency
    fire_proximity: float  # Distance to nearest fire (cells), or inf
    smoke_proximity: float  # Distance to nearest smoke (cells), or inf
    visibility_impact: str  # "normal", "reduced", "severely_reduced"
    safety_assessment: str  # "safe", "at_risk", "in_danger", "critical_danger"
    time_since_awareness: Optional[int] = None  # Steps since agent became aware


def assess_urgency(
    env: MultiHumanGridEnv,
    agent_id: int,
    agent,
    t: int,
    fire_start_time: Optional[int] = None,
) -> UrgencyAssessment:
    """
    Assess the urgency of the current situation for an agent.
    
    Parameters:
    -----------
    env : MultiHumanGridEnv
        The environment
    agent_id : int
        Agent identifier
    agent : HumanAgent
        The agent object
    t : int
        Current simulation step
    fire_start_time : Optional[int]
        Step when fire was first detected/started (for time-based urgency)
    
    Returns:
    --------
    UrgencyAssessment
        Structured urgency assessment
    """
    ax, ay, _, _, _ = get_agent_pose(env, agent_id)
    
    # Get fire and smoke locations
    fire_locs = getattr(env, "fire_locations", set())
    smoke_locs = getattr(env, "smoke_locations", set())
    
    # Calculate distances to nearest fire and smoke
    fire_dist = _min_distance_to_set((ax, ay), fire_locs)
    smoke_dist = _min_distance_to_set((ax, ay), smoke_locs)
    
    # Assess visibility impact (smoke reduces FOV)
    visibility_impact = _assess_visibility_impact(env, agent_id, fire_dist, smoke_dist)
    
    # Get local hazards for context
    hazards = get_local_hazards(env, agent_id, radius=5)
    fire_hazards = [h for h in hazards if "fire" in h.lower()]
    smoke_hazards = [h for h in hazards if "smoke" in h.lower()]
    
    # Calculate urgency score (0.0 to 1.0)
    urgency_score = _calculate_urgency_score(
        fire_dist, smoke_dist, visibility_impact, 
        len(fire_hazards), len(smoke_hazards),
        agent, t, fire_start_time
    )
    
    # Determine urgency level
    if urgency_score >= 0.8:
        urgency_level = "critical"
        safety = "critical_danger"
    elif urgency_score >= 0.6:
        urgency_level = "high"
        safety = "in_danger"
    elif urgency_score >= 0.3:
        urgency_level = "medium"
        safety = "at_risk"
    else:
        urgency_level = "low"
        safety = "safe"
    
    # Identify primary factors
    primary_factors = _identify_primary_factors(
        fire_dist, smoke_dist, visibility_impact,
        len(fire_hazards), len(smoke_hazards),
        agent
    )
    
    # Calculate time since awareness (if agent has seen fire/smoke)
    time_since_awareness = None
    if hasattr(agent, "local_fire_smoke_seen") and agent.local_fire_smoke_seen:
        if hasattr(agent, "fire_awareness_time"):
            time_since_awareness = t - agent.fire_awareness_time
        else:
            # First time we're tracking, set it
            agent.fire_awareness_time = t
            time_since_awareness = 0
    
    return UrgencyAssessment(
        urgency_level=urgency_level,
        urgency_score=urgency_score,
        primary_factors=primary_factors,
        fire_proximity=fire_dist,
        smoke_proximity=smoke_dist,
        visibility_impact=visibility_impact,
        time_since_awareness=time_since_awareness,
        safety_assessment=safety,
    )


def _min_distance_to_set(pos: Tuple[int, int], loc_set: set) -> float:
    """Calculate Manhattan distance to nearest location in set."""
    if not loc_set:
        return float('inf')
    
    px, py = pos
    min_dist = float('inf')
    for (fx, fy) in loc_set:
        dist = abs(px - fx) + abs(py - fy)  # Manhattan distance
        min_dist = min(min_dist, dist)
    
    return min_dist


def _assess_visibility_impact(
    env: MultiHumanGridEnv,
    agent_id: int,
    fire_dist: float,
    smoke_dist: float,
) -> str:
    """Assess how visibility is impacted by fire/smoke proximity."""
    # Use the dynamic FOV range calculation from grid.py
    agent = env.agents[agent_id]
    base_range = agent.config.fov.range_cells
    
    # Check if FOV is reduced due to proximity
    if fire_dist <= 1:
        return "severely_reduced"
    elif fire_dist <= 2 or smoke_dist <= 1:
        return "reduced"
    elif fire_dist <= 4 or smoke_dist <= 3:
        return "slightly_reduced"
    else:
        return "normal"


def _calculate_urgency_score(
    fire_dist: float,
    smoke_dist: float,
    visibility_impact: str,
    num_fire_hazards: int,
    num_smoke_hazards: int,
    agent,
    t: int,
    fire_start_time: Optional[int],
) -> float:
    """
    Calculate urgency score from 0.0 (no urgency) to 1.0 (critical).
    
    Factors:
    - Fire proximity (closer = higher urgency)
    - Smoke proximity (closer = higher urgency)
    - Visibility impact
    - Number of visible hazards
    - Agent's risk perception traits
    - Time since fire started (if known)
    """
    score = 0.0
    
    # Fire proximity factor (0.0 to 0.4)
    if fire_dist < float('inf'):
        if fire_dist <= 1:
            fire_factor = 0.4
        elif fire_dist <= 2:
            fire_factor = 0.3
        elif fire_dist <= 4:
            fire_factor = 0.2
        elif fire_dist <= 6:
            fire_factor = 0.1
        else:
            fire_factor = 0.05
        score += fire_factor
    
    # Smoke proximity factor (0.0 to 0.2)
    if smoke_dist < float('inf'):
        if smoke_dist <= 1:
            smoke_factor = 0.2
        elif smoke_dist <= 2:
            smoke_factor = 0.15
        elif smoke_dist <= 4:
            smoke_factor = 0.1
        else:
            smoke_factor = 0.05
        score += smoke_factor
    
    # Visibility impact factor (0.0 to 0.15)
    visibility_factors = {
        "severely_reduced": 0.15,
        "reduced": 0.1,
        "slightly_reduced": 0.05,
        "normal": 0.0,
    }
    score += visibility_factors.get(visibility_impact, 0.0)
    
    # Multiple hazards factor (0.0 to 0.1)
    total_hazards = num_fire_hazards + num_smoke_hazards
    if total_hazards >= 3:
        score += 0.1
    elif total_hazards >= 2:
        score += 0.05
    
    # Time-based urgency (0.0 to 0.15)
    if fire_start_time is not None:
        time_elapsed = t - fire_start_time
        if time_elapsed > 20:  # Fire has been spreading for a while
            score += 0.15
        elif time_elapsed > 10:
            score += 0.1
        elif time_elapsed > 5:
            score += 0.05
    
    # Agent risk perception modifier
    # Agents with lower risk perception might underestimate urgency
    # Agents with higher risk perception might overestimate urgency
    risk_modifier = _get_risk_perception_modifier(agent)
    score *= risk_modifier
    
    # Cap at 1.0
    return min(1.0, score)


def _get_risk_perception_modifier(agent) -> float:
    """
    Get risk perception modifier based on agent's persona.
    Returns a multiplier (0.7 to 1.3) that adjusts urgency perception.
    """
    # Default modifier
    modifier = 1.0
    
    # Check agent config for risk-related traits
    if hasattr(agent, 'config'):
        cfg = agent.config
        
        # Check persona compact string for risk indicators
        persona_str = getattr(cfg, 'persona_compact', '')
        
        # Lower risk perception -> underestimate urgency (multiply by < 1.0)
        if 'underestimate' in persona_str.lower() or 'skeptical' in persona_str.lower():
            modifier = 0.8
        
        # Higher risk perception -> overestimate urgency (multiply by > 1.0)
        elif 'cautious' in persona_str.lower() or 'risk-averse' in persona_str.lower():
            modifier = 1.2
    
    return modifier


def _identify_primary_factors(
    fire_dist: float,
    smoke_dist: float,
    visibility_impact: str,
    num_fire_hazards: int,
    num_smoke_hazards: int,
    agent,
) -> list[str]:
    """Identify the primary factors contributing to urgency."""
    factors = []
    
    if fire_dist <= 2:
        factors.append("very_close_to_fire")
    elif fire_dist <= 4:
        factors.append("close_to_fire")
    
    if smoke_dist <= 2:
        factors.append("surrounded_by_smoke")
    elif smoke_dist <= 4:
        factors.append("smoke_nearby")
    
    if visibility_impact in ["severely_reduced", "reduced"]:
        factors.append("poor_visibility")
    
    if num_fire_hazards >= 2:
        factors.append("multiple_fire_sources")
    
    if num_smoke_hazards >= 2:
        factors.append("extensive_smoke")
    
    return factors


def format_urgency_for_llm(assessment: UrgencyAssessment) -> str:
    """
    Format urgency assessment as a natural language string for LLM prompts.
    """
    lines = [
        f"Urgency Level: {assessment.urgency_level.upper()} (score: {assessment.urgency_score:.2f})",
        f"Safety Assessment: {assessment.safety_assessment}",
    ]
    
    if assessment.fire_proximity < float('inf'):
        if assessment.fire_proximity <= 1:
            lines.append(f"CRITICAL: Fire is immediately adjacent ({assessment.fire_proximity:.0f} cell away)")
        elif assessment.fire_proximity <= 2:
            lines.append(f"URGENT: Fire is very close ({assessment.fire_proximity:.0f} cells away)")
        elif assessment.fire_proximity <= 4:
            lines.append(f"Fire is nearby ({assessment.fire_proximity:.0f} cells away)")
        else:
            lines.append(f"Fire is at a distance ({assessment.fire_proximity:.0f} cells away)")
    else:
        lines.append("No fire detected in immediate vicinity")
    
    if assessment.smoke_proximity < float('inf'):
        if assessment.smoke_proximity <= 1:
            lines.append(f"Smoke is immediately adjacent ({assessment.smoke_proximity:.0f} cell away)")
        elif assessment.smoke_proximity <= 2:
            lines.append(f"Smoke is very close ({assessment.smoke_proximity:.0f} cells away)")
        else:
            lines.append(f"Smoke is nearby ({assessment.smoke_proximity:.0f} cells away)")
    
    if assessment.visibility_impact != "normal":
        lines.append(f"Visibility is {assessment.visibility_impact.replace('_', ' ')}")
    
    if assessment.time_since_awareness is not None:
        lines.append(f"Time since becoming aware of danger: {assessment.time_since_awareness} steps")
    
    if assessment.primary_factors:
        factors_str = ", ".join(assessment.primary_factors).replace("_", " ")
        lines.append(f"Key factors: {factors_str}")
    
    return "\n".join(lines)

