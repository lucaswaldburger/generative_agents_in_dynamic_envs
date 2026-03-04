from pydantic import BaseModel, Field
from typing import Optional


class IntentDecision(BaseModel):
    """Structured output for high-level intent decisions."""

    intent: str = Field(description="Must be exactly 'ignore' or 'evacuate'")
    action: str = Field(
        description="'stay' if intent is ignore, 'go to <target_location>' if evacuate"
    )
    next_action: str = Field(
        description="Brief natural-language description of what the agent will do next"
    )
    target_location: Optional[str] = Field(
        default=None, description="Evacuation target location, or null if ignoring"
    )
    command: str = Field(
        description="Motion planner command: 'stay' or 'go to <target_location>'"
    )
    reason: str = Field(description="Clear explanation of why this decision was made")


class DirectionDecision(BaseModel):
    """Structured output for mid-level direction choices at intersections."""

    direction: str = Field(
        description="Exactly one of the valid directions (UP, DOWN, LEFT, RIGHT)"
    )
    reason: str = Field(description="Brief explanation of the chosen direction")


class SocialDecision(BaseModel):
    """Structured output for social interaction decisions."""

    talk: bool = Field(description="Whether to talk to the nearby friend")
    new_command: Optional[str] = Field(
        default=None, description="New high-level command string, or null to keep current"
    )
    reason: str = Field(description="1-2 sentence explanation")
