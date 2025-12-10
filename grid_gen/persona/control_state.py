# persona/control_state.py

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
from env.constants import Action


class PlannerLevel(Enum):
    """
    What kind of decision this agent needs next, logically.
    - HIGH is triggered externally (broadcasts, fire) from main.
    - MID is for local choice (e.g. at an intersection).
    - LOW is just executing a short A* segment.
    """
    HIGH = auto()
    MID = auto()
    LOW = auto()


@dataclass
class AgentControlState:
    # High-level “what am I doing?”
    high_intent: Optional[str] = None          # e.g. "evacuate", "stay", "routine"
    high_command: str = "stay"                 # normalized planner cmd: "go to Home_A" / "stay"

    # Mid-level “what direction did I pick?”
    last_mid_direction: Optional[str] = None   # "UP", "DOWN", "LEFT", "RIGHT"

    # Low-level segment: short A* chunk (usually <= FOV)
    segment_actions: List[Action] = field(default_factory=list)

    # What planner level do I logically need next?
    # HIGH is only set from outside when something big happens.
    next_level: PlannerLevel = PlannerLevel.LOW

    def clear_segment(self) -> None:
        self.segment_actions.clear()

    def has_segment(self) -> bool:
        return len(self.segment_actions) > 0

    def pop_next_action(self) -> Action:
        """
        Pops the next low-level action from the segment.
        If empty, returns STAY (caller should treat as 'no segment').
        """
        if not self.segment_actions:
            return Action.STAY
        return self.segment_actions.pop(0)

    def set_new_high_command(self, intent: str, cmd: str) -> None:
        """Convenience for updating high-level decision."""
        self.high_intent = intent
        self.high_command = cmd
        self.clear_segment()
        # After a new high-level command, we usually go to LOW,
        # but MID may be requested when we hit an intersection.
        self.next_level = PlannerLevel.LOW
