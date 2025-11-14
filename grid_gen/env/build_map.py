from __future__ import annotations
from typing import Iterable, Dict, Any, Literal

import gymnasium as gym
import numpy as np



from .load_map import load_map   # your existing loader


DEFAULT_AGENT_COLORS = ["blue", "red", "green", "purple", "yellow", "orange", "grey"]


class MapMultiGridEnv(MultiGridEnv):
    """
    MultiGrid-based env that:
      - builds the grid from your JSON map
      - spawns N agents from personas
      - uses MultiGrid's standard multi-agent API
    """

    def __init__(
        self,
        json_path: str,
        personas: Iterable[Dict[str, Any]],
        max_steps: int = 200,
        agent_view_size: int = 7,
        render_mode: str | None = "human",
        allow_agent_overlap: bool = False,
        joint_reward: bool = False,
        success_termination_mode: Literal["any", "all"] = "any",
        failure_termination_mode: Literal["any", "all"] = "all",
    ):
        self.json_path = json_path
        self.personas = list(personas)

        # Load geometry from your map
        elements, (W, H) = load_map(self.json_path)
        self._elements = elements
        self._W, self._H = W, H

        # Mission text (shown at bottom when rendering)
        mission_space = MissionSpace.from_string(" human evacuation")

        # Initialize MultiGridEnv with N agents
        super().__init__(
            mission_space=mission_space,
            agents=len(self.personas),
            width=W,
            height=H,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=agent_view_size,
            allow_agent_overlap=allow_agent_overlap,
            joint_reward=joint_reward,
            success_termination_mode=success_termination_mode,
            failure_termination_mode=failure_termination_mode,
            render_mode=render_mode,
            agent_pov=False,  # full map view
        )

        # Attach persona metadata + colors to each Agent
        self._init_agents_from_personas()

    # ------------------------------------------------------------------
    # 1) Map personas → Agent state & color
    # ------------------------------------------------------------------
    def _init_agents_from_personas(self):
        """
        Map your personas (ids, start positions, headings, etc.)
        into MultiGrid's Agent objects and joint state.
        """
        for i, (persona, agent) in enumerate(zip(self.personas, self.agents)):
            # position from persona
            start = persona.get("start", {})
            x = int(start.get("x", 0))
            y = int(start.get("y", 0))

            # heading from degrees
            heading_deg = int(persona.get("heading_deg", persona.get("heading", 0)) or 0)
            # 0:right, 1:down, 2:left, 3:up → MultiGrid Direction enum
            dir_idx = int(((heading_deg % 360) + 45) // 90) % 4
            direction = Direction.from_index(dir_idx)

            # color (from persona or from default palette)
            color_name = persona.get(
                "color",
                DEFAULT_AGENT_COLORS[i % len(DEFAULT_AGENT_COLORS)],
            )
            agent.color = Color(color_name)

            # write into joint state
            self.agent_states.pos[i] = (x, y)
            self.agent_states.dir[i] = direction.to_index()
            # carrying is None by default, terminated False by default

    def _gen_grid(self, width: int, height: int):
        """
        Create the grid and place static objects from your map.
        Agent positions are already stored in self.agent_states.
        """
        assert width == self._W and height == self._H
        self.grid = Grid(width, height)

        for obj_type, rgba, color_name, sem_name, (x, y) in self._elements:

            s = sem_name.lower()

            # if s in ("street", "road", "sidewalk"):
            #     obj = Floor(color="grey")
            if s in ("block", "wall", "building"):
                obj = Wall(color="grey")
            elif s in ("park",):
                obj = Floor(color="green")
            elif s in ("fire", "lava"):
                obj = Lava()
            elif s in ("goal", "exit", "safe_zone"):
                obj = Goal(color="green")
            else:
                # default: walkable floor
                obj = Floor(color="blue")

            self.grid.set(x, y, obj)