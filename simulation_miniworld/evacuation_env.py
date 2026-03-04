"""MiniWorld-based 3D building evacuation environment.

Creates a multi-room building using MiniWorld's OpenGL renderer, with:
- Multiple connected rooms (offices, hallways, stairwells)
- Fire and smoke hazards that spread over time
- An exit goal the agent must reach
- First-person 3D observations and a top-down overview
"""
from __future__ import annotations

import math
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from miniworld.entity import Box, Entity
from miniworld.miniworld import MiniWorldEnv


class FireEntity(Entity):
    """Visual marker for a fire cell (a glowing red/orange box)."""

    def __init__(self, size: float = 0.6):
        super().__init__()
        self.color_vec = np.array([1.0, 0.2, 0.0])
        self.size = np.array([size, size * 1.5, size])
        sx, sy, sz = self.size
        self.radius = max(sx, sz) / 2
        self.height = sy

    @property
    def is_static(self):
        return False

    def randomize(self, params, rng):
        pass

    def render(self):
        from pyglet.gl import glColor3f, GL_QUADS, glBegin, glEnd, glVertex3f, \
            glNormal3f, glPushMatrix, glPopMatrix, glTranslatef, glDisable, \
            glEnable, GL_TEXTURE_2D
        glPushMatrix()
        glTranslatef(*self.pos)
        glDisable(GL_TEXTURE_2D)
        glColor3f(*self.color_vec)
        sx, sy, sz = self.size
        hx, hy, hz = sx / 2, sy, sz / 2
        glBegin(GL_QUADS)
        for nx, ny, nz, verts in [
            (0, 0, -1, [(-hx, 0, -hz), (-hx, hy, -hz), (hx, hy, -hz), (hx, 0, -hz)]),
            (0, 0, 1, [(hx, 0, hz), (hx, hy, hz), (-hx, hy, hz), (-hx, 0, hz)]),
            (-1, 0, 0, [(-hx, 0, hz), (-hx, hy, hz), (-hx, hy, -hz), (-hx, 0, -hz)]),
            (1, 0, 0, [(hx, 0, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, 0, hz)]),
            (0, 1, 0, [(-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz)]),
        ]:
            glNormal3f(nx, ny, nz)
            for v in verts:
                glVertex3f(*v)
        glEnd()
        glEnable(GL_TEXTURE_2D)
        glPopMatrix()

    def step(self, delta_time):
        pass


class SmokeEntity(Entity):
    """Semi-transparent grey marker for smoke."""

    def __init__(self, size: float = 0.9):
        super().__init__()
        self.color_vec = np.array([0.55, 0.55, 0.55])
        self.size = np.array([size, size * 0.4, size])
        sx, sy, sz = self.size
        self.radius = max(sx, sz) / 2
        self.height = sy

    @property
    def is_static(self):
        return False

    def randomize(self, params, rng):
        pass

    def render(self):
        from pyglet.gl import glColor3f, GL_QUADS, glBegin, glEnd, glVertex3f, \
            glNormal3f, glPushMatrix, glPopMatrix, glTranslatef, glDisable, \
            glEnable, GL_TEXTURE_2D
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1] + 1.8, self.pos[2])
        glDisable(GL_TEXTURE_2D)
        glColor3f(*self.color_vec)
        sx, sy, sz = self.size
        hx, hy, hz = sx / 2, sy / 2, sz / 2
        glBegin(GL_QUADS)
        for nx, ny, nz, verts in [
            (0, 0, -1, [(-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz)]),
            (0, 0, 1, [(hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz), (-hx, -hy, hz)]),
            (-1, 0, 0, [(-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz), (-hx, -hy, -hz)]),
            (1, 0, 0, [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)]),
            (0, 1, 0, [(-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz)]),
            (0, -1, 0, [(-hx, -hy, hz), (-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz)]),
        ]:
            glNormal3f(nx, ny, nz)
            for v in verts:
                glVertex3f(*v)
        glEnd()
        glEnable(GL_TEXTURE_2D)
        glPopMatrix()

    def step(self, delta_time):
        pass


ROOM_NAMES = [
    "Office_A", "Office_B", "Office_C", "Office_D",
    "Hallway_Main", "Hallway_North", "Hallway_South",
    "Stairwell", "Lobby",
]


class EvacuationBuildingEnv(MiniWorldEnv):
    """A multi-room building with fire/smoke hazards.

    The building layout is a 3x3-ish grid of rooms connected by hallways,
    with a lobby at the ground level containing the exit.

    Parameters
    ----------
    num_rooms : int
        Number of office rooms (2-6).
    building_scale : float
        Multiplier for room sizes.
    fire_spread_rate : float
        Probability of fire spreading to adjacent rooms each step.
    max_episode_steps : int
        Maximum steps before truncation.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        num_rooms: int = 4,
        building_scale: float = 1.0,
        fire_spread_rate: float = 0.03,
        max_episode_steps: int = 500,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        self.num_rooms = min(max(num_rooms, 2), 6)
        self.building_scale = building_scale
        self.fire_spread_rate = fire_spread_rate

        self._room_objects = {}
        self._fire_positions = []
        self._smoke_positions = []
        self._fire_entities = []
        self._smoke_entities = []
        self._room_info = {}
        self.exit_box = None

        super().__init__(
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            obs_width=640,
            obs_height=480,
            window_width=1280,
            window_height=960,
            **kwargs,
        )
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        s = self.building_scale
        self._room_objects = {}
        self._fire_entities = []
        self._smoke_entities = []
        self._fire_positions = []
        self._smoke_positions = []

        lobby = self.add_rect_room(
            min_x=-3 * s, max_x=3 * s,
            min_z=-3 * s, max_z=3 * s,
            wall_tex="brick_wall",
            floor_tex="marble",
            ceil_tex="ceiling_tiles",
        )
        self._room_objects["Lobby"] = lobby
        self._room_info["Lobby"] = {"center": (0.0, 0.0), "has_fire": False, "has_smoke": False}

        hallway_main = self.add_rect_room(
            min_x=-1.5 * s, max_x=1.5 * s,
            min_z=3 * s, max_z=10 * s,
            wall_tex="drywall",
            floor_tex="floor_tiles_bw",
            ceil_tex="ceiling_tile_noborder",
        )
        self._room_objects["Hallway_Main"] = hallway_main
        self._room_info["Hallway_Main"] = {"center": (0.0, 6.5 * s), "has_fire": False, "has_smoke": False}
        self.connect_rooms(lobby, hallway_main, min_x=-1.5 * s, max_x=1.5 * s)

        office_specs = [
            ("Office_A", -7 * s, -1.5 * s,  3 * s,  7 * s, "concrete",    "wood_planks"),
            ("Office_B",  1.5 * s, 7 * s,   3 * s,  7 * s, "drywall",     "wood"),
            ("Office_C", -7 * s, -1.5 * s,  7 * s, 11 * s, "cinder_blocks", "floor_tiles_bw"),
            ("Office_D",  1.5 * s, 7 * s,   7 * s, 11 * s, "concrete",    "marble"),
            ("Office_E", -7 * s, -3 * s,   -3 * s,  0 * s, "brick_wall",  "wood"),
            ("Office_F",  3 * s,  7 * s,   -3 * s,  0 * s, "stucco",      "wood_planks"),
        ]

        for i in range(self.num_rooms):
            name, min_x, max_x, min_z, max_z, wall_tex, floor_tex = office_specs[i]
            room = self.add_rect_room(
                min_x=min_x, max_x=max_x,
                min_z=min_z, max_z=max_z,
                wall_tex=wall_tex,
                floor_tex=floor_tex,
                ceil_tex="ceiling_tile_noborder",
            )
            self._room_objects[name] = room
            cx = (min_x + max_x) / 2
            cz = (min_z + max_z) / 2
            self._room_info[name] = {"center": (cx, cz), "has_fire": False, "has_smoke": False}

            if min_z <= 3 * s and max_z >= 3 * s:
                if max_x <= 0:
                    self.connect_rooms(hallway_main, room, min_z=3.5 * s, max_z=5.5 * s)
                else:
                    self.connect_rooms(hallway_main, room, min_z=3.5 * s, max_z=5.5 * s)
            elif min_z >= 3 * s:
                if max_x <= 0:
                    self.connect_rooms(hallway_main, room, min_z=max(min_z, 7 * s), max_z=min(max_z, 9 * s))
                else:
                    self.connect_rooms(hallway_main, room, min_z=max(min_z, 7 * s), max_z=min(max_z, 9 * s))
            else:
                if max_x <= 0:
                    self.connect_rooms(lobby, room, min_z=-2 * s, max_z=0 * s)
                else:
                    self.connect_rooms(lobby, room, min_z=-2 * s, max_z=0 * s)

        self.exit_box = self.place_entity(
            Box(color="green", size=0.9),
            room=lobby,
            pos=np.array([0.0, 0.0, -2 * s]),
            dir=0,
        )

        # --- Single initial fire in the last office ---
        fire_room_name = list(self._room_objects.keys())[-1]
        fire_info = self._room_info[fire_room_name]
        cx, cz = fire_info["center"]
        fire_info["has_fire"] = True

        for dx, dz in [(0, 0), (0.6, 0.4)]:
            fx, fz = cx + dx, cz + dz
            if any(r.point_inside(np.array([fx, 0, fz])) for r in self.rooms):
                self._spawn_fire(fx, fz)

        # --- Initial smoke around the fire ---
        for fx, fz in list(self._fire_positions):
            for sdx, sdz in [(1.2, 0), (-1.2, 0), (0, 1.2), (0, -1.2)]:
                sx, sz = fx + sdx, fz + sdz
                if any(r.point_inside(np.array([sx, 0, sz])) for r in self.rooms):
                    self._spawn_smoke(sx, sz)

        self.place_agent(room=hallway_main)

    def _spawn_fire(self, x: float, z: float):
        """Place a fire entity at (x, z)."""
        pos = np.array([x, 0.0, z])
        fe = FireEntity(size=0.6)
        fe.pos = pos
        fe.dir = 0
        self.entities.append(fe)
        self._fire_entities.append(fe)
        self._fire_positions.append((x, z))

    def _spawn_smoke(self, x: float, z: float):
        """Place a smoke entity at (x, z)."""
        pos = np.array([x, 0.0, z])
        se = SmokeEntity(size=0.9)
        se.pos = pos
        se.dir = 0
        self.entities.append(se)
        self._smoke_entities.append(se)
        self._smoke_positions.append((x, z))

    def _spread_fire(self):
        """Spread fire in 8 directions (like the 2D grid sim) and generate smoke."""
        new_fires = []
        new_smokes = []
        rng = self.np_random
        spacing = 1.0

        spread_dirs = [
            (-spacing, -spacing), (-spacing, 0), (-spacing, spacing),
            (0, -spacing),                        (0, spacing),
            (spacing, -spacing),  (spacing, 0),   (spacing, spacing),
        ]

        for fx, fz in list(self._fire_positions):
            for ddx, ddz in spread_dirs:
                nx, nz = fx + ddx, fz + ddz

                if not any(r.point_inside(np.array([nx, 0, nz])) for r in self.rooms):
                    continue

                # Fire spread: probabilistic
                already_fire = any(
                    abs(nx - ex) < 0.5 and abs(nz - ez) < 0.5
                    for ex, ez in self._fire_positions
                )
                if not already_fire and rng.random() < self.fire_spread_rate:
                    new_fires.append((nx, nz))

                # Smoke: always generated on non-fire adjacent tiles
                already_smoke = any(
                    abs(nx - sx) < 0.7 and abs(nz - sz) < 0.7
                    for sx, sz in self._smoke_positions
                )
                if not already_fire and not already_smoke:
                    new_smokes.append((nx, nz))

        # Smoke also drifts outward from existing smoke
        for sx, sz in list(self._smoke_positions):
            if rng.random() > 0.05:
                continue
            drift_x = sx + rng.uniform(-1.5, 1.5)
            drift_z = sz + rng.uniform(-1.5, 1.5)
            if any(r.point_inside(np.array([drift_x, 0, drift_z])) for r in self.rooms):
                already = any(
                    abs(drift_x - ex) < 0.7 and abs(drift_z - ez) < 0.7
                    for ex, ez in self._smoke_positions
                )
                if not already:
                    new_smokes.append((drift_x, drift_z))

        for x, z in new_fires:
            self._spawn_fire(x, z)
        for x, z in new_smokes:
            self._spawn_smoke(x, z)

        for name, info in self._room_info.items():
            room = self._room_objects[name]
            info["has_fire"] = any(
                room.point_inside(np.array([fx, 0, fz]))
                for fx, fz in self._fire_positions
            )
            info["has_smoke"] = info["has_fire"] or any(
                room.point_inside(np.array([sx, 0, sz]))
                for sx, sz in self._smoke_positions
            )

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        self._spread_fire()

        agent_pos = self.agent.pos
        for fx, fz in self._fire_positions:
            dist = math.sqrt((agent_pos[0] - fx) ** 2 + (agent_pos[2] - fz) ** 2)
            if dist < 1.2:
                reward -= 1.0
                termination = True
                info["cause"] = "fire"
                return obs, reward, termination, truncation, info

        if self.near(self.exit_box):
            reward += self._reward()
            termination = True
            info["cause"] = "exit_reached"

        info["fire_count"] = len(self._fire_positions)
        info["smoke_count"] = len(self._smoke_positions)
        info["agent_pos"] = tuple(self.agent.pos)
        info["agent_dir"] = self.agent.dir

        return obs, reward, termination, truncation, info

    def get_room_for_pos(self, pos: np.ndarray) -> Optional[str]:
        """Return the room name containing the given position."""
        for name, room in self._room_objects.items():
            if room.point_inside(pos):
                return name
        return None

    def get_room_centers(self) -> dict:
        """Return dict of room_name -> (cx, cz) center coordinates."""
        return {name: info["center"] for name, info in self._room_info.items()}

    def get_hazard_rooms(self) -> Tuple[list, list]:
        """Return lists of room names with fire and smoke."""
        fire_rooms = [n for n, i in self._room_info.items() if i["has_fire"]]
        smoke_rooms = [n for n, i in self._room_info.items() if i["has_smoke"]]
        return fire_rooms, smoke_rooms

    def get_exit_pos(self) -> np.ndarray:
        """Return the exit box position."""
        return self.exit_box.pos.copy()
