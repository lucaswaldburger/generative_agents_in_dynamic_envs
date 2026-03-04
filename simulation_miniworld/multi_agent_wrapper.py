"""Multi-agent wrapper around the MiniWorld evacuation environment.

MiniWorld is inherently single-agent (one camera, one physics body). This
wrapper manages N logical agents by swapping the internal agent position/dir
before rendering or collision checks for each agent, while sharing the same
world geometry, fire/smoke dynamics, and entity list.

Each logical agent gets:
- Its own 3D position (x, y=0, z) and direction (radians)
- A first-person RGB observation from its viewpoint
- A top-down overview (shared, rendered once per step)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from evacuation_env import EvacuationBuildingEnv


@dataclass
class AgentState:
    """Per-agent state tracked by the multi-agent wrapper."""
    agent_id: int
    name: str
    pos: np.ndarray
    direction: float
    color: str = "blue"
    config: object = None
    reached_exit: bool = False
    dead: bool = False

    known_hazard_positions: list = field(default_factory=list)
    current_urgency_assessment: object = None
    social_ignored_friends: set = field(default_factory=set)


class MultiAgentMiniWorldEnv(gym.Env):
    """Manages multiple agents inside one :class:`EvacuationBuildingEnv`.

    Parameters
    ----------
    agent_configs : list
        Persona config objects (from ``load_agent_configs``).
    num_rooms, building_scale, fire_spread_rate, max_steps : various
        Forwarded to :class:`EvacuationBuildingEnv`.
    render_mode : str or None
        ``"rgb_array"`` for headless frame capture.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(
        self,
        agent_configs: list,
        num_rooms: int = 4,
        building_scale: float = 1.0,
        fire_spread_rate: float = 0.03,
        max_steps: int = 500,
        render_mode: Optional[str] = "rgb_array",
    ):
        super().__init__()
        self.agent_configs = agent_configs
        self.num_agents = len(agent_configs)
        self.max_steps = max_steps
        self._step_count = 0

        self._inner = EvacuationBuildingEnv(
            num_rooms=num_rooms,
            building_scale=building_scale,
            fire_spread_rate=fire_spread_rate,
            max_episode_steps=max_steps,
            render_mode=render_mode,
        )

        self.action_space = spaces.MultiDiscrete([4] * self.num_agents)
        single_obs = self._inner.observation_space
        self.observation_space = spaces.Dict({
            f"agent_{i}": single_obs for i in range(self.num_agents)
        })
        self.render_mode = render_mode
        self.agents: list[AgentState] = []

    @property
    def rooms(self):
        return self._inner.rooms

    @property
    def room_info(self):
        return self._inner._room_info

    @property
    def fire_positions(self):
        return self._inner._fire_positions

    @property
    def smoke_positions(self):
        return self._inner._smoke_positions

    def get_room_for_pos(self, pos):
        return self._inner.get_room_for_pos(pos)

    def get_room_centers(self):
        return self._inner.get_room_centers()

    def get_hazard_rooms(self):
        return self._inner.get_hazard_rooms()

    def get_exit_pos(self):
        return self._inner.get_exit_pos()

    def reset(self, seed=None, options=None):
        obs_inner, info = self._inner.reset(seed=seed, options=options)
        self._step_count = 0

        self.agents = []
        s = self._inner.building_scale
        rng = self._inner.np_random

        # Spawn each agent in a different room, away from fire and exit
        fire_rooms, _ = self._inner.get_hazard_rooms()
        exit_pos = self._inner.exit_box.pos
        room_names = [
            n for n in self._inner._room_objects
            if n not in fire_rooms and n != "Lobby"
        ]
        if not room_names:
            room_names = [
                n for n in self._inner._room_objects if n not in fire_rooms
            ]
        if not room_names:
            room_names = list(self._inner._room_objects.keys())

        spawn_rooms = _pick_spread_rooms(room_names, self.num_agents)
        centers = self._inner.get_room_centers()

        for i, cfg in enumerate(self.agent_configs):
            room_name = spawn_rooms[i]
            cx, cz = centers[room_name]
            offset_x = rng.uniform(-0.8, 0.8)
            offset_z = rng.uniform(-0.8, 0.8)
            pos = np.array([cx + offset_x, 0.0, cz + offset_z])
            direction = rng.uniform(-math.pi, math.pi)

            color = getattr(cfg, "color", "blue")
            name = getattr(cfg, "name", f"agent_{i}")

            agent_state = AgentState(
                agent_id=i,
                name=name,
                pos=pos,
                direction=direction,
                color=color,
                config=cfg,
            )
            self.agents.append(agent_state)
            print(f"  [{name}] spawned in {room_name} at ({pos[0]:.1f}, {pos[2]:.1f})")

        multi_obs = self._get_all_observations()
        return multi_obs, info

    def step(self, actions):
        """Execute one step for all agents.

        Parameters
        ----------
        actions : array-like of int
            One action per agent: 0=turn_left, 1=turn_right, 2=move_forward, 3=move_back.
        """
        self._step_count += 1
        rewards = np.zeros(self.num_agents)
        any_terminated = False
        any_truncated = self._step_count >= self.max_steps

        fwd_step = self._inner.params.sample(None, "forward_step")
        turn_step = math.radians(self._inner.params.sample(None, "turn_step"))

        for aid, a in enumerate(self.agents):
            if a.reached_exit or a.dead:
                continue

            action = int(actions[aid])
            if action == 0:  # turn_left
                a.direction += turn_step
            elif action == 1:  # turn_right
                a.direction -= turn_step
            elif action == 2:  # move_forward
                dx = fwd_step * math.cos(a.direction)
                dz = -fwd_step * math.sin(a.direction)
                new_pos = a.pos + np.array([dx, 0, dz])
                if not self._collides(new_pos):
                    a.pos = new_pos
            elif action == 3:  # move_back
                dx = -fwd_step * math.cos(a.direction)
                dz = fwd_step * math.sin(a.direction)
                new_pos = a.pos + np.array([dx, 0, dz])
                if not self._collides(new_pos):
                    a.pos = new_pos

            for fx, fz in self._inner._fire_positions:
                dist = math.sqrt((a.pos[0] - fx) ** 2 + (a.pos[2] - fz) ** 2)
                if dist < 1.2:
                    a.dead = True
                    rewards[aid] = -1.0
                    break

            exit_pos = self._inner.exit_box.pos
            dist_exit = np.linalg.norm(a.pos - exit_pos)
            if dist_exit < 1.5:
                a.reached_exit = True
                rewards[aid] = 1.0

        self._inner._spread_fire()

        all_done = all(a.reached_exit or a.dead for a in self.agents)
        terminated = False
        truncated = any_truncated

        multi_obs = self._get_all_observations()

        info = {
            "step": self._step_count,
            "fire_count": len(self._inner._fire_positions),
            "smoke_count": len(self._inner._smoke_positions),
            "agents_alive": sum(1 for a in self.agents if not a.dead),
            "agents_exited": sum(1 for a in self.agents if a.reached_exit),
        }

        total_reward = float(rewards.sum())
        return multi_obs, total_reward, terminated, truncated, info

    def _collides(self, pos: np.ndarray, radius: float = 0.4) -> bool:
        """Check collision with walls."""
        from miniworld.math import intersect_circle_segs
        if len(self._inner.wall_segs) == 0:
            return False
        px, _, pz = pos
        test_pos = np.array([px, 0, pz])
        return intersect_circle_segs(test_pos, radius, self._inner.wall_segs)

    def _get_all_observations(self) -> dict:
        """Render first-person views for each alive agent."""
        obs = {}
        for aid, a in enumerate(self.agents):
            if a.reached_exit or a.dead:
                obs[f"agent_{aid}"] = np.zeros(
                    self._inner.observation_space.shape, dtype=np.uint8
                )
                continue
            self._inner.agent.pos = a.pos.copy()
            self._inner.agent.dir = a.direction
            try:
                agent_obs = self._inner.render_obs()
                obs[f"agent_{aid}"] = agent_obs
            except Exception:
                obs[f"agent_{aid}"] = np.zeros(
                    self._inner.observation_space.shape, dtype=np.uint8
                )
        return obs

    def render(self) -> np.ndarray:
        """Render a top-down overview with all agents drawn as colored markers."""
        try:
            img, scale = self._inner.render_top_view(render_agent=False, return_scale=True)
        except Exception:
            return np.zeros((960, 1280, 3), dtype=np.uint8)

        img = img.copy()
        x_scale = scale["x_scale"]
        z_scale = scale["z_scale"]
        x_offset = scale["x_offset"]
        z_offset = scale["z_offset"]

        agent_colors = {
            "red": (255, 60, 30),
            "blue": (30, 100, 255),
            "green": (30, 200, 30),
            "yellow": (255, 220, 30),
            "purple": (160, 50, 200),
        }

        h, w = img.shape[:2]
        for a in self.agents:
            px = int(a.pos[0] * x_scale + x_offset)
            pz = int(a.pos[2] * z_scale + z_offset)
            color = agent_colors.get(a.color, (30, 100, 255))
            r = max(6, int(min(x_scale, z_scale) * 0.4))

            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        iy, ix = pz + dy, px + dx
                        if 0 <= iy < h and 0 <= ix < w:
                            img[iy, ix] = color

            # Direction indicator
            dir_len = r + 4
            ddx = int(dir_len * math.cos(a.direction))
            ddz = int(-dir_len * math.sin(a.direction))
            for i in range(dir_len):
                iy = pz + int(ddz * i / dir_len)
                ix = px + int(ddx * i / dir_len)
                for s in range(-1, 2):
                    iy2 = iy + s
                    ix2 = ix + s
                    if 0 <= iy2 < h and 0 <= ix2 < w:
                        img[iy2, ix2] = (255, 255, 0)

            # Label
            if not a.reached_exit and not a.dead:
                label_y = max(0, pz - r - 3)
                for lx in range(max(0, px - r), min(w, px + r + 1)):
                    if 0 <= label_y < h:
                        img[label_y, lx] = color
                    if 0 <= label_y + 1 < h:
                        img[label_y + 1, lx] = color

        return img

    def render_agent_view(self, agent_id: int) -> np.ndarray:
        """Render first-person view for a specific agent."""
        a = self.agents[agent_id]
        self._inner.agent.pos = a.pos.copy()
        self._inner.agent.dir = a.direction
        return self._inner.render_obs()

    def close(self):
        self._inner.close()


def _pick_spread_rooms(room_names: list, n: int) -> list:
    """Pick n rooms, spreading agents as far apart as possible."""
    if n >= len(room_names):
        result = list(room_names)
        while len(result) < n:
            result.append(room_names[len(result) % len(room_names)])
        return result
    step = max(1, len(room_names) // n)
    return [room_names[(i * step) % len(room_names)] for i in range(n)]
