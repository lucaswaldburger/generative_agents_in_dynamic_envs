"""
SmallVille environment using pygame + gymnasium, replacing the original
Phaser.js / Django frontend.  Renders the original Tiled tilemap with
all its visual layers and supports multiple agents.
"""
from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from .constants import Action, AgentConfig, DIR_TO_VEC, DEFAULT_MAX_STEPS
from .tilemap import TiledMapRenderer
from .world_object import SmallvilleAgent


class SmallvilleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        tilemap: TiledMapRenderer,
        agent_configs: List[AgentConfig],
        max_steps: int = DEFAULT_MAX_STEPS,
        render_mode: str | None = "human",
        camera_follow: bool = True,
        viewport_w: int = 1280,
        viewport_h: int = 800,
    ):
        super().__init__()
        self.tilemap = tilemap
        self.agent_configs = agent_configs
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.camera_follow = camera_follow
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h

        self.num_agents = len(agent_configs)
        self.agents: List[SmallvilleAgent] = []

        self.action_space = spaces.MultiDiscrete([len(Action)] * self.num_agents)
        self.observation_space = spaces.Dict({
            "agent_positions": spaces.Box(
                low=0,
                high=max(self.tilemap.map_width, self.tilemap.map_height),
                shape=(self.num_agents, 2),
                dtype=np.int32,
            ),
            "step": spaces.Discrete(self.max_steps + 1),
        })

        self.step_count: int = 0

        self._window: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None

        self._camera_x: float = 0.0
        self._camera_y: float = 0.0

        self._font: Optional[pygame.font.Font] = None

        self._agent_colors = [
            (255, 80, 80), (80, 80, 255), (80, 200, 120),
            (230, 230, 90), (200, 80, 200), (80, 200, 200),
            (255, 165, 0), (200, 200, 200),
        ]

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        self.agents = [SmallvilleAgent.from_config(cfg) for cfg in self.agent_configs]
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action {action}"
        self.step_count += 1

        action_to_heading = {
            Action.RIGHT: 0,
            Action.DOWN: 90,
            Action.LEFT: 180,
            Action.UP: 270,
        }

        for i, agent in enumerate(self.agents):
            act = Action(int(action[i]))
            dx, dy = DIR_TO_VEC[act]
            nx, ny = agent.x + dx, agent.y + dy

            if not self.tilemap.is_blocked(nx, ny):
                agent.x, agent.y = nx, ny
                if act in action_to_heading:
                    agent.heading_deg = action_to_heading[act]

        truncated = self.step_count >= self.max_steps
        return self._get_obs(), 0.0, False, truncated, {}

    def _get_obs(self) -> Dict[str, Any]:
        positions = np.array([[a.x, a.y] for a in self.agents], dtype=np.int32)
        return {"agent_positions": positions, "step": self.step_count}

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_human(return_array=True)
        return None

    def _init_pygame(self) -> None:
        if self._window is not None:
            return
        pygame.init()
        pygame.display.set_caption("SmallVille (pygame)")
        self._window = pygame.display.set_mode(
            (self.viewport_w, self.viewport_h)
        )
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 14, bold=True)

    def _render_human(self, return_array: bool = False):
        self._init_pygame()
        assert self._window is not None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._window = None
                return None

        self._handle_camera_keys()

        tw, th = self.tilemap.tile_width, self.tilemap.tile_height

        if self.camera_follow and self.agents:
            target = self.agents[0]
            target_px = target.x * tw + tw // 2
            target_py = target.y * th + th // 2
            self._camera_x += (target_px - self.viewport_w // 2 - self._camera_x) * 0.15
            self._camera_y += (target_py - self.viewport_h // 2 - self._camera_y) * 0.15

        cam_x = int(self._camera_x)
        cam_y = int(self._camera_y)

        self._window.fill((0, 0, 0))

        bg = self.tilemap.background
        self._window.blit(bg, (-cam_x, -cam_y))

        self._draw_agents(cam_x, cam_y)

        fg = self.tilemap.foreground
        self._window.blit(fg, (-cam_x, -cam_y))

        self._draw_hud()

        assert self._clock is not None
        self._clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

        if return_array:
            frame = pygame.surfarray.array3d(self._window)
            return np.transpose(frame, (1, 0, 2))
        return None

    def _handle_camera_keys(self) -> None:
        keys = pygame.key.get_pressed()
        speed = 16
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self._camera_x -= speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self._camera_x += speed
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self._camera_y -= speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self._camera_y += speed

        max_x = self.tilemap.pixel_width - self.viewport_w
        max_y = self.tilemap.pixel_height - self.viewport_h
        self._camera_x = max(0.0, min(self._camera_x, max_x))
        self._camera_y = max(0.0, min(self._camera_y, max_y))

    def _draw_agents(self, cam_x: int, cam_y: int) -> None:
        tw, th = self.tilemap.tile_width, self.tilemap.tile_height
        assert self._font is not None

        for idx, agent in enumerate(self.agents):
            px = agent.x * tw + tw // 2 - cam_x
            py = agent.y * th + th // 2 - cam_y

            if px < -tw or px > self.viewport_w + tw:
                continue
            if py < -th or py > self.viewport_h + th:
                continue

            color = agent.config.color if agent.config.color != (255, 80, 80) \
                else self._agent_colors[idx % len(self._agent_colors)]

            radius = tw // 2 - 2
            pygame.draw.circle(self._window, color, (px, py), radius)
            pygame.draw.circle(self._window, (255, 255, 255), (px, py), radius, 2)

            initials = "".join(
                w[0].upper() for w in agent.config.name.split() if w
            )[:2]
            txt_surf = self._font.render(initials, True, (255, 255, 255))
            txt_rect = txt_surf.get_rect(center=(px, py))
            self._window.blit(txt_surf, txt_rect)

            if agent.pronunciatio:
                bubble_text = f"{initials}: {agent.pronunciatio}"
                bubble_surf = self._font.render(bubble_text, True, (0, 0, 0))
                bw, bh = bubble_surf.get_size()
                padding = 4
                bubble_bg = pygame.Surface(
                    (bw + padding * 2, bh + padding * 2), pygame.SRCALPHA
                )
                bubble_bg.fill((255, 255, 255, 220))
                bx = px - (bw + padding * 2) // 2
                by = py - radius - bh - padding * 2 - 4
                self._window.blit(bubble_bg, (bx, by))
                self._window.blit(bubble_surf, (bx + padding, by + padding))

    def _draw_hud(self) -> None:
        assert self._font is not None
        step_text = f"Step: {self.step_count}"
        surf = self._font.render(step_text, True, (255, 255, 255))
        bg = pygame.Surface((surf.get_width() + 8, surf.get_height() + 4), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        self._window.blit(bg, (4, 4))
        self._window.blit(surf, (8, 6))

    def close(self) -> None:
        if self._window is not None:
            pygame.quit()
            self._window = None

    # --------------------------------------------------------------- pathfinding
    def astar(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Optional[List[Action]]:
        """A* over the collision grid, returning a list of Actions."""
        if self.tilemap.is_blocked(*goal):
            return None

        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        counter = 0
        heapq.heappush(open_set, (0.0, counter, start))
        came_from: Dict[Tuple[int, int], Tuple[Tuple[int, int], Action]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        moves = [
            (Action.UP, 0, -1),
            (Action.DOWN, 0, 1),
            (Action.LEFT, -1, 0),
            (Action.RIGHT, 1, 0),
        ]

        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current == goal:
                path: List[Action] = []
                node = goal
                while node in came_from:
                    prev, act = came_from[node]
                    path.append(act)
                    node = prev
                path.reverse()
                return path

            for act, dx, dy in moves:
                nb = (current[0] + dx, current[1] + dy)
                if self.tilemap.is_blocked(*nb):
                    continue
                tentative_g = g_score[current] + 1.0
                if tentative_g < g_score.get(nb, float("inf")):
                    g_score[nb] = tentative_g
                    f = tentative_g + heuristic(nb, goal)
                    came_from[nb] = (current, act)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, nb))

        return None
