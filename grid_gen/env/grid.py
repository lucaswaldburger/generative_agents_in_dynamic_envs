# env/grid.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
from .constants import Action, DIR_TO_VEC, DEFAULT_MAX_STEPS, AgentConfig
from .load_map import MapSpec
from .world_object import HumanAgent

class MultiHumanGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 8}

    def __init__(
        self,
        map_spec: MapSpec,
        agent_configs: list[AgentConfig],
        max_steps: int = DEFAULT_MAX_STEPS,
        render_mode: str | None = "human",
        fire_spread_rate: float = 0.05,
        traffic_disappear_mode: bool = True,
        traffic_disappear_rate: float = 0.02,
    ):
        super().__init__()
        self.map_spec = map_spec
        self.agent_configs = agent_configs
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.num_agents = len(agent_configs)
        self.agents: list[HumanAgent] = []

        # joint action space...
        self.action_space = spaces.MultiDiscrete([len(Action)] * self.num_agents)
        self.observation_space = spaces.Dict(
            {
                "agent_positions": spaces.Box(
                    low=0,
                    high=max(self.map_spec.width, self.map_spec.height),
                    shape=(self.num_agents, 2),
                    dtype=np.int32,
                ),
                "fire_grid": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.map_spec.height, self.map_spec.width),
                    dtype=np.int8,
                ),
                # New: Grid for observed smoke
                "smoke_grid": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.map_spec.height, self.map_spec.width),
                    dtype=np.int8,
                ),
                # New: Grid for observed traffic
                "traffic_grid": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.map_spec.height, self.map_spec.width),
                    dtype=np.int8,
                ),
                "step": spaces.Discrete(self.max_steps + 1),
            }
        )

        self.step_count: int = 0

        self._can_enter = {
            int(k): bool(v)
            for k, v in self.map_spec.semantics.get("can_enter", {}).items()
        }
     
        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.cell_size: int = 40  # pixels per grid cell, this can change how big the window is
    
        self.traffic_locations : set[Tuple[int,int]] = set()
        self.max_traffic_locations: int = 20
        self.traffic_disappear_mode = traffic_disappear_mode
        self.traffic_disappear_rate = traffic_disappear_rate 

        for r in map_spec.regions:
            if r['type'] == 'fire':
                fire_start_loc = (r['x'],r['y'])
        self.fire_start_loc = fire_start_loc
        self.fire_locations : set[Tuple[int,int]] = set()
        self.smoke_locations : set[Tuple[int,int]] = set()
        self.fire_spread_rate = fire_spread_rate
        self.fire_start_time: int | None = None  # Track when fire first appears

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0

        self.agents = [HumanAgent.from_config(cfg) for cfg in self.agent_configs]
        self.fire_locations.clear()
        if 0 <= self.fire_start_loc[0] < self.map_spec.width and 0 <= self.fire_start_loc[1] < self.map_spec.height:
             self.fire_locations.add(self.fire_start_loc)
             self.fire_start_time = 0  # Fire starts at step 0
        self.traffic_locations.clear()
        self.smoke_locations.clear()
        self._spawn_new_traffic()
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        self.step_count += 1
        ACTION_TO_HEADING = {
            Action.RIGHT: 0,     # +x
            Action.DOWN: 90,     # +y
            Action.LEFT: 180,    # -x
            Action.UP: 270,      # -y
        }
        # Move each agent
        for i, agent in enumerate(self.agents):
            act = Action(int(action[i]))
            dx, dy = DIR_TO_VEC[act]
            nx, ny = agent.x + dx, agent.y + dy

            if self._can_move_to(nx, ny):
                agent.x, agent.y = nx, ny
                if act in ACTION_TO_HEADING:
                    agent.heading_deg = ACTION_TO_HEADING[act]

        self._update_fire()
        self._update_traffic()

        reward = 0.0
        for agent in self.agents:
            if (agent.x, agent.y) in self.fire_locations:
                # Example: Immediate termination and penalty for being on fire
                terminated = True 
                reward = -10.0 # Huge penalty
                break # Stop checking other agents if one is on fire
        else:
             terminated = False # Only set terminated if an agent is on fire
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            frame = self._render_human(return_array=True)
            return frame
        elif self.render_mode == "ansi":
            return self._render_ansi()
        else:
            return None
        
    def _init_pygame(self):
        if self.window is not None:
            return
        pygame.init()
        w = self.map_spec.width * self.cell_size
        h = self.map_spec.height * self.cell_size
        self.window = pygame.display.set_mode((w, h))
        pygame.display.set_caption("MultiHumanGridEnv")
        self.clock = pygame.time.Clock()

    def _generate_grid(self):
        W, H = self.map_spec.width, self.map_spec.height

        # Background
        self.window.fill((20, 20, 20))

        # Draw base grid (streets)
        for y in range(H):
            for x in range(W):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.window, (50, 50, 50), rect)

        # Draw regions (blocks, park, home, fire, work)
        palette = self.map_spec.raw.get("palette", {})
        for region in self.map_spec.regions:
            x, y = int(region["x"]), int(region["y"])
            w, h = int(region["w"]), int(region["h"])
            rtype = region["type"]

            color_list = palette.get(rtype, [100, 100, 100])
            color = tuple(color_list[:3])  # ignore alpha if present

            rect = pygame.Rect(
                x * self.cell_size,
                y * self.cell_size,
                w * self.cell_size,
                h * self.cell_size,
            )
            pygame.draw.rect(self.window, color, rect)

        # Draw grid lines
        for x in range(W + 1):
            pygame.draw.line(
                self.window,
                (30, 30, 30),
                (x * self.cell_size, 0),
                (x * self.cell_size, H * self.cell_size),
                1,
            )
        for y in range(H + 1):
            pygame.draw.line(
                self.window,
                (30, 30, 30),
                (0, y * self.cell_size),
                (W * self.cell_size, y * self.cell_size),
                1,
            )

    def _generate_agents(self):
        W, H = self.map_spec.width, self.map_spec.height
        # ---------- FOV overlay (semi-transparent, agent-colored) ----------
        fov_surface = pygame.Surface(self.window.get_size(), pygame.SRCALPHA)

        for agent in self.agents:
            fov_cfg = agent.config.fov
            base_rng = int(fov_cfg.range_cells)
            # Use dynamic FOV range that accounts for fire and smoke proximity
            rng = self._get_dynamic_fov_range(agent.x, agent.y, base_rng)
            angle_deg = float(fov_cfg.angle_deg)

            # Heading convention: 0° = right, 90° = down (screen coordinates)
            heading_rad = math.radians(agent.heading_deg)

            # Base direction unit vector
            hx = math.cos(heading_rad)
            hy = math.sin(heading_rad)

            # Color: same as agent, but transparent
            base_r, base_g, base_b = self._agent_rgb(agent.config.color)
            fov_color = (base_r, base_g, base_b, 70)  # last = alpha (0–255)

            ax, ay = agent.x, agent.y

            # Check cells in a square around the agent
            for gy in range(max(0, ay - rng), min(H, ay + rng + 1)):
                for gx in range(max(0, ax - rng), min(W, ax + rng + 1)):
                    dx = gx - ax
                    dy = gy - ay

                    # distance in cells
                    dist = math.hypot(dx, dy)
                    if dist == 0 or dist > rng:
                        continue

                    # direction to this cell
                    vx = dx / dist
                    vy = dy / dist

                    # angle between heading and cell vector
                    dot = max(min(hx * vx + hy * vy, 1.0), -1.0)
                    cell_angle = math.degrees(math.acos(dot))

                    if cell_angle <= angle_deg / 2.0:
                        # inside FOV cone: shade this cell
                        rect = pygame.Rect(
                            gx * self.cell_size,
                            gy * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        )
                        pygame.draw.rect(fov_surface, fov_color, rect)

        self.window.blit(fov_surface, (0, 0))

        for _, agent in enumerate(self.agents):
            cx = agent.x * self.cell_size + self.cell_size // 2
            cy = agent.y * self.cell_size + self.cell_size // 2

            agent_color = self._agent_rgb(agent.config.color)

            pygame.draw.circle(
                self.window,
                agent_color,
                (cx, cy),
                self.cell_size // 3,
            )

    def _render_human(self, return_array: bool = False):
        self._init_pygame()
        assert self.window is not None

        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.window = None
                return None

        self._generate_grid()

        fire_color = (255, 69, 0)
        for fx, fy in self.fire_locations:
             rect = pygame.Rect(
                fx * self.cell_size,
                fy * self.cell_size,
                self.cell_size,
                self.cell_size,
             )
             pygame.draw.rect(self.window, fire_color, rect)
    
    
        smoke_color = (100, 100, 100) # R, G, B, Alpha (100 is semi-transparent)
        for sx, sy in self.smoke_locations:
             rect = pygame.Rect(
                sx * self.cell_size,
                sy * self.cell_size,
                self.cell_size,
                self.cell_size,
             )
             pygame.draw.rect(self.window, smoke_color, rect)

        traffic_color = (150, 150, 150) # Gray
        for tx, ty in self.traffic_locations:
            rect = pygame.Rect(
                tx * self.cell_size,
                ty * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            pygame.draw.rect(self.window, traffic_color, rect)


        self._generate_agents()

        assert self.clock is not None
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

        if return_array:
            frame = pygame.surfarray.array3d(self.window)
            frame = np.transpose(frame, (1, 0, 2))
            return frame
        else:
            return None

    def _agent_rgb(self, color_name: str | None) -> tuple[int, int, int]:
        name = (color_name or "white").lower()
        return {
            "red": (255, 80, 80),
            "blue": (80, 80, 255),
            "green": (80, 200, 120),
            "yellow": (230, 230, 90),
            "white": (240, 240, 240)
        }.get(name, (240, 240, 240))

    def _render_ansi(self) -> str:
        W, H = self.map_spec.width, self.map_spec.height
        char_grid = np.full((H, W), ".", dtype="<U1")

        for region in self.map_spec.regions:
            x, y = int(region["x"]), int(region["y"])
            w, h = int(region["w"]), int(region["h"])
            tile_char = self._region_char(region["type"])
            char_grid[y : y + h, x : x + w] = tile_char

        for fx, fy in self.fire_locations:
                    if 0 <= fx < W and 0 <= fy < H:
                        char_grid[fy, fx] = "F"

        for idx, agent in enumerate(self.agents):
            if 0 <= agent.x < W and 0 <= agent.y < H:
                char_grid[agent.y, agent.x] = str(idx + 1)

        lines = ["".join(row) for row in char_grid]
        txt = "\n".join(lines)
        print(txt)
        return txt

    def close(self):
        pass

    def _can_move_to(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.map_spec.width or y < 0 or y >= self.map_spec.height:
            return False
        if (x, y) in self.traffic_locations:
            return False
        if (x, y) in self.fire_locations:
            return False
        if (x, y) in self.smoke_locations:
            return False
        code = int(self.map_spec.access_grid[y, x])
        return self._can_enter.get(code, False)

    def _region_char(self, region_type: str) -> str:
        return {
            "block": "#",
            "park": "P",
            "home": "H",
            "fire": "F",
            "work": "W",
        }.get(region_type, "?")

    def _get_obs(self) -> Dict[str, Any]:
        W, H = self.map_spec.width, self.map_spec.height
        
        # 1. Agent Positions
        positions = np.array([[a.x, a.y] for a in self.agents], dtype=np.int32)
        
        # 2. Create Global Grids for Dynamic Obstacles
    
        global_fire_grid = np.zeros((H, W), dtype=np.int8)
        for x, y in self.fire_locations:
            if 0 <= y < H and 0 <= x < W:
                global_fire_grid[y, x] = 1

        global_smoke_grid = np.zeros((H, W), dtype=np.int8)
        for x, y in self.smoke_locations:
            if 0 <= y < H and 0 <= x < W:
                global_smoke_grid[y, x] = 1

        global_traffic_grid = np.zeros((H, W), dtype=np.int8)
        for x, y in self.traffic_locations:
            if 0 <= y < H and 0 <= x < W:
                global_traffic_grid[y, x] = 1


        # 3. Calculate Combined Visibility Mask (Union of all Agents' FOV)
        visible_mask = np.zeros((H, W), dtype=np.int8)
        
        for agent in self.agents:
            fov_cfg = agent.config.fov
            # MODIFIED: Use the dynamic range calculated here
            base_rng = int(fov_cfg.range_cells)
            rng = self._get_dynamic_fov_range(agent.x, agent.y, base_rng) 
            
            angle_deg = float(fov_cfg.angle_deg)
            
            # Heading calculation for FOV cone
            # ... (rest of the FOV cone logic remains the same, using the new 'rng' variable)
            
            heading_rad = math.radians(agent.heading_deg)
            hx, hy = math.cos(heading_rad), math.sin(heading_rad)
            ax, ay = agent.x, agent.y
            
            # Always make the agent's current cell visible
            if 0 <= ay < H and 0 <= ax < W:
                visible_mask[ay, ax] = 1

            # Check cells in a square around the agent
            for gy in range(max(0, ay - rng), min(H, ay + rng + 1)):
                for gx in range(max(0, ax - rng), min(W, ax + rng + 1)):
                    dx = gx - ax
                    dy = gy - ay
                    dist = math.hypot(dx, dy)
                    
                    # Check range (uses the dynamic 'rng')
                    if dist <= 0 or dist > rng:
                        continue
                        
                    # Check angle for FOV cone
                    vx = dx / dist
                    vy = dy / dist
                    dot = max(min(hx * vx + hy * vy, 1.0), -1.0)
                    cell_angle = math.degrees(math.acos(dot))

                    if cell_angle <= angle_deg / 2.0:
                        visible_mask[gy, gx] = 1 # Mark cell as visible to at least one agen

        # 4. Apply Mask to Global Grids
        
        # Observed Grid = Global Grid * Visibility Mask
        observed_fire_grid = global_fire_grid * visible_mask
        observed_smoke_grid = global_smoke_grid * visible_mask
        observed_traffic_grid = global_traffic_grid * visible_mask

        return {
            "agent_positions": positions,
            "fire_grid": observed_fire_grid,
            "smoke_grid": observed_smoke_grid,
            "traffic_grid": observed_traffic_grid,
            "step": self.step_count,
        }

    def _update_fire(self):
        """Logic for fire to spread and for smoke generation. Fire can spread to all tiles."""
        new_fire_locations = self.fire_locations.copy()
        new_smoke_locations = set()
        
        # Directions for spreading/smoke generation (8-connectivity)
        spread_directions = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]
        
        W, H = self.map_spec.width, self.map_spec.height
        
        for fx, fy in self.fire_locations:
            for dx, dy in spread_directions:
                nx, ny = fx + dx, fy + dy
                pos = (nx, ny)
                
                # Check bounds
                if not (0 <= nx < W and 0 <= ny < H):
                    continue

                # Check if the potential cell is valid for agent movement (for smoke generation)
                # We use the helper _can_move_to_dynamic_check, which checks static rules and traffic.
                is_movable_tile = self._can_move_to_dynamic_check(nx, ny)
                
                # --- 1. Fire Spreading Logic ---
                # Fire can spread to any tile (including buildings/blocks) that is not already on fire.
                if pos not in self.fire_locations:
                    if self.np_random.random() < self.fire_spread_rate:
                        new_fire_locations.add(pos)
                
                # --- 2. Smoke Generation Logic ---
                # Smoke appears on a movable tile (street/park, etc.) adjacent to fire,
                # provided it is not the actual fire cell itself.
                if is_movable_tile and pos not in self.fire_locations:
                    new_smoke_locations.add(pos)
        
        self.fire_locations = new_fire_locations
        self.smoke_locations = new_smoke_locations

    def _can_move_to_dynamic_check(self, x: int, y: int) -> bool:
        """Checks static boundaries and map access rules, ignoring dynamic obstacles."""
        if x < 0 or x >= self.map_spec.width or y < 0 or y >= self.map_spec.height:
            return False
        
        # Check 2: Cannot move to a traffic cell (traffic is an immediate obstacle)
        if (x, y) in self.traffic_locations:
            return False
        
        # Check 3: Check static map access
        code = int(self.map_spec.access_grid[y, x])
        return self._can_enter.get(code, False)
    
    def _calculate_min_fire_distance(self):
            if not self.fire_locations:
                return float('inf')
            
            min_dist = float('inf')

            min_dists = []
            for agent in self.agents:
                ax, ay = self.agent_pos
                
                for fx, fy in self.fire_locations:
                    dist = abs(ax - fx) + abs(ay - fy)
                    if dist < min_dist:
                        min_dists.append(dist)
                    
            return min_dist
    
    def _get_dynamic_view_size(self, min_dist):
        if min_dist >= 4:
            return 7
        elif min_dist == 3:
            return 5
        else:
            return 3

    def _update_traffic(self):
            """Randomly clears expired traffic and spawns new traffic."""
            
            # 1. Identify Cleared Blocks based on probability
            cleared_blocks = set()
            for pos in self.traffic_locations:
                # Check if a random number [0, 1) is less than the clear probability
                if self.np_random.random() < self.traffic_disappear_rate:
                    cleared_blocks.add(pos)
            
            self.traffic_locations -= cleared_blocks

            # 2. Randomly Spawn New Traffic (to maintain the block count)
            if len(self.traffic_locations) < self.max_traffic_locations:
                self._spawn_new_traffic()
                self._spawn_new_traffic()  # Spawn two blocks per step for more dynamics

    def _spawn_new_traffic(self):
        """Finds a random, valid location and adds a new traffic block."""
        W, H = self.map_spec.width, self.map_spec.height
        
        # Find potential locations (must be a movable tile, not fire, not occupied)
        potential_locs = []
        for x in range(W):
            for y in range(H):
                pos = (x, y)
                
                # Check 1: Static map rules (is it a street, not home/park, etc.)
                if not self._is_street(x, y):
                    continue
                
                # Check 2: Dynamic state rules (not fire, not existing traffic, not agent occupied)
                if pos in self.fire_locations or pos in self.traffic_locations:
                    continue
                if any(a.x == x and a.y == y for a in self.agents):
                    continue

                # NEW CHECK: Must be at least 5 tiles away from the nearest fire
                if self._is_safe_distance_from_fire(x, y, min_safe_dist=5):
                    potential_locs.append(pos)
        
        if potential_locs:
            # Randomly select a spot using numpy's random state for determinism
            idx = self.np_random.integers(0, len(potential_locs))
            tx, ty = potential_locs[idx]
            
            # Add to state (no timer needed)
            self.traffic_locations.add((tx, ty))

    def _is_street(self, x: int, y: int) -> bool:
        """Helper to check if a tile is one where movement is usually allowed and is valid for traffic spawn."""
        if not (0 <= x < self.map_spec.width and 0 <= y < self.map_spec.height):
             return False
             
        code = int(self.map_spec.access_grid[y, x])
        can_enter = self._can_enter.get(code, False)
        
        # New Logic: Check if the tile is part of a restricted region for traffic spawn
        # We assume map_spec.region_grid holds the type of region for the tile.
        # This requires accessing the map_spec.regions data structure. 
        # Since the provided code doesn't show a 'region_grid', we'll rely on the access code 
        # and ensure the can_enter check is sufficient, but we must infer the logic.
        
        # Standard check: must be a tile agents can enter (street, work, park, home)
        if not can_enter:
            return False
            
        # Search the map regions to exclude 'home' and 'park'
        # This is a potentially slow check, but necessary without a pre-computed region_type grid.
        
        is_home_or_park_or_work = False
        for r in self.map_spec.regions:
            x_start, y_start = r['x'], r['y']
            x_end, y_end = x_start + r['w'], y_start + r['h']
            
            # Check if the coordinate (x, y) is inside this region
            if x_start <= x < x_end and y_start <= y < y_end:
                if r['type'] in ('home', 'park', 'work'):
                    is_home_or_park_or_work = True
                    break
        
        if is_home_or_park_or_work:
            return False
            
        # Final check against dynamic fire obstacles
        return (x, y) not in self.fire_locations
    
    def _get_dynamic_fov_range(self, agent_x: int, agent_y: int, base_range: int) -> int:
        """
        Calculates the effective FOV range based on the agent's proximity to the nearest fire and smoke.
        Returns a reduced range if the agent is too close (modeling smoke/stress).
        Smoke has a more gradual impact on visibility than fire.
        """
        # Calculate minimum distance to fire
        fire_dist = float('inf')
        if self.fire_locations:
            for fx, fy in self.fire_locations:
                # Using Manhattan distance for simplicity in a grid environment
                dist = abs(agent_x - fx) + abs(agent_y - fy)
                fire_dist = min(fire_dist, dist)
        
        # Calculate minimum distance to smoke
        smoke_dist = float('inf')
        smoke_locs = getattr(self, "smoke_locations", set())
        if smoke_locs:
            for sx, sy in smoke_locs:
                dist = abs(agent_x - sx) + abs(agent_y - sy)
                smoke_dist = min(smoke_dist, dist)

        # Dynamic FOV reduction logic:
        # Fire has immediate severe impact, smoke has more gradual impact
        # Calculate reduction from both and use the most restrictive (minimum FOV)
        
        fire_reduced_range = base_range
        smoke_reduced_range = base_range
        
        # Fire proximity: severe reduction
        if fire_dist <= 1:
            fire_reduced_range = 1  # Can only see own tile and maybe immediate neighbors
        elif fire_dist <= 2:
            fire_reduced_range = 2  # Reduced visibility
        elif fire_dist <= 4:
            # Moderate reduction for nearby fire
            fire_reduced_range = max(3, base_range // 2)
        
        # Smoke proximity: more gradual reduction
        if smoke_dist <= 1:
            # Very close smoke: significant reduction
            smoke_reduced_range = max(2, base_range // 2)
        elif smoke_dist <= 2:
            # Close smoke: moderate reduction
            smoke_reduced_range = max(3, int(base_range * 0.7))
        elif smoke_dist <= 3:
            # Nearby smoke: slight reduction
            smoke_reduced_range = max(4, int(base_range * 0.85))
        elif smoke_dist <= 4:
            # Distant smoke: minimal reduction
            smoke_reduced_range = max(5, int(base_range * 0.9))
        
        # Use the most restrictive (minimum) FOV from fire or smoke
        return min(fire_reduced_range, smoke_reduced_range, base_range)
        
    def _get_min_fire_distance(self, x: int, y: int) -> float:
        """Calculates the Manhattan distance from (x, y) to the nearest fire location."""
        if not self.fire_locations:
            return float('inf')
        
        min_dist = float('inf')
        for fx, fy in self.fire_locations:
            # Using Manhattan distance (L1 norm)
            dist = abs(x - fx) + abs(y - fy)
            min_dist = min(min_dist, dist)
            
        return min_dist

    def _is_safe_distance_from_fire(self, x: int, y: int, min_safe_dist: int = 5) -> bool:
        """Checks if the location is at least 'min_safe_dist' away from the nearest fire."""
        # If there's no fire, it's safe
        if not self.fire_locations:
            return True
        
        distance = self._get_min_fire_distance(x, y)
        
        return distance >= min_safe_dist