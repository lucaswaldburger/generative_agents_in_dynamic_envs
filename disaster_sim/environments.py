from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
import numpy as np
import itertools
from objects import FireObstacle, TrafficObstacle, SmokeObstacle

# class BaseEnv(MiniGridEnv):
#     def __init__(self, size=16, max_steps=100, **kwargs):
#         self.size = size

#         mission_space = MissionSpace(
#             mission_func=lambda: "get to the green goal without touching the fire"
#         )
#         render_mode = kwargs.pop("render_mode", "rgb_array")
        
#         super().__init__(
#             mission_space=mission_space,
#             grid_size=size,
#             max_steps=max_steps,
#             agent_view_size=base_view_size,
#             render_mode=render_mode,
#             **kwargs
#         )

#     def _get_dynamic_view_size(self, min_dist):
#         if min_dist >= 4:
#             return 7 
#         elif min_dist == 3:
#             return 5
#         else:
#             return 3 
    
        
# class FireEnv(BaseEnv):
#     def __init__(self, size=16, max_steps=100, spread_rate=0.05, **kwargs):
#         super().__init__(size=size, max_steps=max_steps, **kwargs)
#         self.spread_rate = spread_rate
#         self.fire_locations = []

#     def _gen_grid(self, width, height):
#         self.grid = Grid(width, height)
#         self.grid.wall_rect(0, 0, width, height)

#         self.place_agent()
        
#         goal = Goal()
#         self.place_obj(goal, top=(width - 2, height - 2)) 
#         for i, j in itertools.product(range(width), range(height)):
#             cell = self.grid.get(i, j)
#             if isinstance(cell, Goal):
#                 self.goal_pos = (i, j)
#                 break

#         start_x = self.width // 2
#         start_y = self.height // 2
#         self.grid.set(start_x, start_y, FireObstacle())
        
#         self.fire_locations = [(start_x, start_y)]
#     def _calculate_min_fire_distance(self):
#         if not self.fire_locations:
#             return float('inf')
        
#         min_dist = float('inf')
#         ax, ay = self.agent_pos
        
#         for fx, fy in self.fire_locations:
#             dist = abs(ax - fx) + abs(ay - fy)
#             if dist < min_dist:
#                 min_dist = dist
                
#         return min_dist

#     def gen_obs(self):
#         min_dist = self._calculate_min_fire_distance()
#         new_view_size = self._get_dynamic_view_size(min_dist)
#         self.agent_view_size = new_view_size
        
#         self._last_view_size = new_view_size

#         obs = super().gen_obs()
#         return obs
        
#     def step(self, action):
#         obs, reward, terminated, truncated, info = super().step(action)
        
#         info['view_size'] = self._last_view_size
        
#         if isinstance(self.grid.get(*self.agent_pos), FireObstacle):
#             print("AGENT TOUCHED FIRE! 🔥")
#             reward = -1.0
#             terminated = True 

#         if not terminated and not truncated:
#             self._spread_fire()

#         return obs, reward, terminated, truncated, info
    
#     def _spread_fire(self):
#         new_fires = []
        
#         current_fire_locs = list(self.fire_locations) 
        
#         for fx, fy in current_fire_locs:
#             for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#                 nx, ny = fx + dx, fy + dy
                
#                 in_bounds = (nx >= 0 and nx < self.width and 
#                              ny >= 0 and ny < self.height)

#                 if (in_bounds and 
#                     self.grid.get(nx, ny) is None and
#                     (nx, ny) != tuple(self.agent_pos)): 
                    
#                     if self.np_random.uniform() < self.spread_rate:
#                         self.grid.set(nx, ny, FireObstacle())
#                         new_fires.append((nx, ny))
        
#         self.fire_locations.extend(new_fires)

class FireSmokeEnv(MiniGridEnv):
    def __init__(self, size=16, max_steps=100, spread_rate=0.05, **kwargs):
        self.size = size
        self.spread_rate = spread_rate
        base_view_size = kwargs.pop('agent_view_size', 7) 

        mission_space = MissionSpace(
            mission_func=lambda: "get to the green goal without touching the fire"
        )
        
        render_mode = kwargs.pop("render_mode", "rgb_array")
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            agent_view_size=base_view_size,
            render_mode=render_mode,
            **kwargs
        )
        
        self.fire_locations = []
        self.smoke_locations = []
        self._last_view_size = base_view_size
        self.goal_pos = None 

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.place_agent()
        goal = Goal()
        self.place_obj(goal, top=(width - 2, height - 2)) 
        for i, j in itertools.product(range(width), range(height)):
            cell = self.grid.get(i, j)
            if isinstance(cell, Goal):
                self.goal_pos = (i, j)
                break
        start_x = self.width // 2
        start_y = self.height // 2
        self.grid.set(start_x, start_y, FireObstacle())
        self.fire_locations = [(start_x, start_y)]
    
    def _calculate_min_fire_distance(self):
        if not self.fire_locations:
            return float('inf')
        
        min_dist = float('inf')
        ax, ay = self.agent_pos
        
        for fx, fy in self.fire_locations:
            dist = abs(ax - fx) + abs(ay - fy)
            if dist < min_dist:
                min_dist = dist
                
        return min_dist

    def _get_dynamic_view_size(self, min_dist):
        if min_dist >= 4:
            return 7
        elif min_dist == 3:
            return 5
        else: # min_dist <= 2
            return 3

    def gen_obs(self):
        min_dist = self._calculate_min_fire_distance()
        new_view_size = self._get_dynamic_view_size(min_dist)
        
        self.agent_view_size = new_view_size
        
        self._last_view_size = new_view_size

        obs = super().gen_obs()
        return obs
        
    def _smart_action_choice(self):
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        
        best_score = float('inf')
        best_action = self.actions.forward
        
        for action in [self.actions.left, self.actions.right, self.actions.forward]:
            
            current_score = float('inf')
            
            if action == self.actions.forward:
                dx = [1, 0, -1, 0][self.agent_dir]
                dy = [0, 1, 0, -1][self.agent_dir]
                next_pos = (ax + dx, ay + dy)
                
                in_bounds = (next_pos[0] >= 0 and next_pos[0] < self.width and 
                             next_pos[1] >= 0 and next_pos[1] < self.height)

                if in_bounds:
                    cell = self.grid.get(*next_pos)
                    if isinstance(cell, FireObstacle):
                        current_score = float('inf') 
                    elif cell is not None and not cell.can_overlap():
                        current_score = float('inf') 
                    else:
                        current_score = abs(next_pos[0] - gx) + abs(next_pos[1] - gy)
                else:
                    current_score = float('inf') 

            elif action in [self.actions.left, self.actions.right]:                
                new_dir = (self.agent_dir + (1 if action == self.actions.right else -1)) % 4
                dx = [1, 0, -1, 0][new_dir]
                dy = [0, 1, 0, -1][new_dir]
                simulated_fwd_pos = (ax + dx, ay + dy)
                
                dist_after_turn_and_move = abs(simulated_fwd_pos[0] - gx) + abs(simulated_fwd_pos[1] - gy)
                
                current_score = dist_after_turn_and_move + 1
            
            if current_score < best_score:
                best_score = current_score
                best_action = action

        return best_action

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options) 
        self._spread_fire(initial_placement=True)
        info['view_size'] = self._last_view_size
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info['view_size'] = self._last_view_size        
        if isinstance(self.grid.get(*self.agent_pos), FireObstacle):
            print("AGENT TOUCHED FIRE! 🔥")
            reward = -1.0
            terminated = True 

        if not terminated and not truncated:
            self._spread_fire()

        return obs, reward, terminated, truncated, info
    
    def _spread_fire(self, initial_placement=False):
        for loc in self.smoke_locations:
            if not isinstance(self.grid.get(*loc), FireObstacle):
                self.grid.set(*loc, None)
        self.smoke_locations.clear()
        
        new_fires = []
        if not initial_placement:
            current_fire_locs = list(self.fire_locations) 
            
            spread_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for fx, fy in current_fire_locs:
                for dx, dy in spread_deltas:
                    nx, ny = fx + dx, fy + dy
                    
                    in_bounds = (1 <= nx < self.width - 1 and 
                                 1 <= ny < self.height - 1)

                    if (in_bounds and 
                        self.grid.get(nx, ny) is None and
                        (nx, ny) != tuple(self.agent_pos) and
                        (nx, ny) != self.goal_pos): 
                        
                        if self.np_random.uniform() < self.spread_rate:
                            self.grid.set(nx, ny, FireObstacle())
                            new_fires.append((nx, ny))
            
            self.fire_locations.extend(new_fires)
        
        potential_smoke_locs = set() 
        neighbor_deltas = [
            (-1, 0), (1, 0), (0, -1), (0, 1), 
            (-1, -1), (1, 1), (-1, 1), (1, -1)
        ]

        for fx, fy in self.fire_locations:
            for dx, dy in neighbor_deltas:
                sx, sy = fx + dx, fy + dy
                
                if (1 <= sx < self.width - 1 and 
                    1 <= sy < self.height - 1 and 
                    self.grid.get(sx, sy) is None and 
                    (sx, sy) != tuple(self.agent_pos) and 
                    (sx, sy) != self.goal_pos):
                    
                    potential_smoke_locs.add((sx, sy))

        for sx, sy in potential_smoke_locs:
            if self.grid.get(sx, sy) is None:
                self.grid.set(sx, sy, SmokeObstacle())
                self.smoke_locations.append((sx, sy))

class TrafficSimEnv(MiniGridEnv):
    def __init__(self,
                 size=16,
                 max_steps=100,
                 static_disappearing_mode=False,
                 disappear_prob=0.1,
                 traffic_concentration_pct=1.0,
                 traffic_concentration_align=0.5,
                 target_concentration_x=None,
                 concentration_span=None, **kwargs):
        self.traffic_objects = []
        self.traffic_lane_configs = []
        base_view_size = kwargs.pop('agent_view_size', 7) 

        self.static_disappearing_mode = static_disappearing_mode
        self.disappear_prob = disappear_prob 
        self.traffic_concentration_pct = traffic_concentration_pct 
        self.traffic_concentration_align = traffic_concentration_align
        self.target_concentration_x = target_concentration_x
        self.concentration_span = concentration_span
        if self.static_disappearing_mode:
            mission = "navigate to the green goal while avoiding static, randomly reappearing traffic"
        else:
            mission = "navigate to the green goal while avoiding the moving traffic"
        self.mission_space = MissionSpace(mission_func=lambda: mission)
        self.agent_start_pos = (1, size - 2)
        self.agent_start_dir = 3

        super().__init__(
            mission_space=self.mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs
        )
        self._last_view_size = base_view_size
        self.goal_pos = None 

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(), width - 2, 1)
        for i, j in itertools.product(range(width), range(height)):
            cell = self.grid.get(i, j)
            if isinstance(cell, Goal):
                self.goal_pos = (i, j)
                break
        self._add_traffic(row=4, color='red', initial_dir=1)
        self._add_traffic(row=6, color='red', initial_dir=-1)
        self._add_traffic(row=8, color='red', initial_dir=1)
        self._add_traffic(row=2, color='red', initial_dir=1)
        self._add_traffic(row=9, color='red', initial_dir=1)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir

        self._reset_traffic()
        self.grid.set(self.agent_start_pos[0], self.agent_start_pos[1], None)

    def _add_traffic(self, row, color, initial_dir):
        lane_config = {
            'row': row,
            'color': color,
            'initial_dir': initial_dir,
            'lane_start_x': 1,
            'lane_end_x': self.width - 2,
        }
        self.traffic_lane_configs.append(lane_config)

    def _get_concentration_zone(self, lane_start_x, lane_end_x):
        if self.target_concentration_x is not None and self.concentration_span is not None:
            center_x = self.target_concentration_x
            span = self.concentration_span
            half_span = span // 2
            
            zone_start_x = center_x - half_span
            zone_end_x = center_x + (span - 1) - half_span
            
            zone_start_x = max(lane_start_x, zone_start_x)
            zone_end_x = min(lane_end_x, zone_end_x)

            if zone_start_x > zone_end_x:
                zone_start_x = min(lane_end_x, center_x)
                zone_end_x = zone_start_x
            
            return zone_start_x, zone_end_x
        
        else:            
            lane_length = lane_end_x - lane_start_x + 1
            concentrated_width = max(1, int(lane_length * self.traffic_concentration_pct))
            remaining_width = lane_length - concentrated_width
            offset = int(remaining_width * self.traffic_concentration_align)
            
            zone_start_x = lane_start_x + offset
            zone_end_x = zone_start_x + concentrated_width - 1
            
            zone_end_x = min(zone_end_x, lane_end_x)
            
            return zone_start_x, zone_end_x

    def _reset_traffic(self):
        for traffic in self.traffic_objects:
            self.grid.set(*traffic.cur_pos, None)
            
        self.traffic_objects = []
        
        for config in self.traffic_lane_configs:
            traffic = TrafficObstacle(config['color'], config['initial_dir'])
            
            traffic.lane_row = config['row']
            traffic.lane_start_x = config['lane_start_x']
            traffic.lane_end_x = config['lane_end_x']

            zone_start_x, zone_end_x = self._get_concentration_zone(
                traffic.lane_start_x, traffic.lane_end_x
            )
            
            random_col = self.np_random.integers(zone_start_x, zone_end_x + 1)
            random_pos = (random_col, traffic.lane_row)
            
            if self.grid.get(*random_pos) is None and random_pos != self.agent_pos:
                traffic.cur_pos = random_pos
                self.grid.set(random_pos[0], random_pos[1], traffic)
                self.traffic_objects.append(traffic)

    def _update_traffic_state(self):
        if self.static_disappearing_mode:
            active_traffic_objects = self.traffic_objects
            self.traffic_objects = []
            traffic_configs_for_respawn = []
            for traffic in active_traffic_objects:
                if np.random.rand() < self.disappear_prob:
                    self.grid.set(*traffic.cur_pos, None)
                    config = next((c for c in self.traffic_lane_configs if c['row'] == traffic.lane_row), None)
                    if config:
                        traffic_configs_for_respawn.append(config)
                else:
                    self.traffic_objects.append(traffic)
            active_rows = {t.lane_row for t in self.traffic_objects}
            all_respawn_configs = traffic_configs_for_respawn
            for config in self.traffic_lane_configs:
                if config['row'] not in active_rows:
                    all_respawn_configs.append(config)
            for config in all_respawn_configs:
                if np.random.rand() < self.disappear_prob:
                    lane_row = config['row']
                    lane_start_x = config['lane_start_x']
                    lane_end_x = config['lane_end_x']
                    zone_start_x, zone_end_x = self._get_concentration_zone(
                        lane_start_x, lane_end_x
                    )
                    random_col = self.np_random.integers(zone_start_x, zone_end_x + 1)
                    random_pos = (random_col, lane_row)
                    if self.grid.get(*random_pos) is None and random_pos != self.agent_pos:
                        new_traffic = TrafficObstacle(config['color'], config['initial_dir'])
                        new_traffic.lane_row = lane_row
                        new_traffic.lane_start_x = lane_start_x
                        new_traffic.lane_end_x = lane_end_x
                        new_traffic.cur_pos = random_pos

                        self.grid.set(random_pos[0], random_pos[1], new_traffic)
                        self.traffic_objects.append(new_traffic)
        
        else:
            for traffic in self.traffic_objects:
                old_pos = traffic.cur_pos
                self.grid.set(old_pos[0], old_pos[1], None)
                new_x = old_pos[0] + traffic.current_dir
                new_y = old_pos[1]
                if new_x > traffic.lane_end_x or new_x < traffic.lane_start_x:
                    traffic.current_dir *= -1
                    new_x = old_pos[0] + traffic.current_dir # Re-calculate new x
                traffic.cur_pos = (new_x, new_y)
                self.grid.set(new_x, new_y, traffic)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        agent_pos_after_action = self.agent_pos
        traffic_hit_by_agent = self.grid.get(*agent_pos_after_action)
        if traffic_hit_by_agent and traffic_hit_by_agent.type == 'ball':
             reward = -1.0
             terminated = True
             return obs, reward, terminated, truncated, info
        self._update_traffic_state()
        traffic_hit_agent = self.grid.get(*agent_pos_after_action)
        if traffic_hit_agent and traffic_hit_agent.type == 'ball':
             reward = -1.0
             terminated = True
             obs = self.gen_obs()
             return obs, reward, terminated, truncated, info
        if not terminated and not truncated:
            reward = 0.0
        
        return obs, reward, terminated, truncated, info
    
    def _smart_action_choice(self):
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        
        best_score = float('inf')
        best_action = self.actions.forward
        
        for action in [self.actions.left, self.actions.right, self.actions.forward]:
            
            current_score = float('inf')
            
            if action == self.actions.forward:
                dx = [1, 0, -1, 0][self.agent_dir]
                dy = [0, 1, 0, -1][self.agent_dir]
                next_pos = (ax + dx, ay + dy)
                
                in_bounds = (next_pos[0] >= 0 and next_pos[0] < self.width and 
                            next_pos[1] >= 0 and next_pos[1] < self.height)

                if in_bounds:
                    cell = self.grid.get(*next_pos)
                    
                    if isinstance(cell, TrafficObstacle):
                        current_score = float('inf') 
                    elif cell is not None and not cell.can_overlap():
                        current_score = float('inf') 
                    else:
                        current_score = abs(next_pos[0] - gx) + abs(next_pos[1] - gy)
                else:
                    current_score = float('inf') 

            elif action in [self.actions.left, self.actions.right]:                
                new_dir = (self.agent_dir + (1 if action == self.actions.right else -1)) % 4
                
                dx = [1, 0, -1, 0][new_dir]
                dy = [0, 1, 0, -1][new_dir]
                simulated_fwd_pos = (ax + dx, ay + dy)
                
                dist_after_turn_and_move = abs(simulated_fwd_pos[0] - gx) + abs(simulated_fwd_pos[1] - gy)
                
                current_score = dist_after_turn_and_move + 1
            
            if current_score < best_score:
                best_score = current_score
                best_action = action

        return best_action
        
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options) 
        info['view_size'] = self._last_view_size
        return obs, info