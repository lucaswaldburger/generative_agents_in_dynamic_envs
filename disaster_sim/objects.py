from minigrid.core.world_object import WorldObj, Lava, Goal
from minigrid.core.world_object import point_in_rect, fill_coords
from minigrid.core.constants import COLOR_NAMES
from globals import COLOR_TO_RGB
import numpy as np

class FireObstacle(Lava):
    def __init__(self, color='red'): 
        super().__init__() 
        self.color = color
        self.type = 'fire' 
        
    def can_overlap(self):
        return True
        
    def encode(self):
        return (10, 4, 0)
    
class TrafficObstacle(WorldObj):
    def __init__(self, color, initial_dir, name="car"):
        super().__init__('ball', color)
        self.initial_dir = initial_dir
        self.current_dir = initial_dir
        self.name = name

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLOR_TO_RGB[self.color]
        
        fill_coords(img, point_in_rect(0.1, 0.2, 0.9, 0.8), c)
        
        wheel_color = (c * 0.5).astype(np.uint8)
        fill_coords(img, point_in_rect(0.15, 0.1, 0.35, 0.2), wheel_color)
        fill_coords(img, point_in_rect(0.65, 0.1, 0.85, 0.2), wheel_color)
        fill_coords(img, point_in_rect(0.15, 0.8, 0.35, 0.9), wheel_color)
        fill_coords(img, point_in_rect(0.65, 0.8, 0.85, 0.9), wheel_color)

class SmokeObstacle(WorldObj):
    def __init__(self, color='grey'): 
        super().__init__("ball", color) 
        self.color = color
        self.type = 'smoke'
        
    def can_overlap(self):
        return True
        
    def encode(self):
        return (8, COLOR_NAMES.index(self.color), 0)

    def render(self, r: np.ndarray) -> np.ndarray:
        color_rgb = np.array([100, 100, 100])
        
        s = r.shape[0] // 4
        e = r.shape[0] * 3 // 4
        r[s:e, s:e, :] = color_rgb
        return r