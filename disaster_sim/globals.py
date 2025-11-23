
import time
import random
from gymnasium.envs.registration import register
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo


COLOR_TO_RGB = {
    'red': np.array([255, 50, 0], dtype=np.uint8),
    'green': np.array([0, 255, 0], dtype=np.uint8),
    'blue': np.array([0, 0, 255], dtype=np.uint8),
    'purple': np.array([170, 0, 255], dtype=np.uint8),
    'yellow': np.array([255, 255, 0], dtype=np.uint8),
    'grey': np.array([100, 100, 100], dtype=np.uint8),
}

MAX_STEPS = 1_000
SPEED = 0.5