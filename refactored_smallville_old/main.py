"""
Minimal SmallVille runner using pygame + gymnasium.
No Django frontend server required.

Usage:
    python main.py                    # interactive camera mode
    python main.py --demo             # demo with random agent walks
    python main.py --record out.mp4   # record to video file
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from env.constants import Action, AgentConfig, DIR_TO_VEC, COLLISION_BLOCK_ID
from env.tilemap import TiledMapRenderer
from env.smallville_env import SmallvilleEnv
from env.world_object import SmallvilleAgent

GENERATIVE_AGENTS_ROOT = Path(__file__).resolve().parent.parent / "generative_agents"
ASSETS_ROOT = (
    GENERATIVE_AGENTS_ROOT
    / "environment" / "frontend_server" / "static_dirs" / "assets"
)
THE_VILLE = ASSETS_ROOT / "the_ville"
VISUALS_DIR = THE_VILLE / "visuals"
MATRIX_DIR = THE_VILLE / "matrix"


def load_spawning_locations() -> Dict[str, Tuple[int, int]]:
    """Parse the spawning_location_maze.csv and spawning_location_blocks.csv
    to map persona names to tile coordinates."""
    blocks_path = MATRIX_DIR / "special_blocks" / "spawning_location_blocks.csv"
    maze_path = MATRIX_DIR / "maze" / "spawning_location_maze.csv"

    block_id_to_name: Dict[int, str] = {}
    with blocks_path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            parts = [p.strip() for p in row]
            if len(parts) >= 5:
                # Store "sector / arena / spawn" for persona matching
                block_id_to_name[int(parts[0])] = f"{parts[2]} / {parts[3]}"

    maze_width = 140
    locations: Dict[str, Tuple[int, int]] = {}
    with maze_path.open() as f:
        raw = f.read()
    values = [v.strip() for v in raw.split(",")]
    for idx, v in enumerate(values):
        if not v or v == "0":
            continue
        gid = int(v)
        if gid in block_id_to_name:
            room_name = block_id_to_name[gid]
            col = idx % maze_width
            row = idx // maze_width
            if room_name not in locations:
                locations[room_name] = (col, row)

    return locations


def load_sector_info() -> Dict[int, str]:
    """Parse sector_blocks.csv for sector name lookup."""
    path = MATRIX_DIR / "special_blocks" / "sector_blocks.csv"
    result: Dict[int, str] = {}
    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            parts = [p.strip() for p in row]
            if len(parts) >= 3:
                result[int(parts[0])] = parts[2]
    return result


def build_default_agents() -> List[AgentConfig]:
    """Create agent configs from known SmallVille personas and their spawn points."""
    spawns = load_spawning_locations()

    persona_colors = {
        "Isabella Rodriguez": (255, 80, 80),
        "Klaus Mueller": (80, 80, 255),
        "Maria Lopez": (80, 200, 120),
        "Abigail Chen": (230, 230, 90),
        "Rajiv Patel": (200, 80, 200),
    }

    agents: List[AgentConfig] = []
    for name, color in persona_colors.items():
        first = name.split()[0].lower()
        last = name.split()[-1].lower()
        room_candidates = [
            k for k in spawns
            if first in k.lower() or last in k.lower()
        ]
        if room_candidates:
            key = room_candidates[0]
            x, y = spawns[key]
        else:
            x, y = 60, 44
        agents.append(AgentConfig(name=name, start_x=x, start_y=y, color=color))

    if not agents:
        agents.append(AgentConfig(name="Agent 1", start_x=60, start_y=44, color=(255, 80, 80)))

    return agents


def random_walk_step(
    env: SmallvilleEnv, rng: np.random.Generator
) -> np.ndarray:
    """Generate one step of random walk for all agents using A* to random goals."""
    actions = np.zeros(env.num_agents, dtype=np.int64)

    for i, agent in enumerate(env.agents):
        if not hasattr(agent, "_path") or not agent._path:
            for _ in range(50):
                gx = rng.integers(0, env.tilemap.map_width)
                gy = rng.integers(0, env.tilemap.map_height)
                if not env.tilemap.is_blocked(gx, gy):
                    path = env.astar((agent.x, agent.y), (gx, gy))
                    if path and len(path) > 0:
                        agent._path = path
                        break
            else:
                agent._path = []

        if hasattr(agent, "_path") and agent._path:
            actions[i] = int(agent._path.pop(0))
        else:
            actions[i] = int(Action.STAY)

    return actions


def run_interactive(env: SmallvilleEnv) -> None:
    """Run with keyboard camera controls. Agents stay in place."""
    obs, info = env.reset()
    env.render()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        actions = np.zeros(env.num_agents, dtype=np.int64)
        obs, _, _, truncated, _ = env.step(actions)
        env.render()

        if truncated:
            break

    env.close()


def run_demo(env: SmallvilleEnv, steps: int = 500) -> None:
    """Run a demo where agents random-walk with A*."""
    rng = np.random.default_rng(42)
    obs, info = env.reset()
    env.render()

    for t in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                env.close()
                return

        actions = random_walk_step(env, rng)
        obs, _, terminated, truncated, _ = env.step(actions)
        env.render()
        time.sleep(0.05)

        if terminated or truncated:
            break

    env.close()


def run_record(env: SmallvilleEnv, output_path: str, steps: int = 300) -> None:
    """Record a video of the demo."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("Install imageio to record: pip install imageio imageio-ffmpeg")
        return

    rng = np.random.default_rng(42)
    env_rec = SmallvilleEnv(
        tilemap=env.tilemap,
        agent_configs=env.agent_configs,
        max_steps=steps,
        render_mode="rgb_array",
        camera_follow=env.camera_follow,
        viewport_w=env.viewport_w,
        viewport_h=env.viewport_h,
    )
    obs, _ = env_rec.reset()

    writer = imageio.get_writer(output_path, fps=env.metadata["render_fps"])
    print(f"Recording {steps} frames to {output_path}...")
    for t in range(steps):
        actions = random_walk_step(env_rec, rng)
        obs, _, terminated, truncated, _ = env_rec.step(actions)
        frame = env_rec.render()
        if frame is not None:
            writer.append_data(frame)
        if terminated or truncated:
            break
        if t % 50 == 0:
            print(f"  frame {t}/{steps}")
    writer.close()
    env_rec.close()
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SmallVille pygame viewer")
    parser.add_argument("--demo", action="store_true", help="Run random-walk demo")
    parser.add_argument("--record", type=str, default=None, help="Record to mp4")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps")
    parser.add_argument("--width", type=int, default=1280, help="Viewport width")
    parser.add_argument("--height", type=int, default=800, help="Viewport height")
    parser.add_argument("--no-follow", action="store_true", help="Disable camera follow")
    args = parser.parse_args()

    map_json = VISUALS_DIR / "the_ville_jan7.json"
    collision_csv = MATRIX_DIR / "maze" / "collision_maze.csv"

    if not map_json.exists():
        print(f"ERROR: Map JSON not found at {map_json}")
        print("Make sure generative_agents is cloned at ../generative_agents/")
        sys.exit(1)
    if not collision_csv.exists():
        print(f"ERROR: Collision CSV not found at {collision_csv}")
        sys.exit(1)

    if args.record:
        import os
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.display.set_mode((1, 1))

    print("Loading tilemap...")
    tilemap = TiledMapRenderer(
        map_json_path=str(map_json),
        collision_csv_path=str(collision_csv),
        collision_block_id=COLLISION_BLOCK_ID,
    )

    print("Loading tileset assets (this may take a moment)...")
    tilemap.load_assets()
    print(f"Map size: {tilemap.map_width}x{tilemap.map_height} tiles "
          f"({tilemap.pixel_width}x{tilemap.pixel_height} px)")

    agent_configs = build_default_agents()
    print(f"Agents: {[a.name for a in agent_configs]}")

    render_mode = "rgb_array" if args.record else "human"
    env = SmallvilleEnv(
        tilemap=tilemap,
        agent_configs=agent_configs,
        max_steps=args.steps,
        render_mode=render_mode,
        camera_follow=not args.no_follow,
        viewport_w=args.width,
        viewport_h=args.height,
    )

    if args.record:
        run_record(env, args.record, steps=args.steps)
    elif args.demo:
        run_demo(env, steps=args.steps)
    else:
        run_interactive(env)


if __name__ == "__main__":
    main()
