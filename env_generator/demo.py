"""
CLI demo – generate a procedural environment, render it, and optionally
record an mp4 video with wandering agents.

Usage::

    python -m env_generator.demo --layout village --seed 42 --steps 300
    python -m env_generator.demo --layout urban --width 80 --height 60
    python -m env_generator.demo --layout campus --export-csv campus_output/
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from .generator import WorldGenerator
from .env import Action


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Procedural environment generator demo")
    p.add_argument("--layout", type=str, default="village",
                   choices=["village", "urban", "campus"],
                   help="Layout strategy to use")
    p.add_argument("--width", type=int, default=80)
    p.add_argument("--height", type=int, default=60)
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--world-name", type=str, default=None)
    p.add_argument("--agents", nargs="*", default=["Alice", "Bob", "Charlie"])
    p.add_argument("--render-mode", type=str, default="rgb_array",
                   choices=["human", "rgb_array"])
    p.add_argument("--window-w", type=int, default=1280)
    p.add_argument("--window-h", type=int, default=800)
    p.add_argument("--zoom", type=float, default=None)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--output", type=str, default="generated_env.mp4")
    p.add_argument("--export-csv", type=str, default=None,
                   help="Export to SmallVille CSV format at this path")
    p.add_argument("--sprites", action="store_true", default=False)
    p.add_argument("--assets-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    gen = WorldGenerator()

    extra_kw = {}
    if args.world_name:
        extra_kw["world_name"] = args.world_name
    extra_kw["tile_size"] = args.tile_size

    print(f"Generating {args.layout} environment ({args.width}x{args.height}, "
          f"seed={args.seed})...")
    config = gen.generate(
        layout=args.layout, width=args.width, height=args.height,
        seed=args.seed, **extra_kw)
    print(f"  World: {config.world_name}")
    print(f"  Zones: {len(config.zones)}, Buildings: {len(config.buildings)}, "
          f"Parks: {len(config.parks)}, Spawns: {len(config.spawns)}")

    if args.export_csv:
        out_path = gen.export_csv(config, args.export_csv)
        print(f"  Exported CSV to: {out_path}")

    env = gen.build_env(
        config, agent_names=args.agents, render_mode=args.render_mode,
        window_w=args.window_w, window_h=args.window_h,
        use_sprites=args.sprites, assets_dir=args.assets_dir)
    obs, info = env.reset()

    if args.zoom is not None:
        env._zoom = args.zoom
    else:
        map_px_w = env.renderer.pixel_width
        map_px_h = env.renderer.pixel_height
        env._zoom = min(args.window_w / map_px_w, args.window_h / map_px_h) * 0.95
    env._camera_x = -(env.renderer.pixel_width * env._zoom - args.window_w) / 2
    env._camera_y = -(env.renderer.pixel_height * env._zoom - args.window_h) / 2

    locations = _build_location_list(env)
    print(f"  Available locations: {len(locations)}")
    print(f"  Agents: {[a.name for a in env.agents]}")
    print(f"  Recording {args.steps} steps to {args.output} ({args.fps} fps)\n")

    import imageio
    w_out = args.window_w if args.window_w % 2 == 0 else args.window_w + 1
    h_out = args.window_h if args.window_h % 2 == 0 else args.window_h + 1
    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=1)

    rng = random.Random(args.seed)
    t_start = time.time()
    step = 0

    try:
        while step < args.steps:
            if args.render_mode == "human":
                if not env.handle_pygame_events():
                    break
                import pygame
                keys = pygame.key.get_pressed()
                if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
                    break

            for i, agent in enumerate(env.agents):
                if not agent.path and step % 40 == 0 and locations:
                    target_addr = rng.choice(locations)
                    env.move_agent_to_address(i, target_addr)
                    parts = target_addr.split(":")
                    short = " > ".join(parts[1:]) if len(parts) >= 2 else target_addr
                    agent.description = f"→ {short[:50]}"

            actions = [Action.STAY] * len(env.agents)
            obs, reward, terminated, truncated, info = env.step(actions)

            frame = env.render()
            if frame is not None:
                if frame.shape[0] != h_out or frame.shape[1] != w_out:
                    frame = np.pad(frame, (
                        (0, max(0, h_out - frame.shape[0])),
                        (0, max(0, w_out - frame.shape[1])),
                        (0, 0),
                    ))[:h_out, :w_out]
                writer.append_data(frame)

            step += 1
            if step % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  step {step}/{args.steps} "
                      f"({elapsed:.0f}s, {step / elapsed:.1f} steps/s)")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        writer.close()
        env.close()
        elapsed = time.time() - t_start
        print(f"\nDone: {step} steps in {elapsed:.1f}s")
        print(f"Video saved to: {args.output}")


def _build_location_list(env) -> List[str]:
    seen = set()
    locations = []
    for addr in sorted(env.maze.address_tiles.keys()):
        if addr.startswith("<spawn_loc>"):
            continue
        parts = addr.split(":")
        if len(parts) >= 2:
            short = " > ".join(parts[1:])
            if short not in seen:
                seen.add(short)
                locations.append(addr)
    return locations


if __name__ == "__main__":
    main()
