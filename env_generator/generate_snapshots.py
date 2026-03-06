"""
Generate 15 environments (5 village, 5 urban, 5 campus) and save a PNG
snapshot of each, rendered with the SmallVille CuteRPG tileset sprites.

Usage:
    conda activate cs294
    python -m env_generator.generate_snapshots
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from PIL import Image

from .generator import WorldGenerator
from .generated_map import GeneratedMap
from .renderer import ProceduralRenderer


OUTPUT_DIR = Path(__file__).parent.parent / "generated_environments"
ASSETS_DIR = Path(__file__).parent.parent / "refactored_smallville" / "assets"

CONFIGS = {
    "village": [
        {"seed": 10, "width": 90, "height": 65},
        {"seed": 27, "width": 100, "height": 70},
        {"seed": 42, "width": 80, "height": 60},
        {"seed": 55, "width": 110, "height": 75},
        {"seed": 73, "width": 85, "height": 65},
    ],
    "urban": [
        {"seed": 3,  "width": 70, "height": 55},
        {"seed": 18, "width": 80, "height": 60},
        {"seed": 33, "width": 90, "height": 65},
        {"seed": 50, "width": 75, "height": 55},
        {"seed": 66, "width": 85, "height": 60},
    ],
    "campus": [
        {"seed": 5,  "width": 90, "height": 70},
        {"seed": 21, "width": 100, "height": 75},
        {"seed": 37, "width": 85, "height": 65},
        {"seed": 48, "width": 95, "height": 70},
        {"seed": 61, "width": 80, "height": 60},
    ],
}


def render_to_png(gen_map: GeneratedMap, out_path: Path, assets_dir: Path):
    renderer = ProceduralRenderer(gen_map, assets_dir=assets_dir)
    surf = renderer.build_map_surface()
    raw = pygame.image.tostring(surf, "RGB")
    img = Image.frombytes("RGB", (renderer.pixel_width, renderer.pixel_height), raw)
    img.save(str(out_path))


def main():
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((1, 1))

    gen = WorldGenerator()
    assets = ASSETS_DIR

    if not (assets / "the_ville" / "visuals").exists():
        print(f"WARNING: Tileset assets not found at {assets}", flush=True)
        print("  Falling back to flat-colour rendering.", flush=True)

    for layout, variants in CONFIGS.items():
        layout_dir = OUTPUT_DIR / layout
        layout_dir.mkdir(parents=True, exist_ok=True)

        for idx, params in enumerate(variants, start=1):
            seed = params["seed"]
            w = params["width"]
            h = params["height"]

            config = gen.generate(layout=layout, width=w, height=h,
                                  seed=seed, tile_size=32)
            gen_map = gen.build_map(config)

            fname = f"{layout}_{idx}_{config.world_name.replace(' ', '_')}.png"
            out_path = layout_dir / fname
            render_to_png(gen_map, out_path, assets)

            n_bld = len(config.buildings)
            n_zones = len(config.zones)
            n_parks = len(config.parks)
            n_addr = len(gen_map.address_tiles)
            print(f"  [{layout:>7s} {idx}] {config.world_name:25s} "
                  f"{w}x{h}  seed={seed:3d}  "
                  f"buildings={n_bld:2d}  zones={n_zones:2d}  "
                  f"parks={n_parks}  addresses={n_addr:3d}  -> {out_path.name}",
                  flush=True)

    pygame.quit()
    print(f"\nAll snapshots saved to: {OUTPUT_DIR}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
