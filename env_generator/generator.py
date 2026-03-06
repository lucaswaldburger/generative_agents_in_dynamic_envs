"""
WorldGenerator – top-level orchestrator for procedural environment generation.

Usage::

    from env_generator import WorldGenerator

    gen = WorldGenerator()
    config = gen.generate(layout="village", width=80, height=60, seed=42)
    env    = gen.build_env(config, agent_names=["Alice", "Bob"])
    gen.export_csv(config, "output/my_village")
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import WorldConfig
from .generated_map import GeneratedMap
from .renderer import ProceduralRenderer
from .env import GeneratedEnv
from .exporter import CSVExporter
from .layouts import LAYOUT_REGISTRY, LayoutStrategy


class WorldGenerator:
    """Generates worlds via pluggable layout strategies."""

    def __init__(self):
        self._layouts: Dict[str, LayoutStrategy] = dict(LAYOUT_REGISTRY)

    @property
    def available_layouts(self) -> List[str]:
        return list(self._layouts.keys())

    def register_layout(self, name: str, strategy: LayoutStrategy):
        self._layouts[name] = strategy

    def generate(
        self,
        layout: str = "village",
        width: int = 80,
        height: int = 60,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> WorldConfig:
        """Generate a :class:`WorldConfig` using the named layout strategy."""
        if layout not in self._layouts:
            raise ValueError(
                f"Unknown layout {layout!r}. "
                f"Available: {self.available_layouts}"
            )
        rng = random.Random(seed)
        return self._layouts[layout].generate(width, height, rng, **kwargs)

    @staticmethod
    def build_map(config: WorldConfig) -> GeneratedMap:
        return GeneratedMap(config)

    @staticmethod
    def build_env(
        config: WorldConfig,
        agent_names: Optional[List[str]] = None,
        render_mode: str = "human",
        window_w: int = 1280,
        window_h: int = 800,
        use_sprites: bool = False,
        assets_dir: Optional[str] = None,
    ) -> GeneratedEnv:
        gen_map = GeneratedMap(config)
        return GeneratedEnv(
            gen_map,
            agent_names=agent_names,
            render_mode=render_mode,
            window_w=window_w,
            window_h=window_h,
            use_sprites=use_sprites,
            assets_dir=assets_dir,
        )

    @staticmethod
    def export_csv(config: WorldConfig, output_dir: str | Path) -> Path:
        gen_map = GeneratedMap(config)
        return CSVExporter(gen_map).export(output_dir)
