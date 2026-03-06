"""
Layout strategies for procedural world generation.

Each strategy implements ``generate(width, height, rng, **kw) -> WorldConfig``.
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import (
    BuildingConfig,
    CarLaneConfig,
    IntersectionConfig,
    PaletteConfig,
    ParkConfig,
    RoadConfig,
    SpawnConfig,
    WorldConfig,
    ZoneConfig,
)

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _rects_overlap(ax: int, ay: int, aw: int, ah: int,
                   bx: int, by: int, bw: int, bh: int,
                   margin: int = 0) -> bool:
    return not (ax + aw + margin <= bx or bx + bw + margin <= ax or
                ay + ah + margin <= by or by + bh + margin <= ay)


def _pick_color(rng: random.Random, base: Tuple[int, int, int],
                spread: int = 40) -> Tuple[int, int, int]:
    return tuple(
        max(0, min(255, c + rng.randint(-spread, spread))) for c in base
    )


_VILLAGE_BUILDING_POOL = [
    ("General Store", "shop"), ("Bakery", "shop"), ("Blacksmith", "workshop"),
    ("Tavern", "tavern"), ("Inn", "inn"), ("Church", "church"),
    ("Town Hall", "government"), ("Schoolhouse", "school"),
    ("Apothecary", "pharmacy"), ("Mill", "workshop"), ("Stables", "stables"),
    ("Library", "library"), ("Cottage", "residential"),
    ("Farmhouse", "residential"), ("Woodworker", "workshop"),
    ("Market Stall", "shop"), ("Well House", "utility"),
    ("Potter", "workshop"), ("Weaver", "workshop"), ("Chapel", "church"),
]

_VILLAGE_OBJECTS = {
    "shop": ["counter", "shelves", "crate"],
    "tavern": ["bar counter", "table", "chair"],
    "inn": ["front desk", "bed", "wardrobe"],
    "church": ["altar", "pew", "candle stand"],
    "government": ["desk", "notice board", "chair"],
    "school": ["desk", "blackboard", "bookshelf"],
    "pharmacy": ["counter", "medicine shelf", "mortar"],
    "workshop": ["workbench", "tool rack", "stool"],
    "stables": ["hay bale", "trough", "saddle rack"],
    "library": ["bookshelf", "reading desk", "chair"],
    "residential": ["bed", "table", "fireplace"],
    "utility": ["well", "bucket"],
}

_URBAN_BUILDING_POOL = [
    ("Apartments", "residential"), ("Diner", "restaurant"),
    ("Bookstore", "shop"), ("Gym", "gym"), ("Office Tower", "office"),
    ("Cafe", "cafe"), ("Pharmacy", "pharmacy"), ("Bar", "bar"),
    ("School", "school"), ("Grocery", "shop"), ("Art Studio", "studio"),
    ("Hospital", "hospital"), ("Police Station", "government"),
    ("Fire Station", "government"), ("Condos", "residential"),
    ("Pizza Place", "restaurant"), ("Library", "library"),
    ("Lofts", "residential"), ("Yoga Studio", "gym"),
    ("Noodle House", "restaurant"), ("Tech Office", "office"),
    ("Lounge", "bar"), ("Repair Shop", "shop"),
    ("Laundromat", "shop"), ("Community Center", "community"),
    ("Music Venue", "entertainment"), ("Mall", "shop"),
    ("Cinema", "entertainment"), ("Hotel", "hotel"),
    ("Sushi Bar", "restaurant"), ("City Hall", "government"),
    ("Post Office", "government"), ("Bus Station", "transit"),
    ("Market Square", "shop"), ("Night Club", "entertainment"),
    ("Train Station", "transit"),
]

_CAMPUS_BUILDING_POOL = [
    ("Science Hall", "academic"), ("Lecture Center", "academic"),
    ("Engineering Lab", "academic"), ("Humanities Building", "academic"),
    ("Math & CS Building", "academic"), ("Art Building", "academic"),
    ("Student Center", "community"), ("Main Library", "library"),
    ("Campus Bookstore", "shop"), ("Dining Hall", "cafeteria"),
    ("Dorm A", "dormitory"), ("Dorm B", "dormitory"),
    ("Dorm C", "dormitory"), ("Admin Building", "government"),
    ("Health Center", "clinic"), ("Recreation Center", "gym"),
    ("Auditorium", "entertainment"), ("Research Lab", "academic"),
    ("Greenhouse", "academic"), ("Music Building", "academic"),
    ("Coffee Shop", "cafe"),
]

_CAMPUS_OBJECTS = {
    "academic": ["desk", "whiteboard", "projector", "chair"],
    "library": ["bookshelf", "reading desk", "computer terminal"],
    "shop": ["counter", "shelves", "register"],
    "cafeteria": ["serving counter", "table", "chair", "vending machine"],
    "dormitory": ["bed", "desk", "closet", "lamp"],
    "government": ["reception desk", "filing cabinet", "chair"],
    "clinic": ["exam table", "medicine cabinet", "reception desk"],
    "gym": ["treadmill", "weight rack", "locker"],
    "entertainment": ["stage", "seating", "sound booth"],
    "cafe": ["espresso machine", "counter", "table"],
    "community": ["lounge chair", "bulletin board", "table"],
}


# ──────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────

class LayoutStrategy(ABC):
    @abstractmethod
    def generate(self, width: int, height: int, rng: random.Random,
                 **kwargs: Any) -> WorldConfig:
        ...


# ══════════════════════════════════════════════════════════════════════
# VILLAGE LAYOUT
# ══════════════════════════════════════════════════════════════════════

class VillageLayout(LayoutStrategy):
    """Organic village: winding paths, scattered buildings, town square, park."""

    def generate(self, width: int, height: int, rng: random.Random,
                 **kwargs: Any) -> WorldConfig:
        name = kwargs.get("world_name") or rng.choice([
            "Oakwood Village", "Willowbrook", "Riverbend", "Maple Hollow",
            "Hearthstone", "Pinecrest", "Eldergrove", "Sunvale",
        ])
        cfg = WorldConfig(
            world_name=name, width=width, height=height,
            tile_size=kwargs.get("tile_size", 16), layout_type="village",
            palette=PaletteConfig(
                road=(165, 140, 100),
                sidewalk=(145, 130, 100),
                grass=(75, 150, 65),
                park=(50, 145, 55),
                park_path=(190, 175, 140),
                building=(140, 120, 90),
                block=(75, 150, 65),
                fence=(130, 100, 60),
            ),
        )

        cx, cy = width // 2, height // 2

        # --- town square zone ---
        sq_w = rng.randint(6, 10)
        sq_h = rng.randint(5, 8)
        sq_x = cx - sq_w // 2
        sq_y = cy - sq_h // 2
        cfg.zones.append(ZoneConfig("Town Square", sq_x, sq_y, sq_w, sq_h,
                                    zone_type="plaza", collision=False))

        # --- main roads (cross through center) ---
        road_thickness = 2
        cfg.roads.append(RoadConfig("Main Road", 0, cy - road_thickness // 2,
                                    width, road_thickness, "horizontal"))
        cfg.roads.append(RoadConfig("Cross Road", cx - road_thickness // 2, 0,
                                    road_thickness, height, "vertical"))

        # --- park ---
        park_w = rng.randint(8, 14)
        park_h = rng.randint(6, 10)
        park_quadrant = rng.choice(["NE", "NW", "SE", "SW"])
        if park_quadrant == "NE":
            park_x = cx + rng.randint(3, 6)
            park_y = rng.randint(2, max(3, cy - park_h - 3))
        elif park_quadrant == "NW":
            park_x = rng.randint(2, max(3, cx - park_w - 3))
            park_y = rng.randint(2, max(3, cy - park_h - 3))
        elif park_quadrant == "SE":
            park_x = cx + rng.randint(3, 6)
            park_y = cy + rng.randint(3, 6)
        else:
            park_x = rng.randint(2, max(3, cx - park_w - 3))
            park_y = cy + rng.randint(3, 6)

        park_x = max(1, min(park_x, width - park_w - 1))
        park_y = max(1, min(park_y, height - park_h - 1))

        trees = []
        for _ in range(rng.randint(5, 12)):
            tx = park_x + rng.randint(1, park_w - 2)
            ty = park_y + rng.randint(1, park_h - 2)
            trees.append((tx, ty))

        park_paths = [
            {"x": park_x + park_w // 2, "y": park_y, "w": 1, "h": park_h},
            {"x": park_x, "y": park_y + park_h // 2, "w": park_w, "h": 1},
        ]
        cfg.parks.append(ParkConfig(
            rng.choice(["Village Green", "Willow Park", "Old Pond Park",
                        "Meadow Garden", "Founders' Park"]),
            park_x, park_y, park_w, park_h, trees, park_paths,
        ))

        # --- buildings ---
        placed: List[Tuple[int, int, int, int]] = [
            (sq_x, sq_y, sq_w, sq_h),
            (park_x, park_y, park_w, park_h),
        ]
        building_pool = list(_VILLAGE_BUILDING_POOL)
        rng.shuffle(building_pool)
        n_buildings = kwargs.get("n_buildings", rng.randint(10, 18))

        for i in range(min(n_buildings, len(building_pool))):
            bname, btype = building_pool[i]
            bw = rng.randint(3, 6)
            bh = rng.randint(3, 5)
            zone_name = self._pick_zone_name(rng, bname)

            for _ in range(80):
                bx = rng.randint(2, width - bw - 2)
                by = rng.randint(2, height - bh - 2)
                if any(_rects_overlap(bx, by, bw, bh, *r, margin=2) for r in placed):
                    continue
                if _rects_overlap(bx, by, bw, bh, 0, cy - 1, width, road_thickness + 2, margin=0):
                    continue
                if _rects_overlap(bx, by, bw, bh, cx - 1, 0, road_thickness + 2, height, margin=0):
                    continue
                placed.append((bx, by, bw, bh))
                objs = _VILLAGE_OBJECTS.get(btype, ["table", "chair"])
                cfg.buildings.append(BuildingConfig(
                    name=bname, x=bx, y=by, w=bw, h=bh,
                    building_type=btype,
                    color=_pick_color(rng, cfg.palette.building, 35),
                    zone=zone_name,
                    game_objects=list(objs),
                ))
                zone_exists = any(z.name == zone_name for z in cfg.zones)
                if not zone_exists:
                    zx = max(0, bx - 1)
                    zy = max(0, by - 1)
                    zw = min(bw + 2, width - zx)
                    zh = min(bh + 2, height - zy)
                    cfg.zones.append(ZoneConfig(zone_name, zx, zy, zw, zh,
                                                zone_type="district", collision=False))
                break

        # --- spawn points ---
        for si in range(kwargs.get("n_spawns", 6)):
            for _ in range(40):
                sx = rng.randint(1, width - 2)
                sy = rng.randint(1, height - 2)
                if not any(_rects_overlap(sx, sy, 1, 1, *r) for r in placed):
                    cfg.spawns.append(SpawnConfig(f"sp-{si + 1}", sx, sy))
                    break

        return cfg

    @staticmethod
    def _pick_zone_name(rng: random.Random, building_name: str) -> str:
        prefixes = ["North", "South", "East", "West", "Old", "New", "Upper", "Lower"]
        suffixes = ["Quarter", "District", "Lane", "Row", "Corner"]
        return f"{rng.choice(prefixes)} {rng.choice(suffixes)}"


# ══════════════════════════════════════════════════════════════════════
# URBAN LAYOUT
# ══════════════════════════════════════════════════════════════════════

class UrbanLayout(LayoutStrategy):
    """Grid-based city: blocks separated by roads, intersections, car lanes."""

    def generate(self, width: int, height: int, rng: random.Random,
                 **kwargs: Any) -> WorldConfig:
        name = kwargs.get("world_name") or rng.choice([
            "Metro City", "Harborview", "Steel Heights", "Newburgh",
            "Riverton", "Summit City", "Bayshore", "Iron Gate",
        ])
        cfg = WorldConfig(
            world_name=name, width=width, height=height,
            tile_size=kwargs.get("tile_size", 16), layout_type="urban",
            palette=PaletteConfig(
                road=(70, 70, 75), sidewalk=(160, 160, 155),
                block=(55, 55, 65), building=(90, 85, 100),
                park=(45, 140, 55), park_path=(180, 170, 140),
                crosswalk=(220, 220, 215), lane_mark=(200, 200, 60),
            ),
        )

        road_w = kwargs.get("road_width", 3)
        block_w = kwargs.get("block_width", rng.randint(10, 14))
        block_h = kwargs.get("block_height", rng.randint(8, 11))

        cols = max(1, (width - road_w) // (block_w + road_w))
        rows = max(1, (height - road_w) // (block_h + road_w))

        x_offset = (width - (cols * block_w + (cols + 1) * road_w)) // 2
        y_offset = (height - (rows * block_h + (rows + 1) * road_w)) // 2
        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)

        # pick one block to be a park
        park_row = rng.randint(0, rows - 1)
        park_col = rng.randint(0, cols - 1)

        block_idx = 0
        block_letter = ord("A")
        building_pool = list(_URBAN_BUILDING_POOL)
        rng.shuffle(building_pool)
        bp_idx = 0

        for r in range(rows):
            for c in range(cols):
                bx = x_offset + road_w + c * (block_w + road_w)
                by = y_offset + road_w + r * (block_h + road_w)
                bx = min(bx, width - block_w)
                by = min(by, height - block_h)

                if r == park_row and c == park_col:
                    trees = []
                    for _ in range(rng.randint(6, 14)):
                        tx = bx + rng.randint(1, block_w - 2)
                        ty = by + rng.randint(1, block_h - 2)
                        trees.append((tx, ty))
                    park_paths = [
                        {"x": bx + block_w // 2, "y": by, "w": 2, "h": block_h},
                        {"x": bx, "y": by + block_h // 2, "w": block_w, "h": 2},
                    ]
                    cfg.parks.append(ParkConfig(
                        rng.choice(["Central Park", "City Park", "Liberty Square",
                                    "Union Park", "Memorial Green"]),
                        bx, by, block_w, block_h, trees, park_paths,
                    ))
                    continue

                zone_name = f"Block {chr(block_letter)}"
                block_letter += 1
                if block_letter > ord("Z"):
                    block_letter = ord("A")
                cfg.zones.append(ZoneConfig(zone_name, bx, by, block_w, block_h,
                                            zone_type="block", collision=True))

                inner_placed: List[Tuple[int, int, int, int]] = []
                n_bldg = rng.randint(2, 4)
                for _ in range(n_bldg):
                    if bp_idx >= len(building_pool):
                        rng.shuffle(building_pool)
                        bp_idx = 0
                    bname, btype = building_pool[bp_idx]
                    bp_idx += 1

                    bld_w = rng.randint(3, min(6, block_w - 2))
                    bld_h = rng.randint(2, min(4, block_h - 2))

                    for _ in range(30):
                        bld_x = bx + rng.randint(1, max(1, block_w - bld_w - 1))
                        bld_y = by + rng.randint(1, max(1, block_h - bld_h - 1))
                        if any(_rects_overlap(bld_x, bld_y, bld_w, bld_h, *ir, margin=1)
                               for ir in inner_placed):
                            continue
                        inner_placed.append((bld_x, bld_y, bld_w, bld_h))
                        cfg.buildings.append(BuildingConfig(
                            name=bname, x=bld_x, y=bld_y, w=bld_w, h=bld_h,
                            building_type=btype,
                            color=_pick_color(rng, cfg.palette.building, 50),
                            zone=zone_name,
                        ))
                        break

                block_idx += 1

        # --- intersections at road crossings ---
        for r in range(rows + 1):
            for c in range(cols + 1):
                ix = x_offset + c * (block_w + road_w)
                iy = y_offset + r * (block_h + road_w)
                street_names = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]
                ave_names = ["Main", "Central", "Park", "South", "North", "Oak", "Elm", "Pine"]
                sn = street_names[c % len(street_names)]
                an = ave_names[r % len(ave_names)]
                cfg.intersections.append(IntersectionConfig(
                    f"{sn} & {an}", ix, iy, road_w, road_w,
                ))

        # --- car lanes along every road ---
        for r in range(rows + 1):
            ry = y_offset + r * (block_h + road_w) + road_w // 2
            if 0 <= ry < height:
                cfg.car_lanes.append(CarLaneConfig(
                    f"H-{r} East", [(0, ry), (width - 1, ry)],
                    _pick_color(rng, (180, 60, 60), 40),
                ))
                cfg.car_lanes.append(CarLaneConfig(
                    f"H-{r} West", [(width - 1, ry + 1), (0, ry + 1)],
                    _pick_color(rng, (60, 60, 180), 40),
                ))
        for c in range(cols + 1):
            cx = x_offset + c * (block_w + road_w) + road_w // 2
            if 0 <= cx < width:
                cfg.car_lanes.append(CarLaneConfig(
                    f"V-{c} South", [(cx, 0), (cx, height - 1)],
                    _pick_color(rng, (60, 160, 60), 40),
                ))
                cfg.car_lanes.append(CarLaneConfig(
                    f"V-{c} North", [(cx + 1, height - 1), (cx + 1, 0)],
                    _pick_color(rng, (160, 160, 60), 40),
                ))

        # --- spawns on sidewalks near buildings ---
        for si in range(kwargs.get("n_spawns", 8)):
            for _ in range(60):
                sx = rng.randint(1, width - 2)
                sy = rng.randint(1, height - 2)
                on_road = False
                for zone in cfg.zones:
                    if zone.x <= sx < zone.x + zone.w and zone.y <= sy < zone.y + zone.h:
                        on_road = True
                        break
                for park in cfg.parks:
                    if park.x <= sx < park.x + park.w and park.y <= sy < park.y + park.h:
                        on_road = True
                        break
                if not on_road:
                    cfg.spawns.append(SpawnConfig(f"sp-{si + 1}", sx, sy))
                    break

        return cfg


# ══════════════════════════════════════════════════════════════════════
# CAMPUS LAYOUT
# ══════════════════════════════════════════════════════════════════════

class CampusLayout(LayoutStrategy):
    """University campus: central quad, academic buildings, dorms, paths."""

    def generate(self, width: int, height: int, rng: random.Random,
                 **kwargs: Any) -> WorldConfig:
        name = kwargs.get("world_name") or rng.choice([
            "Oak Hill College", "Riverside University", "Summit Institute",
            "Greenfield Academy", "Crestwood University", "Lakeside College",
        ])
        cfg = WorldConfig(
            world_name=name, width=width, height=height,
            tile_size=kwargs.get("tile_size", 16), layout_type="campus",
            palette=PaletteConfig(
                road=(150, 145, 135),
                sidewalk=(180, 175, 165),
                grass=(70, 155, 60),
                park=(55, 160, 50),
                park_path=(195, 185, 155),
                building=(130, 115, 95),
                parking=(95, 95, 100),
            ),
        )

        cx, cy = width // 2, height // 2

        # --- central quad ---
        quad_w = rng.randint(max(6, width // 5), max(8, width // 3))
        quad_h = rng.randint(max(5, height // 5), max(7, height // 3))
        quad_x = cx - quad_w // 2
        quad_y = cy - quad_h // 2

        trees = []
        for _ in range(rng.randint(6, 15)):
            tx = quad_x + rng.randint(1, quad_w - 2)
            ty = quad_y + rng.randint(1, quad_h - 2)
            trees.append((tx, ty))

        cfg.parks.append(ParkConfig(
            "Main Quad", quad_x, quad_y, quad_w, quad_h, trees,
            [
                {"x": quad_x + quad_w // 2, "y": quad_y, "w": 2, "h": quad_h},
                {"x": quad_x, "y": quad_y + quad_h // 2, "w": quad_w, "h": 2},
            ],
        ))

        # --- campus perimeter paths ---
        path_y_top = max(1, quad_y - 2)
        path_y_bot = min(height - 2, quad_y + quad_h + 1)
        cfg.roads.append(RoadConfig("North Walk", 1, path_y_top, width - 2, 1, "horizontal"))
        cfg.roads.append(RoadConfig("South Walk", 1, path_y_bot, width - 2, 1, "horizontal"))
        path_x_left = max(1, quad_x - 2)
        path_x_right = min(width - 2, quad_x + quad_w + 1)
        cfg.roads.append(RoadConfig("West Walk", path_x_left, 1, 1, height - 2, "vertical"))
        cfg.roads.append(RoadConfig("East Walk", path_x_right, 1, 1, height - 2, "vertical"))

        # --- buildings around the quad perimeter ---
        placed: List[Tuple[int, int, int, int]] = [
            (quad_x, quad_y, quad_w, quad_h),
        ]
        building_pool = list(_CAMPUS_BUILDING_POOL)
        rng.shuffle(building_pool)
        n_buildings = kwargs.get("n_buildings", rng.randint(10, 16))

        regions = [
            ("North Campus", 2, 2, width - 4, max(1, quad_y - 4)),
            ("South Campus", 2, quad_y + quad_h + 3, width - 4,
             max(1, height - quad_y - quad_h - 5)),
            ("West Campus", 2, 2, max(1, quad_x - 4), height - 4),
            ("East Campus", quad_x + quad_w + 3, 2,
             max(1, width - quad_x - quad_w - 5), height - 4),
        ]

        for i in range(min(n_buildings, len(building_pool))):
            bname, btype = building_pool[i]
            bw = rng.randint(4, 7)
            bh = rng.randint(3, 5)
            region_name, rx, ry, rw, rh = regions[i % len(regions)]

            for _ in range(60):
                bx = rx + rng.randint(0, max(0, rw - bw))
                by = ry + rng.randint(0, max(0, rh - bh))
                bx = max(1, min(bx, width - bw - 1))
                by = max(1, min(by, height - bh - 1))
                if any(_rects_overlap(bx, by, bw, bh, *r, margin=2) for r in placed):
                    continue
                placed.append((bx, by, bw, bh))
                objs = _CAMPUS_OBJECTS.get(btype, ["desk", "chair"])
                cfg.buildings.append(BuildingConfig(
                    name=bname, x=bx, y=by, w=bw, h=bh,
                    building_type=btype,
                    color=_pick_color(rng, cfg.palette.building, 30),
                    zone=region_name,
                    game_objects=list(objs),
                ))
                zone_exists = any(z.name == region_name for z in cfg.zones)
                if not zone_exists:
                    cfg.zones.append(ZoneConfig(
                        region_name, rx, ry, rw, rh,
                        zone_type="district", collision=False,
                    ))
                break

        # --- parking lot ---
        lot_w = rng.randint(5, 8)
        lot_h = rng.randint(3, 5)
        for _ in range(40):
            lx = rng.randint(1, width - lot_w - 1)
            ly = rng.randint(1, height - lot_h - 1)
            if not any(_rects_overlap(lx, ly, lot_w, lot_h, *r, margin=1) for r in placed):
                placed.append((lx, ly, lot_w, lot_h))
                cfg.zones.append(ZoneConfig("Parking Lot", lx, ly, lot_w, lot_h,
                                            zone_type="parking", collision=False))
                break

        # --- spawns ---
        for si in range(kwargs.get("n_spawns", 8)):
            for _ in range(40):
                sx = rng.randint(quad_x, quad_x + quad_w - 1)
                sy = rng.randint(quad_y, quad_y + quad_h - 1)
                cfg.spawns.append(SpawnConfig(f"sp-{si + 1}", sx, sy))
                break

        return cfg


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────

LAYOUT_REGISTRY: Dict[str, LayoutStrategy] = {
    "village": VillageLayout(),
    "urban": UrbanLayout(),
    "campus": CampusLayout(),
}
