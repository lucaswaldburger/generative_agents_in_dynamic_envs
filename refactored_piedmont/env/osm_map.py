"""
OSMMap – Downloads OpenStreetMap data for a given place, rasterizes streets
and buildings to a 2-D tile grid, labels buildings with real names/types,
and exposes the same interface as refactored_city's CityMap / SmallVille's Maze.

Grid value encoding:
    0 = background (collision)
    1 = street (walkable)
    2 = building (walkable)
    3 = traffic signal / stop sign (walkable)
    4 = safe zone — park / open space (walkable)
    5 = water (collision)
    6 = railway (collision)
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from shapely.geometry import Point

# grid value constants
GRID_BG = 0
GRID_STREET = 1
GRID_BUILDING = 2
GRID_INTERSECTION = 3
GRID_SAFE_ZONE = 4
GRID_WATER = 5
GRID_RAILWAY = 6


def _safe_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val)


_CACHE_VERSION = 3

def _cache_path(place_name: str, grid_size: int, cache_dir: Path) -> Path:
    key = f"{place_name}|{grid_size}|v{_CACHE_VERSION}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return cache_dir / f"osm_cache_{h}.json"


def _coord_to_grid(
    x: float, y: float,
    min_x: float, max_x: float,
    min_y: float, max_y: float,
    size: int,
) -> Tuple[int, int]:
    """Convert lon/lat to (row, col) in the raster grid."""
    norm_x = (x - min_x) / (max_x - min_x) if max_x != min_x else 0.5
    norm_y = (y - min_y) / (max_y - min_y) if max_y != min_y else 0.5
    col = int(norm_x * (size - 1))
    row = int((1 - norm_y) * (size - 1))
    col = max(0, min(size - 1, col))
    row = max(0, min(size - 1, row))
    return row, col


def _nearest_street_name(
    centroid_x: float, centroid_y: float,
    edge_segments: List[Tuple[str, float, float]],
) -> str:
    """Find the name of the nearest street edge to a lon/lat point."""
    best_name = "Unknown Street"
    best_dist = float("inf")
    for name, ex, ey in edge_segments:
        d = (centroid_x - ex) ** 2 + (centroid_y - ey) ** 2
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


ROAD_HALF_WIDTH: Dict[str, int] = {
    "motorway": 2, "motorway_link": 2,
    "trunk": 2, "trunk_link": 2,
    "primary": 1, "primary_link": 1,
    "secondary": 1, "secondary_link": 1,
    "tertiary": 1, "tertiary_link": 1,
    "residential": 0,
    "unclassified": 0,
    "living_street": 0,
    "service": 0,
    "footway": 0,
    "path": 0,
    "cycleway": 0,
    "pedestrian": 0,
    "steps": 0,
    "track": 0,
}

POP_DENSITY: Dict[str, float] = {
    "apartments": 4.0,
    "residential": 2.0,
    "house": 2.0,
    "detached": 1.5,
    "commercial": 1.0,
    "office": 3.0,
    "retail": 1.0,
    "industrial": 0.5,
    "school": 10.0,
    "hospital": 5.0,
    "church": 2.0,
    "building": 1.5,
}


class OSMMap:
    """2-D tile map built from OpenStreetMap data for a given place."""

    TILE_PX = 4

    def __init__(
        self,
        place_name: str = "Piedmont, California, USA",
        grid_size: int = 250,
        cache_dir: Optional[str | Path] = None,
    ):
        self.place_name = place_name
        self.grid_size = grid_size
        self.maze_width = grid_size
        self.maze_height = grid_size
        self.sq_tile_size = self.TILE_PX

        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        cached = self._load_cache()
        if cached is not None:
            self._from_cache(cached)
        else:
            self._build_from_osm()
            self._save_cache()

        self._build_tile_structures()

    # ------------------------------------------------------------------
    # OSM download + rasterisation
    # ------------------------------------------------------------------

    def _build_from_osm(self):
        import osmnx as ox
        from skimage.draw import line, polygon as skpoly
        import geopandas as gpd

        print(f"Downloading streets for '{self.place_name}'...")
        G = ox.graph_from_place(self.place_name, network_type="walk")
        nodes, edges = ox.graph_to_gdfs(G)

        print("Downloading buildings...")
        buildings = ox.features_from_place(self.place_name, tags={"building": True})

        print("Downloading POIs (amenities, shops, leisure)...")
        pois = gpd.GeoDataFrame()
        for tag_key in ("amenity", "shop", "leisure"):
            try:
                feat = ox.features_from_place(self.place_name, tags={tag_key: True})
                if not feat.empty:
                    feat["_poi_type_key"] = tag_key
                    pois = gpd.pd.concat([pois, feat], ignore_index=True)
            except Exception:
                pass

        print("Downloading traffic signals and stop signs...")
        try:
            traffic_controls = ox.features_from_place(
                self.place_name,
                tags={"highway": ["traffic_signals", "stop"]},
            )
        except Exception:
            traffic_controls = gpd.GeoDataFrame()

        print("Downloading parks and open spaces...")
        safe_zone_polys = gpd.GeoDataFrame()
        for tag_dict in ({"leisure": "park"}, {"landuse": "grass"}):
            try:
                feat = ox.features_from_place(self.place_name, tags=tag_dict)
                if not feat.empty:
                    safe_zone_polys = gpd.pd.concat(
                        [safe_zone_polys, feat], ignore_index=True
                    )
            except Exception:
                pass

        print("Downloading water bodies and waterways...")
        water_polys = gpd.GeoDataFrame()
        for tag_dict in ({"natural": "water"}, {"water": True}):
            try:
                feat = ox.features_from_place(self.place_name, tags=tag_dict)
                if not feat.empty:
                    water_polys = gpd.pd.concat(
                        [water_polys, feat], ignore_index=True
                    )
            except Exception:
                pass
        waterways = gpd.GeoDataFrame()
        try:
            waterways = ox.features_from_place(
                self.place_name, tags={"waterway": True}
            )
        except Exception:
            pass

        print("Downloading railways...")
        railways = gpd.GeoDataFrame()
        try:
            railways = ox.features_from_place(
                self.place_name, tags={"railway": "rail"}
            )
        except Exception:
            pass

        min_x, max_x = nodes["x"].min(), nodes["x"].max()
        min_y, max_y = nodes["y"].min(), nodes["y"].max()
        sz = self.grid_size

        grid = np.zeros((sz, sz), dtype=int)
        street_width_grid = np.zeros((sz, sz), dtype=int)

        # --- collect named street edges for sector assignment ---
        named_edge_points: List[Tuple[str, float, float]] = []
        for _, row_data in edges.iterrows():
            street_name = _safe_str(row_data.get("name", ""))
            if not street_name:
                continue
            u, v = row_data.name[0], row_data.name[1]
            mx = (nodes.loc[u, "x"] + nodes.loc[v, "x"]) / 2
            my = (nodes.loc[u, "y"] + nodes.loc[v, "y"]) / 2
            named_edge_points.append((street_name, mx, my))

        # --- rasterize streets with road-hierarchy widths ---
        print("Rasterizing streets (with road hierarchy)...")
        for _, row_data in edges.iterrows():
            u, v = row_data.name[0], row_data.name[1]
            r0, c0 = _coord_to_grid(
                nodes.loc[u, "x"], nodes.loc[u, "y"],
                min_x, max_x, min_y, max_y, sz,
            )
            r1, c1 = _coord_to_grid(
                nodes.loc[v, "x"], nodes.loc[v, "y"],
                min_x, max_x, min_y, max_y, sz,
            )
            rr, cc = line(r0, c0, r1, c1)

            hw_tag = _safe_str(row_data.get("highway", ""))
            if isinstance(hw_tag, list):
                hw_tag = hw_tag[0] if hw_tag else ""
            half_w = ROAD_HALF_WIDTH.get(hw_tag, 0)
            pixel_width = 1 + 2 * half_w

            for pr, pc in zip(rr, cc):
                for dr in range(-half_w, half_w + 1):
                    for dc in range(-half_w, half_w + 1):
                        nr, nc = pr + dr, pc + dc
                        if 0 <= nr < sz and 0 <= nc < sz:
                            if grid[nr, nc] == GRID_BG:
                                grid[nr, nc] = GRID_STREET
                            street_width_grid[nr, nc] = max(
                                street_width_grid[nr, nc], pixel_width
                            )

        # --- rasterize safe zones (parks / open space) before buildings ---
        print("Rasterizing safe zones...")
        safe_zone_meta: List[Dict[str, Any]] = []
        if not safe_zone_polys.empty:
            for _, sz_row in safe_zone_polys.iterrows():
                geom = sz_row.geometry
                if geom is None:
                    continue
                polys = []
                if geom.geom_type == "Polygon":
                    polys = [geom]
                elif geom.geom_type == "MultiPolygon":
                    polys = list(geom.geoms)
                for poly in polys:
                    x_coords, y_coords = poly.exterior.xy
                    grid_rows, grid_cols = [], []
                    for x, y in zip(x_coords, y_coords):
                        r, c = _coord_to_grid(x, y, min_x, max_x, min_y, max_y, sz)
                        grid_rows.append(r)
                        grid_cols.append(c)
                    fill_rr, fill_cc = skpoly(grid_rows, grid_cols, shape=grid.shape)
                    if len(fill_rr) == 0:
                        continue
                    mask = grid[fill_rr, fill_cc] == GRID_BG
                    grid[fill_rr[mask], fill_cc[mask]] = GRID_SAFE_ZONE

                    name = _safe_str(sz_row.get("name", ""))
                    if not name:
                        name = "Open Space"
                    centroid = poly.centroid
                    cr, ccol = _coord_to_grid(
                        centroid.x, centroid.y, min_x, max_x, min_y, max_y, sz
                    )
                    safe_zone_meta.append({
                        "name": name,
                        "centroid_row": int(cr),
                        "centroid_col": int(ccol),
                    })

        # --- rasterize buildings and collect metadata ---
        print("Rasterizing buildings...")
        building_meta: List[Dict[str, Any]] = []
        bld_idx = 0

        poi_lookup: Dict[int, Dict[str, str]] = {}
        if not pois.empty and not buildings.empty:
            try:
                for poi_i, poi_row in pois.iterrows():
                    poi_geom = poi_row.geometry
                    if poi_geom is None:
                        continue
                    poi_pt = poi_geom.centroid if poi_geom.geom_type != "Point" else poi_geom
                    poi_name = _safe_str(poi_row.get("name", ""))
                    poi_type = _safe_str(poi_row.get("_poi_type_key", ""))
                    poi_val = _safe_str(poi_row.get(poi_type, "")) if poi_type else ""
                    if not poi_name:
                        continue
                    for bld_i, bld_row in buildings.iterrows():
                        if bld_row.geometry is not None and bld_row.geometry.contains(poi_pt):
                            poi_lookup[bld_i] = {"name": poi_name, "type": poi_val}
                            break
            except Exception:
                pass

        for bld_i, bld_row in buildings.iterrows():
            geom = bld_row.geometry
            if geom is None or geom.geom_type != "Polygon":
                continue

            x_coords, y_coords = geom.exterior.xy
            grid_rows, grid_cols = [], []
            for x, y in zip(x_coords, y_coords):
                r, c = _coord_to_grid(x, y, min_x, max_x, min_y, max_y, sz)
                grid_rows.append(r)
                grid_cols.append(c)

            rr, cc = skpoly(grid_rows, grid_cols, shape=grid.shape)
            if len(rr) == 0:
                continue
            grid[rr, cc] = GRID_BUILDING

            centroid = geom.centroid
            cx, cy = centroid.x, centroid.y

            raw_name = _safe_str(bld_row.get("name", ""))
            raw_type = _safe_str(bld_row.get("building", ""))
            if raw_type == "yes":
                raw_type = "building"

            if bld_i in poi_lookup:
                poi = poi_lookup[bld_i]
                if not raw_name:
                    raw_name = poi["name"]
                if raw_type in ("", "building") and poi["type"]:
                    raw_type = poi["type"]

            sector = _nearest_street_name(cx, cy, named_edge_points) if named_edge_points else "Piedmont"

            if not raw_name:
                bld_idx += 1
                raw_name = f"Building #{bld_idx}"

            cr, ccol = _coord_to_grid(cx, cy, min_x, max_x, min_y, max_y, sz)

            levels_raw = _safe_str(bld_row.get("building:levels", ""))
            try:
                levels = max(1, int(float(levels_raw)))
            except (ValueError, TypeError):
                levels = 1 if raw_type in ("house", "detached") else 2

            btype = raw_type or "building"
            tile_count = max(1, len(rr))
            density = POP_DENSITY.get(btype, 1.5)
            population = max(1, int(levels * tile_count * density))

            is_safe = btype in ("hospital", "school")

            building_meta.append({
                "name": raw_name,
                "type": btype,
                "sector": sector,
                "centroid_row": int(cr),
                "centroid_col": int(ccol),
                "rows": rr.tolist(),
                "cols": cc.tolist(),
                "levels": levels,
                "population": population,
                "safe_zone": is_safe,
            })

        # --- rasterize water bodies and waterways (value = 5) ---
        print("Rasterizing water...")
        if not water_polys.empty:
            for _, w_row in water_polys.iterrows():
                geom = w_row.geometry
                if geom is None:
                    continue
                polys = []
                if geom.geom_type == "Polygon":
                    polys = [geom]
                elif geom.geom_type == "MultiPolygon":
                    polys = list(geom.geoms)
                for poly in polys:
                    x_coords, y_coords = poly.exterior.xy
                    grid_rows, grid_cols = [], []
                    for x, y in zip(x_coords, y_coords):
                        r, c = _coord_to_grid(x, y, min_x, max_x, min_y, max_y, sz)
                        grid_rows.append(r)
                        grid_cols.append(c)
                    fill_rr, fill_cc = skpoly(grid_rows, grid_cols, shape=grid.shape)
                    if len(fill_rr) > 0:
                        grid[fill_rr, fill_cc] = GRID_WATER

        if not waterways.empty:
            for _, w_row in waterways.iterrows():
                geom = w_row.geometry
                if geom is None:
                    continue
                coords = []
                if geom.geom_type == "LineString":
                    coords = list(geom.coords)
                elif geom.geom_type == "MultiLineString":
                    for ls in geom.geoms:
                        coords.extend(list(ls.coords))
                for i in range(len(coords) - 1):
                    r0, c0 = _coord_to_grid(
                        coords[i][0], coords[i][1],
                        min_x, max_x, min_y, max_y, sz
                    )
                    r1, c1 = _coord_to_grid(
                        coords[i + 1][0], coords[i + 1][1],
                        min_x, max_x, min_y, max_y, sz
                    )
                    rr, cc = line(r0, c0, r1, c1)
                    for pr, pc in zip(rr, cc):
                        for dr in range(-1, 2):
                            for dc in range(-1, 2):
                                nr, nc = pr + dr, pc + dc
                                if 0 <= nr < sz and 0 <= nc < sz:
                                    grid[nr, nc] = GRID_WATER

        # --- rasterize railways (value = 6) ---
        print("Rasterizing railways...")
        if not railways.empty:
            for _, r_row in railways.iterrows():
                geom = r_row.geometry
                if geom is None:
                    continue
                coords = []
                if geom.geom_type == "LineString":
                    coords = list(geom.coords)
                elif geom.geom_type == "MultiLineString":
                    for ls in geom.geoms:
                        coords.extend(list(ls.coords))
                for i in range(len(coords) - 1):
                    r0, c0 = _coord_to_grid(
                        coords[i][0], coords[i][1],
                        min_x, max_x, min_y, max_y, sz
                    )
                    r1, c1 = _coord_to_grid(
                        coords[i + 1][0], coords[i + 1][1],
                        min_x, max_x, min_y, max_y, sz
                    )
                    rr, cc = line(r0, c0, r1, c1)
                    for pr, pc in zip(rr, cc):
                        for dr in range(-1, 2):
                            for dc in range(-1, 2):
                                nr, nc = pr + dr, pc + dc
                                if 0 <= nr < sz and 0 <= nc < sz:
                                    grid[nr, nc] = GRID_RAILWAY

        # --- rasterize traffic signals / stop signs (value = 3) ---
        print("Rasterizing traffic signals...")
        intersection_meta: List[Dict[str, Any]] = []
        if not traffic_controls.empty:
            for _, ctrl in traffic_controls.iterrows():
                geom = ctrl.geometry
                if geom is None or geom.geom_type != "Point":
                    continue
                x, y = geom.x, geom.y
                r, c = _coord_to_grid(x, y, min_x, max_x, min_y, max_y, sz)

                hw_type = _safe_str(ctrl.get("highway", ""))
                signal_type = "traffic_signals" if hw_type == "traffic_signals" else "stop"

                r_lo = max(0, r - 1)
                r_hi = min(sz, r + 2)
                c_lo = max(0, c - 1)
                c_hi = min(sz, c + 2)
                grid[r_lo:r_hi, c_lo:c_hi] = GRID_INTERSECTION

                intersection_meta.append({
                    "type": signal_type,
                    "row": int(r),
                    "col": int(c),
                })

        total_pop = sum(bm["population"] for bm in building_meta)
        print(
            f"Rasterized {len(building_meta)} buildings (total pop ~{total_pop}), "
            f"{len(safe_zone_meta)} safe zones, "
            f"{len(intersection_meta)} traffic controls, grid {sz}x{sz}"
        )

        self._grid = grid
        self._street_width_grid = street_width_grid
        self._building_meta = building_meta
        self._intersection_meta = intersection_meta
        self._safe_zone_meta = safe_zone_meta
        self._bounds = (float(min_x), float(max_x), float(min_y), float(max_y))

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _save_cache(self):
        cp = _cache_path(self.place_name, self.grid_size, self.cache_dir)
        data = {
            "place_name": self.place_name,
            "grid_size": self.grid_size,
            "grid": self._grid.tolist(),
            "street_width_grid": self._street_width_grid.tolist(),
            "building_meta": self._building_meta,
            "intersection_meta": self._intersection_meta,
            "safe_zone_meta": self._safe_zone_meta,
            "bounds": list(self._bounds),
        }
        cp.write_text(json.dumps(data))
        print(f"Cache saved to {cp}")

    def _load_cache(self) -> Optional[Dict]:
        cp = _cache_path(self.place_name, self.grid_size, self.cache_dir)
        if not cp.exists():
            return None
        try:
            data = json.loads(cp.read_text())
            if data.get("grid_size") != self.grid_size:
                return None
            print(f"Loaded cached map from {cp}")
            return data
        except Exception:
            return None

    def _from_cache(self, data: Dict):
        self._grid = np.array(data["grid"], dtype=int)
        self._street_width_grid = np.array(
            data.get("street_width_grid", np.zeros_like(self._grid)), dtype=int
        )
        self._building_meta = data["building_meta"]
        self._intersection_meta = data.get("intersection_meta", [])
        self._safe_zone_meta = data.get("safe_zone_meta", [])
        self._bounds = tuple(data["bounds"])

    # ------------------------------------------------------------------
    # Build tile / collision / address structures
    # ------------------------------------------------------------------

    def _build_tile_structures(self):
        sz = self.grid_size

        pixel_to_bld: Dict[Tuple[int, int], int] = {}
        for idx, bm in enumerate(self._building_meta):
            for r, c in zip(bm.get("rows", []), bm.get("cols", [])):
                pixel_to_bld[(r, c)] = idx

        self.tiles: List[List[Dict[str, Any]]] = []
        self.collision_maze: List[List[str]] = []

        intersection_lookup: Dict[Tuple[int, int], str] = {}
        for im in self._intersection_meta:
            r, c = im["row"], im["col"]
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    intersection_lookup[(r + dr, c + dc)] = im["type"]

        safe_zone_lookup: Dict[Tuple[int, int], str] = {}
        for szm in self._safe_zone_meta:
            safe_zone_lookup[(szm["centroid_row"], szm["centroid_col"])] = szm["name"]

        for row in range(sz):
            tile_row: List[Dict[str, Any]] = []
            col_row: List[str] = []
            for col in range(sz):
                val = int(self._grid[row, col])
                bld_idx = pixel_to_bld.get((row, col))

                if val == GRID_BG:
                    td = {
                        "world": "Piedmont",
                        "sector": "",
                        "arena": "",
                        "game_object": "",
                        "spawning_location": "",
                        "collision": True,
                        "events": set(),
                        "tile_type": "background",
                    }
                    col_row.append("1")
                elif val == GRID_STREET:
                    td = {
                        "world": "Piedmont",
                        "sector": "Streets",
                        "arena": "",
                        "game_object": "",
                        "spawning_location": "",
                        "collision": False,
                        "events": set(),
                        "tile_type": "street",
                    }
                    col_row.append("0")
                elif val == GRID_INTERSECTION:
                    sig_type = intersection_lookup.get((row, col), "traffic_signals")
                    td = {
                        "world": "Piedmont",
                        "sector": "Streets",
                        "arena": "",
                        "game_object": sig_type,
                        "spawning_location": "",
                        "collision": False,
                        "events": set(),
                        "tile_type": "intersection",
                    }
                    col_row.append("0")
                elif val == GRID_SAFE_ZONE:
                    sz_name = safe_zone_lookup.get((row, col), "")
                    td = {
                        "world": "Piedmont",
                        "sector": "Safe Zones",
                        "arena": sz_name if sz_name else "Open Space",
                        "game_object": "park",
                        "spawning_location": "",
                        "collision": False,
                        "events": set(),
                        "tile_type": "safe_zone",
                    }
                    col_row.append("0")
                elif val == GRID_WATER:
                    td = {
                        "world": "Piedmont",
                        "sector": "",
                        "arena": "",
                        "game_object": "",
                        "spawning_location": "",
                        "collision": True,
                        "events": set(),
                        "tile_type": "water",
                    }
                    col_row.append("1")
                elif val == GRID_RAILWAY:
                    td = {
                        "world": "Piedmont",
                        "sector": "",
                        "arena": "",
                        "game_object": "",
                        "spawning_location": "",
                        "collision": True,
                        "events": set(),
                        "tile_type": "railway",
                    }
                    col_row.append("1")
                elif val == GRID_BUILDING:
                    if bld_idx is not None:
                        bm = self._building_meta[bld_idx]
                        sector = bm["sector"]
                        arena = bm["name"]
                        game_obj = bm["type"]
                        population = bm.get("population", 1)
                        is_safe = bm.get("safe_zone", False)
                    else:
                        sector = ""
                        arena = ""
                        game_obj = ""
                        population = 1
                        is_safe = False

                    td = {
                        "world": "Piedmont",
                        "sector": sector,
                        "arena": arena,
                        "game_object": game_obj,
                        "spawning_location": "",
                        "collision": False,
                        "events": set(),
                        "tile_type": "building",
                        "population": population,
                        "safe_zone": is_safe,
                    }
                    col_row.append("0")
                else:
                    td = {
                        "world": "Piedmont",
                        "sector": "",
                        "arena": "",
                        "game_object": "",
                        "spawning_location": "",
                        "collision": True,
                        "events": set(),
                        "tile_type": "background",
                    }
                    col_row.append("1")

                tile_row.append(td)
            self.tiles.append(tile_row)
            self.collision_maze.append(col_row)

        self.address_tiles: Dict[str, Set[Tuple[int, int]]] = {}
        for row in range(sz):
            for col in range(sz):
                t = self.tiles[row][col]
                addresses: List[str] = []
                if t["sector"]:
                    addresses.append(f'{t["world"]}:{t["sector"]}')
                if t["arena"]:
                    addresses.append(f'{t["world"]}:{t["sector"]}:{t["arena"]}')
                if t["game_object"]:
                    addresses.append(
                        f'{t["world"]}:{t["sector"]}:{t["arena"]}:{t["game_object"]}'
                    )
                for addr in addresses:
                    self.address_tiles.setdefault(addr, set()).add((col, row))

        for bm in self._building_meta:
            bm.pop("rows", None)
            bm.pop("cols", None)

    # ------------------------------------------------------------------
    # Public interface (matches Maze / CityMap)
    # ------------------------------------------------------------------

    def access_tile(self, tile: Tuple[int, int]) -> Dict[str, Any]:
        return self.tiles[tile[1]][tile[0]]

    def get_tile_path(self, tile: Tuple[int, int], level: str) -> str:
        t = self.tiles[tile[1]][tile[0]]
        path = t["world"]
        if level == "world":
            return path
        path += f":{t['sector']}"
        if level == "sector":
            return path
        path += f":{t['arena']}"
        if level == "arena":
            return path
        path += f":{t['game_object']}"
        return path

    def get_nearby_tiles(
        self, tile: Tuple[int, int], vision_r: int
    ) -> List[Tuple[int, int]]:
        left = max(0, tile[0] - vision_r)
        right = min(self.maze_width - 1, tile[0] + vision_r + 1)
        top = max(0, tile[1] - vision_r)
        bottom = min(self.maze_height - 1, tile[1] + vision_r + 1)
        return [(x, y) for x in range(left, right) for y in range(top, bottom)]

    def add_event_from_tile(self, curr_event: Any, tile: Tuple[int, int]):
        self.tiles[tile[1]][tile[0]]["events"].add(curr_event)

    def remove_event_from_tile(self, curr_event: Any, tile: Tuple[int, int]):
        evts = self.tiles[tile[1]][tile[0]]["events"].copy()
        for ev in evts:
            if ev == curr_event:
                self.tiles[tile[1]][tile[0]]["events"].remove(ev)

    def remove_subject_events_from_tile(self, subject: str, tile: Tuple[int, int]):
        evts = self.tiles[tile[1]][tile[0]]["events"].copy()
        for ev in evts:
            if ev[0] == subject:
                self.tiles[tile[1]][tile[0]]["events"].remove(ev)

    def turn_coordinate_to_tile(self, px_coordinate: Tuple[int, int]) -> Tuple[int, int]:
        x = math.ceil(px_coordinate[0] / self.sq_tile_size)
        y = math.ceil(px_coordinate[1] / self.sq_tile_size)
        return (x, y)

    @property
    def named_buildings(self) -> List[Dict[str, Any]]:
        return [
            bm for bm in self._building_meta
            if not bm["name"].startswith("Building #")
        ]

    @property
    def all_buildings(self) -> List[Dict[str, Any]]:
        return list(self._building_meta)
