import json


def _iter_rect_cells(x, y, w, h):
    for cx in range(x, x + w):
        for cy in range(y, y + h):
            yield cx, cy

def load_map(json_path):
    """
    Returns:
      elements: list of (obj_type, rgba_tuple, color_name, (x,y))
      size: (W, H)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    palette = data.get("colors") or data.get("palette") or {}
    wall_c = tuple(palette.get("wall",  (110,130,140)))
    fire_c = tuple(palette.get("fire",  (255,165,0)))      # can be RGBA
    goal_c = tuple(palette.get("goal",  (0,220,0)))
    park_c = tuple(palette.get("park",  goal_c))
    home_c = tuple(palette.get("home",  (230,0,230)))
    work_c = tuple(palette.get("work",  (0,0,255)))

    elements = []

    for w in data.get("walls", []) or data.get("streets_as_walls", []):
        if w.get("shape") == "rect":
            for x, y in _iter_rect_cells(w["x"], w["y"], w["w"], w["h"]):
                elements.append(("wall", wall_c, "grey", (x, y)))

    for l in data.get("lava", []):
        elements.append(("lava", fire_c, "red", (l["x"], l["y"])))

    for g in data.get("goal", []):
        elements.append(("goal", goal_c, "green", (g["x"], g["y"])))


    def region_to_tile(rtype):
        if rtype == "block":
            return ("wall",  wall_c, "grey")
        if rtype == "park":
            return ("floor", park_c, "green")
        if rtype == "home":
            return ("floor", home_c, "purple")
        if rtype == "work":
            return ("floor", work_c, "blue")
        if rtype == "fire":
            return ("lava",  fire_c, "red")
        # default
        return ("floor", goal_c, "yellow")

    for r in data.get("regions", []):
        if r.get("shape") != "rect":
            continue
        obj_type, rgba, color_name = region_to_tile(r.get("type", "block"))
        for x, y in _iter_rect_cells(r["x"], r["y"], r["w"], r["h"]):
            elements.append((obj_type, rgba, color_name, (x, y)))

    width, height = int(data["width"]), int(data["height"])
    return elements, (width, height)
