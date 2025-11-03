import json

def load_map(json_path):
    """
    Loads geometry and colors from a JSON map file.
    Returns:
        elements: list of (obj_type, color_rgb_tuple, (x, y))
        size: (width, height)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    colors = data["colors"]
    elements = []

    # --- walls ---
    for w in data.get("walls", []):
        if w["shape"] == "rect":
            for x in range(w["x"], w["x"] + w["w"]):
                for y in range(w["y"], w["y"] + w["h"]):
                    elements.append(("wall", tuple(colors["wall"]), (x, y)))

    # --- lava (fire) ---
    for l in data.get("lava", []):
        elements.append(("lava", tuple(colors["lava"]), (l["x"], l["y"])))

    # --- goal (safe zones) ---
    for g in data.get("goal", []):
        elements.append(("goal", tuple(colors["goal"]), (g["x"], g["y"])))

    width, height = data["width"], data["height"]
    return elements, (width, height)
