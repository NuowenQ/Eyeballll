#!/usr/bin/env python3
"""Visualise an eyeball map YAML from the backups folder.

Usage:
    python visualise.py eyeball_A.yaml
    python visualise.py eyeball_A          # .yaml is optional
    python visualise.py --list             # list available backups
    python visualise.py eyeball_A --save out.png --no-show

The YAML file is looked up inside ../backups relative to this script; you may
also pass an absolute or relative path to any other YAML file.
"""

import argparse
import math
import os
import sys

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

TITLE = "COURSE"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKUPS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "backups"))


def resolve_yaml_path(name):
    """Resolve a user-supplied name to a YAML file path.

    Tries, in order: the path as given, the path inside the backups folder,
    and the same with a .yaml extension appended.
    """
    candidates = []
    for base in (name, os.path.join(BACKUPS_DIR, name)):
        candidates.append(base)
        if not base.lower().endswith((".yaml", ".yml")):
            candidates.append(base + ".yaml")

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find '{name}'. Looked in {BACKUPS_DIR} and as a direct path.\n"
        f"Available backups: {', '.join(list_backups()) or '(none)'}"
    )


def list_backups():
    """Return the sorted list of YAML filenames in the backups folder."""
    if not os.path.isdir(BACKUPS_DIR):
        return []
    return sorted(
        f for f in os.listdir(BACKUPS_DIR) if f.lower().endswith((".yaml", ".yml"))
    )


def load_coordinates(yaml_file):
    """Load coordinates from a YAML file."""
    print(f"Loading coordinates from: {yaml_file}")
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)
    return data["map"]


def create_octagon_obstacle(coordinates, octagon_radius=1.35):
    """Create octagon obstacle coordinates for the "octagon" point."""
    octagon_point = None
    for coord in coordinates:
        if coord["name"] == "octagon":
            octagon_point = coord
            break

    if octagon_point is None:
        print("Warning: 'octagon' coordinate not found")
        return None

    center_north = octagon_point["x"]
    center_east = octagon_point["y"]

    octagon_vertices = []
    num_sides = 8
    for i in range(num_sides):
        angle = 2 * math.pi * i / num_sides + math.pi / 8
        vertex_north = center_north + octagon_radius * math.cos(angle)
        vertex_east = center_east + octagon_radius * math.sin(angle)
        octagon_vertices.append((vertex_north, vertex_east))

    return {
        "name": "octagon_obstacle",
        "center": (center_north, center_east),
        "vertices": octagon_vertices,
        "radius": octagon_radius,
    }


def create_slalom_gates(coordinates, gate_spacing=2.0, gate_width=3.0,
                        layer_2_offset=0.0, layer_3_offset=0.0):
    """Create slalom gate coordinates for the "slalom" point."""
    slalom_start = None
    for coord in coordinates:
        if coord["name"] == "slalom":
            slalom_start = coord
            break

    if slalom_start is None:
        print("Warning: 'slalom' coordinate not found")
        return []

    yaw_rad = math.radians(slalom_start.get("yaw", 0.0))
    start_north = slalom_start["x"]
    start_east = slalom_start["y"]

    gates = []
    for i in range(3):
        distance_forward = gate_spacing * i
        gate_center_north = start_north + distance_forward * math.cos(yaw_rad)
        gate_center_east = start_east + distance_forward * math.sin(yaw_rad)

        perp_yaw_rad = yaw_rad + math.pi / 2
        half_width = gate_width / 2

        left_north = gate_center_north + half_width * math.cos(perp_yaw_rad)
        left_east = gate_center_east + half_width * math.sin(perp_yaw_rad)
        right_north = gate_center_north - half_width * math.cos(perp_yaw_rad)
        right_east = gate_center_east - half_width * math.sin(perp_yaw_rad)

        gates.append({
            "name": f"slalom_gate_{i + 1}",
            "center": (gate_center_north, gate_center_east),
            "left_end": (left_north, left_east),
            "right_end": (right_north, right_east),
            "gate_number": i + 1,
        })

    for key in ("center", "left_end", "right_end"):
        gates[1][key] = (gates[1][key][0], gates[1][key][1] + layer_2_offset)
        gates[2][key] = (gates[2][key][0], gates[2][key][1] + layer_3_offset)

    return gates


def create_gate_end_obstacle(coordinates, obstacle_setback=2.0, obstacle_half_width=1.5):
    """Create obstacle coordinate for the "gate" point."""
    gate_end = None
    for coord in coordinates:
        if coord["name"] == "gate":
            gate_end = coord
            break

    if gate_end is None:
        print("Warning: 'gate' coordinate not found")
        return None

    yaw_rad = math.radians(gate_end.get("yaw", 0.0))
    gate_north = gate_end["x"]
    gate_east = gate_end["y"]

    setback_north = gate_north - obstacle_setback * math.cos(yaw_rad)
    setback_east = gate_east - obstacle_setback * math.sin(yaw_rad)

    perp_yaw_rad = yaw_rad + math.pi / 2
    obstacle_start_north = setback_north + obstacle_half_width * math.cos(perp_yaw_rad)
    obstacle_start_east = setback_east + obstacle_half_width * math.sin(perp_yaw_rad)
    obstacle_end_north = setback_north - obstacle_half_width * math.cos(perp_yaw_rad)
    obstacle_end_east = setback_east - obstacle_half_width * math.sin(perp_yaw_rad)

    return {
        "name": "gate_end_obstacle",
        "start": (obstacle_start_north, obstacle_start_east),
        "end": (obstacle_end_north, obstacle_end_east),
        "gate_name": "gate_end",
    }


def layout_labels_margin(plot_x, plot_y, names, xlim, ylim,
                         left_frac=0.16, right_frac=0.16, band_pad=1.0):
    """Lay labels out in the empty left/right margins with leader lines.

    Points are split into a left and a right group (by their East position),
    and within each group the labels are stacked evenly down a vertical lane in
    the empty margin. This guarantees no two labels overlap no matter how tight
    the points cluster; a leader line (drawn by the caller) ties each label back
    to its coordinate.

    Parameters:
    plot_x, plot_y: arrays of point positions (matplotlib axes coords)
    names: label strings (unused for placement, kept for a consistent signature)
    xlim, ylim: axis limits (min, max)
    left_frac, right_frac: lane position as a fraction of axis width in from
        each edge
    band_pad: vertical padding kept clear at the top/bottom of the lane (m)

    Returns:
    list of (x, y, ha) tuples, one per point, where ha is the text alignment.
    """
    n = len(names)
    width = xlim[1] - xlim[0]
    left_x = xlim[0] + left_frac * width
    right_x = xlim[1] - right_frac * width
    top = ylim[1] - band_pad
    bot = ylim[0] + band_pad

    # Split into halves by East so each lane serves the nearer points.
    order = sorted(range(n), key=lambda i: (plot_x[i], plot_y[i]))
    half = n // 2
    left_idx = order[:half]
    right_idx = order[half:]

    result = [None] * n

    def place(indices, lane_x, ha):
        # Stack evenly down the lane, ordered by the point's North position so
        # leader lines run roughly parallel and don't cross.
        ordered = sorted(indices, key=lambda i: plot_y[i])
        m = len(ordered)
        for k, i in enumerate(ordered):
            ly = (top + bot) / 2 if m == 1 else bot + (top - bot) * k / (m - 1)
            result[i] = (lane_x, ly, ha)

    place(left_idx, left_x, "left")
    place(right_idx, right_x, "left")
    return result


def plot_ned_coordinates_with_obstacles(coordinates, figsize=(12, 8), save_path=None,
                                        obstacle_setback=2.0, obstacle_half_width=1.5,
                                        gate_spacing=2.0, gate_width=3.0, octagon_radius=1.35,
                                        slalom_layer_2_offset=0.0, slalom_layer_3_offset=0.0,
                                        show=True):
    """Create a 2D visualization of coordinates with gate obstacles and octagon."""
    fig, ax = plt.subplots(figsize=figsize)

    XLIM = (-20, 20)
    YLIM = (-1, 20)

    x_coords, y_coords, names, yaws = [], [], [], []
    for coord in coordinates:
        x_coords.append(coord["x"])
        y_coords.append(coord["y"])
        names.append(coord["name"])
        yaws.append(coord.get("yaw", 0))

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    # Display: X-axis=East(Y), Y-axis=North(X)
    plot_x = y_coords
    plot_y = x_coords

    ax.scatter(plot_x, plot_y, c="red", s=50, alpha=0.7,
               edgecolors="darkred", linewidth=2, zorder=5)

    # Move the labels out into the empty left/right margins and connect each
    # one back to its coordinate with a thin leader line, so tightly clustered
    # points stay individually readable.
    label_positions = layout_labels_margin(plot_x, plot_y, names, XLIM, YLIM)
    for i, name in enumerate(names):
        lx, ly, ha = label_positions[i]
        ax.annotate(name, xy=(plot_x[i], plot_y[i]), xytext=(lx, ly),
                    textcoords="data",
                    fontsize=6, ha=ha, va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.6,
                                    alpha=0.7, shrinkA=2, shrinkB=2),
                    zorder=9)

    arrow_length = 1.0
    for i, yaw in enumerate(yaws):
        yaw_rad = math.radians(yaw)
        dx_ned = arrow_length * math.sin(yaw_rad)
        dy_ned = arrow_length * math.cos(yaw_rad)
        arrow = FancyArrowPatch((plot_x[i], plot_y[i]),
                                (plot_x[i] + dx_ned, plot_y[i] + dy_ned),
                                arrowstyle="->", mutation_scale=20,
                                color="blue", linewidth=2, zorder=4)
        ax.add_patch(arrow)

    # Torpedo visualization line
    torpedo_coord = None
    for coord in coordinates:
        if coord["name"] == "torpedo_with_yaw":
            torpedo_coord = coord
            break

    if torpedo_coord:
        torpedo_yaw_rad = math.radians(torpedo_coord.get("yaw", 0))
        torpedo_north = torpedo_coord["x"]
        torpedo_east = torpedo_coord["y"]
        line_half_length = 0.31

        start_north = torpedo_north - line_half_length * math.sin(torpedo_yaw_rad)
        start_east = torpedo_east + line_half_length * math.cos(torpedo_yaw_rad)
        end_north = torpedo_north + line_half_length * math.sin(torpedo_yaw_rad)
        end_east = torpedo_east - line_half_length * math.cos(torpedo_yaw_rad)

        ax.plot([start_east, end_east], [start_north, end_north],
                "orange", linewidth=4, alpha=0.9, solid_capstyle="round",
                label="Torpedo (60cm)", zorder=8)

    # Octagon obstacle
    octagon = create_octagon_obstacle(coordinates, octagon_radius)
    if octagon:
        plot_vertices = [(e, n) for n, e in octagon["vertices"]]
        octagon_patch = Polygon(plot_vertices, closed=True,
                                facecolor="purple", alpha=0.6,
                                edgecolor="purple", linewidth=2,
                                label="Octagon Obstacle", zorder=6)
        ax.add_patch(octagon_patch)

    # Slalom gates
    slalom_gates = create_slalom_gates(coordinates, gate_spacing, gate_width,
                                       layer_2_offset=slalom_layer_2_offset,
                                       layer_3_offset=slalom_layer_3_offset)
    for gate in slalom_gates:
        center_north, center_east = gate["center"]
        left_north, left_east = gate["left_end"]
        right_north, right_east = gate["right_end"]

        ax.plot(center_east, center_north, "ro", markersize=6, markeredgecolor="black",
                label="Slalom Gates" if gate == slalom_gates[0] else "", zorder=7)
        ax.plot([left_east, right_east], [left_north, right_north],
                "wo", markersize=6, markeredgecolor="black", markeredgewidth=1, zorder=7)

    # Gate end obstacle
    obstacle = create_gate_end_obstacle(coordinates, obstacle_setback, obstacle_half_width)
    if obstacle:
        start_north, start_east = obstacle["start"]
        end_north, end_east = obstacle["end"]
        ax.plot([start_east, end_east], [start_north, end_north],
                "r-", linewidth=8, alpha=0.8, solid_capstyle="round",
                label="Gate End Obstacle", zorder=7)

    ax.set_xlabel("East (Y) [m]", fontsize=12, fontweight="bold")
    ax.set_ylabel("North (X) [m]", fontsize=12, fontweight="bold")
    ax.set_title(TITLE, fontsize=14, fontweight="bold")

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)

    ax.grid(True, alpha=0.7, linestyle="-", linewidth=0.3 * 10, color="gray")

    grid_spacing = 2.8
    x_min, x_max = XLIM
    x_neg_count = int(np.ceil(abs(x_min) / grid_spacing))
    x_pos_count = int(np.ceil(abs(x_max) / grid_spacing))
    x_ticks = np.concatenate([
        np.arange(0, -x_neg_count * grid_spacing - grid_spacing / 2, -grid_spacing)[::-1],
        np.arange(0, x_pos_count * grid_spacing + grid_spacing / 2, grid_spacing),
    ])
    x_ticks = x_ticks[(x_ticks >= x_min) & (x_ticks <= x_max)]

    y_min, y_max = YLIM
    y_neg_count = int(np.ceil(abs(y_min) / grid_spacing))
    y_pos_count = int(np.ceil(abs(y_max) / grid_spacing))
    y_ticks = np.concatenate([
        np.arange(0, -y_neg_count * grid_spacing - grid_spacing / 2, -grid_spacing)[::-1],
        np.arange(0, y_pos_count * grid_spacing + grid_spacing / 2, grid_spacing),
    ])
    y_ticks = y_ticks[(y_ticks >= y_min) & (y_ticks <= y_max)]

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_aspect("equal")

    ax.plot(0, 0, "ko", markersize=8, label="Origin (0,0)", zorder=5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()

    # Coordinate summary
    print("\nCoordinate Summary:")
    print("-" * 50)
    print("Display: X-axis=East(Y), Y-axis=North(X)")
    for coord in coordinates:
        print(f"{coord['name']}: N={coord['x']:.1f}, E={coord['y']:.1f}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualise an eyeball map YAML from the backups folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file", nargs="?",
                        help="YAML file in the backups folder (e.g. eyeball_A.yaml, "
                             ".yaml optional), or a path to any YAML file")
    parser.add_argument("--list", action="store_true",
                        help="list available backup files and exit")
    parser.add_argument("--save", metavar="PATH", default=None,
                        help="save the figure to this path")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open an interactive window")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.list:
        backups = list_backups()
        if backups:
            print(f"Available backups in {BACKUPS_DIR}:")
            for name in backups:
                print(f"  {name}")
        else:
            print(f"No YAML backups found in {BACKUPS_DIR}")
        return 0

    if not args.file:
        print("Error: no file specified. Use --list to see available backups.")
        return 1

    try:
        yaml_path = resolve_yaml_path(args.file)
        coords = load_coordinates(yaml_path)
        plot_ned_coordinates_with_obstacles(
            coords,
            figsize=(12, 8),
            save_path=args.save,
            obstacle_setback=0.0,
            obstacle_half_width=1.5,
            gate_spacing=2.0,
            gate_width=3.0,
            octagon_radius=1.35,
            slalom_layer_2_offset=-0.2,
            slalom_layer_3_offset=-0.2,
            show=not args.no_show,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
