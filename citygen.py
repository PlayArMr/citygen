#!/usr/bin/env python3
"""
Transport Map Generator
------------------------
Downloads a road network, transit infrastructure (rail/subway/tram/monorail/
bus routes), and water features (rivers/lakes/coastline) around a city
centre, then renders a stylized dark map.

Requires: osmnx >= 2.0, matplotlib, geopandas (comes with osmnx)
"""

import sys
import threading
import time
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_RADIUS_KM = 20.0
DEFAULT_DPI = 300  # 600 was print-poster overkill for line art; 300 is plenty
DEFAULT_THEME = "tokyo-night"
DEFAULT_RESOLUTION = "square"

# figsize is in INCHES, actual pixel output = figsize * dpi. These presets
# pick sane inch dimensions for common wallpaper/print aspect ratios so
# people don't have to think in inches. Width x height, matches how you'd
# describe a screen (16:9 landscape, 9:16 phone portrait, etc).
RESOLUTIONS = {
    "square": (20, 20),        # original default, good for posters/prints
    "desktop-16-9": (24, 13.5),   # standard widescreen monitor
    "desktop-21-9": (28, 12),     # ultrawide monitor
    "phone-9-16": (13.5, 24),     # phone lock screen / portrait wallpaper
    "phone-9-19-5": (12, 26),     # taller modern phone aspect ratio
}

# Every theme needs every one of these keys. Keep them all fully defined per
# theme rather than falling back to a global default - a half-themed map
# looks like a bug, not a feature.
THEMES = {
    "tokyo-night": {
        "background": "#0D1117",
        "street": "#8B95A1",
        "arterial": "#C97A75",
        "major_road": "#F85149",
        "rail": "#5DE0C4",
        "tram": "#8FE8D8",
        "bus_route": "#7EC8E3",
        "water": "#1B3A4B",
        "water_line": "#2C5F74",
        "coastline": "#2C5F74",
    },
    "deccan-dusk": {
        "background": "#1A0F0D",
        "street": "#B08E7E",
        "arterial": "#D98E5F",
        "major_road": "#E85D3D",
        "rail": "#5FA8A0",
        "tram": "#8FC9C2",
        "bus_route": "#C97B5A",
        "water": "#2B3A3A",
        "water_line": "#4A6363",
        "coastline": "#4A6363",
    },
    "monochrome": {
        "background": "#0A0A0A",
        "street": "#707070",
        "arterial": "#9A9A9A",
        "major_road": "#EDEDED",
        "rail": "#B0B0B0",
        "tram": "#9A9A9A",
        "bus_route": "#5F5F5F",
        "water": "#161616",
        "water_line": "#2A2A2A",
        "coastline": "#2A2A2A",
    },
    "cyberpunk": {
        "background": "#050014",
        "street": "#5A4A8C",
        "arterial": "#FF2E9A",
        "major_road": "#00F0FF",
        "rail": "#FF2E9A",
        "tram": "#B026FF",
        "bus_route": "#39FF14",
        "water": "#0A0033",
        "water_line": "#1E1A5E",
        "coastline": "#1E1A5E",
    },
    "blueprint": {
        "background": "#0A2E4D",
        "street": "#6B9AC4",
        "arterial": "#7FB2E5",
        "major_road": "#FFFFFF",
        "rail": "#FF9F5A",
        "tram": "#FFC490",
        "bus_route": "#5C9BD1",
        "water": "#062038",
        "water_line": "#154064",
        "coastline": "#154064",

    },
}

MAJOR_ROADS = {"motorway", "motorway_link", "trunk", "trunk_link"}
ARTERIAL_ROADS = {"primary", "primary_link", "secondary", "secondary_link"}

# Passenger rail infrastructure (not just "railway=rail" which includes freight sidings etc.)
RAIL_INFRA_TYPES = ["rail", "subway", "monorail", "funicular"]
TRAM_INFRA_TYPES = ["tram", "light_rail"]

# Water tags: polygons (natural=water, landuse=reservoir, natural=coastline area)
# and linear waterways (river/stream/canal)
WATERWAY_LINE_TYPES = ["river", "stream", "canal", "tidal_channel"]


# ============================================================
# HELPERS
# ============================================================

def has_road_type(highway, road_types):
    if isinstance(highway, list):
        return any(road_type in road_types for road_type in highway)
    return highway in road_types


def safe_filename(name):
    return (
        name.lower()
        .strip()
        .replace(" ", "_")
        .replace(",", "")
        .replace("/", "_")
    )


def prompt_float(prompt_text, default):
    raw = input(f"{prompt_text} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  Not a number, using default ({default}).")
        return default


def prompt_yes_no(prompt_text, default_yes=True):
    suffix = "[Y/n]" if default_yes else "[y/N]"
    raw = input(f"{prompt_text} {suffix}: ").strip().lower()
    if not raw:
        return default_yes
    return raw in {"y", "yes"} if not default_yes else raw not in {"n", "no"}


class Spinner:
    """
    Minimal zero-dependency ANSI terminal spinner.

    Runs a rotating cursor + elapsed time on the current line while a slow
    blocking call (network fetch, render, etc.) happens on the main thread.
    Use as a context manager:

        with Spinner("Downloading road network"):
            G = ox.graph_from_bbox(bbox, network_type="drive")

    On exit it clears the line and prints "<label>... done (Xs)".
    If an exception occurs inside the block, it clears the line and
    prints "<label>... failed" before letting the exception propagate,
    so tracebacks don't get mangled mid-line.
    """

    FRAMES = ["|", "/", "-", "\\"]

    def __init__(self, label, interval=0.1):
        self.label = label
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop_event.is_set():
                break
            elapsed = time.time() - self._start_time
            sys.stdout.write(f"\r{self.label}... {frame} ({elapsed:0.1f}s)")
            sys.stdout.flush()
            time.sleep(self.interval)

    def __enter__(self):
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

        elapsed = time.time() - self._start_time
        # \r + pad with spaces to stomp out any leftover spinner text, then \r again
        clear_width = len(self.label) + 20
        sys.stdout.write("\r" + " " * clear_width + "\r")

        if exc_type is None:
            print(f"{self.label}... done ({elapsed:0.1f}s)")
        else:
            print(f"{self.label}... failed ({elapsed:0.1f}s)")

        return False  # never swallow exceptions


def safe_features_from_bbox(bbox_tuple, tags, label):
    """
    Wraps ox.features_from_bbox with version-tolerant bbox ordering and
    graceful failure (returns None instead of nuking the whole run when
    a feature type simply doesn't exist in the area, which is common and
    NOT an error worth crashing over).

    osmnx >= 2.0 expects bbox as (left, bottom, right, top) i.e.
    (west, south, east, north) - matching a standard GeoPandas/Shapely bbox.
    Older osmnx (<2.0) used (north, south, east, west). We detect version
    and adapt so this script survives an osmnx upgrade/downgrade.
    """
    west, south, east, north = bbox_tuple

    try:
        major_version = int(ox.__version__.split(".")[0])
    except (ValueError, AttributeError):
        major_version = 2  # assume modern if we can't tell

    try:
        if major_version >= 2:
            result = ox.features_from_bbox((west, south, east, north), tags=tags)
        else:
            result = ox.features_from_bbox(north, south, east, west, tags=tags)
        return result
    except InsufficientResponseError:
        print(f"  No {label} features found in this area (that's fine).")
        return None
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a CLI tool
        print(f"  Could not fetch {label} features: {exc}")
        return None


try:
    from osmnx._errors import InsufficientResponseError
except ImportError:
    # Fallback shim for older osmnx where this exception may not exist
    class InsufficientResponseError(Exception):
        pass


# ============================================================
# USER INPUT
# ============================================================

def get_user_input():
    print()
    print("===================================")
    print("      TRANSPORT MAP GENERATOR")
    print("===================================")
    print()

    city = input("City: ").strip()
    if not city:
        raise ValueError("You need to enter a city.")

    radius_km = prompt_float("Radius around city centre in km", DEFAULT_RADIUS_KM)
    if radius_km <= 0:
        raise ValueError("Radius must be positive.")

    theme_names = list(THEMES.keys())
    print(f"Available themes: {', '.join(theme_names)}")
    theme_input = input(f"Theme [{DEFAULT_THEME}]: ").strip().lower()
    if not theme_input:
        theme_name = DEFAULT_THEME
    elif theme_input in THEMES:
        theme_name = theme_input
    else:
        print(f"  Unknown theme '{theme_input}', using default ({DEFAULT_THEME}).")
        theme_name = DEFAULT_THEME

    resolution_names = list(RESOLUTIONS.keys())
    print(f"Available resolutions: {', '.join(resolution_names)}")
    resolution_input = input(f"Resolution [{DEFAULT_RESOLUTION}]: ").strip().lower()
    if not resolution_input:
        resolution_name = DEFAULT_RESOLUTION
    elif resolution_input in RESOLUTIONS:
        resolution_name = resolution_input
    else:
        print(f"  Unknown resolution '{resolution_input}', using default ({DEFAULT_RESOLUTION}).")
        resolution_name = DEFAULT_RESOLUTION

    include_rail = prompt_yes_no("Include rail/subway/monorail?", default_yes=True)
    include_tram = prompt_yes_no("Include trams/light rail?", default_yes=True)
    include_bus = prompt_yes_no("Include bus routes? (slow, can be dense)", default_yes=False)
    include_water = prompt_yes_no("Include water features (rivers/lakes/coastline)?", default_yes=True)

    dpi_input = prompt_float("Output DPI (300 = normal, 600 = huge files)", DEFAULT_DPI)
    dpi = int(dpi_input)

    format_input = input("Output format [png/pdf/svg] [png]: ").strip().lower()
    output_format = format_input if format_input else "png"
    if output_format not in {"png", "pdf", "svg"}:
        raise ValueError("Output format must be png, pdf, or svg.")

    return {
        "city": city,
        "radius_km": radius_km,
        "theme_name": theme_name,
        "resolution_name": resolution_name,
        "include_rail": include_rail,
        "include_tram": include_tram,
        "include_bus": include_bus,
        "include_water": include_water,
        "dpi": dpi,
        "output_format": output_format,
    }


# ============================================================
# DATA FETCHING
# ============================================================

def geocode_city(city):
    print()
    try:
        with Spinner(f"Finding {city}"):
            latitude, longitude = ox.geocode(city)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not geocode '{city}'. Check spelling, or try "
            f"'City, Country' for disambiguation. Original error: {exc}"
        ) from exc
    print(f"City centre: {latitude:.5f}, {longitude:.5f}")
    return latitude, longitude


def compute_bbox_for_aspect(center, radius_km, figsize):
    """
    Build a lat/lon bbox around `center` shaped to match the target figsize's
    aspect ratio, rather than always pulling a square area.

    Without this, a phone-portrait figsize (e.g. 13.5x24in) still gets a
    SQUARE chunk of map data (since ox.graph_from_point's `dist` is radial,
    i.e. same distance in every direction), and matplotlib's equal-aspect
    axes then only fills the middle square portion of the tall canvas -
    leaving a huge blank strip top or bottom. Shaping the query bbox to
    match the canvas means the map data actually fills the frame.

    radius_km is treated as the HALF-HEIGHT (or half-width, whichever is
    smaller) of the requested area, then the other axis is scaled by the
    figsize ratio.
    """
    width_in, height_in = figsize
    aspect = width_in / height_in  # >1 = wider than tall, <1 = taller than wide

    if aspect >= 1:
        # Wider than tall: keep radius_km as half-height, widen the width.
        half_height_km = radius_km
        half_width_km = radius_km * aspect
    else:
        # Taller than wide: keep radius_km as half-width, extend the height.
        half_width_km = radius_km
        half_height_km = radius_km / aspect

    lat, lon = center

    # Rough conversion: 1 degree latitude ~= 111.32 km, everywhere.
    # 1 degree longitude ~= 111.32 km * cos(latitude), shrinks toward poles.
    import math

    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat))
    km_per_deg_lon = max(km_per_deg_lon, 1e-6)  # guard near-poles div-by-zero

    delta_lat = half_height_km / km_per_deg_lat
    delta_lon = half_width_km / km_per_deg_lon

    north = lat + delta_lat
    south = lat - delta_lat
    east = lon + delta_lon
    west = lon - delta_lon

    return (west, south, east, north)


def download_road_network(center, radius_km, figsize):
    print()
    bbox = compute_bbox_for_aspect(center, radius_km, figsize)
    west, south, east, north = bbox

    try:
        major_version = int(ox.__version__.split(".")[0])
    except (ValueError, AttributeError):
        major_version = 2  # assume modern if we can't tell

    try:
        with Spinner("Downloading road network"):
            if major_version >= 2:
                G = ox.graph_from_bbox((west, south, east, north), network_type="drive")
            else:
                G = ox.graph_from_bbox(north, south, east, west, network_type="drive")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to download road network: {exc}") from exc
    print("Nodes:", len(G.nodes))
    print("Edges:", len(G.edges))
    return G


def get_bbox_from_graph(G):
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    west = nodes.geometry.x.min()
    east = nodes.geometry.x.max()
    south = nodes.geometry.y.min()
    north = nodes.geometry.y.max()
    return (west, south, east, north)


def download_transit(bbox, include_rail, include_tram, include_bus):
    """
    Returns a dict of {label: GeoDataFrame or None}.
    Rail/tram are queried as physical infrastructure (railway=*) since that's
    what OSM actually models reliably. Bus is queried as route *relations*
    (route=bus) which represent actual passenger routes, not just any
    highway=bus_stop node.
    """
    layers = {}

    if include_rail:
        print()
        with Spinner("Downloading rail/subway/monorail infrastructure"):
            rail = safe_features_from_bbox(bbox, {"railway": RAIL_INFRA_TYPES}, "rail")
        if rail is not None and not rail.empty:
            rail = rail[rail.geometry.geom_type.isin(["LineString", "MultiLineString"])]
            print("Rail features:", len(rail))
        layers["rail"] = rail

    if include_tram:
        print()
        with Spinner("Downloading tram/light rail infrastructure"):
            tram = safe_features_from_bbox(bbox, {"railway": TRAM_INFRA_TYPES}, "tram")
        if tram is not None and not tram.empty:
            tram = tram[tram.geometry.geom_type.isin(["LineString", "MultiLineString"])]
            print("Tram features:", len(tram))
        layers["tram"] = tram

    if include_bus:
        print()
        with Spinner("Downloading bus routes (route relations, can be slow)"):
            bus = safe_features_from_bbox(bbox, {"route": "bus"}, "bus route")
        if bus is not None and not bus.empty:
            raw_count = len(bus)
            bus = bus[bus.geometry.geom_type.isin(["LineString", "MultiLineString"])]
            print(f"Bus route features: {len(bus)} (of {raw_count} raw results)")
            if len(bus) == 0:
                print(
                    "  Note: bus route relations came back but none had "
                    "usable line geometry (only points/polygons or none at "
                    "all). This is a known limitation - OSM bus routes are "
                    "relations of way segments, and osmnx doesn't always "
                    "resolve them into a single drawable line per version. "
                    "Bus layer will be empty on this render."
                )
        elif bus is not None and bus.empty:
            print("Bus route features: 0 (query succeeded but returned no data for this area)")
        layers["bus"] = bus

    return layers


def download_water(bbox):
    print()

    with Spinner("Downloading water bodies"):
        water_polys = safe_features_from_bbox(
            bbox,
            {"natural": ["water", "bay"], "landuse": ["reservoir", "basin"]},
            "water body",
        )
    if water_polys is not None and not water_polys.empty:
        water_polys = water_polys[
            water_polys.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ]
        print("Water body features:", len(water_polys))

    with Spinner("Downloading waterways (rivers/streams)"):
        waterways = safe_features_from_bbox(
            bbox, {"waterway": WATERWAY_LINE_TYPES}, "waterway"
        )
    if waterways is not None and not waterways.empty:
        waterways = waterways[
            waterways.geometry.geom_type.isin(["LineString", "MultiLineString"])
        ]
        print("Waterway (river/stream) features:", len(waterways))

    with Spinner("Downloading coastline"):
        coastline = safe_features_from_bbox(
            bbox, {"natural": "coastline"}, "coastline"
        )
    if coastline is not None and not coastline.empty:
        coastline = coastline[
            coastline.geometry.geom_type.isin(["LineString", "MultiLineString"])
        ]
        print("Coastline features:", len(coastline))

    return {
        "polygons": water_polys,
        "waterways": waterways,
        "coastline": coastline,
    }

# ============================================================
# STYLING
# ============================================================

def style_road_edges(G, theme):
    edge_colors = []
    edge_widths = []

    # Fixed absolute linewidths look fine on dense city grids (thousands of
    # overlapping segments create visual weight even at sub-pixel widths)
    # but make sparse small-town networks vanish entirely - there's nothing
    # for a 0.1px line to blend into. Scale minor-street width down slightly
    # as edge count grows, so small towns get a strong boost and megacities
    # like Mumbai don't turn into a muddy blob of overlapping thick lines.
    edge_count = len(G.edges)

    if edge_count < 5000:
        street_width = 0.75       # small town: needs to be clearly visible
    elif edge_count < 20000:
        street_width = 0.55       # mid-size city
    else:
        street_width = 0.40       # dense megacity: thinner so it doesn't mud out

    for _, _, _, data in G.edges(keys=True, data=True):
        highway = data.get("highway", "")

        if has_road_type(highway, MAJOR_ROADS):
            edge_colors.append(theme["major_road"])
            edge_widths.append(1.10)
        elif has_road_type(highway, ARTERIAL_ROADS):
            edge_colors.append(theme["arterial"])
            edge_widths.append(0.45)
        else:
            edge_colors.append(theme["street"])
            edge_widths.append(street_width)

    return edge_colors, edge_widths


# ============================================================
# RENDERING
# ============================================================

def render_map(G, transit_layers, water_layers, output_path, dpi, output_format, theme, city_label, figsize):
    edge_colors, edge_widths = style_road_edges(G, theme)

    background = theme["background"]

    fig, ax = plt.subplots(figsize=figsize, facecolor=background)
    ax.set_facecolor(background)

    # --- Water, base layer everything else sits on top of ---
    if water_layers:
        polys = water_layers.get("polygons")
        if polys is not None and not polys.empty:
            polys.plot(ax=ax, color=theme["water"], linewidth=0, zorder=1)

        waterways = water_layers.get("waterways")
        if waterways is not None and not waterways.empty:
            waterways.plot(ax=ax, color=theme["water_line"], linewidth=0.6, zorder=1, alpha=0.9)

        coastline = water_layers.get("coastline")
        if coastline is not None and not coastline.empty:
            coastline.plot(ax=ax, color=theme["coastline"], linewidth=0.8, zorder=1)

    # --- Roads ---
    ox.plot_graph(
        G,
        ax=ax,
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        bgcolor=background,
        show=False,
        close=False,
    )

    # --- Redraw major roads on top so they don't get buried under local streets ---
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    major_edges = edges[edges["highway"].apply(lambda h: has_road_type(h, MAJOR_ROADS))]
    if not major_edges.empty:
        major_edges.plot(ax=ax, color=theme["major_road"], linewidth=1.10, alpha=1.0, zorder=6)

    # --- Transit layers on top of roads ---
    bus = transit_layers.get("bus")
    if bus is not None and not bus.empty:
        bus.plot(ax=ax, color=theme["bus_route"], linewidth=0.6, alpha=0.85, zorder=4)

    tram = transit_layers.get("tram")
    if tram is not None and not tram.empty:
        tram.plot(ax=ax, color=theme["tram"], linewidth=0.5, alpha=0.9, zorder=5)

    rail = transit_layers.get("rail")
    if rail is not None and not rail.empty:
        rail.plot(ax=ax, color=theme["rail"], linewidth=0.6, alpha=0.95, zorder=6)

    ax.set_axis_off()
    ax.margins(0)

    # Reserve a thin strip at the bottom of the figure for the city caption.
    # Without this, subplots_adjust(bottom=0) leaves zero room and the text
    # either overlaps the map edge or gets clipped by bbox_inches="tight".
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.035)

    fig.text(
        0.5,
        0.012,
        city_label.upper(),
        ha="center",
        va="bottom",
        fontsize=22,
        color=theme["major_road"],
        family="monospace",
        alpha=0.9,
    )

    print()

    save_kwargs = dict(bbox_inches="tight", pad_inches=0, facecolor=background)
    with Spinner("Rendering and saving map"):
        if output_format == "png":
            fig.savefig(output_path, dpi=dpi, **save_kwargs)
        else:
            fig.savefig(output_path, format=output_format, **save_kwargs)

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    try:
        config = get_user_input()

        latitude, longitude = geocode_city(config["city"])
        center = (latitude, longitude)

        figsize = RESOLUTIONS[config["resolution_name"]]

        G = download_road_network(center, config["radius_km"], figsize)
        bbox = get_bbox_from_graph(G)

        transit_layers = download_transit(
            bbox, config["include_rail"], config["include_tram"], config["include_bus"]
        )

        water_layers = {}
        if config["include_water"]:
            water_layers = download_water(bbox)

        theme = THEMES[config["theme_name"]]

        filename = (
            safe_filename(config["city"])
            + "_"
            + config["theme_name"]
            + "_"
            + config["resolution_name"]
            + "_transport_map."
            + config["output_format"]
        )
        output_path = Path(filename)

        render_map(
            G,
            transit_layers,
            water_layers,
            output_path,
            config["dpi"],
            config["output_format"],
            theme,
            config["city"],
            figsize,
        )

        print()
        print("==============================")
        print("COMPLETE")
        print("==============================")
        print()
        print(f"Saved: {output_path.resolve()}")

    except (ValueError, RuntimeError) as exc:
        print()
        print(f"Error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("Cancelled.")
        sys.exit(130)


if __name__ == "__main__":
    main()