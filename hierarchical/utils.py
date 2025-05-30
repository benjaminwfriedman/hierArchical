import plotly.graph_objects as go
from collections import defaultdict
from .items import BaseItem, Element
from .helpers import random_color, generate_id
from typing import List, Union
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from shapely.plotting import plot_line

def plot_items(items,
               show_coords=False,
               color_by_class=False,
               color_by_attribute=None,
               flatten_to_elements=False):
    """
    Plot the B-rep surfaces of multiple BaseItems, grouped and colored by attribute or class.

    Args:
        items (List[BaseItem])
        show_coords (bool)
        color_by_class (bool)
        color_by_attribute (str)
        flatten_to_elements (bool)
    """
    from collections import defaultdict
    import plotly.graph_objects as go

    fig = go.Figure()

    # Flatten to elements if needed
    if flatten_to_elements:
        def flatten(item):
            if isinstance(item, Element):
                return [item]
            elif hasattr(item, "sub_items"):
                flattened = []
                for sub in item.sub_items:
                    flattened.extend(flatten(sub))
                return flattened
            return []

        all_elements = []
        for top in items:
            all_elements.extend(flatten(top))
        items = all_elements

    # Determine grouping key
    def get_group_key(item):
        if color_by_class:
            return type(item).__name__
        elif color_by_attribute:
            return getattr(item, color_by_attribute, "unknown")
        return "default"

    # Group items by key
    grouped_items = defaultdict(list)
    for item in items:
        grouped_items[get_group_key(item)].append(item)

    # Assign colors
    keys = sorted(grouped_items.keys())
    key_colors = {key: random_color(seed=idx + 10) for idx, key in enumerate(keys)}

    # Plot each group
    for key, group in grouped_items.items():
        vertices = []
        faces = []
        vert_count = 0

        for item in group:
            brep = item.geometry.brep_data
            surfaces = brep.get("surfaces", [])
            for surf in surfaces:
                vs = surf.get("vertices", [])
                offset = len(vertices)
                vertices.extend(vs)

                if len(vs) >= 3:
                    for i in range(1, len(vs) - 1):
                        faces.append((offset, offset + i, offset + i + 1))

        if not vertices or not faces:
            continue

        x, y, z = zip(*vertices)
        i, j, k = zip(*faces)

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.5,
            color=key_colors[key],
            name=str(key),  # legend entry
            hoverinfo='skip',
            showlegend=True
        ))

        if show_coords:
            labels = [f"({round(xi, 2)}, {round(yi, 2)}, {round(zi, 2)})" for xi, yi, zi in vertices]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="text",
                text=labels,
                showlegend=False,
                hoverinfo="none",
                textfont=dict(size=9, color="black")
            ))

    fig.update_layout(
        title="B-rep Visualization of Elements",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=True
    )

    fig.show()



def print_bill_of_materials(items):
    """
    Prints a Bill of Materials (BOM) by aggregating volumes per material.
    """
    material_totals = defaultdict(lambda: {"volume": 0.0, "percent": 0.0})

    def recurse_materials(item):
        for mat, data in item.materials.items():
            material_totals[mat]["volume"] += data.get("volume", 0.0)
        if hasattr(item, "sub_objects"):
            for child in item.sub_objects:
                recurse_materials(child)

    for item in items:
        recurse_materials(item)

    total_volume = sum(data["volume"] for data in material_totals.values())

    print("\n📦 Bill of Materials")
    print("-" * 40)
    for mat, data in material_totals.items():
        percent = (data["volume"] / total_volume * 100) if total_volume > 0 else 0
        print(f"{mat.title():<15} | Volume: {data['volume']:.3f} m³ | {percent:.1f}%")
    print("-" * 40)
    print(f"{'Total':<15} | Volume: {total_volume:.3f} m³")


def list_all_elements(items: List[BaseItem]) -> List[Element]:
    """
    Recursively collects all Element instances from a list of BaseItems.

    Args:
        items: List of top-level BaseItems (Object, Component, Element)

    Returns:
        List of Element instances found in the hierarchy.
    """
    all_elements = []

    def recurse(item: BaseItem):
        if isinstance(item, Element):
            all_elements.append(item)
        elif hasattr(item, "sub_items") and item.sub_items:
            for sub in item.sub_items:
                recurse(sub)

    for top in items:
        recurse(top)

    return all_elements

def print_parts_report(objects: List[BaseItem]) -> None:
    """
    Prints a summary report of parts grouped by type and material
    for a given list of top-level BaseItems.
    """
    elements = list_all_elements(objects)
    summary = defaultdict(int)

    for e in elements:
        key = f"{e.type} ({e.material})"
        summary[key] += 1

    print("\n🧾 Bill of Materials")
    print("-" * 40)
    for item, count in sorted(summary.items()):
        print(f"{count} X {item}")

import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, Point, MultiLineString, MultiPolygon
from shapely.plotting import plot_line, plot_polygon, plot_points

def plot_shapely_geometries(geometries, color='black'):
    """
    Plot a list of shapely geometry objects using shapely's built-in plotting tools.

    Args:
        geometries (list): List of shapely geometry objects (LineString, Polygon, etc).
        color (str): Color for drawing the geometries.
    """
    fig, ax = plt.subplots()

    for geom in geometries:
        if isinstance(geom, LineString):
            plot_line(geom, ax=ax, color=color)
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                plot_line(line, ax=ax, color=color)
        elif isinstance(geom, Polygon):
            plot_polygon(geom, ax=ax, add_points=False, facecolor='none', edgecolor=color)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                plot_polygon(poly, ax=ax, add_points=False, facecolor='none', edgecolor=color)
        elif isinstance(geom, Point):
            plot_points(geom, ax=ax, color=color)
        else:
            print(f"Unsupported geometry type: {type(geom)}")

    ax.set_aspect('equal', 'box')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Shapely Geometries")
    plt.grid(True)
    plt.show()


def plot_topologic_objects(objects: List):
    from topologicpy.Topology import Topology

    Topology.Show(
        objects,
        renderer="browser"
    )


