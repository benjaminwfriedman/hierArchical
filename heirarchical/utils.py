import plotly.graph_objects as go
from collections import defaultdict
from .items import BaseItem, Element
from .helpers import random_color, generate_id
from typing import List, Union
import plotly.graph_objects as go

def plot_items(items, show_coords=False, color_by_class=False, flatten_to_elements=False):
    """
    Plot the B-rep surfaces of multiple BaseItems, optionally flattening to elements.

    Args:
        items (List[BaseItem]): List of items (Element, Component, Object).
        show_coords (bool): Whether to annotate vertex coordinates in the plot.
        color_by_class (bool): Whether to group items by their class and color them consistently.
        flatten_to_elements (bool): Whether to recursively extract all elements before plotting.
    """
    from collections import defaultdict

    fig = go.Figure()

    # Optionally flatten to elements
    if flatten_to_elements:
        all_elements = []

        def recurse(item: BaseItem):
            if isinstance(item, Element):
                all_elements.append(item)
            elif hasattr(item, "sub_items") and item.sub_items:
                for sub in item.sub_items:
                    recurse(sub)

        for top in items:
            recurse(top)

        items = all_elements

    # Assign consistent colors by class or element type
    class_colors = {}
    if color_by_class:
        types = sorted(set(type(item).__name__ if not flatten_to_elements else item.type for item in items))
        for idx, t in enumerate(types):
            class_colors[t] = random_color(seed=idx + 3)

    for idx, item in enumerate(items):
        brep = item.geometry.brep_data
        surfaces = brep.get("surfaces", [])
        vertices = []
        faces = []

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
        label = item.type if flatten_to_elements else type(item).__name__
        color = class_colors[label] if color_by_class else random_color(seed=idx)

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.5,
            color=color,
            name=f"{item.name} ({label})" if color_by_class else item.name,
            hovertext=item.name,
            hoverinfo='text'
        ))

        if show_coords:
            labels = [f"({round(xi, 2)}, {round(yi, 2)}, {round(zi, 2)})" for xi, yi, zi in vertices]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="text",
                text=labels,
                textposition="top center",
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

