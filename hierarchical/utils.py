import plotly.graph_objects as go
from collections import defaultdict
from .items import BaseItem, Element
from .helpers import random_color, generate_id
from typing import List, Union
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from shapely.plotting import plot_line
import plotly.io as pio
pio.renderers.default = "browser"


def plot_items(items,
               show_coords=False,
               color_by_class=False,
               color_by_attribute=None,
               color_by_item=True,
               flatten_to_elements=False):
    """
    Plot the B-rep surfaces of multiple BaseItems, grouped and colored by attribute or class.

    Args:
        items (List[BaseItem])
        show_coords (bool)
        color_by_class (bool)
        color_by_attribute (str)
        color_by_item (bool): If True, each item gets its own color
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
        if color_by_item:
            return id(item)  # Each item gets unique ID
        elif color_by_class:
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
    
    if color_by_item:
        # Generate unique colors for each item
        key_colors = {key: random_color(seed=hash(key) % 10000) for key in keys}
    else:
        # Use consistent colors for groups
        key_colors = {key: random_color(seed=idx + 10) for idx, key in enumerate(keys)}

    # Plot each group
    for key, group in grouped_items.items():
        vertices = []
        faces = []
        
        for item in group:
            # Check if item has healed geometry (occ_face)
            if hasattr(item.geometry, 'occ_face') and item.geometry.occ_face:
                # Use mesh_data from healed geometry
                item_vertices = item.geometry.mesh_data.get("vertices", [])
                item_faces = item.geometry.mesh_data.get("faces", [])
                
                if item_vertices and item_faces:
                    offset = len(vertices)
                    vertices.extend(item_vertices)
                    
                    # Adjust face indices with offset
                    for face in item_faces:
                        faces.append(tuple(idx + offset for idx in face))
            else:
                # Fall back to brep_data
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

        # Create legend name
        if color_by_item:
            # Use item name or class name for legend
            item_name = getattr(group[0], 'name', None) or type(group[0]).__name__
            legend_name = f"{item_name} ({id(group[0])})"
        else:
            legend_name = str(key)

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.5,
            color=key_colors[key],
            name=legend_name,
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
    return fig

# def random_color(seed=None):
#     """Generate a random color for plotting"""
#     import random
#     if seed is not None:
#         random.seed(seed)
    
#     # Generate bright, distinct colors
#     colors = [
#         '#636efa', '#EF553B', '#00cc96', '#ab63fa', '#FFA15A',
#         '#19d3f3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
#         '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
#     ]
    
#     if seed is not None:
#         return colors[seed % len(colors)]
#     else:
#         return random.choice(colors)



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



# import plotly.graph_objects as go
import plotly.express as px
# from plotly.subplots import make_subplots
# import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union

def plot_opencascade_shapes(shapes: Union[List, Any], 
                                show_faces=True, 
                                show_edges=True, 
                                show_vertices=False,
                                face_opacity=0.7,
                                edge_width=2,
                                colors=None,
                                labels=None,
                                title="OpenCascade Shapes"):
    """
    Visualize one or more OpenCascade shapes using Plotly
    
    Args:
        shapes: Single OpenCascade TopoDS_Shape or list of shapes
        show_faces: Whether to show face surfaces
        show_edges: Whether to show edges
        show_vertices: Whether to show vertices
        face_opacity: Opacity of faces (0-1)
        edge_width: Width of edge lines
        colors: List of colors for each shape (optional)
        labels: List of labels for each shape (optional)
        title: Plot title
    """
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    
    # Convert single shape to list
    if not isinstance(shapes, list):
        shapes = [shapes]
    
    fig = go.Figure()
    
    # Default colors if not provided
    if colors is None:
        colors = px.colors.qualitative.Set1[:len(shapes)]
        if len(shapes) > len(colors):
            colors = colors * (len(shapes) // len(colors) + 1)
    
    # Default labels if not provided
    if labels is None:
        labels = [f"Shape {i+1}" for i in range(len(shapes))]
    
    for shape_idx, shape in enumerate(shapes):
        shape_color = colors[shape_idx % len(colors)]
        shape_label = labels[shape_idx % len(labels)]
        
        # Mesh the shape for tessellation
        try:
            mesh = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)
            mesh.Perform()
        except Exception as e:
            print(f"Warning: Could not mesh shape {shape_idx}: {e}")
            continue
        
        # Add faces as mesh3d
        if show_faces:
            face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
            face_count = 0
            while face_explorer.More():
                face = face_explorer.Current()
                vertices, triangles = extract_face_mesh(face)
                
                if len(vertices) > 0 and len(triangles) > 0:
                    x, y, z = zip(*vertices)
                    i, j, k = zip(*triangles)
                    
                    # Show legend only for first face of each shape
                    show_legend = face_count == 0
                    legend_name = f"{shape_label} (faces)" if show_legend else None
                    
                    fig.add_trace(go.Mesh3d(
                        x=x, y=y, z=z,
                        i=i, j=j, k=k,
                        opacity=face_opacity,
                        color=shape_color,
                        name=legend_name,
                        showlegend=show_legend,
                        legendgroup=f"shape_{shape_idx}",
                        hovertemplate=f"<b>{shape_label}</b><br>" +
                                    "Face %{text}<br>" +
                                    "X: %{x:.2f}<br>" +
                                    "Y: %{y:.2f}<br>" +
                                    "Z: %{z:.2f}<extra></extra>",
                        text=[f"{face_count}"] * len(x)
                    ))
                    face_count += 1
                
                face_explorer.Next()
        
        # Add edges as lines
        if show_edges:
            edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            edge_count = 0
            edge_points = []
            
            while edge_explorer.More():
                edge = edge_explorer.Current()
                points = extract_edge_points(edge)
                
                if len(points) > 1:
                    edge_points.extend(points)
                    edge_points.append(None)  # Break in line for separate edges
                
                edge_explorer.Next()
                edge_count += 1
            
            if edge_points:
                # Remove trailing None
                if edge_points[-1] is None:
                    edge_points = edge_points[:-1]
                
                x_edges, y_edges, z_edges = zip(*[p for p in edge_points if p is not None])
                
                fig.add_trace(go.Scatter3d(
                    x=x_edges, y=y_edges, z=z_edges,
                    mode='lines',
                    line=dict(color='black', width=edge_width),
                    name=f"{shape_label} (edges)",
                    legendgroup=f"shape_{shape_idx}",
                    hovertemplate=f"<b>{shape_label}</b><br>" +
                                "Edge<br>" +
                                "X: %{x:.2f}<br>" +
                                "Y: %{y:.2f}<br>" +
                                "Z: %{z:.2f}<extra></extra>"
                ))
        
        # Add vertices as points
        if show_vertices:
            vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
            vertex_points = []
            
            while vertex_explorer.More():
                vertex = vertex_explorer.Current()
                point = BRep_Tool.Pnt(vertex)
                vertex_points.append((point.X(), point.Y(), point.Z()))
                vertex_explorer.Next()
            
            if vertex_points:
                x_verts, y_verts, z_verts = zip(*vertex_points)
                
                fig.add_trace(go.Scatter3d(
                    x=x_verts, y=y_verts, z=z_verts,
                    mode='markers',
                    marker=dict(color='red', size=5),
                    name=f"{shape_label} (vertices)",
                    legendgroup=f"shape_{shape_idx}",
                    hovertemplate=f"<b>{shape_label}</b><br>" +
                                "Vertex<br>" +
                                "X: %{x:.2f}<br>" +
                                "Y: %{y:.2f}<br>" +
                                "Z: %{z:.2f}<extra></extra>"
                ))
    
    # Configure layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    fig.show()
    return fig

def extract_face_mesh(face):
    """Extract triangular mesh from a face"""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.Poly import Poly_Triangulation
    
    try:
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        
        if triangulation is None:
            return [], []
        
        # Get transformation from location
        transformation = location.Transformation()
        
        # Extract vertices
        vertices = []
        for i in range(1, triangulation.NbNodes() + 1):
            node = triangulation.Node(i)
            # Apply transformation if present
            if not location.IsIdentity():
                node.Transform(transformation)
            vertices.append((node.X(), node.Y(), node.Z()))
        
        # Extract triangles (convert from 1-based to 0-based indexing)
        triangles = []
        for i in range(1, triangulation.NbTriangles() + 1):
            triangle = triangulation.Triangle(i)
            n1, n2, n3 = triangle.Get()
            triangles.append((n1-1, n2-1, n3-1))
        
        return vertices, triangles
        
    except Exception as e:
        print(f"Error extracting face mesh: {e}")
        return [], []

def extract_edge_points(edge, num_points=50):
    """Extract points along an edge for visualization"""
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GCPnts import GCPnts_UniformAbscissa
    
    try:
        curve_adaptor = BRepAdaptor_Curve(edge)
        
        # Use uniform abscissa for better point distribution
        uniform_abscissa = GCPnts_UniformAbscissa()
        uniform_abscissa.Initialize(curve_adaptor, num_points)
        
        points = []
        if uniform_abscissa.IsDone():
            for i in range(1, uniform_abscissa.NbPoints() + 1):
                param = uniform_abscissa.Parameter(i)
                point = curve_adaptor.Value(param)
                points.append((point.X(), point.Y(), point.Z()))
        else:
            # Fallback to parameter-based sampling
            first_param = curve_adaptor.FirstParameter()
            last_param = curve_adaptor.LastParameter()
            
            for i in range(num_points):
                param = first_param + (last_param - first_param) * i / (num_points - 1)
                point = curve_adaptor.Value(param)
                points.append((point.X(), point.Y(), point.Z()))
        
        return points
        
    except Exception as e:
        print(f"Error extracting edge points: {e}")
        return []