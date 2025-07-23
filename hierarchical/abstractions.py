from tqdm import tqdm
from hierarchical.items import Element, Component, Wall, Deck, Window, Door, Object, BaseItem
from hierarchical.relationships import AdjacentTo, Relationship, Creates
from hierarchical.geometry import Geometry
from hierarchical.helpers import test_healing_validation
from collections import defaultdict
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from copy import deepcopy
from hierarchical.utils import generate_id, plot_shapely_geometries, plot_topologic_objects, plot_items, random_color, plot_opencascade_shapes
import matplotlib.pyplot as plt
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Vertex import Vertex
from topologicpy.Cell import Cell
from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE
from OCC.Core.TopoDS import topods
from OCC.Core import TopoDS
from math import comb


from itertools import combinations
import plotly.graph_objects as go
import kuzu
import uuid
from uuid import uuid4
from openai import OpenAI
from OCC.Core.BRepTools import breptools
import os
import numpy as np



client = OpenAI(api_key="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")



# Abstractions are things like spaces and zones. They are not elements but are 
# defined by elements. They are used to group elements together in ways that are meaningful 
# and speak to how people experience the building. For example, a room is a space that is defined by
# walls, floors, and ceilings. (Maybe boundary abstractions)

# Idea 1: We create boundaries out of our walls and floors -> We make spaces out of boundaries?

# Idea 2: We create spaces out of our walls and floors -> We make boundaries out of spaces?

import numpy as np
from scipy.spatial import Delaunay

def triangulate_polygon_3d(vertices_3d, normal=None):
    """
    Triangulate a planar polygon defined by 3D vertices.

    Args:
        vertices_3d (list of tuple): List of 3D points (x, y, z).
        normal (np.array): Optional normal vector of the polygon plane. If not given, it's computed.

    Returns:
        list of tuple: Face indices (i, j, k) as triangles into the input vertex list.
    """
    vertices_3d = np.array(vertices_3d)

    # Compute normal if not provided
    if normal is None:
        v1 = vertices_3d[1] - vertices_3d[0]
        v2 = vertices_3d[2] - vertices_3d[0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)

    # Create 2D projection basis (u, v)
    u = vertices_3d[1] - vertices_3d[0]
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    def project_to_2d(p):
        rel = p - vertices_3d[0]
        return [np.dot(rel, u), np.dot(rel, v)]

    points_2d = np.array([project_to_2d(p) for p in vertices_3d])

    # Triangulate in 2D
    delaunay = Delaunay(points_2d)
    return [tuple(triangle) for triangle in delaunay.simplices]


@dataclass
class Boundary(BaseItem):
    """Represents a space boundary with its properties and geometry."""
    
    is_access_boundary: bool = False
    is_visual_boundary: bool = False
    base_item: Optional[BaseItem] = None
    height: float = 0.0
    normal_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    adjacent_spaces: List[str] = field(default_factory=list)
    
    # Override parent defaults if needed
    relationships: List[Relationship] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        # Calculate boundary properties based on type
        if self.type == 'full':
            self.is_access_boundary = True
            self.is_visual_boundary = True
        elif self.type == 'partial':
            self.is_access_boundary = True
            self.is_visual_boundary = False
        elif self.type == 'open':
            self.is_access_boundary = False
            self.is_visual_boundary = False

    def _find_vertex_indexes(self, target_points: List[Tuple[float, float, float]], tolerance: float = 0.01) -> List[Optional[int]]:
        """
        Find vertex indexes in geometry that match the target points.
        
        Args:
            target_points: List of 3D points to find in the vertex list
            tolerance: Maximum distance for considering points as matching
            
        Returns:
            List of vertex indexes (None if not found)
        """
        if not self.geometry or not self.geometry.get_vertices():
            return [None] * len(target_points)

        vertices = self.geometry.get_vertices()
        vertex_indexes = []

        for target_point in target_points:
            target_array = np.array(target_point)
            best_index = None
            min_distance = float('inf')

            for i, vertex in enumerate(vertices):
                vertex_array = np.array(vertex)
                distance = np.linalg.norm(vertex_array - target_array)

                if distance < tolerance and distance < min_distance:
                    min_distance = distance
                    best_index = i

            vertex_indexes.append(best_index)

        return vertex_indexes

    def _analyze_geometry_edges(self, tolerance: float = 0.01) -> Dict[str, Dict]:
        """
        Analyze the geometry to identify the boundary edges.
        
        Args:
            tolerance: Tolerance for comparing coordinates
            
        Returns:
            Dictionary containing edge information
        """
        if not self.geometry or not self.geometry.get_vertices():
            return {}

        vertices = list(self.geometry.get_vertices())
        if len(vertices) < 4:
            return {}

        # Convert to numpy arrays for easier manipulation
        vertex_arrays = [np.array(v) for v in vertices]

        # Find min and max Z coordinates to identify bottom and top edges
        z_coords = [v[2] for v in vertices]
        min_z = min(z_coords)
        max_z = max(z_coords)

        # Separate vertices into bottom and top groups
        bottom_vertices = []
        top_vertices = []

        for i, vertex in enumerate(vertices):
            if abs(vertex[2] - min_z) <= tolerance:
                bottom_vertices.append((i, vertex))
            elif abs(vertex[2] - max_z) <= tolerance:
                top_vertices.append((i, vertex))

        if len(bottom_vertices) < 2 or len(top_vertices) < 2:
            return {}

        # Sort bottom vertices by X coordinate, then by Y if X is equal
        bottom_vertices.sort(key=lambda x: (x[1][0], x[1][1]))
        top_vertices.sort(key=lambda x: (x[1][0], x[1][1]))

        # Identify corner vertices
        bottom_left = bottom_vertices[0]
        bottom_right = bottom_vertices[-1]
        top_left = top_vertices[0]
        top_right = top_vertices[-1]

        return {
            'bottom': {
                'start_point': bottom_left[1],
                'end_point': bottom_right[1],
                'start_vertex_index': bottom_left[0],
                'end_vertex_index': bottom_right[0],
                'edge_type': 'bottom'
            },
            'top': {
                'start_point': top_left[1],
                'end_point': top_right[1],
                'start_vertex_index': top_left[0],
                'end_vertex_index': top_right[0],
                'edge_type': 'top'
            },
            'left': {
                'start_point': bottom_left[1],
                'end_point': top_left[1],
                'start_vertex_index': bottom_left[0],
                'end_vertex_index': top_left[0],
                'edge_type': 'left'
            },
            'right': {
                'start_point': bottom_right[1],
                'end_point': top_right[1],
                'start_vertex_index': bottom_right[0],
                'end_vertex_index': top_right[0],
                'edge_type': 'right'
            }
        }

    def get_top_edge(self) -> Dict:
        """
        Get the top edge of the boundary as a line segment with vertex indexes.
        
        Returns:
            Dictionary with edge coordinates and vertex indexes
        """
        edges = self._analyze_geometry_edges()
        return edges.get('top', {})

    def get_bottom_edge(self) -> Dict:
        """
        Get the bottom edge of the boundary as a line segment with vertex indexes.
        
        Returns:
            Dictionary with edge coordinates and vertex indexes
        """
        edges = self._analyze_geometry_edges()
        return edges.get('bottom', {})

    def get_left_edge(self) -> Dict:
        """
        Get the left edge of the boundary as a line segment with vertex indexes.
        
        Returns:
            Dictionary with edge coordinates and vertex indexes
        """
        edges = self._analyze_geometry_edges()
        return edges.get('left', {})

    def get_right_edge(self) -> Dict:
        """
        Get the right edge of the boundary as a line segment with vertex indexes.
        
        Returns:
            Dictionary with edge coordinates and vertex indexes
        """
        edges = self._analyze_geometry_edges()
        return edges.get('right', {})

    def get_all_edges(self) -> Dict[str, Dict]:
        """
        Get all four edges with their vertex indexes.
        
        Returns:
            Dictionary containing all edges with their vertex indexes
        """
        return self._analyze_geometry_edges()

    def update_vertex_by_index(self, vertex_index: int, new_coordinates: Tuple[float, float, float]) -> bool:
        """
        Update a specific vertex by its index.
        
        Args:
            vertex_index: Index of the vertex to update
            new_coordinates: New coordinates for the vertex
            
        Returns:
            True if successful, False if failed
        """
        if not self.geometry or vertex_index is None:
            return False

        try:
            vertices = list(self.geometry.get_vertices())
            if 0 <= vertex_index < len(vertices):
                vertices[vertex_index] = new_coordinates

                # Update the geometry
                faces = self.geometry.get_faces()
                self.geometry.mesh_data = {
                    "vertices": vertices,
                    "faces": faces
                }
                self.geometry._generate_brep_from_mesh()
                return True
        except Exception as e:
            print(f"Error updating vertex {vertex_index}: {e}")

        return False

    def extend_edge_to_point(self, edge_type: str, target_point: Tuple[float, float, float]) -> bool:
        """
        Extend a specific edge to a target point by moving its vertices.
        
        Args:
            edge_type: 'top', 'bottom', 'left', or 'right'
            target_point: Point to extend the edge toward
            
        Returns:
            True if successful, False if failed
        """
        # Get the edge with vertex indexes from geometry analysis
        edges = self._analyze_geometry_edges()

        if edge_type not in edges:
            return False

        edge = edges[edge_type]

        # Update vertices to extend toward target point
        success = True

        if edge.get('start_vertex_index') is not None:
            # Calculate new position for start vertex
            target_array = np.array(target_point)
            new_start = tuple(target_array)

            success &= self.update_vertex_by_index(edge['start_vertex_index'], new_start)

        if edge.get('end_vertex_index') is not None:
            # Calculate new position for end vertex
            target_array = np.array(target_point)
            new_end = tuple(target_array)

            success &= self.update_vertex_by_index(edge['end_vertex_index'], new_end)

        return success

    def get_geometry_bounds(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get the bounding box of the geometry.
        
        Returns:
            Dictionary with min and max coordinates
        """
        if not self.geometry or not self.geometry.get_vertices():
            return {}

        vertices = list(self.geometry.get_vertices())
        if not vertices:
            return {}

        # Find min and max for each coordinate
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        z_coords = [v[2] for v in vertices]

        return {
            'min_point': (min(x_coords), min(y_coords), min(z_coords)),
            'max_point': (max(x_coords), max(y_coords), max(z_coords)),
            'dimensions': (max(x_coords) - min(x_coords), 
                          max(y_coords) - min(y_coords), 
                          max(z_coords) - min(z_coords))
        }

    def get_start_point_bottom(self) -> Tuple[float, float, float]:
        """
        Get the start point of the bottom edge from the geometry
        
        Returns:
            Tuple with coordinates of the start point
        """
        bottom_edge = self.get_bottom_edge()
        return bottom_edge.get('start_point', (0.0, 0.0, 0.0))

    def get_end_point_bottom(self) -> Tuple[float, float, float]:
        """
        Get the end point of the bottom edge from the geometry
        
        Returns:
            Tuple with coordinates of the end point
        """
        bottom_edge = self.get_bottom_edge()
        return bottom_edge.get('end_point', (0.0, 0.0, 0.0))


@dataclass
class Space:
    """
    Represents a space in the building model, defined by its boundaries and properties.
    """

    # A human-readable name for the item
    name: str

    geometry: Geometry
    boundaries: List[Boundary] = field(default_factory=list)
    volume: float = 0.0
    area: float = 0.0
    relationships: Dict[str, List[Relationship]] = field(default_factory=lambda: defaultdict(list))
    topology: Optional[Cell] = None  # Topologic cell representing the space
    # A unique UUID
    id: str = field(default_factory=lambda: str(uuid4()))

    def centoid(self) -> Tuple[float, float, float]:
        """
        Calculate the centroid of the space based on its geometry.
        
        Returns:
            Tuple with coordinates of the centroid
        """
        if not self.geometry or not self.geometry.get_vertices():
            return (0.0, 0.0, 0.0)

        vertices = np.array(self.geometry.get_vertices())
        return tuple(np.mean(vertices, axis=0))

from abc import ABC, abstractmethod
class Graph(ABC):
    """
    Abstract base class representing a graph structure using KuzuDB.
    Subclasses should implement create_graph to define schema and initial data.
    """
    def __init__(self, db_path: str = "./demo_db"):
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self._initialize_graph()

    def _initialize_graph(self):
        """
        Internal method to call create_graph once during initialization.
        """
        self.create_graph()

    @abstractmethod
    def create_graph(self):
        """
        Create the graph schema and structure in the database.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement create_graph method.")

    def add_node(self, label: str, node_id: str = None, **attributes):
        """
        Add a node of a given label with attributes to the graph.
        """
        node_id = node_id or str(uuid.uuid4())
        all_attrs = {'id': node_id, **attributes}

        flat_attrs = {}
        for k, v in all_attrs.items():
            if v is None:
                continue
            elif isinstance(v, np.generic):
                flat_attrs[k] = v.item()
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    if subv is not None:
                        if isinstance(subv, np.generic):
                            subv = subv.item()
                        flat_attrs[subk] = subv  # ✅ no prefix
            else:
                flat_attrs[k] = v

        attr_str = ", ".join(f"{k}: {self._format_value(v)}" for k, v in flat_attrs.items())
        query = f"CREATE (:{label} {{ {attr_str} }})"
        self.conn.execute(query)
        print(query)
        return node_id

    def add_edge(self, from_id: str, to_id: str, rel_type: str, from_label: str = "Node", to_label: str = "Node", **attributes):
        """
        Add a relationship between two nodes by ID.
        """
        attr_str = ""
        if attributes:
            attr_str = "{" + ", ".join([f"{k}: {self._format_value(v)}" for k, v in attributes.items()]) + "}"

        query = f"""
        MATCH (a:{from_label} {{id: '{from_id}'}}), (b:{to_label} {{id: '{to_id}'}})
        CREATE (a)-[:{rel_type} {attr_str}]->(b)
        """
        self.conn.execute(query)

    def _format_value(self, val):
        """
        Format values for Cypher strings: handle numbers, strings, bools, etc.
        """
        if isinstance(val, str):
            return f"'{val}'"
        if isinstance(val, bool):
            return "true" if val else "false"
        return str(val)

    def query_to_string(self, query: str) -> str:
        """
        Execute a raw Cypher query against the graph database and return the results as a string.
        """
        try:
            result = self.conn.execute(query)
        except Exception as e:
            raise Exception(f"Query failed: {e}")

        if result.has_next():
            rows = []
            while result.has_next():
                row = result.get_next()
                rows.append(str(row))
            return "\n".join(rows)
        else:
            return "No results."

    def query(self, query: str):
        """
        Execute a Cypher query and return the result set.
        """
        try:
            return self.conn.execute(query)
        except Exception as e:
            raise Exception(f"Query failed: {e}")

    def get_node_types(self) -> List[str]:
        """
        Get a list of all node labels (types) used in the graph.
        """
        query = """
        MATCH (n)
        RETURN DISTINCT labels(n)[0] AS node_type
        """
        result = self.query(query)
        return [row['node_type'] for row in result.get_all()] if result else []

    def get_node_types_to_string(self) -> str:
        """
        Get a string representation of all node types in the graph.
        """
        query = """
        MATCH (n)
        RETURN DISTINCT label(n)
        """
        result = self.query(query)
        if result and result.has_next():
            rows = []
            while result.has_next():
                row = result.get_next()
                rows.append(str(row[0]))  # or `str(row)` if you want the full row object
            return ", ".join(rows)
        return "No node types found."

    def get_relationship_types(self) -> List[str]:
        """
        Get a list of all relationship types used in the graph.
        """
        query = """
        MATCH ()-[r]->()
        RETURN DISTINCT type(r) AS rel_type
        """
        result = self.query(query)
        return [row['rel_type'] for row in result.get_all()] if result else []

    def get_relationship_types_to_string(self) -> str:
        """
        Get a string representation of all relationship types in the graph.
        Assumes each relationship has a 'type' property.
        """
        query = """
        MATCH ()-[r]->()
        RETURN DISTINCT label(r) AS rel_type
        """
        result = self.query(query)
        if result and result.has_next():
            rows = []
            while result.has_next():
                row = result.get_next()
                rows.append(str(row[0]))
            return ", ".join(rows)
        return "No relationship types found."

    def get_connection_schema(self) -> List[Tuple[str, str, str]]:
        """
        Infers all distinct connection patterns from the graph in the form:
        (source_label, relationship_type, target_label)
        Assumes relationship type is stored as a property named 'type'.
        """
        query = """
        MATCH (a)-[r]->(b)
        RETURN DISTINCT label(a) AS source, label(r) AS rel_type, label(b) AS target
        """
        result = self.query(query)
        schema = []
        if result and result.has_next():
            while result.has_next():
                row = result.get_next()
                schema.append((str(row[0]), str(row[1]), str(row[2])))
        return schema

    def get_connection_schema_string(self) -> str:
        """
        Returns a string listing all connection types in the format:
        'Source --[REL]--> Target'
        """
        schema = self.get_connection_schema()
        if not schema:
            return "No connections found."
        return "\n".join(f"{src} --[{rel}]--> {tgt}" for src, rel, tgt in schema)


class BuildingGraph(Graph):
    """
    Represents the building graph structure in KuzuDB.
    Contains nodes for objects and edges for relationships.
    """
    def create_graph(self):
        """
        Create the building graph schema in the database.
        """
        # create node tables
        self.conn.execute("CREATE NODE TABLE IF NOT EXISTS Space(id STRING PRIMARY KEY, name STRING, volume FLOAT, centroid_x FLOAT, centroid_y FLOAT, centroid_z FLOAT)")
        # create node tables for other object types
        self.conn.execute("CREATE NODE TABLE IF NOT EXISTS Object(id STRING PRIMARY KEY, type STRING, volume FLOAT, centroid_x FLOAT, centroid_y FLOAT, centroid_z FLOAT)")
        self.conn.execute("CREATE NODE TABLE IF NOT EXISTS Element(id STRING PRIMARY KEY, type STRING, volume FLOAT, centroid_x FLOAT, centroid_y FLOAT, centroid_z FLOAT)")
        self.conn.execute("CREATE NODE TABLE IF NOT EXISTS Component(id STRING PRIMARY KEY, type STRING, volume FLOAT, centroid_x FLOAT, centroid_y FLOAT, centroid_z FLOAT)")
        self.conn.execute("CREATE NODE TABLE IF NOT EXISTS Boundary(id STRING PRIMARY KEY, boundary_id STRING, type STRING, is_access_boundary BOOL, is_visual_boundary BOOL, centroid_x FLOAT, centroid_y FLOAT, centroid_z FLOAT)")

        # create relationship tables
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS OBJECT_ADJACENT_TO(FROM Object TO Object, type STRING)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS BOUNDARY_ADJACENT_TO(FROM Boundary TO Boundary, type STRING)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS OBJECT_CREATES_BOUNDARY(FROM Object TO Boundary, type STRING)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS BOUNDARY_CREATES_SPACE(FROM Boundary TO Space, type STRING)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS SPACE_ADJACENT_TO(FROM Space TO Space, type STRING)")




class Model:
    """
    A class representing a building model, which can
    - contain objects and their components and elements.
    - contain spaces and zones.
    - contain relationships between all types of items
    """
    def __init__(self, name):
        self.name = name
        self.elements = {}
        self.components = {}
        self.objects = {}
        self.spaces = {}
        self.zones = {}
        self.relationships = defaultdict(list)
        self.boundaries = {}
        # Boundary Graph - Might collapse with building_graph in future
        self.boundary_graph = nx.Graph()
        self.id = str(uuid.uuid4())  # Unique identifier for the model
        self.building_graph = BuildingGraph(db_path=f"./building_dbs/{self.id}_building_graph.db")  # Initialize the building graph


    @classmethod
    def from_objects(cls, name, objects):
        """
        Create a model from a list of objects.
        """
        model = cls(name)
        for obj in objects:
            if isinstance(obj, Element):
                model.elements[obj.id] = obj
            elif isinstance(obj, Component):
                model.components[obj.id] = obj
            elif isinstance(obj, Object):
                model.objects[obj.id] = obj

        """
        Add object ids as nodes in the graph and enrich it with features about the nodes like:
        - type
        - length
        - width
        - height
        - volume

        """

        for obj_id, obj in model.objects.items():
            # Attempt to extract geometric features if they exist
            features = {
                "type": type(obj).__name__,  # class name, e.g., Wall, Door, etc.
                # "length": getattr(obj, "length", None),
                # "width": getattr(obj, "width", None),
                # "height": getattr(obj, "height", None),
                "volume": obj.geometry.compute_volume() if obj.geometry else 0.0,
                "centroid_x": obj.get_centroid().x,
                "centroid_y": obj.get_centroid().y,
                "centroid_z": obj.get_centroid().z,
                # "object": obj  # preserve full object for further use
            }

            model.building_graph.add_node("Object", node_id=obj.id, **features)

        model.create_object_adjacency_relationships(tolerance=0.001)
        model.infer_bounds(dimentions='3d')
        model.infer_spaces(dimentions='3d')
        model.generate_adjacency_graph()


        return model

    def create_object_adjacency_relationships(self, tolerance=0.01):
        """
        Add adjacency relationships between objects in the model to the building_graph.
        """
        for obj in self.objects.values():
            adjacent_items = obj.find_adjacent_items(self.objects.values(), tolerance=tolerance)
            for adjacent_item in adjacent_items:
                rel = AdjacentTo(obj.id, adjacent_item.id)
                self.relationships[obj.id].append(rel)

                # Add edge to graph
                self.building_graph.add_edge(obj.id, adjacent_item.id, "OBJECT_ADJACENT_TO", from_label="Object", to_label="Object")


    def heal_boundaries(self, tolerance=15.0):
        """Heal boundaries with comprehensive shape fixing and gap filling"""
        from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_Sewing, BRepBuilderAPI_MakePolygon, 
                                            BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeShell)
        from OCC.Core.gp import gp_Pnt, gp_Pln, gp_Dir, gp_Vec
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE
        from OCC.Core.ShapeFix import (ShapeFix_Shape, ShapeFix_Wireframe, 
                                    ShapeFix_Shell, ShapeFix_FixSmallFace)
        from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
        from OCC.Core.ShapeUpgrade import ShapeUpgrade_RemoveInternalWires
        from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
        from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
        from OCC.Core import TopoDS


        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing

        
        all_boundaries = list(self.boundaries.values())
        
        # Step 1: Create initial faces with better polygon construction
        initial_faces = []
        for boundary in all_boundaries:
            face = self._create_robust_face(boundary, tolerance)
            if face:
                initial_faces.append(face)

        print(f"Created {len(initial_faces)} initial faces")

        # Define face groups with indices to track overlaps
        # face_group_indices = [
        #     [0, 1, 2, 3, 7, 9],      # Group 1
        #     [3, 4, 5, 6, 8, 10]      # Group 2 (face 3 is shared)
        # ]

        shape_groups = self.find_enclosed_shape_groups(initial_faces)

        shape_groups.sort(key=lambda x: x['face_count'], reverse=True)

        filtered_shape_groups = []
        for current_group in shape_groups:
            current_faces = set(current_group['face_indices'])
            is_subset = False
            
            # Check if current group is subset of any already selected group
            for selected_group in filtered_shape_groups:
                selected_faces = set(selected_group['face_indices'])
                if current_faces.issubset(selected_faces):
                    is_subset = True
                    break
            
            # Only add if it's not a subset
            if not is_subset:
                filtered_shape_groups.append(current_group)

        face_group_indices = [group['face_indices'] for group in filtered_shape_groups]

        healed_face_groups = []
        # face_mapping = {i: [] for i in range(len(initial_faces))}
        face_mapping = {i: [] for i in range(len(initial_faces))}

        for group_idx, face_indices in enumerate(face_group_indices):
            face_indices = face_group_indices[group_idx]
            face_group = [initial_faces[i] for i in face_indices]

            # Create and configure the sewing tool
            sewer = BRepBuilderAPI_Sewing(0.1)
            sewer.SetNonManifoldMode(True)
            for face in face_group:
                sewer.Add(face)
            sewer.Perform()
            sewn_shape = sewer.SewedShape()

            healed_faces = []
            face_explorer = TopExp_Explorer(sewn_shape, TopAbs_FACE)

            while face_explorer.More():
                face = TopoDS.topods.Face(face_explorer.Current())
                healed_faces.append(face)
                face_explorer.Next()

            healed_face_groups.append(healed_faces)

            # map original face indices to healed faces
            for i in range(len(healed_faces)):
                original_idx = face_indices[i]
                face_mapping[original_idx].append(healed_faces[i])
        
        # if there are two versions of the same face, we need to merge them
        all_healed_faces = []
        for original_idx, healed_faces in face_mapping.items():
            if len(healed_faces) == 1:
                all_healed_faces.append(healed_faces[0])
            elif len(healed_faces) > 1:
                # Merge faces with a sewing operation
                sewer = BRepBuilderAPI_Sewing(0.1)
                for face in healed_faces:
                    sewer.Add(face)
                sewer.Perform()
                merged_shape = sewer.SewedShape()
                
                # Extract the merged face
                face_explorer = TopExp_Explorer(merged_shape, TopAbs_FACE)
                if face_explorer.More():
                    all_healed_faces.append(TopoDS.topods.Face(face_explorer.Current()))

        # Step 9: Update boundary geometries with improved vertex extraction
        self._update_boundary_geometries(all_boundaries, all_healed_faces)
        
        return all_healed_faces

    def _create_robust_face(self, boundary, tolerance):
        """Create a robust face from boundary with error handling"""
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.ShapeFix import ShapeFix_Wire
        
        try:
            vertices = boundary.geometry.get_vertices()
            if len(vertices) < 3:
                return None
            
            # Create polygon with validation
            polygon = BRepBuilderAPI_MakePolygon()
            points = []
            
            for vertex in vertices:
                if hasattr(vertex, 'x'):
                    point = gp_Pnt(vertex.x, vertex.y, vertex.z)
                else:
                    point = gp_Pnt(float(vertex[0]), float(vertex[1]), float(vertex[2]))
                points.append(point)
                polygon.Add(point)
            
            # Ensure polygon is closed
            if not polygon.Wire().Closed():
                polygon.Close()
            
            if not polygon.IsDone():
                return None
            
            # Fix the wire before creating face
            wire = polygon.Wire()
            wire_fixer = ShapeFix_Wire()
            wire_fixer.SetPrecision(tolerance)
            wire_fixer.Load(wire)
            wire_fixer.Perform()
            fixed_wire = wire_fixer.Wire()
            
            # Create face
            face_maker = BRepBuilderAPI_MakeFace(fixed_wire)
            if face_maker.IsDone():
                return face_maker.Face()
            
        except Exception as e:
            print(f"Error creating face for boundary: {e}")
        
        return None

    

    def get_face_normal_and_center(self, face):
        """Get the center point and outward normal of a face"""
        
        # Get face centroid
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        centroid = props.CentreOfMass()
        center = np.array([centroid.X(), centroid.Y(), centroid.Z()])
        
        # Get surface and normal at center
        surface = BRep_Tool.Surface(face)
        umin, umax, vmin, vmax = breptools.UVBounds(face)
        u_center = (umin + umax) / 2
        v_center = (vmin + vmax) / 2
        
        try:
            props_surface = GeomLProp_SLProps(surface, u_center, v_center, 1, 1e-6)
            if props_surface.IsNormalDefined():
                normal = props_surface.Normal()
                normal_vec = np.array([normal.X(), normal.Y(), normal.Z()])
                normal_vec = normal_vec / np.linalg.norm(normal_vec)
            else:
                normal_vec = np.array([0, 0, 1])
        except:
            normal_vec = np.array([0, 0, 1])
        
        return center, normal_vec, -normal_vec

    def find_convergence_point(self, face_centers, face_normals):
        """Find where inward-pointing normals converge"""
        
        if len(face_centers) < 3:
            return None, float('inf')
        
        # For each pair of faces, find intersection of inward normal lines
        intersection_points = []
        
        for i in range(len(face_centers)):
            for j in range(i + 1, len(face_centers)):
                center_i, center_j = face_centers[i], face_centers[j]
                inward_i, inward_j = -face_normals[i], -face_normals[j]
                
                # Solve line intersection: center_i + t*inward_i ≈ center_j + s*inward_j
                w = center_i - center_j
                a = np.dot(inward_i, inward_i)
                b = np.dot(inward_i, inward_j)
                c = np.dot(inward_j, inward_j)
                d = np.dot(inward_i, w)
                e = np.dot(inward_j, w)
                
                denom = a * c - b * b
                if abs(denom) > 1e-10:
                    t = (b * e - c * d) / denom
                    s = (a * e - b * d) / denom
                    
                    point_i = center_i + t * inward_i
                    point_j = center_j + s * inward_j
                    intersection_points.append((point_i + point_j) / 2)
        
        if not intersection_points:
            return None, float('inf')
        
        # Find average convergence point
        convergence_point = np.mean(intersection_points, axis=0)
        
        # Calculate how well normals converge (lower is better)
        convergence_error = 0
        for point in intersection_points:
            convergence_error += np.linalg.norm(point - convergence_point)
        
        return convergence_point, convergence_error / len(intersection_points)

    # Extract your existing logic into a standalone function
    def process_single_face_combination(self, face_indices, face_data, faces, group_size, max_convergence_error, 
                                    find_convergence_point_func, edges_are_close_func):
        """
        Process a single face combination - this is your existing inner loop logic.
        """
        valid_groups = []
        
        # Your existing code with minimal changes:
        # Try both normal orientations for each face
        face_group_data = [face_data[i] for i in face_indices]
        
        # For each face, we can choose either normal1 or normal2
        # Try all 2^n combinations
        for combo_bits in range(2**len(face_group_data)):
            centers = []
            normals = []
            
            for bit_pos, (center, normal1, normal2, _) in enumerate(face_group_data):
                centers.append(center)
                # Use bit to choose which normal direction
                if (combo_bits >> bit_pos) & 1:
                    normals.append(normal2)
                else:
                    normals.append(normal1)
            
            # Find convergence point for this combination
            conv_point, error = find_convergence_point_func(centers, normals)
            
            # Add directional validation after the convergence point check
            if conv_point is not None and error < max_convergence_error:
                # ... [rest of your existing validation logic] ...
                # [I'm keeping the structure but truncating for brevity]
                
                # Check for directional coverage (6 cardinal directions)
                directions = {
                    'north': [0, 1, 0],   # +Y
                    'south': [0, -1, 0],  # -Y  
                    'east': [1, 0, 0],    # +X
                    'west': [-1, 0, 0],   # -X
                    'up': [0, 0, 1],      # +Z
                    'down': [0, 0, -1]    # -Z
                }
                
                direction_coverage = {direction: False for direction in directions}
                
                valid = True
                distances = []
                
                for center, normal in zip(centers, normals):
                    to_face = center - conv_point
                    distance = np.linalg.norm(to_face)
                    distances.append(distance)
                    
                    if distance > 1e-10:
                        to_face_norm = to_face / distance
                        if np.dot(normal, to_face_norm) < 0.3:  # Should point away
                            valid = False
                            break
                    
                    # Check which direction this normal covers
                    for dir_name, dir_vec in directions.items():
                        if np.dot(normal, dir_vec) > 0.7:  # Strong alignment
                            direction_coverage[dir_name] = True
                
                #Require coverage in at least 4 directions for a valid box
                covered_directions = sum(direction_coverage.values())
                if covered_directions < 4:
                    valid = False

                if not direction_coverage['up'] or not direction_coverage['down']:
                    valid = False
                
                # Also check for opposing pairs
                opposing_pairs = 0
                for i in range(len(normals)):
                    for j in range(i + 1, len(normals)):
                        if np.dot(normals[i], normals[j]) < -0.7:
                            opposing_pairs += 1
                
                if opposing_pairs < 2:  # Need at least 2 opposing pairs
                    valid = False

                # Check that all faces have one edge within a tolerance of at least one other face edge
                edge_proximity_valid = True
                proximity_tolerance = 0.1  # Adjust as needed
                
                if edge_proximity_valid:  # Only check if previous validations passed
                    face_has_nearby_edge = [False] * len(face_indices)
                    
                    for i, face_idx_i in enumerate(face_indices):
                        face_i = faces[face_idx_i]
                        
                        # Get edges from face i
                        edges_i = []
                        edge_explorer_i = TopExp_Explorer(face_i, TopAbs_EDGE)
                        while edge_explorer_i.More():
                            edge_i = topods.Edge(edge_explorer_i.Current())
                            edges_i.append(edge_i)
                            edge_explorer_i.Next()
                        
                        # Check against all other faces
                        for j, face_idx_j in enumerate(face_indices):
                            if i == j:
                                continue
                                
                            face_j = faces[face_idx_j]
                            
                            # Get edges from face j
                            edges_j = []
                            edge_explorer_j = TopExp_Explorer(face_j, TopAbs_EDGE)
                            while edge_explorer_j.More():
                                edge_j = topods.Edge(edge_explorer_j.Current())
                                edges_j.append(edge_j)
                                edge_explorer_j.Next()
                            
                            # Check if any edge from face i is close to any edge from face j
                            for edge_i in edges_i:
                                for edge_j in edges_j:
                                    if self.edges_are_close(edge_i, edge_j, proximity_tolerance):
                                        face_has_nearby_edge[i] = True
                                        break
                                if face_has_nearby_edge[i]:
                                    break
                            if face_has_nearby_edge[i]:
                                break
                    
                    # All faces must have at least one nearby edge
                    if not all(face_has_nearby_edge):
                        edge_proximity_valid = False
                
                if not edge_proximity_valid:
                    valid = False
                        
                # At the end, if valid, add to results
                if valid and len(distances) > 0:
                    avg_dist = np.mean(distances)
                    dist_variance = np.var(distances) / (avg_dist**2) if avg_dist > 0 else float('inf')
                    
                    if dist_variance < 1.0:  # Reasonable consistency
                        valid_group = {
                            'face_indices': list(face_indices),
                            'faces': [faces[i] for i in face_indices],
                            'convergence_point': conv_point,
                            'error': error,
                            'distances': distances,
                            'centers': centers,
                            'normals': normals,
                            'face_count': group_size
                        }
                        valid_groups.append(valid_group)
        
        return valid_groups
    
    def process_combinations_multiprocessing(self, faces, face_data, group_size, max_convergence_error):
        """
        Replace your existing loop with this multiprocessing version.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        
        total_combinations = comb(len(faces), group_size)

        ## TODO set workers as a config

        max_workers = min(multiprocessing.cpu_count(), 8)  # Don't overwhelm the system
        
        valid_groups_for_size = []
        
        # Submit all combinations to the process pool
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_combination = {}
            for face_indices in combinations(range(len(faces)), group_size):
                future = executor.submit(
                    self.process_single_face_combination,
                    face_indices,
                    face_data,
                    faces,
                    group_size,
                    max_convergence_error,
                    self.find_convergence_point,  # Pass method as function
                    self.edges_are_close          # Pass method as function
                )
                future_to_combination[future] = face_indices
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_combination), 
                                total=total_combinations,
                                desc=f"Processing combinations (workers={max_workers})"):
                try:
                    valid_groups = future.result()
                    valid_groups_for_size.extend(valid_groups)
                except Exception as exc:
                    face_indices = future_to_combination[future]
                    print(f'Combination {face_indices} generated exception: {exc}')
        
        return valid_groups_for_size

    def group_faces_by_normal_convergence(self, faces, max_convergence_error=3):


        def plot_convergence(centers, normals, convergence_point):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for center, normal in zip(centers, normals):
                ax.quiver(center[0], center[1], center[2], 
                            normal[0], normal[1], normal[2], 
                            length=0.5, normalize=True, color='b', alpha=0.5)
                ax.scatter(center[0], center[1], center[2], color='r', s=50)
            if convergence_point is not None:
                ax.scatter(convergence_point[0], convergence_point[1], convergence_point[2], color='g', s=100, label='Convergence Point')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title(f'Convergence Point for Group')
            plt.legend()
            plt.show()

        plot = True  # Set to True to visualize convergence
        # Get face data
        face_data = []
        for i, face in enumerate(faces):
            center, normal, neg_normal = self.get_face_normal_and_center(face)
            face_data.append((center, normal, neg_normal, i))
        
        all_valid_groups = []
        # used_faces = set()
        
        # Try different group sizes, starting with larger groups
        for group_size in range(min(6, len(faces)), 5, -1):
            # if len(used_faces) >= len(faces) - 2:
            #     break
            print('Testing group size:', group_size)
            valid_groups_for_size = []
            
            total_combinations = comb(len(faces), group_size)

            valid_groups_for_size = self.process_combinations_multiprocessing(
                faces, face_data, group_size, max_convergence_error
            )
            
            # Add all valid groups for this size
            for group in valid_groups_for_size:
                # Only add if faces aren't already used
                # if not any(face_idx in used_faces for face_idx in group['face_indices']):
                all_valid_groups.append(group)
                # used_faces.update(group['face_indices'])
        
        return all_valid_groups

    # Main function
    def find_enclosed_shape_groups(self, faces):
        """Find groups of faces that likely form enclosed shapes"""

        groups = self.group_faces_by_normal_convergence(faces)

        # Sort by lower error first and then by larger face count


        groups.sort(key=lambda x: (x['error'], -x['face_count']))


        min_error = min([g['error'] for g in groups])
        max_face_count = max([g['face_count'] for g in groups])

        # Select groups with minimum error and maximum face count
        groups = [g for g in groups if g['error'] == min_error and g['face_count'] == max_face_count]

        
        
        print(f"Found {len(groups)} potential enclosed shapes:")
        for i, group in enumerate(groups):
            print(f"  Group {i+1}: {len(group['faces'])} faces")
            print(f"    Convergence error: {group['error']:.4f}")
            print(f"    Face indices: {group['face_indices']}")
            print(f"    Convergence point: {group['convergence_point']}")
        
        return groups

    def edges_are_close(self, edge1, edge2, tolerance):
        """Check if two edges have points within tolerance distance"""
        
        # Sample points along each edge
        def sample_edge_points(edge, num_samples=10):
            points = []
            curve, first_param, last_param = BRep_Tool.Curve(edge)
            if curve:
                for i in range(num_samples):
                    param = first_param + i * (last_param - first_param) / (num_samples - 1)
                    point = curve.Value(param)
                    points.append(np.array([point.X(), point.Y(), point.Z()]))
            return points
        
        points1 = sample_edge_points(edge1)
        points2 = sample_edge_points(edge2)
        
        # Check if any point from edge1 is close to any point from edge2
        for p1 in points1:
            for p2 in points2:
                if np.linalg.norm(p1 - p2) < tolerance:
                    return True
        return False

    def _update_boundary_geometries(self, boundaries, healed_faces):
        """Update boundary geometries with improved vertex ordering"""
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopTools import TopTools_ListOfShape
        from OCC.Core.ShapeAnalysis import ShapeAnalysis_WireOrder
        
        for i, boundary in enumerate(boundaries):
            if i < len(healed_faces):
                try:
                    # Extract vertices in proper order by following edges
                    vertices = self._extract_ordered_vertices(healed_faces[i])
                    
                    if len(vertices) >= 3:
                        # Create triangulated faces for mesh data
                        faces = []
                        for j in range(1, len(vertices) - 1):
                            faces.append((0, j, j + 1))
                        
                        boundary.geometry.mesh_data["vertices"] = vertices
                        boundary.geometry.mesh_data["faces"] = faces
                        boundary.geometry.oc_geometry = healed_faces[i]
                        
                except Exception as e:
                    print(f"Error updating boundary {i}: {e}")

    def _extract_ordered_vertices(self, face):
        """Extract vertices from face in proper order following wire edges"""
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core import TopoDS
        
        vertices = []
        
        # Get the outer wire of the face
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        if wire_explorer.More():
            wire = wire_explorer.Current()
            
            # Follow edges in order
            edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
            while edge_explorer.More():
                edge = edge_explorer.Current()
                
                # Get vertices of this edge
                vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
                edge_vertices = []
                while vertex_explorer.More():
                    vertex = vertex_explorer.Current()
                    point = BRep_Tool.Pnt(vertex)
                    edge_vertices.append((point.X(), point.Y(), point.Z()))
                    vertex_explorer.Next()
                
                # Add first vertex if this is the first edge, otherwise add second
                if len(vertices) == 0 and len(edge_vertices) > 0:
                    vertices.append(edge_vertices[0])
                if len(edge_vertices) > 1:
                    vertices.append(edge_vertices[1])
                
                edge_explorer.Next()
        
        return vertices

    def extract_healed_faces(self, healed_shape):
        """Extract individual faces from healed shape with validation"""
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        
        faces = []
        face_explorer = TopExp_Explorer(healed_shape, TopAbs_FACE)
        
        while face_explorer.More():
            face = face_explorer.Current()
            
            # Validate face before adding
            analyzer = BRepCheck_Analyzer(face)
            if analyzer.IsValid():
                faces.append(face)
            else:
                print("Warning: Invalid face detected and skipped")
            
            face_explorer.Next()
        
        return faces



    def infer_bounds(self, dimentions: str = "3d"):
        """
        Find bounds of spaces in the model by exploring walls and wall like objects to determine where spaces
        are located and split up.
        
        Bounds can include walls that extend from the floor to the ceiling, these would be full boundaries and are access and visual boundaries. They can include partial boundaries
        that are not full height these could be a access boundary but not a visual boundary and they could be open boundaries which are infered boundaries usually because walls may extend 
        towards each other from different sides of the space but not meet like in a art gallery. 

        Returns:
            list: A list of boundary objects representing the bounds of the spaces.
        """

        ## TODO infer 3D bounds as well (EG Floors and Ceilings)

        # Find wall objects in the model that are of class Wall or inharit from Wall
        wall_objects = [obj for obj in self.objects.values() if isinstance(obj, Wall)]

        deck_objects = [obj for obj in self.objects.values() if isinstance(obj, Deck)]
        # Find wall objects in the model that are of class Wall or inharit from Wall

        # for each wall determine if its a full height wall based on its adjacency relationships to decks at the top and the bottom

        decks_connected_to_walls = {}
        for wall in wall_objects:
            decks = []
            # find the wall in the building graph and find its adjacency relationships
            wall_id = wall.id
            wall_relationships = self.relationships[wall_id]
            for r in wall_relationships:
                # if the relationship type is AdjacentTo
                if isinstance(r, AdjacentTo):
                    # check if the target is a deck
                    target = self.objects[r.target]
                    if isinstance(target, Deck):
                        decks.append(target)
                        decks_connected_to_walls[target.id] = target
            # determine the max z distance of the decks
            max_z = max([deck.get_centroid().z for deck in decks])
            min_z = min([deck.get_centroid().z for deck in decks])

            # determine the % of this span that the wall height covers
            wall_height = wall.get_height()
            wall_span = max_z - min_z
            wall_height_ratio = wall_height / wall_span

            # if the wall height is greater than 90% of the span then its a full height wall
            if wall_height_ratio > 0.7:
                boundary = Boundary(
                    name=wall.name,
                    type='full',
                    geometry=wall.get_centerplane_geometry(),
                    is_access_boundary=True,
                    is_visual_boundary=True,
                    base_item=wall,
                    height=wall_height,
                    normal_vector=wall.get_centerplane_normal_vector(),
                    adjacent_spaces=[]  # This will be filled later
                )

                self.boundaries[boundary.id] = boundary
                self.boundary_graph.add_node(boundary.id,
                                             type=boundary.type,
                                             geometry=boundary.geometry,
                                             is_access_boundary=boundary.is_access_boundary,
                                             is_visual_boundary=boundary.is_visual_boundary,
                                             base_item=boundary.base_item,
                                             height=boundary.height,
                                             normal_vector=boundary.normal_vector, 
                                             centroid_x=wall.get_centroid().x,
                                             centroid_y=wall.get_centroid().y,
                                             centroid_z=wall.get_centroid().z)

                features = {
                    'boundary_id': boundary.id,
                    'type': boundary.type,
                    'is_access_boundary': boundary.is_access_boundary,
                    'is_visual_boundary': boundary.is_visual_boundary,
                    'centroid_x': wall.get_centroid().x,
                    'centroid_y': wall.get_centroid().y,
                    'centroid_z': wall.get_centroid().z
                }


                self.building_graph.add_node('Boundary', node_id=boundary.id, features=features)

                rel = Creates(wall.id, boundary.id)
                boundary.relationships.append(rel)
                self.relationships[boundary.id].append(rel)
                self.building_graph.add_edge(wall.id, boundary.id, "OBJECT_CREATES_BOUNDARY", from_label='Object', to_label='Boundary')

                # add boundary_id to the walls boundary_id attribute
                wall.boundary_id = boundary.id

            elif wall_height_ratio < 0.7:
                boundary = Boundary(
                    id=generate_id('boundary'),
                    type='full',
                    geometry=wall.get_centerplane_geometry(),
                    is_access_boundary=True,
                    is_visual_boundary=True,
                    base_item=wall,
                    height=wall_height,
                    normal_vector=wall.get_centerplane_normal_vector(),
                    adjacent_spaces=[]  # This will be filled later
                )

                self.boundaries[boundary.id] = boundary
                self.boundary_graph.add_node(boundary.id,
                                             type=boundary.type,
                                             geometry=boundary.geometry,
                                             is_access_boundary=boundary.is_access_boundary,
                                             is_visual_boundary=boundary.is_visual_boundary,
                                             base_item=boundary.base_item,
                                             height=boundary.height,
                                             normal_vector=boundary.normal_vector, 
                                             centroid_x=wall.get_centroid().x,
                                             centroid_y=wall.get_centroid().y,
                                             centroid_z=wall.get_centroid().z)

                # add boundary_id to the walls boundary_id attribute
                wall.boundary_id = boundary.id

                features = {
                    'boundary_id': boundary.id,
                    'type': boundary.type,
                    'is_access_boundary': boundary.is_access_boundary,
                    'is_visual_boundary': boundary.is_visual_boundary,
                    'centroid_x': wall.get_centroid().x,
                    'centroid_y': wall.get_centroid().y,
                    'centroid_z': wall.get_centroid().z
                }

                self.building_graph.add_node('Boundary', node_id=boundary.id, features=features)

                # add relatinoship between the wall and the boundary
                rel = Creates(wall.id, boundary.id)
                boundary.relationships.append(rel)
                self.relationships[boundary.id].append(rel)
                self.building_graph.add_edge(wall.id, boundary.id, "OBJECT_CREATES_BOUNDARY", from_label='Object', to_label='Boundary')                                             

            else:
                boundary = Boundary(
                    id=generate_id('boundary'),
                    type='open',
                    geometry=wall.get_centerplane_geometry(),
                    is_access_boundary=False,
                    is_visual_boundary=False,
                    base_item=wall,
                    height=wall_height,
                    normal_vector=wall.get_centerplane_normal_vector(),
                    adjacent_spaces=[]  # This will be filled later
                )

                self.boundaries[boundary.id] = boundary
                self.boundary_graph.add_node(boundary.id,
                                            type=boundary.type,
                                            geometry=boundary.geometry,
                                            is_access_boundary=boundary.is_access_boundary,
                                            is_visual_boundary=boundary.is_visual_boundary,
                                            base_item=boundary.base_item,
                                            height=boundary.height,
                                            normal_vector=boundary.normal_vector,
                                            centroid_x=wall.get_centroid().x,
                                            centroid_y=wall.get_centroid().y,
                                            centroid_z=wall.get_centroid().z)


                # add boundary_id to the walls boundary_id attribute
                wall.boundary_id = boundary.id

                features = {
                    'boundary_id': boundary.id,
                    'type': boundary.type,
                    'is_access_boundary': boundary.is_access_boundary,
                    'is_visual_boundary': boundary.is_visual_boundary,
                    'centroid_x': wall.get_centroid().x,
                    'centroid_y': wall.get_centroid().y,
                    'centroid_z': wall.get_centroid().z
                }

                self.building_graph.add_node('Boundary', node_id=boundary.id, features=features)

                rel = Creates(wall.id, boundary.id)
                boundary.relationships.append(rel)
                self.relationships[boundary.id].append(rel)
                self.building_graph.add_edge(wall.id, boundary.id, "OBJECT_CREATES_BOUNDARY", from_label='Object', to_label='Boundary')

        if dimentions == '3d':
            for deck in deck_objects:
                # get the deck center plane geometry
                deck_geometry = deck.get_centerplane_geometry()
                # create a boundary for the deck
                boundary = Boundary(
                    name=deck.name,
                    type='deck',
                    geometry=deck_geometry,
                    is_access_boundary=True,
                    is_visual_boundary=True,
                    base_item=deck,
                    height=deck.get_height(),
                    normal_vector=deck.get_centerplane_normal_vector(),
                    adjacent_spaces=[]  # This will be filled later
                )
                self.boundaries[boundary.id] = boundary
                self.boundary_graph.add_node(boundary.id,
                                                type=boundary.type,
                                                geometry=boundary.geometry,
                                                is_access_boundary=boundary.is_access_boundary,
                                                is_visual_boundary=boundary.is_visual_boundary,
                                                base_item=boundary.base_item,
                                                height=boundary.height,
                                                normal_vector=boundary.normal_vector, 
                                                centroid_x=deck.get_centroid().x,
                                                centroid_y=deck.get_centroid().y,
                                                centroid_z=deck.get_centroid().z)
                # add boundary_id to the decks boundary_id attribute
                deck.boundary_id = boundary.id

                features = {
                    'boundary_id': boundary.id,
                    'type': boundary.type,
                    'is_access_boundary': boundary.is_access_boundary,
                    'is_visual_boundary': boundary.is_visual_boundary,
                    'centroid_x': deck.get_centroid().x,
                    'centroid_y': deck.get_centroid().y,
                    'centroid_z': deck.get_centroid().z
                }

                self.building_graph.add_node('Boundary', node_id=boundary.id, features=features)

                # add relatinoship between the deck and the boundary
                rel = Creates(deck.id, wall.id)
                boundary.relationships.append(rel)
                self.relationships[boundary.id].append(rel)
                self.building_graph.add_edge(deck.id, boundary.id, "OBJECT_CREATES_BOUNDARY", from_label='Object', to_label='Boundary')
                # Don't process decks
                


        # Now lets heal the boundaries by finding intersections and extending them
        # self.heal_boundaries(dimentions=dimentions)
        occ_faces = self.heal_boundaries(tolerance=25.0)

        # test_healing_validation(self.boundaries, occ_faces)

        # Now lets create the adjacency relationships between boundaries
        for boundary_id, boundary in self.boundaries.items():
            # Find adjacent boundaries based on their geometry
            for other_boundary_id, other_boundary in self.boundaries.items():
                if boundary_id == other_boundary_id:
                    continue
                # Check if the boundaries intersect this needs to be a mesh intersect
                if boundary.geometry.mesh_intersects(other_boundary.geometry):
                    # Create adjacency relationship
                    rel = AdjacentTo(boundary_id, other_boundary_id)
                    boundary.relationships.append(rel)

                    self.relationships[boundary_id].append(rel)
                    self.boundary_graph.add_edge(boundary_id, other_boundary_id, relationship=rel.type)

                    # Add the other boundary to the adjacent spaces list
                    boundary.adjacent_spaces.append(other_boundary_id)
                    other_boundary.adjacent_spaces.append(boundary_id)

                    # Add to the building graph
                    self.building_graph.add_edge(boundary_id, other_boundary_id, 'BOUNDARY_ADJACENT_TO', from_label='Boundary', to_label='Boundary')

        print(f"Boundaries inferred: {len(self.boundaries)}")



    def infer_spaces(self, dimentions: str = "3d") -> List[Space]:
        """
        Infer spaces by finding boundary cycles in the boundary graph (networkx) whose edges form closed loops.
        This method will create Space objects from the boundaries and their relationships.
        Returns:
            List[Space]: A list of Space objects representing the inferred spaces.
        """
        space_counter = 0
        if dimentions == '2d':



            # Function to cut a line at given points
            from shapely.geometry import Polygon, LineString, Point
            from shapely.ops import linemerge

            def cut_line_at_points(line, points):
                # First coords of line
                coords = list(line.coords)

                # Keep list coords where to cut (cuts = 1)
                cuts = [0] * len(coords)
                cuts[0] = 1
                cuts[-1] = 1

                # Add the coords from the points
                coords += [list(p.coords)[0] for p in points]    
                cuts += [1] * len(points)        

                # Calculate the distance along the line for each point    
                dists = [line.project(Point(p)) for p in coords]    
                # sort the coords/cuts based on the distances    
                # see http://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list    
                coords = [p for (d, p) in sorted(zip(dists, coords))]    
                cuts = [p for (d, p) in sorted(zip(dists, cuts))]          

                # generate the Lines    
                #lines = [LineString([coords[i], coords[i+1]]) for i in range(len(coords)-1)]    
                lines = []        

                for i in range(len(coords)-1):    
                    if cuts[i] == 1:    
                        # find next element in cuts == 1 starting from index i + 1   
                        j = cuts.index(1, i + 1)    
                        lines.append(LineString(coords[i:j+1]))            

                return lines


            # # Step 1: Find and normalize unique cycles
            # all_cycles = list(nx.simple_cycles(self.boundary_graph))
            # unique_cycles = deduplicate_cycles_by_nodes(all_cycles)

            # find basis_cycles in the boundary graph
            unique_cycles = list(nx.cycle_basis(self.boundary_graph))

            print(f"Found {len(unique_cycles)} cycles in the boundary graph.")
            for cycle in unique_cycles:
                # determine if the cycle is a valid space by checking if the normal vectors of the boundaries are consistent
                cycle_boundaries = [self.boundaries[b_id] for b_id in cycle]
                if not cycle_boundaries:
                    continue
                # check if you plot the bottom edges of the boundaries in the cycle they form a closed polygon
                bottom_edges = [boundary.get_bottom_edge() for boundary in cycle_boundaries]
                if not bottom_edges:
                    continue
                # Check if the bottom edges form a closed polygon
                edge_lines = []
                for edge in bottom_edges:
                    start = edge['start_point']
                    end = edge['end_point']
                    edge_lines.append(LineString([start, end]))
                # Create a polygon from the bottom edges by extracting their start and end points and combining them as vertices

                # edge_vertices = [edge['start_point'] for edge in bottom_edges] + [edge['end_point'] for edge in bottom_edges]
                # edge_vertices_ordered = Geometry.order_vertices_by_angle(edge_vertices)
                # # add the first vertex to the end to close the polygon
                # if edge_vertices_ordered[0] != edge_vertices_ordered[-1]:
                #     edge_vertices_ordered.append(edge_vertices_ordered[0])

                # # combined_line = LineString(edge_vertices_ordered)
                # combined_line = linemerge(edge_lines)
                # Check if the combined line is closed

                # turn edge lines into a topologicpy Edge object

                topologic_edges = []
                for edge in edge_lines:
                    s_v = Vertex.ByCoordinates(x=edge.coords[0][0], y=edge.coords[0][1], z=edge.coords[0][2])
                    e_v = Vertex.ByCoordinates(x=edge.coords[1][0], y=edge.coords[1][1], z=edge.coords[1][2])
                    topologic_edges.append(Edge.ByStartVertexEndVertex(s_v, e_v))

                face = Face.ByEdges(topologic_edges, tolerance=0.01)

                if face:


                    cell = Cell.ByThickenedFace(face, thickness=max([boundary.height for boundary in cycle_boundaries]))
                    # Create geometry from the cell
                    geometry = Geometry.from_topology(cell)
                    space = Space(
                        name="Space {}".format(space_counter),
                        geometry=geometry,
                        boundaries=cycle_boundaries,
                        area=Face.Area(face) # Assuming area as a proxy for area in 2D
                    )

                    cell_dict = Topology.Dictionary(cell)
                    cell_dict = Dictionary.SetValueAtKey(cell_dict, 'space_id', space.id)
                    cell = Topology.SetDictionary(cell, cell_dict)
                    space.topology = cell

                    self.spaces[space.id] = space
                    space_counter += 1

                    # Add the space to the building graph
                    features = {
                        'name': space.name,
                        'volume': space.geometry.compute_volume(),
                        'centroid_x': space.geometry.get_centroid().x,
                        'centroid_y': space.geometry.get_centroid().y,
                        'centroid_z': space.geometry.get_centroid().z
                    }
                    self.building_graph.add_node('Space', node_id=space.id, features=features)

                    # add relationship between bounds and spaces
                    for boundary in space.boundaries:
                        rel = Creates(boundary.id, space.id)
                        boundary.relationships.append(rel)
                        self.relationships[boundary.id].append(rel)
                        self.building_graph.add_edge(boundary.id, space.id, "BOUNDARY_CREATES_SPACE", from_label='Boundary', to_label='Space')


                else:
                    # see if we can heal the boundaries by trimming them to form a closed polygon
                    print(f"Cycle {cycle} does not form a valid polygon, attempting to heal boundaries...")
                    # Attempt to heal boundaries by extending edges to form a closed polygon
                    edge_graph = nx.Graph()
                    points = []
                    cleaned_edges = {i: edge for i, edge in enumerate(edge_lines)}
                    for i in range(len(edge_lines)):
                        for j in range(len(edge_lines)):
                            edge = cleaned_edges[i]
                            other_edge = cleaned_edges[j]
                            if edge == other_edge:
                                continue
                            if edge.intersects(other_edge):
                                intersection = edge.intersection(other_edge)
                                if isinstance(intersection, Point):
                                    # split the edge at the intersection point
                                    cut_lines = cut_line_at_points(edge, [intersection])

                                    # find the largest line segment
                                    largest_line = max(cut_lines, key=lambda l: l.length)
                                    cleaned_edges[i]=largest_line
                                elif isinstance(intersection, LineString):
                                    # If intersection is a line, take its endpoints
                                    continue

                            else:
                                continue # No intersection, keep original edge
                    # get edges as a list
                    cleaned_edges = list(cleaned_edges.values())
                    topologic_edges = []
                    for edge in cleaned_edges:
                        s_v = Vertex.ByCoordinates(x=edge.coords[0][0], y=edge.coords[0][1], z=edge.coords[0][2])
                        e_v = Vertex.ByCoordinates(x=edge.coords[1][0], y=edge.coords[1][1], z=edge.coords[1][2])
                        topologic_edges.append(Edge.ByStartVertexEndVertex(s_v, e_v))

                    face = Face.ByEdges(topologic_edges, tolerance=0.01, silent=True)
                    if face:
                        space_height = max([boundary.height for boundary in cycle_boundaries])
                        cell = Cell.ByThickenedFace(face, thickness=space_height)

                        # Create geometry from the cell
                        geometry = Geometry.from_topology(cell)
                        space = Space(
                            name="Space {}".format(space_counter),
                            boundaries=cycle_boundaries,
                            area=Face.Area(face),
                            geometry=geometry,  # Assuming area as a proxy for area in 2D
                        )

                        cell_dict = Topology.Dictionary(cell)
                        cell_dict = Dictionary.SetValueAtKey(cell_dict, 'space_id', space.id)
                        cell = Topology.SetDictionary(cell, cell_dict)
                        space.topology = cell

                        self.spaces[space.id] = space
                        space_counter += 1

                        # Add the space to the building graph
                        features = {
                            'name': space.name,
                            'volume': space.geometry.compute_volume(),
                            'centroid_x': space.geometry.get_centroid().x,
                            'centroid_y': space.geometry.get_centroid().y,
                            'centroid_z': space.geometry.get_centroid().z
                        }  

                        self.building_graph.add_node('Space', node_id=space.id, features=features)

                        # add relationship between bounds and spaces
                        for boundary in space.boundaries:
                            rel = Creates(boundary.id, space.id)
                            boundary.relationships.append(rel)
                            self.relationships[boundary.id].append(rel)
                            self.building_graph.add_edge(boundary.id, space.id, "BOUNDARY_CREATES_SPACE", from_label='Boundary', to_label='Space')

        elif dimentions == '3d':
            from itertools import combinations
            def process_face_combo(combo):
                # Local import of Face and Cell if needed, or pass them as globals depending on your env
                try:
                    normals = [Face.Normal(face) for face in combo]
                    normals = np.array(normals)
                    net = np.sum(normals, axis=0)
                    magnitude = np.linalg.norm(net)

                    if magnitude > 1:
                        return (combo, magnitude, None)  # skip
                    else:
                        cell = Cell.ByFaces(list(combo), tolerance=0.01)
                        return (combo, magnitude, cell)

                except Exception as e:
                    return (combo, float("inf"), None)  # error path

            def occ_to_topologic(occ_shape):
                from OCC.Core.BRepTools import breptools
                # generate temp file
                temp_path = f'temp_{uuid4()}.brep'
                # write occ to brep file with oi string
                breptools.Write(boundary.geometry.oc_geometry, temp_path)

                topology = Topology.ByBREPPath(temp_path)

                # Clean up the temp file
                os.remove(temp_path)

                # Return the topology
                return topology
            
            # TODO idea: How about we create topologic face objects from every boundary geometry and then we try all combinations of 4+ 
            # and find those that create a closed volume?


            topologic_faces = []
            for boundary in self.boundaries.values():
                if not hasattr(boundary, 'geometry') or not boundary.geometry:
                    continue
                # # Create a topologic face from the boundary geometry
                # topologic_vertices = [Vertex.ByCoordinates(x=v[0], y=v[1], z=v[2]) for v in boundary.geometry.get_vertices()]
                # topologic_face = Face.ByVertices(topologic_vertices, tolerance=0.01, silent=True)

                topologic_face = occ_to_topologic(boundary.geometry.oc_geometry)
                if topologic_face:
                    face_dict = Topology.Dictionary(topologic_face)
                    face_dict = Dictionary.SetValueAtKey(face_dict, 'boundary_id', boundary.id)
                    topologic_face = Topology.SetDictionary(topologic_face, face_dict)


                    topologic_faces.append(topologic_face)

            # find all combinations of 4+ faces that form a closed volume

            # Step 1: Find all combinations of 4+ faces
            all_combinations = []
            for r in range(5, 15):
                for combo in combinations(topologic_faces, r):
                    all_combinations.append(combo)
            print(f"Found {len(all_combinations)} combinations of 4+ faces.")
            # Step 2: Check if each combination forms a closed volume
            magnitudes = []
            for combo in all_combinations:

                # determine if all faces in the combo have normal vectors that point inward
                normals = [Face.Normal(face) for face in combo]

                normals = np.array(normals)

                net = np.sum(normals, axis=0)

                magnitude = np.linalg.norm(net)
                magnitudes.append(magnitude)
                if magnitude > 2:
                    # print(f"Combination {combo} does not form a closed volume, skipping...")
                    continue


                cell = Cell.ByFaces(list(combo), tolerance=1.0, silent=True)

                if cell:
                    # Create geometry from the cell
                    geometry = Geometry.from_topology(cell)


                    # check if the geometry is the same as an existing space geometry
                    existing_space_overlaps = [
                        space for space in self.spaces.values() if space.geometry.bbox_intersects(geometry, return_overlap_percent=True) > 0.3
                    ]
                    if len(existing_space_overlaps) > 0:
                        print(f"Found existing space with geometry")
                        continue
                    space_id = generate_id('space')

                    # add space_id to the cell
                    cell_dict = Topology.Dictionary(cell)
                    cell_dict = Dictionary.SetValueAtKey(cell_dict, 'space_id', space_id)
                    cell = Topology.SetDictionary(cell, cell_dict)

                    # get cell boundaries 
                    boundaries = []
                    
                    for face in combo:
                        face_dict = Topology.Dictionary(face)
                        boundary_id = Dictionary.ValueAtKey(face_dict, 'boundary_id')
                        if boundary_id is not None:
                            boundary = self.boundaries.get(boundary_id)
                            if boundary:
                                boundaries.append(boundary)

                    space = Space(
                        name="Space {}".format(space_counter),
                        geometry=geometry,
                        boundaries=boundaries,
                        volume=Geometry.compute_volume(geometry)  # Assuming volume as a proxy for area in 3D
                    )

                    cell_dict = Topology.Dictionary(cell)
                    cell_dict = Dictionary.SetValueAtKey(cell_dict, 'space_id', space.id)
                    cell = Topology.SetDictionary(cell, cell_dict)
                    space.topology = cell

                    self.spaces[space.id] = space
                    space_counter += 1

                    # Add the space to the building graph
                    features = {
                        'name': space.name,
                        'volume': space.geometry.compute_volume(),
                        'centroid_x': space.geometry.get_centroid().x,
                        'centroid_y': space.geometry.get_centroid().y,
                        'centroid_z': space.geometry.get_centroid().z
                    }  
                    self.building_graph.add_node('Space', node_id=space.id, features=features)

                    # add relationship between bounds and spaces
                    for boundary in space.boundaries:
                        rel = Creates(boundary.id, space.id)
                        boundary.relationships.append(rel)
                        self.relationships[boundary.id].append(rel)
                        self.building_graph.add_edge(boundary.id, space.id, "BOUNDARY_CREATES_SPACE", from_label='Boundary', to_label='Space')

    def generate_adjacency_graph(self):
        """
        Generates an adjacency graph using topologicpy and converts it into a graph with relationships
        """
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Graph import Graph

        def get_neighbors_by_attribute(G, key, value):
            # Find the node with the matching attribute value
            target_node = None
            for node, data in G.nodes(data=True):
                if data.get(key) == value:
                    target_node = node
                    break
            
            if target_node is None:
                return f"No node found with {key} = {value}"

            # Get the neighbors of the node
            neighbors = list(G.neighbors(target_node))
            return neighbors

        spaces = self.spaces.values()
        if not spaces:
            print("No spaces found to generate adjacency graph.")
            return
        space_topologies = [space.topology for space in spaces if space.topology is not None]
        if not space_topologies:
            print("No valid space topologies found to generate adjacency graph.")
            return

        space_complex = CellComplex.ByCells(space_topologies, tolerance=0.01, transferDictionaries=True)

        # Create a graph from the space complex
        space_graph = Graph.ByTopology(space_complex, tolerance=0.01)

        space_graph_nx = Graph.NetworkXGraph(space_graph)

        self.space_adjacency_graph = space_graph_nx

        # Add adjcacency edges to the building graph
        for space_id in self.spaces:
            space = self.spaces[space_id]
            # find the neighbors of the node with attribute "space_id" == space.id
            neighbors = get_neighbors_by_attribute(space_graph_nx, 'space_id', space.id)
            for neighbor in neighbors:
                neighbor_space_id = space_graph_nx.nodes[neighbor]['space_id']
                if neighbor_space_id is not None:
                    # Add an edge to the building graph
                    self.building_graph.add_edge(space.id, neighbor_space_id, 'SPACE_ADJACENT_TO', from_label='Space', to_label='Space')
                
    def show_boundaries(self):
        """
        Display boundaries using Plotly for better 3D interaction.
        """
        import plotly.graph_objects as go
        import numpy as np

        fig = go.Figure()

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        color_idx = 0

        for boundary in self.boundaries.values():
            # Extract geometry data
            vertices = boundary.geometry.get_vertices()
            faces = boundary.geometry.get_faces()

            if not vertices or not faces:
                continue

            x, y, z = zip(*vertices)
            i, j, k = zip(*faces)

            # Create mesh3d trace for the boundary
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                opacity=0.5,
                color=colors[color_idx % len(colors)],
                name=f"Boundary: {boundary.id}",
                hoverinfo='skip',
                showlegend=True
            ))

            color_idx += 1
        fig.update_layout(
            title="Building Boundaries",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
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


    def show(self,
         show_coords=False,
         color_by_class=True,
         color_by_attribute=None,
         flatten_to_elements=False,
         item_types=None,
         show_building_graph=False,
         graph_color_by='type'):
        """
        Plot the B-rep surfaces of items in the model, grouped and colored by attribute or class.
        Optionally show the building graph alongside the model objects.

        Args:
            show_coords (bool): Show coordinate labels on vertices
            color_by_class (bool): Color items by their class name
            color_by_attribute (str): Color items by a specific attribute
            flatten_to_elements (bool): Flatten all items to elements before plotting
            item_types (list): Which item types to include ('objects', 'components', 'elements'). 
                            If None, includes all available types.
            show_building_graph (bool): Show the building graph alongside model objects
            graph_color_by (str): Node attribute to color graph nodes by. Default is 'type'.
        """
        from collections import defaultdict
        import plotly.graph_objects as go
        import networkx as nx
        from .helpers import random_color  # Assuming this exists in helpers

        fig = go.Figure()

        # Collect items from model based on item_types
        items = []
        if item_types is None:
            item_types = ['objects', 'components', 'elements']

        if 'objects' in item_types:
            items.extend(self.objects.values())
        if 'components' in item_types:
            items.extend(self.components.values())
        if 'elements' in item_types:
            items.extend(self.elements.values())

        # Plot model objects (B-rep surfaces)
        if items:
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

                for item in group:
                    # Check if item has geometry and brep_data
                    if not hasattr(item, 'geometry') or not hasattr(item.geometry, 'brep_data'):
                        continue

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
                    name=f"Objects: {str(key)}",  # legend entry
                    hoverinfo='skip',
                    showlegend=True
                ))

                if show_coords:
                    labels = [f"({round(xi, 2)}, {round(yi, 2)}, {round(zi, 2)})" 
                            for xi, yi, zi in vertices]
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode="text",
                        text=labels,
                        showlegend=False,
                        hoverinfo="none",
                        textfont=dict(size=9, color="black")
                    ))

        # Add building graph visualization if requested
        if show_building_graph and hasattr(self, 'building_graph'):
            # Get centroid coordinates for positioning
            centroid_x = nx.get_node_attributes(self.building_graph, 'centroid_x')
            centroid_y = nx.get_node_attributes(self.building_graph, 'centroid_y')
            centroid_z = nx.get_node_attributes(self.building_graph, 'centroid_z')

            all_nodes = list(self.building_graph.nodes)

            # Check if centroid data is available
            has_centroids = all(node in centroid_x and node in centroid_y and node in centroid_z 
                            for node in all_nodes)

            if has_centroids:
                # Extract the node attribute values for coloring
                node_attrs = nx.get_node_attributes(self.building_graph, graph_color_by)

                # Create color mapping for graph nodes
                unique_values = list(set(node_attrs.values()))
                graph_node_colors = {val: random_color(seed=hash(str(val)) % 100) 
                                for val in unique_values}

                # Group nodes by attribute value for separate traces
                node_groups = defaultdict(list)
                for node in all_nodes:
                    attr_value = node_attrs.get(node, "unknown")
                    node_groups[attr_value].append(node)

                # Plot nodes grouped by attribute
                for attr_value, nodes in node_groups.items():
                    if not nodes:
                        continue

                    x_coords = [centroid_x[node] for node in nodes]
                    y_coords = [centroid_y[node] for node in nodes]
                    z_coords = [centroid_z[node] for node in nodes]
                    node_labels = [str(node) for node in nodes]

                    fig.add_trace(go.Scatter3d(
                        x=x_coords,
                        y=y_coords,
                        z=z_coords,
                        mode='markers+text',
                        marker=dict(
                            size=12,
                            color=graph_node_colors.get(attr_value, 'gray'),
                            opacity=0.8
                        ),
                        text=node_labels,
                        textposition="middle center",
                        textfont=dict(size=10, color="white"),
                        name=f"Graph {graph_color_by}: {attr_value}",
                        hovertemplate=f"Node: %{{text}}<br>{graph_color_by}: {attr_value}<br>x: %{{x}}<br>y: %{{y}}<br>z: %{{z}}<extra></extra>",
                        showlegend=True
                    ))

                # Plot edges
                edge_x, edge_y, edge_z = [], [], []
                edge_labels = nx.get_edge_attributes(self.building_graph, 'relationship')

                for edge in self.building_graph.edges():
                    node1, node2 = edge
                    if node1 in centroid_x and node2 in centroid_x:
                        # Add edge line
                        edge_x.extend([centroid_x[node1], centroid_x[node2], None])
                        edge_y.extend([centroid_y[node1], centroid_y[node2], None])
                        edge_z.extend([centroid_z[node1], centroid_z[node2], None])

                # Add edges as a single trace
                if edge_x:
                    fig.add_trace(go.Scatter3d(
                        x=edge_x,
                        y=edge_y,
                        z=edge_z,
                        mode='lines',
                        line=dict(color='black', width=2),
                        name="Graph Edges",
                        hoverinfo='skip',
                        showlegend=True
                    ))

                # Add edge labels if they exist
                if edge_labels:
                    edge_label_x, edge_label_y, edge_label_z, edge_label_text = [], [], [], []
                    for edge, label in edge_labels.items():
                        node1, node2 = edge
                        if node1 in centroid_x and node2 in centroid_x:
                            # Position label at midpoint of edge
                            mid_x = (centroid_x[node1] + centroid_x[node2]) / 2
                            mid_y = (centroid_y[node1] + centroid_y[node2]) / 2
                            mid_z = (centroid_z[node1] + centroid_z[node2]) / 2

                            edge_label_x.append(mid_x)
                            edge_label_y.append(mid_y)
                            edge_label_z.append(mid_z)
                            edge_label_text.append(str(label))

                    if edge_label_text:
                        fig.add_trace(go.Scatter3d(
                            x=edge_label_x,
                            y=edge_label_y,
                            z=edge_label_z,
                            mode='text',
                            text=edge_label_text,
                            textfont=dict(size=8, color="white"),
                            name="Edge Labels",
                            hoverinfo='skip',
                            showlegend=False
                        ))
            else:
                print("Warning: Building graph nodes don't have centroid coordinates. Skipping graph visualization.")

        elif show_building_graph:
            print("Warning: No building graph found in model or building_graph parameter is False.")

        # Set title based on what's being shown
        title_parts = []
        if items:
            title_parts.append("B-rep Model Objects")
        if show_building_graph and hasattr(self, 'building_graph'):
            title_parts.append("Building Graph")

        title = " and ".join(title_parts) if title_parts else "Model Visualization"

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
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


    def show_boundaries_graph(self):
        """
        Display the boundaries graph as a 3D network using Plotly.
        Node positions will be inferred from face centroids if available,
        otherwise a spring layout will be used.
        """
        G = self.boundary_graph
        pos = {}

        for node in G.nodes:
            obj = self.boundaries.get(node)
            if obj and hasattr(obj, "geometry"):
                verts = obj.geometry.get_vertices()
                center = np.mean(np.array(verts), axis=0)
                pos[node] = center
            else:
                pos = nx.spring_layout(G, dim=3, seed=42)
                break  # fallback if any missing geometry

        edge_x, edge_y, edge_z = [], [], []
        for u, v in G.edges():
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='gray', width=2),
            hoverinfo='none'
        )

        node_x, node_y, node_z, text = [], [], [], []
        for node, (x, y, z) in pos.items():
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_data = G.nodes[node]
            text.append(f"{node}<br>{node_data.get('type', '')}")

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(size=6, color='lightblue'),
            text=text,
            hoverinfo='text',
            textposition='top center'
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(title='Boundaries Graph (3D)',
                        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                        margin=dict(l=0, r=0, b=0, t=40))


        fig.show()

    def show_spaces_graph(self):
        """
        Display the space adjacency graph in 3D using Plotly.
        Positions are inferred from space geometry centroid or node attributes.
        """
        if not hasattr(self, 'space_adjacency_graph'):
            print("No space adjacency graph found. Please generate it first.")
            return

        G = self.space_adjacency_graph
        pos = {}

        for node in G.nodes:
            node_data = G.nodes[node]
            space_id = node_data.get('space_id', None)
            space = self.spaces.get(space_id, None)

            coord = None
            if space and hasattr(space, "geometry") and hasattr(space.geometry, "get_centroid"):
                coord = space.geometry.get_centroid()
            elif all(k in node_data for k in ('x', 'y', 'z')):
                coord = (node_data['x'], node_data['y'], node_data['z'])

            if coord:
                pos[node] = coord

        if not pos:
            print("No valid coordinates found for any nodes.")
            return

        # Edges
        edge_x, edge_y, edge_z = [], [], []
        for u, v in G.edges():
            if u in pos and v in pos:
                _0 = pos[u]
                _1 = pos[v]
                x0 = _0.x if hasattr(_0, 'x') else _0[0]
                y0 = _0.y if hasattr(_0, 'y') else _0[1]
                z0 = _0.z if hasattr(_0, 'z') else _0[2]
                x1 = _1.x if hasattr(_1, 'x') else _1[0]
                y1 = _1.y if hasattr(_1, 'y') else _1[1]
                z1 = _1.z if hasattr(_1, 'z') else _1[2]

                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                edge_z += [z0, z1, None]

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='gray', width=2),
            hoverinfo='none'
        )

        # Nodes
        node_x, node_y, node_z, text = [], [], [], []
        for node, (x, y, z) in pos.items():
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            label = G.nodes[node].get('type') or G.nodes[node].get('category', '')
            node_space_id = G.nodes[node].get('space_id', '')
            if node_space_id:
                label = f"{node_space_id}"
            else:
                label = f"{node}"

            text.append(label)

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(size=6, color='lightgreen'),
            text=text,
            hoverinfo='text',
            textposition='top center'
        )

        # get the bounding box of the model
        bbox = self.get_bounding_box()
        if bbox:
            min_x, min_y, min_z, max_x, max_y, max_z = bbox
            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                title='Spaces Graph (3D)',
                scene=dict(
                    xaxis=dict(range=[min_x, max_x], title='X'),
                    yaxis=dict(range=[min_y, max_y], title='Y'),
                    zaxis=dict(range=[min_z, max_z], title='Z')
                ),
                margin=dict(l=0, r=0, b=0, t=40)
            )
        else:

            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                title='Spaces Graph (3D)',
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                margin=dict(l=0, r=0, b=0, t=40)
            )

        fig.show()



    def show_objects(self, **kwargs):
        """Convenience method to show only objects from the model."""
        return self.show(item_types=['objects'], **kwargs)


    def show_components(self, **kwargs):
        """Convenience method to show only components from the model."""
        return self.show(item_types=['components'], **kwargs)


    def show_elements(self, **kwargs):
        """Convenience method to show only elements from the model."""
        return self.show(item_types=['elements'], **kwargs)


    def show_all_as_elements(self, **kwargs):
        """Convenience method to show all items flattened to elements."""
        return self.show(flatten_to_elements=True, **kwargs)

    def show_spaces(self):
        """
        Display the inferred spaces in the model using Plotly.
        Each space is represented as a 3D mesh with its boundaries.
        """
        import plotly.graph_objects as go

        fig = go.Figure()

        for space in self.spaces.values():
            # Extract geometry data
            vertices = space.geometry.get_vertices()
            faces = space.geometry.get_faces()

            if not vertices or not faces:
                continue

            x, y, z = zip(*vertices)
            i, j, k = zip(*faces)

            # Create mesh3d trace for the space
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                opacity=0.5,
                color='lightblue',
                name=f"Space: {space.id}",
                hoverinfo='skip',
                showlegend=True
            ))

        fig.update_layout(
            title="Inferred Spaces",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            showlegend=True
        )

        fig.show()

    def show_building_graph(self, color_by='type'):
        """
        Display the building graph (Kuzu graph) using Plotly.
        Args:
            color_by (str): Node attribute to color nodes by. Default is 'type'.
        """
        # Check if building graph exists
        if not hasattr(self, 'building_graph'):
            print("No building graph found in model.")
            return

        # Get node positions and attributes
        query = "MATCH (n) RETURN DISTINCT n"
        result = self.building_graph.query(query)
        pos = {}
        
        # Initialize Plotly figure
        fig = go.Figure()

        # get all nodes from the results
        if not result:
            print("No nodes found in the building graph.")
            return
        
        if result.has_next():
            nodes = []
            while result.has_next():
                row = result.get_next()
                nodes.append(row)
            
        else:
            return
        

        for node in nodes:
            node_id = node[0]['id']
            if 'centroid_x' in node[0] and 'centroid_y' in node[0] and 'centroid_z' in node[0]:
                pos[node_id] = (
                    node[0]['centroid_x'],
                    node[0]['centroid_y'],
                    node[0]['centroid_z']
                )
            else:
                pos[node_id] = (0, 0, 0)

            # Get node attributes for coloring
            if color_by in node[0]:
                node[0]['color'] = node[0][color_by]
            else:
                node[0]['color'] = 'default'

        # Create a NetworkX graph from the building graph
        G = nx.Graph()
        for node in nodes:
            node_id = node[0]['id']
            G.add_node(node_id, **node[0])

        # Add edges
        edges_result = self.building_graph.query("MATCH (n)-[r]->(m) RETURN n.id AS source, m.id AS target, r")
        if edges_result.has_next():
            edges = []
            while edges_result.has_next():
                row = edges_result.get_next()
                edges.append(row)
            
        else:
            return
        
        for edge in edges:
            source = edge[0]
            target = edge[1]
            G.add_edge(source, target, relationship=edge[2]['_label'])

        # Create color mapping for nodes
        unique_colors = set(nx.get_node_attributes(G, 'color').values())
        color_map = {color: random_color(seed=hash(color) % 100) for color in unique_colors}

                # Create color mapping for nodes
        unique_labels = set(nx.get_node_attributes(G, '_label').values())
        color_map = {label: random_color(seed=hash(label) % 100) for label in unique_labels}

        # Create color mapping for edge types
        unique_edge_types = set(nx.get_edge_attributes(G, 'relationship').values())
        edge_color_map = {edge_type: random_color(seed=hash(edge_type) % 100) for edge_type in unique_edge_types}
        
        # Define edge styles for different relationship types
        edge_styles = {
            'contains': dict(width=3, dash='solid'),
            'connects_to': dict(width=2, dash='dash'),
            'adjacent_to': dict(width=2, dash='dot'),
            'supports': dict(width=4, dash='solid'),
            'flows_through': dict(width=2, dash='dashdot'),
            'part_of': dict(width=2, dash='solid'),
            'default': dict(width=2, dash='solid')
        }

        # --- Edge traces grouped by relationship type ---
        for edge_type in unique_edge_types:
            edge_x, edge_y, edge_z, edge_text = [], [], [], []
            
            for u, v, data in G.edges(data=True):
                if data.get("relationship") == edge_type and u in pos and v in pos:
                    x0, y0, z0 = pos[u]
                    x1, y1, z1 = pos[v]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]
                    edge_z += [z0, z1, None]
                    edge_text.append(f"{u} → {v}<br>{edge_type}")

            if edge_x:  # Only add trace if there are edges of this type
                style = edge_styles.get(edge_type, edge_styles['default'])
                
                edge_trace = go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode='lines',
                    line=dict(
                        color=edge_color_map[edge_type], 
                        width=style['width'],
                        dash=style['dash']
                    ),
                    hoverinfo='text',
                    text=edge_text,
                    name=f'Edge: {edge_type}',
                    # legendgroup='edges'
                )
                
                fig.add_trace(edge_trace)

        # --- Node traces grouped by _label ---
        for label in unique_labels:
            node_x, node_y, node_z, text = [], [], [], []
            for node, (x, y, z) in pos.items():
                if G.nodes[node].get('_label') == label:
                    node_x.append(x)
                    node_y.append(y)
                    node_z.append(z)
                    text.append(f"{node}<br>{label}<br>x={x}<br>y={y}<br>z={z}")
            
            fig.add_trace(go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers+text',
                marker=dict(size=6, color=color_map[label], opacity=0.8),
                text=text,
                textposition="top center",
                textfont=dict(size=10, color="black"),
                hoverinfo='text',
                name=label
            ))

           

        # Set layout
        fig.update_layout(
            title='Building Graph (3D)',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
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


    def get_bounding_box(self):
        """
        Get the bounding box of the model by combining all objects, components, and elements.
        Returns:
            tuple: (min_x, min_y, min_z, max_x, max_y, max_z)
        """
        if not self.objects and not self.components and not self.elements:
            return None

        all_geometries = []
        for item in self.objects.values():
            if hasattr(item, 'geometry'):
                all_geometries.append(item.geometry)
        for item in self.components.values():
            if hasattr(item, 'geometry'):
                all_geometries.append(item.geometry)
        for item in self.elements.values():
            if hasattr(item, 'geometry'):
                all_geometries.append(item.geometry)

        all_vertices = []
        for geometry in all_geometries:
            if hasattr(geometry, 'get_vertices'):
                vertices = geometry.get_vertices()
                all_vertices.extend(vertices)
        if not all_vertices:
            return None
        all_vertices = np.array(all_vertices)
        min_x, min_y, min_z = np.min(all_vertices, axis=0)
        max_x, max_y, max_z = np.max(all_vertices, axis=0)
        return (min_x, min_y, min_z, max_x, max_y, max_z)

    def ask(self, question: str, **kwargs) -> str:
        """
        Ask a question about the model by using LLM to generate and run a Cypher query on the building graph.

        Leverages:
        - self.building_graph.get_node_types_to_string()
        - self.building_graph.get_relationship_types_to_string()
        - self.building_graph.get_connection_schema_string()
        - self.building_graph.query() to execute the generated query
        
        Args:
            question (str): The natural language question to ask.
            **kwargs: Additional options like OpenAI parameters.
        
        Returns:
            str: The answer generated from the graph query result.
        """
        from openai import OpenAI
        
        client = OpenAI(api_key="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")

        # Step 1: Get schema context
        node_types = self.building_graph.get_node_types_to_string()
        rel_types = self.building_graph.get_relationship_types_to_string()
        connections = self.building_graph.get_connection_schema_string()

        # Step 2: Construct system prompt
        system_prompt = f"""
    You are an expert in querying a building model stored in a graph database.
    Here is the graph schema:
    - Node Types: {node_types}
    - Relationship Types: {rel_types}
    - Connection Patterns:\n{connections}

    Given a user's question, return a Cypher query that can be run on this graph to answer it.
    Always use the available types and relationships, and return only the query.
    """

        # Step 3: Get Cypher query from OpenAI
        response = client.chat.completions.create(model=kwargs.get("model", "gpt-4"),
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"User question: {question}"}
        ],
        temperature=kwargs.get("temperature", 0),
        max_tokens=200)
        cypher_query = response.choices[0].message.content.strip()

        # Step 4: Execute query on graph
        try:
            result = self.building_graph.query_to_string(cypher_query)
        except Exception as e:
            return f"Error running query: {e}\nQuery:\n{cypher_query}"

        if not result:
            return f"No result found.\nQuery:\n{cypher_query}"

        # Step 5: Format result
        result_str = str(result)

        # Step 6: Ask OpenAI to interpret the result
        interpret_prompt = f"""
    You wrote and executed the following Cypher query:

    {cypher_query}

    It returned the following result:

    {result_str}

    Answer the user's question using the result in natural language:
    "{question}"
    """

        answer_response = client.chat.completions.create(model=kwargs.get("model", "gpt-4"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes database query results into natural language."},
            {"role": "user", "content": interpret_prompt.strip()}
        ],
        temperature=kwargs.get("temperature", 0.2),
        max_tokens=300)

        final_answer = answer_response.choices[0].message.content.strip()

        return f"Answer: {final_answer}"
