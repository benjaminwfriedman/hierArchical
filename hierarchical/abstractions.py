from tqdm import tqdm
from hierarchical.items import Element, Component, Wall, Deck, Window, Door, Object, BaseItem
from hierarchical.relationships import AdjacentTo, Relationship, Creates, Supports, SupportedBy, FlowsTo, FlowsFrom, Above, Below, InFrontOf, Behind, LeftOf, RightOf, IsPartOf, Contains 
from hierarchical.geometry import Geometry
from hierarchical.helpers import test_healing_validation
from collections import defaultdict
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from hierarchical.utils import generate_id, random_color
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
from dotenv import load_dotenv


from itertools import combinations
import plotly.graph_objects as go
import kuzu
import uuid
from uuid import uuid4
from openai import OpenAI
from OCC.Core.BRepTools import breptools
import os
import numpy as np

import numpy as np
from scipy.spatial import Delaunay

# Load environment variables
load_dotenv()

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

client = OpenAI(api_key=OPEN_AI_API_KEY)


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

    # space attributes dictionary
    attributes: Dict[str, any] = field(default_factory=dict)


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

    def update_node(self, node_id: str, label: str = "Node", **attributes):
        """
        Update an existing node's attributes.
        
        Args:
            node_id: The ID of the node to update
            label: The label/type of the node (default: "Node")
            **attributes: Key-value pairs of attributes to update
        """
        if not attributes:
            print("No attributes provided for update")
            return
        
        # Process attributes similar to add_node
        flat_attrs = {}
        for k, v in attributes.items():
            if v is None:
                continue
            elif isinstance(v, np.generic):
                flat_attrs[k] = v.item()
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    if subv is not None:
                        if isinstance(subv, np.generic):
                            subv = subv.item()
                        flat_attrs[subk] = subv
            else:
                flat_attrs[k] = v
        
        if not flat_attrs:
            print("No valid attributes to update")
            return
        
        # Build SET clause
        set_clauses = [f"n.{k} = {self._format_value(v)}" for k, v in flat_attrs.items()]
        set_str = ", ".join(set_clauses)
        
        query = f"""
        MATCH (n:{label} {{id: '{node_id}'}})
        SET {set_str}
        RETURN n
        """
        
        try:
            result = self.conn.execute(query)
            if result.has_next():
                updated_node = result.get_next()
            else:
                pass
        except Exception as e:
            print(f"Failed to update node: {e}")
            print(f"Query: {query}")
       
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

    def query_to_string(self, query: str, return_type: str = 'list') -> str:
        """
        Execute a raw Cypher query against the graph database and return the results as a string.
        
        Args:
            query: Cypher query to execute
            return_type: 'list' for line-separated results, 'dict' for column:value mapping
        """
        try:
            result = self.conn.execute(query)
        except Exception as e:
            raise Exception(f"Query failed: {e}")

        if result.has_next():
            if return_type.lower() == 'dict':
                # Get column names from the Kuzu result
                columns = result.get_column_names()
                rows = []
                while result.has_next():
                    # For dict format, return first row as key-value pairs
                    row = result.get_next()
                    row_dict = {}
                    
                    # Kuzu rows are typically lists/tuples, map to column names
                    for i, column_name in enumerate(columns):
                        value = row[i] if i < len(row) else None
                        row_dict[column_name] = value

                    rows.append(row_dict)
                return ",".join(str(r) for r in rows)
            
            else:
                # Default: return as line-separated list (current behavior)
                rows = []
                while result.has_next():
                    row = result.get_next()
                    rows.append(str(row))
                return "\n".join(rows)
        else:
            if return_type.lower() == 'dict':
                return str({})
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

    def get_node_labels_to_string(self) -> str:
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

    def get_node_types_to_string(self) -> str:
        """
        Get a string representation of all node types in the graph.
        """
        return self.query_to_string("MATCH (n) RETURN DISTINCT n.type, label(n)")

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
        self.conn.execute("CREATE NODE TABLE IF NOT EXISTS Space(id STRING PRIMARY KEY, name STRING, volume FLOAT, centroid_x FLOAT, centroid_y FLOAT, centroid_z FLOAT, omniclass_space_type STRING)")
        # create node tables for other object types
        self.conn.execute("CREATE NODE TABLE IF NOT EXISTS Object(id STRING PRIMARY KEY, type STRING, volume FLOAT, centroid_x FLOAT, centroid_y FLOAT, centroid_z FLOAT, length FLOAT, width FLOAT, height FLOAT)")
        self.conn.execute("CREATE NODE TABLE IF NOT EXISTS Element(id STRING PRIMARY KEY, type STRING, volume FLOAT, centroid_x FLOAT, centroid_y FLOAT, centroid_z FLOAT)")
        self.conn.execute("CREATE NODE TABLE IF NOT EXISTS Component(id STRING PRIMARY KEY, type STRING, volume FLOAT, centroid_x FLOAT, centroid_y FLOAT, centroid_z FLOAT)")
        self.conn.execute("CREATE NODE TABLE IF NOT EXISTS Boundary(id STRING PRIMARY KEY, boundary_id STRING, type STRING, is_access_boundary BOOL, is_visual_boundary BOOL, centroid_x FLOAT, centroid_y FLOAT, centroid_z FLOAT)")

        # create relationship tables
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS OBJECT_ADJACENT_TO(FROM Object TO Object, type STRING)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS OBJECT_EMBEDDED_IN(FROM Object TO Object, type STRING)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS OBJECT_EMBEDS(FROM Object TO Object, type STRING)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS OBJECT_CONTAINS_COMPONENT(FROM Object TO Component, type STRING)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS COMPONENT_IS_PART_OF(FROM Component TO Object, type STRING)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS COMPONENT_CONTAINS_ELEMENT(FROM Component TO Element, type STRING)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS ELEMENT_IS_PART_OF(FROM Element TO Component, type STRING)")
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
    def from_objects(cls, name, objects, existing_spaces=None):
        """
        Create a model from a list of objects.
        """
        model = cls(name)
        for obj in objects:
            if isinstance(obj, Element):
                model.elements[obj.id] = obj
            elif isinstance(obj, Component):
                model.components[obj.id] = obj
                for item in obj.sub_items:
                    if isinstance(item, Element):
                        model.elements[item.id] = item
            elif isinstance(obj, Object):
                model.objects[obj.id] = obj
                for item in obj.sub_items:
                    if isinstance(item, Element):
                        model.elements[item.id] = item
                    elif isinstance(item, Component):
                        model.components[item.id] = item
                        for sub_item in item.sub_items:
                            if isinstance(sub_item, Element):
                                model.elements[sub_item.id] = sub_item
                            elif isinstance(sub_item, Component):
                                model.components[sub_item.id] = sub_item
                                for sub_sub_item in sub_item.sub_items:
                                    if isinstance(sub_sub_item, Element):
                                        model.elements[sub_sub_item.id] = sub_sub_item

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
                "type": type(obj).__name__.lower(),  # class name, e.g., Wall, Door, etc.
                # "length": getattr(obj, "length", None),
                # "width": getattr(obj, "width", None),
                # "height": getattr(obj, "height", None),
                "volume": obj.geometry.compute_volume() if obj.geometry else 0.0,
                "centroid_x": obj.get_centroid().x,
                "centroid_y": obj.get_centroid().y,
                "centroid_z": obj.get_centroid().z,
                "length": obj.get_length(),
                "width": obj.get_width(),
                "height": obj.get_height()
                # "object": obj  # preserve full object for further use
            }

            model.building_graph.add_node("Object", node_id=obj.id, **features)
            if obj.sub_items:
                for component in obj.sub_items:
                    if isinstance(component, Component):
                        model.building_graph.add_node("Component", node_id=component.id, type=component.type.lower(),
                                                    volume=component.geometry.compute_volume() if component.geometry else 0.0,
                                                    centroid_x=component.get_centroid().x,
                                                    centroid_y=component.get_centroid().y,
                                                    centroid_z=component.get_centroid().z)
                        model.building_graph.add_edge(obj.id, component.id, "OBJECT_CONTAINS_COMPONENT", from_label="Object", to_label="Component")
                        model.building_graph.add_edge(component.id, obj.id, "COMPONENT_IS_PART_OF", from_label="Component", to_label="Object")

                        # add relationship to relationships
                        model.relationships[obj.id].append(Contains(source=obj.id, target=component.id))
                        model.relationships[component.id].append(IsPartOf(source=component.id, target=obj.id))


                for element in component.sub_items:
                    if isinstance(element, Element):
                        model.building_graph.add_node("Element", node_id=element.id, type=element.type.lower(),
                                                        volume=element.geometry.compute_volume() if element.geometry else 0.0,
                                                        centroid_x=element.get_centroid().x,
                                                        centroid_y=element.get_centroid().y,
                                                        centroid_z=element.get_centroid().z)
                        
                        model.building_graph.add_edge(component.id, element.id, "COMPONENT_CONTAINS_ELEMENT", from_label="Component", to_label="Element")
                        model.building_graph.add_edge(element.id, component.id, "ELEMENT_IS_PART_OF", from_label="Element", to_label="Component")

                        # add relationship to relationships
                        model.relationships[component.id].append(Contains(source=component.id, target=element.id))
                        model.relationships[element.id].append(IsPartOf(source=element.id, target=component.id))
        else:
            print(f"Object {obj_id} has no sub_items, skipping.")


        model.create_object_adjacency_relationships(tolerance=0.001)
        model.create_object_embedded_relationships()  # Uses default 95% threshold
        model.infer_bounds()
        model.infer_spaces(existing_spaces=existing_spaces)
        model.generate_adjacency_graph()
        model.apply_ontologies()


        return model

    @classmethod
    def from_ifc(cls, ifc_file: str):
        """
        Create a model from an IFC file.
        """
        import ifcopenshell
        import ifcopenshell.geom
        import ifcopenshell.util.element
        import ifcopenshell.util.placement
        import ifcopenshell.util.shape
        from pathlib import Path
        from contextlib import contextmanager

        from hierarchical.items import Wall, Deck, Window, Door

        @contextmanager
        def ifc_file_context(filepath):
            ifc_file = None
            try:
                ifc_file = ifcopenshell.open(filepath)
                yield ifc_file
            finally:
                if ifc_file:
                    del ifc_file  # Don't use close(), just delete
                    import gc
                    gc.collect()

        # Fix: ifc_file is a path string, use it directly
        with ifc_file_context(ifc_file) as model:
            if not model:
                raise ValueError(f"Failed to open IFC file: {ifc_file}")

            # Extract the building elements
            element_types = ["IfcWall", "IfcSlab", "IfcWindow", "IfcDoor", "IfcColumn", "IfcBeam", "IfcSpace", "IFCWallStandardCase", "IfcPlate", "IFCCovering", "IfcSpace"]
            objects = []
            for element_type in element_types:
                objects.extend(model.by_type(element_type))

            # Convert IFC objects to hierarchical objects
            hierarchical_objects = []
            hierarchical_spaces = []
            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS, True)

            # deduplicate ifc_objs based on ifc_obj.GlobalId

            seen_global_ids = set()
            objects = [ifc_obj for ifc_obj in objects 
                        if ifc_obj.GlobalId not in seen_global_ids 
                        and not seen_global_ids.add(ifc_obj.GlobalId)]
            
            for ifc_obj in objects:
                try:
                    # Create geometry
                    shape = ifcopenshell.geom.create_shape(settings, ifc_obj)
                    geom = shape.geometry

                    if isinstance(geom, ifcopenshell.ifcopenshell_wrapper.Triangulation):
                        vertices = geom.verts
                        faces = geom.faces
                        mesh_data = {
                            "vertices": [(vertices[i], vertices[i+1], vertices[i+2]) for i in range(0, len(vertices), 3)],
                            "faces": [(faces[i], faces[i+1], faces[i+2]) for i in range(0, len(faces), 3)]
                        }
                        geometry = Geometry(mesh_data=mesh_data)
                    else:
                        print(f"Unsupported geometry type for {ifc_obj.Name}: {type(geom)}")
                        continue

                    # Create appropriate hierarchical object
                    obj_name = ifc_obj.Name or f"Unnamed_{ifc_obj.is_a()}"
                    
                    if ifc_obj.is_a("IfcWall"):
                        wall = Wall(name=obj_name, type="Wall", geometry=geometry)
                        hierarchical_objects.append(wall)
                    elif ifc_obj.is_a("IFCWallStandardCase"):
                        wall = Wall(name=obj_name, type="Wall", geometry=geometry)
                        hierarchical_objects.append(wall)
                    elif ifc_obj.is_a("IfcPlate"):
                        wall = Wall(name=obj_name, type="Wall", geometry=geometry)
                        hierarchical_objects.append(wall)
                    elif ifc_obj.is_a("IfcSlab"):
                        deck = Deck(name=obj_name, type="Deck", geometry=geometry)
                        hierarchical_objects.append(deck)
                    elif ifc_obj.is_a("IFCCovering"):
                        deck = Deck(name=obj_name, type="Deck", geometry=geometry)
                        hierarchical_objects.append(deck)
                    elif ifc_obj.is_a("IfcWindow"):
                        window = Window(name=obj_name, type="Window", geometry=geometry)
                        hierarchical_objects.append(window)
                    elif ifc_obj.is_a("IfcDoor"):
                        door = Door(name=obj_name, type="Door", geometry=geometry)
                        hierarchical_objects.append(door)
                    elif ifc_obj.is_a("IfcColumn"):
                        column = Column(name=obj_name, type="Column", geometry=geometry)
                        hierarchical_objects.append(column)
                    elif ifc_obj.is_a("IfcBeam"):
                        beam = Beam(name=obj_name, type="Beam", geometry=geometry)
                        hierarchical_objects.append(beam)
                    elif ifc_obj.is_a("IfcSpace"):
                        space = Space(name=obj_name, geometry=geometry)
                        hierarchical_spaces.append(space)
                    else:
                        print(f"Unhandled IFC object type: {ifc_obj.is_a()}")
                        
                except Exception as e:
                    print(f"Error processing {ifc_obj.Name or 'Unnamed'} ({ifc_obj.is_a()}): {e}")
                    continue

        # Fix: Move this outside the loop and context manager
        hierarchical_model = cls.from_objects(name=Path(ifc_file).stem, objects=hierarchical_objects, existing_spaces=hierarchical_spaces)
        return hierarchical_model

    ## TODO: Implement method to load from Speckle
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

    def create_object_embedded_relationships(self, intersection_threshold=50.0):
        """
        Add embedded_in relationships between objects in the model based on geometric intersection.
        Only embeddable objects (doors, windows) with >50% intersection are considered embedded.

        Args:
            intersection_threshold: Minimum overlap percentage to consider as embedded (default: 95%)
        """
        from .relationships import EmbeddedIn, Embeds
        
        for obj in self.objects.values():
            # Only embeddable objects can be embedded in other objects
            if not getattr(obj, 'embeddable', False):
                continue
                
            # Find all objects that this embeddable object intersects with (excluding itself)
            other_objects = [other for other in self.objects.values() if other != obj]
            
            for other_obj in other_objects:
                # Check intersection percentage between embeddable object and other object
                overlap_percent = obj.intersects_with(other_obj, return_overlap_percent=True)
                
                # Only create embedded relationship if overlap exceeds threshold
                if overlap_percent >= intersection_threshold:
                    # Create embedded_in relationship from embeddable obj to other_obj
                    embedded_rel = EmbeddedIn(source=obj.id, target=other_obj.id)
                    self.relationships[obj.id].append(embedded_rel)
                    
                    # Create inverse embeds relationship from other_obj to embeddable obj
                    embeds_rel = Embeds(source=other_obj.id, target=obj.id)
                    self.relationships[other_obj.id].append(embeds_rel)
                    
                    # Add edges to graph (use existing relationship types)
                    self.building_graph.add_edge(obj.id, other_obj.id, "OBJECT_EMBEDDED_IN", from_label="Object", to_label="Object")
                    self.building_graph.add_edge(other_obj.id, obj.id, "OBJECT_EMBEDS", from_label="Object", to_label="Object")


    def extend_face(self, face, extention):
        from topologicpy.Face import Face
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Topology import Topology


        # translate the face to the xy plane
        face_translated = Topology.Translate(face, x=0, y=0, z=-Topology.Centroid(face).Z())
        # simplify
        simplified_face = Face.Simplify(face_translated)
        # create a new face by extending the original face with ByOffset
        extended_face = Face.ByOffset(simplified_face, extention)
        # translate the extended face back to its original position
        extended_face = Topology.Translate(extended_face, x=0, y=0, z=Topology.Centroid(face).Z())

        return extended_face

    def heal_boundaries(self, tolerance=15.0, version='occ'):
        """Heal boundaries with comprehensive shape fixing and gap filling"""
        if version == 'occ':
            import tempfile
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
            from OCC.Core.BRepOffset import BRepOffset_Analyse
            from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape
            from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume
            from OCC.Core.GeomAbs import GeomAbs_Intersection
            from OCC.Core.BRepOffset import BRepOffset_Skin
            from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffset
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            from OCC.Core.BRepTools import breptools_OuterWire
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_WIRE
            from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Splitter
            from OCC.Core.BRep import BRep_Builder
            from OCC.Core.TopoDS import TopoDS_Compound
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.TopTools import TopTools_ListOfShape
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.GeomLib import GeomLib_IsPlanarSurface
            from OCC.Core.gp import gp_Pln, gp_Vec, gp_Pnt
            from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            from OCC.Core.TopTools import TopTools_ListOfShape
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.BRepTools import breptools
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_SOLID

            # TNaming imports
            from OCC.Core.TNaming import TNaming_Builder, TNaming_NamedShape
            from OCC.Core.TDF import TDF_Data, TDF_Label, TDF_ChildIterator
            from OCC.Core.TDataStd import TDataStd_Name, TDataStd_Integer
            from OCC.Core.TCollection import TCollection_ExtendedString

            from topologicpy.Topology import Topology
            import math

            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
            from hierarchical.utils import plot_opencascade_shapes
            from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
            from OCC.Core.BOPAlgo import BOPAlgo_RemoveFeatures
            from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
            from OCC.Core.BRepGProp import brepgprop

            from topologicpy.CellComplex import CellComplex
            from topologicpy.Shell import Shell
            from topologicpy.Face import Face

            from hierarchical.utils import transfer_topologic_dict
            

          # Initialize TNaming document for tracking
            doc = TDF_Data()
            main_label = doc.Root()

            # Create labels for different stages
            initial_faces_label = main_label.NewChild()
            offset_faces_label = main_label.NewChild()
            merged_faces_label = main_label.NewChild()
            volume_faces_label = main_label.NewChild()

            # Simplified helper functions that don't use TDataStd at all
            def set_label_name(label, name_string):
                """Helper function - just returns the name string, no TDataStd calls"""
                return name_string

            def set_label_integer(label, int_value):
                """Helper function - just returns the integer, no TDataStd calls"""
                return int_value

            # Create a label tracking dictionary using label IDs as keys
            label_metadata = {}
            label_metadata[id(initial_faces_label)] = {"name": "InitialFaces", "stage": "initial", "label": initial_faces_label}
            label_metadata[id(offset_faces_label)] = {"name": "OffsetFaces", "stage": "offset", "label": offset_faces_label}
            label_metadata[id(merged_faces_label)] = {"name": "MergedFaces", "stage": "merged", "label": merged_faces_label}
            label_metadata[id(volume_faces_label)] = {"name": "VolumeFaces", "stage": "volume", "label": volume_faces_label}

            # Tracking dictionaries
            face_genealogy = {}  # Maps face_id -> [boundary_id, stage, operations]
            boundary_to_faces = {}  # Maps boundary_id -> list of final face_ids

            def register_face(face, boundary_id, stage, parent_label, operation="created", parent_face_id=None):
                """Register a face in the TNaming system with genealogy tracking"""
                face_label = parent_label.NewChild()
                builder = TNaming_Builder(face_label)
                builder.Generated(face)
                
                # Create unique face ID
                face_id = id(face)
                
                # Set name with boundary and stage info - no TDataStd calls
                name = f"Boundary_{boundary_id}_{stage}_{face_id}"
                name_attr = set_label_name(face_label, name)  # Just returns the string
                
                # Store boundary ID - no TDataStd calls
                int_attr = set_label_integer(face_label, boundary_id)  # Just returns the int
                
                # Track genealogy in Python dictionaries
                if face_id not in face_genealogy:
                    face_genealogy[face_id] = {
                        'boundary_id': boundary_id,
                        'stage': stage,
                        'operations': [],
                        'label': face_label,
                        'label_id': id(face_label),
                        'name': name,
                        'name_attr': name_attr,
                        'int_attr': int_attr,
                        'face_object': face
                    }
                
                face_genealogy[face_id]['operations'].append({
                    'operation': operation,
                    'parent_face_id': parent_face_id,
                    'stage': stage
                })
                
                return face_id, face_label


            def track_face_transformation(original_face_id, new_face, new_stage, parent_label, operation="transformed"):
                """Track when a face is transformed into a new face"""
                if original_face_id in face_genealogy:
                    boundary_id = face_genealogy[original_face_id]['boundary_id']
                    new_face_id, new_label = register_face(
                        new_face, boundary_id, new_stage, parent_label, operation, original_face_id
                    )
                    return new_face_id
                return None

            def find_faces_by_boundary(boundary_id, stage=None):
                """Find all faces belonging to a specific boundary at a given stage"""
                matching_faces = []
                for face_id, info in face_genealogy.items():
                    if info['boundary_id'] == boundary_id:
                        if stage is None or info['stage'] == stage:
                            matching_faces.append((face_id, info))
                return matching_faces

            def get_final_faces_for_boundary(boundary_id):
                """Get the final faces that originated from a specific boundary"""
                # Find the latest stage for this boundary
                boundary_faces = find_faces_by_boundary(boundary_id)
                if not boundary_faces:
                    return []
                
                # Group by stage and get the latest
                stages = {}
                for face_id, info in boundary_faces:
                    stage = info['stage']
                    if stage not in stages:
                        stages[stage] = []
                    stages[stage].append((face_id, info))
                
                # Return faces from the latest stage
                latest_stage = max(stages.keys()) if stages else None
                return stages.get(latest_stage, [])


            def get_plane_normal_and_point(face):
                """Extract plane normal and point from a face"""
                try:
                    surface = BRep_Tool.Surface(face)
                    
                    # Check if it's planar
                    if GeomLib_IsPlanarSurface(surface):
                        # Get plane parameters
                        plane = surface.GetObject().Plane()
                        normal = plane.Axis().Direction()
                        point = plane.Location()
                        
                        return (normal.X(), normal.Y(), normal.Z()), (point.X(), point.Y(), point.Z())
                except:
                    pass
                return None, None

            def are_coplanar(face1, face2, tolerance=1e-6):
                """Check if two faces are coplanar within tolerance"""
                normal1, point1 = get_plane_normal_and_point(face1)
                normal2, point2 = get_plane_normal_and_point(face2)
                
                if normal1 is None or normal2 is None:
                    return False
                
                # Check if normals are parallel (or anti-parallel)
                dot_product = abs(normal1[0]*normal2[0] + normal1[1]*normal2[1] + normal1[2]*normal2[2])
                if abs(dot_product - 1.0) > tolerance:
                    return False
                
                # Check if points lie on the same plane
                # Vector from point1 to point2
                vec_12 = (point2[0]-point1[0], point2[1]-point1[1], point2[2]-point1[2])
                
                # Dot product with normal should be ~0 if points are coplanar
                distance = abs(vec_12[0]*normal1[0] + vec_12[1]*normal1[1] + vec_12[2]*normal1[2])
                
                return distance < tolerance

            def group_coplanar_faces(offset_faces, tolerance=1e-6):
                """Group faces that are coplanar"""
                valid_faces = [f for f in offset_faces if f is not None]
                groups = []
                used = set()
                
                for i, face1 in enumerate(valid_faces):
                    if i in used:
                        continue
                        
                    # Start new group with this face
                    group = [face1]
                    used.add(i)
                    
                    # Find all other faces coplanar with this one
                    for j, face2 in enumerate(valid_faces):
                        if j in used or j <= i:
                            continue
                            
                        if are_coplanar(face1, face2, tolerance):
                            group.append(face2)
                            used.add(j)
                    
                    groups.append(group)
                
                print(f"Grouped {len(valid_faces)} faces into {len(groups)} coplanar groups")
                return groups

            def merge_coplanar_group(face_group):
                """Merge a group of coplanar faces using boolean union"""
                if len(face_group) == 1:
                    return face_group[0]
                
                result = face_group[0]
                
                for face in face_group[1:]:
                    try:
                        # Use Fuse to union the faces
                        fuse_op = BRepAlgoAPI_Fuse(result, face)
                        fuse_op.Build()
                        
                        if fuse_op.IsDone():
                            result = fuse_op.Shape()
                        else:
                            print("Warning: Face merge failed, keeping separate")
                            # If merge fails, we'll just keep them separate
                    except Exception as e:
                        print(f"Warning: Exception during face merge: {e}")
                
                return result

            def preprocess_coplanar_faces(face_id_pairs, tolerance=0.1):
                """
                Merge coplanar faces before splitting to reduce artifacts
                """
                # Group coplanar faces
                coplanar_groups = group_coplanar_faces(face_id_pairs, tolerance)
                
                # Merge each group and track genealogy
                merged_faces_with_ids = []
                for group in coplanar_groups:
                    if len(group) == 1:
                        # Single face, no merge needed
                        face, face_id = group[0]
                        merged_faces_with_ids.append((face, face_id))
                    else:
                        # Merge group - this returns a tuple (merged_face, source_ids)
                        merged_result = merge_coplanar_group(group)
                        merged_face, source_ids = merged_result  # Unpack the tuple properly
                        
                        # Extract faces from the merged result (could be compound)
                        if merged_face:
                            explorer = TopExp_Explorer(merged_face, TopAbs_FACE)  # Now merged_face is just the shape
                            while explorer.More():
                                current_face = explorer.Current()
                                # Register this as a merged face from multiple sources
                                boundary_id = face_genealogy[source_ids[0]]['boundary_id']  # Use first source's boundary
                                merged_face_id = register_face(
                                    current_face, boundary_id, "merged", merged_faces_label, 
                                    operation=f"merged_from_{len(source_ids)}_faces"
                                )[0]
                                
                                # Record the merge operation genealogy
                                face_genealogy[merged_face_id]['operations'].append({
                                    'operation': 'merged_from',
                                    'source_face_ids': source_ids,
                                    'stage': 'merged'
                                })
                                
                                merged_faces_with_ids.append((current_face, merged_face_id))
                                explorer.Next()
                
                print(f"Merged down to {len(merged_faces_with_ids)} faces")
                return merged_faces_with_ids

            def offset_opencascade_face(face, offset):
                # Extract the outer wire of the face
                outer_wire = breptools_OuterWire(face)

                # Create 2D offset with sharp join type in constructor
                offset_maker = BRepOffsetAPI_MakeOffset(outer_wire, GeomAbs_Intersection)

                # No need for AddWire since we passed the wire in constructor
                offset_maker.Perform(offset)

                if offset_maker.IsDone():
                    offset_wire = offset_maker.Shape()
                    
                    # Convert wire back to face
                    face_maker = BRepBuilderAPI_MakeFace(offset_wire)
                    if face_maker.IsDone():
                        offset_face = face_maker.Face()
                        return offset_face
                    
            def split_faces_by_faces(offset_faces, tolerance=0.01):
                """
                Use BRepAlgoAPI_Splitter to split all offset faces by each other
                This creates a cell complex by finding all intersections
                """
                
                # Create a compound of all offset faces using BRep_Builder
                builder = BRep_Builder()
                compound = TopoDS_Compound()
                builder.MakeCompound(compound)
                
                # Add all offset faces to the compound
                for face in offset_faces:
                    if face is not None:  # Skip any None faces from failed offsets
                        builder.Add(compound, face)
                
                # Create the splitter
                splitter = BRepAlgoAPI_Splitter()
                
                # Set tolerance
                splitter.SetFuzzyValue(tolerance)

                # Create TopTools_ListOfShape for arguments and tools
                arguments_list = TopTools_ListOfShape()
                tools_list = TopTools_ListOfShape()
                
                # Add each face individually as both argument and tool
                for face in offset_faces:
                    if face is not None:
                        arguments_list.Append(face)
                        tools_list.Append(face)
            
                # Set arguments and tools
                splitter.SetArguments(arguments_list)
                splitter.SetTools(tools_list)
                
                # Perform the splitting operation
                splitter.Build()
                
                # Add the compound as both arguments and tools
                # Arguments = shapes to be split
                # Tools = shapes to split by
                
                
                if splitter.IsDone():
                    result_shape = splitter.Shape()
                    
                    # Extract all resulting faces
                    split_faces = []
                    explorer = TopExp_Explorer(result_shape, TopAbs_FACE)
                    
                    while explorer.More():
                        split_face = explorer.Current()
                        split_faces.append(split_face)
                        explorer.Next()
                    
                    print(f"Splitter created {len(split_faces)} faces from {len(offset_faces)} input faces")
                    return split_faces
                else:
                    print("Splitter operation failed")
                    return None

            
            
            # MAIN PROCESS STARTS HERE
            all_boundaries = list(self.boundaries.values())
            
            # Step 1: Create initial faces with TNaming registration
            initial_faces_with_ids = []
            initial_topologic_faces_with_ids = []
            for i, boundary in enumerate(all_boundaries):
                face = self._create_robust_face(boundary, tolerance)
                if face:
                    # Register in TNaming system
                    face_id, face_label = register_face(face, boundary.id, "initial", initial_faces_label)
                    initial_faces_with_ids.append((face, face_id))

                topology = boundary.topologic
                
                # remove any co-linear edges
                topology = Shell.RemoveCollinearEdges(topology)
               
                topologic_face = Face.ByShell(topology)

                # transfer dict
                topologic_face = transfer_topologic_dict(topology, topologic_face)

                dict = Topology.Dictionary(topologic_face)

                dict = Dictionary.SetValueAtKey(dict, "boundary_id", boundary.id)

                topologic_face = Topology.SetDictionary(topologic_face, dict)

                initial_topologic_faces_with_ids.append(topologic_face)


            print(f"Created {len(initial_faces_with_ids)} initial faces")

            # Step 2: Create offset faces with tracking
            offset_faces_with_ids = []
            for face, face_id in initial_faces_with_ids:
                offset_face = offset_opencascade_face(face, 0.2)
                if offset_face:
                    # Track transformation
                    new_face_id = track_face_transformation(
                        face_id, offset_face, "offset", offset_faces_label, "offset_operation"
                    )
                    offset_faces_with_ids.append((offset_face, new_face_id))

            # Step 3: Merge coplanar faces with tracking
            merged_faces_with_ids = preprocess_coplanar_faces(offset_faces_with_ids)

            # Step 4: Create volumes
            volume_maker = BOPAlgo_MakerVolume()

            # Add faces to volume maker
            faces_only = [face for face, face_id in merged_faces_with_ids]
            for face in faces_only:
                volume_maker.AddArgument(face)

            # Set AvoidInternalShapes to true
            volume_maker.SetAvoidInternalShapes(True)
            volume_maker.SetIntersect(True)
            volume_maker.SetFuzzyValue(0.01)
            # Perform the volume creation
            volume_maker.Perform()

            # Check if operation was successful
            if volume_maker.HasErrors():
                print("Error creating volumes:")
                print(volume_maker.GetReport())
            else:
                # Get the resulting shape(s)
                result = volume_maker.Shape()
                
                # Extract volumes and their faces with tracking
                explorer = TopExp_Explorer(result, TopAbs_SOLID)
                volumes = []
                final_face_tracking = []
                all_found_faces = []
                boundary_to_healed_faces = {}
                healed_faces = []
                while explorer.More():
                    solid = explorer.Current()

                    # unifier = ShapeUpgrade_UnifySameDomain(solid, True, True, True)
                    # unifier.SetLinearTolerance(0.1)  # Adjust based on your model scale
                    # unifier.SetAngularTolerance(0.1)
                    # unifier.Build()
                    # unified_solid = unifier.Shape()

                    # healer = ShapeFix_Shape()
                    # healer.Init(unified_solid)
                    # healer.Perform()
                    # healed_solid = healer.Shape()

                    volumes.append(solid)
                    
                    # Simplified but more robust approach
                    face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
                    volume_faces = []

                    # while face_explorer.More():
                    #     face = face_explorer.Current()
                    #     healed_faces.append(face)
                    #     volume_faces.append(face)
                    #     all_found_faces.append(face)
                    #     # Find the most likely source face by geometric properties
                    #     best_match_boundary_id = None
                    #     best_match_face_id = None
                        
                    #     # Get geometric properties of the current face
                    #     from OCC.Core.GProp import GProp_GProps
                    #     from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
                        
                    #     current_props = GProp_GProps()
                    #     brepgprop_SurfaceProperties(face, current_props)
                    #     current_center = current_props.CentreOfMass()
                    #     current_area = current_props.Mass()
                        
                    #     min_distance = float('inf')

                    #     def get_vert_dist_between_faces(face, initial_face):
                    #         # Calculate distance between centers
                                
                    #         vertices1, vertices2 = [], []

                    #         # Get vertices from face1
                    #         v_exp1 = TopExp_Explorer(face, TopAbs_VERTEX)
                    #         while v_exp1.More():
                    #             pt = BRep_Tool.Pnt(v_exp1.Current())
                    #             vertices1.append((pt.X(), pt.Y(), pt.Z()))
                    #             v_exp1.Next()
                            
                    #         # Get vertices from face2  
                    #         v_exp2 = TopExp_Explorer(initial_face, TopAbs_VERTEX)
                    #         while v_exp2.More():
                    #             pt = BRep_Tool.Pnt(v_exp2.Current())
                    #             vertices2.append((pt.X(), pt.Y(), pt.Z()))
                    #             v_exp2.Next()
                            
                    #         if not vertices1 or not vertices2:
                    #             return float('inf')
                            
                    #         # Calculate average minimum distance
                    #         total_dist = 0.0
                    #         for v1 in vertices1:
                    #             min_dist = min(((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2)**0.5 
                    #                         for v2 in vertices2)
                    #             total_dist += min_dist

                    #         avg_dist = total_dist / len(vertices1)
                    #         return avg_dist


                    #     initial_face_distances = []
                    #     for initial_face, initial_face_id in initial_faces_with_ids:
                    #         # initial_props = GProp_GProps()
                    #         # brepgprop_SurfaceProperties(initial_face, initial_props)
                    #         # initial_center = initial_props.CentreOfMass()
                    #         # distance = current_center.Distance(initial_center)  # Fixed: use initial_center

                    #         distance = get_vert_dist_between_faces(face, initial_face)
                    #         co_planer = are_coplanar(face, initial_face)
                    #         if co_planer:
                    #             co_planer_penalty = 1
                            
                    #         else:
                    #             co_planer_penalty = 100

                    #         distance = distance * co_planer_penalty

                    #         initial_face_distances.append({
                    #             "id": initial_face_id,
                    #             "face": initial_face,
                    #             "distance": distance
                    #         })

                    #     # Sort by distance (closest first)
                    #     initial_face_distances.sort(key=lambda x: x['distance'])
                                                    


                        
                    #     # Compare with all merged faces to find the closest match
                    #     for face_item in initial_face_distances[:10]:
                    #         try:
                    #             initial_face_id = face_item['id']
                    #             initial_face = face_item['face']

                    #             initial_props = GProp_GProps()
                    #             brepgprop_SurfaceProperties(initial_face, initial_props)
                    #             merged_center = initial_props.CentreOfMass()
                    #             merged_area = initial_props.Mass()

                    #             # intersection_op = BRepAlgoAPI_Common(face, initial_face)
                    #             # intersection_op.SetFuzzyValue(0.01)
                    #             # intersection_op.Build()

                    #             # if not intersection_op.IsDone():
                    #             #     return 0.0
        
                    #             # intersection_shape = intersection_op.Shape()
                                
                    #             # # Calculate area of intersection
                    #             # intersection_area = 0.0
                    #             # face_explorer = TopExp_Explorer(intersection_shape, TopAbs_FACE)
                                
                    #             # while face_explorer.More():
                    #             #     intersect_face = TopoDS.Face(face_explorer.Current())
                    #             #     face_props = GProp_GProps()
                    #             #     brepgprop.SurfaceProperties(intersect_face, face_props)
                    #             #     intersection_area += face_props.Mass()
                    #             #     face_explorer.Next()
                                
                                
                    #             area_ratio = abs(current_area - merged_area) / max(current_area, merged_area, 1e-10)
                    #             avg_dist = get_vert_dist_between_faces(face, initial_face)
                    #             co_planer = are_coplanar(face, initial_face)
                    #             if co_planer:
                    #                 co_planer_penalty = -1
                    #             else:
                    #                 co_planer_penalty = 1
                    #             # Combined metric: distance + area difference
                    #             # combined_metric = avg_dist + area_ratio * 100  # Weight area difference

                    #             combined_metric = avg_dist + co_planer_penalty * 10_000
                                
                    #             # combined_metric = avg_dist
                                
                    #             if combined_metric < min_distance:
                    #                 min_distance = combined_metric
                    #                 best_match_boundary_id = face_genealogy[initial_face_id]['boundary_id']
                    #                 best_match_face_id = initial_face_id
                    #                 best_match_face = initial_face
                                    
                    #         except:
                    #             continue
                        
                    #     # Fallback to first available if no good match found
                    #     if best_match_boundary_id is None and merged_faces_with_ids:
                    #         best_match_boundary_id = face_genealogy[merged_faces_with_ids[0][1]]['boundary_id']
                    #         best_match_face_id = merged_faces_with_ids[0][1]
                        
                    #     # Register the final face
                    #     if best_match_boundary_id is not None:
                    #         final_face_id = register_face(
                    #             face, best_match_boundary_id, "final_volume", volume_faces_label, 
                    #             operation="volume_boundary", parent_face_id=best_match_face_id
                    #         )[0]
                            
                    #         volume_faces.append((face, final_face_id))
                    #         final_face_tracking.append((face, final_face_id))

                    #         try:
                    #             boundary_to_healed_faces[best_match_boundary_id]
                    #             pass
                    #         except Exception as e:
                    #             boundary_to_healed_faces[best_match_boundary_id] = []

                            
                    #         boundary_to_healed_faces[best_match_boundary_id].append(face)

                        
                    #     face_explorer.Next()
                    
                    # print(f"Volume has {len(volume_faces)} faces")
                    explorer.Next()
            # TODO Apply Healed Face Information back to original Boundaries

            # Create a CC out of the faces

            geom_object = Geometry()
            volumes_topologic_cells = [geom_object._opencascade_to_topologic(v) for v in volumes]

            cc = CellComplex.ByCells(volumes_topologic_cells)
            healed_faces_from_cc = Topology.Faces(cc)

            offset_faces_from_cc = []
            for f in healed_faces_from_cc:
                try:
                    simplified_face = Face.Simplify(f)
                    offset_face = Face.ByOffset(simplified_face, offset=-0.1, numWorkers=1)
                    offset_faces_from_cc.append(offset_face)

                except TypeError as e:
                    pass


        
            # healed_faces_with_ids = Topology.Inherit(healed_faces_from_cc, initial_topologic_faces_with_ids, exclusive=False, tolerance=0.1)
            all_healed_boundaries_dict = {}
            all_healed_boundaries = []
            for h_f in healed_faces_from_cc:
                h_f_geom = Geometry.from_topology(h_f)
                
                vertex_distances = []
                for i_f in initial_topologic_faces_with_ids:
                    i_f_dict = Topology.Dictionary(i_f)
                    i_f_dict = Dictionary.PythonDictionary(i_f_dict)
                    initial_boundary_id = i_f_dict['boundary_id']

                    distance = h_f_geom.average_vertex_distance(Geometry.from_topology(i_f)) ** 1.5
                    
                    if distance == 3.228202347412661:
                        print('hi')
                    
                    co_planer = h_f_geom.is_coplanar(Geometry.from_topology(i_f))

                    if co_planer:
                        co_planer_penalty = 1
                    else:
                        co_planer_penalty = 1_000

                    cost = distance * co_planer_penalty
                    vertex_distances.append(
                        {'distance': cost,
                         'distance_score': distance,
                         'co_planer_penalty': co_planer_penalty,
                         'face': i_f,
                         'boundary_id': initial_boundary_id}
                    )

                # Sort by distance (closest first)
                vertex_distances.sort(key=lambda x: x['distance'])
                                                    
                selected_face = vertex_distances[0]


                        
                original_boundary = self.boundaries[selected_face['boundary_id']]

                geom = Geometry.from_topology(selected_face['face'])

                new_boundary = Boundary(
                    name=original_boundary.name,
                    type=boundary.type,
                    is_access_boundary=boundary.is_access_boundary,
                    is_visual_boundary=boundary.is_visual_boundary,
                    base_item=boundary.base_item,
                    geometry=h_f_geom
                )
                
                all_healed_boundaries.append(new_boundary)
                all_healed_boundaries_dict[new_boundary.id] = new_boundary


            return all_healed_boundaries_dict, all_healed_boundaries

            # for original_id, healed_faces in boundary_to_healed_faces.items():
                # f len(healed_faces) == 1:
                #     all_healed_faces.append(healed_faces[0])
                # elif len(healed_faces) > 1:
                #     # Merge faces with a sewing operation
                #     sewer = BRepBuilderAPI_Sewing(0.1)
                #     for face in healed_faces:
                #         sewer.Add(face)
                #     sewer.Perform()
                #     merged_shape = sewer.SewedShape()
                #     i
                #     # Extract the merged face
                #     face_explorer = TopExp_Explorer(merged_shape, TopAbs_FACE)
                #     if face_explorer.More():
                #         all_healed_faces.append(TopoDS.topods.Face(face_explorer.Current()))

            # Step 9: Update boundary geometries with improved vertex extraction
            # self._update_boundary_geometries(all_boundaries, all_healed_faces)

            #     original_boundary = self.boundaries[original_id]

            #     for face in healed_faces:

            #         geom = Geometry.from_occ(face)

            #         new_boundary = Boundary(
            #             name=original_boundary.name,
            #             type=boundary.type,
            #             is_access_boundary=boundary.is_access_boundary,
            #             is_visual_boundary=boundary.is_visual_boundary,
            #             base_item=boundary.base_item,
            #             geometry=geom
            #         )

            #         all_healed_faces.append(new_boundary)


                    
            
            # return all_healed_faces
        if version == 'topologic':
            from hierarchical.utils import topology_to_dict
            from topologicpy.Face import Face
            from topologicpy.Vertex import Vertex
            from topologicpy.Cell import Cell
            from topologicpy.Wire import Wire
            from topologicpy.CellComplex import CellComplex
            from topologicpy.Topology import Topology
            from topologicpy.Cluster import Cluster
            from topologicpy.CSG import CSG
            from topologicpy.Shell import Shell
            from hierarchical.utils import transfer_topologic_dict
            from topologicpy.Helper import Helper
            from topologicpy.Graph import Graph as Topologic_Graph


            def intersection_edges_from_faces(faces, tol=1e-4, angle_prune_degree=1.0, skip_near_parallel=False, use_bbox_prune=True, merge=False):
                from itertools import combinations
                from math import cos, pi
                from topologicpy.CSG import CSG
                from topologicpy.Topology import Topology
                from topologicpy.Cluster import Cluster
                from topologicpy.Face import Face
                from topologicpy.Vector import Vector
                from topologicpy.Wire import Wire

                def _bbox_overlay(a, b, tol=1e-9):

                    def _brute_force_bounding_box(face):
                        vertices = Topology.Vertices(face)
                        if not vertices:
                            return None
                        x = [Vertex.X(v) for v in vertices]
                        y = [Vertex.Y(v) for v in vertices]
                        z = [Vertex.Z(v) for v in vertices]

                        min_x = min(x)
                        max_x = max(x)
                        min_y = min(y)
                        max_y = max(y)
                        min_z = min(z)
                        max_z = max(z)

                        # Create 8 corner vertices for the bounding box
                        v1 = Vertex.ByCoordinates(min_x, min_y, min_z)  # Bottom-front-left
                        v2 = Vertex.ByCoordinates(max_x, min_y, min_z)  # Bottom-front-right
                        v3 = Vertex.ByCoordinates(max_x, max_y, min_z)  # Bottom-back-right
                        v4 = Vertex.ByCoordinates(min_x, max_y, min_z)  # Bottom-back-left
                        v5 = Vertex.ByCoordinates(min_x, min_y, max_z)  # Top-front-left
                        v6 = Vertex.ByCoordinates(max_x, min_y, max_z)  # Top-front-right
                        v7 = Vertex.ByCoordinates(max_x, max_y, max_z)  # Top-back-right
                        v8 = Vertex.ByCoordinates(min_x, max_y, max_z)  # Top-back-left
                        
                        
                        # create wires for the edges of the bounding box
                        wires = [
                            Wire.ByVertices([v1, v2]),
                            Wire.ByVertices([v2, v3]),
                            Wire.ByVertices([v3, v4]),
                            Wire.ByVertices([v4, v1]),
                            Wire.ByVertices([v5, v6]),
                            Wire.ByVertices([v6, v7]),
                            Wire.ByVertices([v7, v8]),
                            Wire.ByVertices([v8, v5]),
                            Wire.ByVertices([v1, v5]),
                            Wire.ByVertices([v2, v6]),
                            Wire.ByVertices([v3, v7]),
                            Wire.ByVertices([v4, v8])
                        ] 

                        cell = Cell.ByWires(wires)

                        

                        return cell
                    
                    try:
                        ca = Topology.BoundingBox(a, tolerance=0.0000001)
                        cb = Topology.BoundingBox(b, tolerance=0.00000001)
                        if not ca:
                            ca = _brute_force_bounding_box(a)
                        if not cb:
                            cb = _brute_force_bounding_box(b)

                        va = Topology.Vertices(ca)
                        vb = Topology.Vertices(cb)

                        ax = [Vertex.X(v) for v in va]
                        ay = [Vertex.Y(v) for v in va]
                        az = [Vertex.Z(v) for v in va]

                        bx = [Vertex.X(v) for v in vb]
                        by = [Vertex.Y(v) for v in vb]
                        bz = [Vertex.Z(v) for v in vb]

                        ax0, ax1 = min(ax), max(ax)
                        ay0, ay1 = min(ay), max(ay)
                        az0, az1 = min(az), max(az)

                        bx0, bx1 = min(bx), max(bx)
                        by0, by1 = min(by), max(by)
                        bz0, bz1 = min(bz), max(bz)

                        return (ax0 <= bx1 + tol and ax1 + tol >= bx0 and
                                ay0 <= by1 + tol and ay1 + tol >= by0 and
                                az0 <= bz1 + tol and az1 + tol >= bz0)
                    except Exception as e:
                        return False

                # compute normals if possible 
                normals = {}
                for f in faces:
                    try:
                        normals[f] = Vector.Normalize(Face.Normal(f))
                    except:
                        normals[f] = None

                # Threshold for near parallelism
                cos_thresh = cos(angle_prune_degree * pi / 180.0)

                collected_edges = []
                for fa, fb in combinations(faces, 2):
                    if use_bbox_prune and not _bbox_overlay(fa, fb):
                        continue

                    try:
                        na, nb = normals[fa], normals[fb]
                        if skip_near_parallel and na and nb:
                            if abs(Vector.Dot(na, nb)) > cos_thresh:
                                continue
                    except:
                        pass

                    # intersection
                    try:
                        res = Topology.Intersect(fa, fb, tolerance=tol)
                        if not res:
                            continue
                        if Topology.IsInstance(res, "Face"):
                            es = Topology.Edges(res)
                        elif Topology.IsInstance(res, "Edge"):
                            es = [res]
                        if es:
                            collected_edges.extend(es)
                    except Exception as e:
                        print(f"Error computing intersection between faces: {e}")
                        continue

                # Now intersect edges to find manifold points
                collected_intersected_wires = []
                for ea, eb in combinations(collected_edges, 2):
                    # intersection
                    ea_verts = Topology.Vertices(ea)
                    eb_verts = Topology.Vertices(eb)

                    ea_x1 = Vertex.X(ea_verts[0])
                    ea_y1 = Vertex.Y(ea_verts[0])
                    ea_x2 = Vertex.X(ea_verts[1])
                    ea_y2 = Vertex.Y(ea_verts[1])

                    eb_x1 = Vertex.X(eb_verts[0])
                    eb_y1 = Vertex.Y(eb_verts[0])
                    eb_x2 = Vertex.X(eb_verts[1])
                    eb_y2 = Vertex.Y(eb_verts[1])
                                    
                    try:
                        res = Topology.Intersect(ea, eb, tolerance=tol)
                        if not res:
                            continue
                        if Topology.IsInstance(res, "Edge"): 
                            # es = Topology.Vertices(res)
                            pass
                        elif Topology.IsInstance(res, "Vertex"):
                            # for each edge create a wire using original end points and the new intersection point
                            wire_ea = Wire.ByVertices([ea_verts[0], res, ea_verts[1]])
                            wire_eb = Wire.ByVertices([eb_verts[0], res, eb_verts[1]])
                            es = [wire_ea, wire_eb]
                        if es:
                            collected_intersected_wires.extend(es)
                    except Exception as e:
                        print(f"Error computing intersection between faces: {e}")
                        continue

                if not collected_intersected_wires:
                    return []
                
                if not merge:
                    return collected_intersected_wires
                
                # Merge / Clean
                try:
                    merged = Topology.SelfMerge(Cluster.ByTopologies(collected_intersected_wires), tolerance=tol)
                    if Topology.IsInstance(merged, "Cluster"):
                        return Topology.Edges(merged) or []
                    if Topology.IsInstance(merged, "Edge"):
                        return [merged]
                    
                    # Best Effort Extraction

                    return Topology.Edges(merged) or []
                except Exception as e:
                    return collected_intersected_wires


                   
                                                




            def find_shared_edges_simple(faces):
                    edge_to_faces = defaultdict(list)

                    # get all edges
                    all_edges = set()
                    for face in faces:
                        edges = Topology.Edges(face)
                        for edge in edges:
                            all_edges.add(edge)
                    
                    
                    for edge in all_edges:
                        edge_to_faces[edge] = []
                        # Iterate through each face and check edges
                        for face_idx, face in enumerate(faces):
                            # Check if edge is a super topology of each face
                            super_topologies = Topology.SuperTopologies(edge, face, 'Face')
                            if super_topologies:
                                edge_to_faces[edge].extend(super_topologies)
                    
                    # Analyze the results
                    naked_edges = []
                    shared_edges = []
                    overlapping_edges = []
                    
                    for edge_key, face_indices in edge_to_faces.items():
                        if len(face_indices) == 1:
                            naked_edges.append((edge_key, face_indices[0]))
                        elif len(face_indices) == 2:
                            shared_edges.append((edge_key, face_indices))
                        else:
                            overlapping_edges.append((edge_key, face_indices))
                    
                    print(f"Naked edges: {len(naked_edges)}")
                    print(f"Shared edges: {len(shared_edges)}")
                    print(f"Overlapping edges: {len(overlapping_edges)}")
                    
                    # Show some examples
                    if naked_edges:
                        print(f"Example naked edge on face {naked_edges[0][1]}")
                    if shared_edges:
                        print(f"Example shared edge between faces {shared_edges[0][1]}")
                    if overlapping_edges:
                        print(f"Example overlapping edge on faces {overlapping_edges[0][1]}")
                    
                    return edge_to_faces
            
            def are_all_directions_collinear(directions, tolerance=0.01):
                """
                Check if all direction vectors are collinear (parallel or anti-parallel).
                """
                from topologicpy.Vector import Vector

                if len(directions) <= 1:
                    return True

                # Use first direction as reference
                ref_dir = Vector.Normalize(directions[0])
                if not ref_dir:
                    return False

                for direction in directions[1:]:
                    norm_dir = Vector.Normalize(direction)
                    if not norm_dir:
                        continue

                    # Check dot product - should be ±1 for collinear vectors
                    dot = abs(Vector.Dot(ref_dir, norm_dir))
                    if dot < (1.0 - tolerance):
                        return False

                return True


            topologic_faces = []
            all_boundaries = list(self.boundaries.values())
            
            # Step 1: Create initial faces with better polygon construction
            topologic_faces = []
            for boundary in all_boundaries:
                topology = boundary.topologic
                
                # remove any co-linear edges
                topology = Shell.RemoveCollinearEdges(topology)
               
                topologic_face = Face.ByShell(topology)

                # transfer dict
                topologic_face = transfer_topologic_dict(topology, topologic_face)


                topologic_faces.append(topologic_face)
            

            # # Desired Number of Vertices for each face (e.g. 4)
            # desired_n_verts = 4

            # # Maximum Allowed Number of Attemps to simplify the faces and reduce the nunmber of their vertices (e.g. 20)
            # max_attempts = 20

            # simplified_faces = []
            # for f in topologic_faces:
            #     n = len(Topology.Vertices(f))
            #     tol = 0.0000001
            #     m = 0
            #     f_simplified = f
            #     while n > desired_n_verts and m < max_attempts:
            #         f_simplified = Face.Simplify(f, tolerance=tol)
            #         f_simplified = transfer_topologic_dict(f, f_simplified)
            #         n = len(Topology.Vertices(f_simplified))
            #         tol = tol*10
            #         m = m + 1

            #     simplified_faces.append(f_simplified)

            simplified_faces = topologic_faces

            print(" ")
            for f in simplified_faces:
                verts = Topology.Vertices(f)
                print(" AFTER:", len(verts))

            # Try to create a CellComplex
            cc = None
            offset = 0.5
            attempts = 0
            max_attempts = 10
            cell_complexes = []
            while attempts < max_attempts:
                expanded_faces = [transfer_topologic_dict(f, Face.ByOffset(f, offset=-offset)) for f in simplified_faces]
                
                wires = intersection_edges_from_faces(expanded_faces, tol=0.001, angle_prune_degree=1.0, skip_near_parallel=True, use_bbox_prune=False, merge=False)
                print(f"Len Wires: {len(wires)}")

                # Create cluster from wires to get unified topology
                wire_cluster = Cluster.ByTopologies(wires)
                
                # Self-merge to connect shared vertices
                merged_topology = Topology.SelfMerge(wire_cluster, transferDictionaries=True, tolerance=0.1)
                wires = Topology.Wires(merged_topology)
                print(f"Len Wires: {len(wires)}")

                # Extract vertices and edges
                vertices = Topology.Vertices(merged_topology)
                edges = Topology.Edges(merged_topology)
                print(len(edges))
                merged_edges = Cluster.ByTopologies(edges)
                merged_edges = Topology.SelfMerge(merged_edges, transferDictionaries=True, tolerance=0.1)
                edges = Topology.Edges(merged_edges)
                print(len(edges))

                # Create graph
                graph = Topologic_Graph.ByVerticesEdges(vertices, edges)
                graph_degree_scores = Topologic_Graph.DegreeCentrality(graph, key='degree')
                vertices = Topologic_Graph.Vertices(graph)
                vertices_degree = []
                v_i = 0
                for vertex in vertices:
                    d = Topology.Dictionary(vertex)
                    d = Dictionary.SetValueAtKey(d, "degree", graph_degree_scores[v_i])
                    d = Dictionary.SetValueAtKey(d, "degree_str", str(graph_degree_scores[v_i]))
                    vertex = Topology.SetDictionary(vertex, d)
                    vertices_degree.append(vertex)
                    v_i += 1

                
                # Find vertices with degree 1 (dangling)
                vertices_to_remove = []
                for vertex in vertices:
                    edges = Topologic_Graph.Edges(graph, vertices=[vertex])
                    edge_directions = []

                    for edge in edges:
                        direction = Edge.Direction(edge)
                        edge_directions.append(direction)

                    # Check if all directions are parallel/anti-parallel
                    if are_all_directions_collinear(edge_directions):
                        vertices_to_remove.append(vertex)
                    
                
                # Remove dangling vertices iteratively
                while vertices_to_remove:
                    v_count = 0
                    for vertex in vertices_to_remove:
                        graph = Topologic_Graph.RemoveVertex(graph, vertex)
                    vertices_to_remove = []
                    # # Check for new dangling vertices after removal
                    # vertices = Topologic_Graph.Vertices(graph)
                    # vertices_to_remove = []
                    # for vertex in vertices:
                    #     degree = Topologic_Graph.VertexDegree(graph, vertex, tolerance=tolerance)
                    #     if degree == 1:
                    #         vertices_to_remove.append(vertex)
                
                # Convert cleaned graph back to wires
                cleaned_edges = Topologic_Graph.Edges(graph)
                cleaned_wires = []
                
                # Group connected edges into wires
                edge_groups = group_connected_edges(cleaned_edges, tolerance)
                for edge_group in edge_groups:
                    wire = Wire.ByEdges(edge_group, orient=True, tolerance=tolerance)
                    if wire:
                        cleaned_wires.append(wire)
                
                return cleaned_wires
                
                #     print("Found edges:", edges)
                # edges = []
                # for e in edges:
                #     verts = Topology.Vertices(e)
                #     wire = Wire.ByVertices(verts)
                #     wires.append(wire)


                try:
                    cc = CellComplex.ByFaces(expanded_faces, silent=True)
                    if Topology.IsInstance(cc, "CellComplex"):

                        cells = Topology.Cells(cc)
                        cell_volumns = [Cell.Volume(c) for c in cells]
                        # get the mean volumn
                        mean_volumn = sum(cell_volumns) / len(cell_volumns) if cell_volumns else 0
                        # get the index of volumes that are greater than the mean volumn
                        large_cells_indices = [i for i, v in enumerate(cell_volumns) if v > mean_volumn]
                        # get the cells that are larger than the mean volumn
                        large_cells = [cells[i] for i in large_cells_indices]

                        if len(large_cells) > 2:
                            # create a new cell complex with the large cells
                            cc = CellComplex.ByCells(large_cells, silent=True)

                            cell_complexes.append(cc)
                
                except Exception as e:
                    pass

                attempts += 1
                offset += 0.5
                print(f"attempt {attempts}")

            n_cells = [len(Topology.Cells(c)) for c in cell_complexes]
            cell_complexes = Helper.Sort(cell_complexes, n_cells)
            cc = cell_complexes[0] 

            cc = Topology.RemoveCollinearEdges(cc, tolerance=0.001, silent=True)
            # get the faces from the CellComplex
            healed_faces = Topology.Faces(cc)
            
            healed_faces_info = Topology.Inherit(healed_faces, topologic_faces, keys='id', exclusive=True, tolerance=0.1, silent=False)

            
            '''
            Now we will update the boundary geometries with the geometries of the healed faces
            If boundaries were split or merged in the process of created the cell complex, we will either add or
            remove boundaries from the model accordingly and link them back to their respective base objects
            '''
            ## Something happens there and we loose some boundaries
            healed_boundaries = []
            # iterate through the healed topologic faces
            for face in healed_faces:
                # get the face dictionary
                face_dict = topology_to_dict(face)
                # check if the face has an id
                if 'id' in face_dict:
                    boundary_id = face_dict['id']
                    all_boundary_ids = [b.id for b in all_boundaries]
                    # check if the boundary_id exists in all_boundary_ids
                    if boundary_id in all_boundary_ids:
                        for boundary in all_boundaries:
                            if boundary.id == boundary_id:
                                # create a boundary with information from the original boundary
                                new_boundary = Boundary(
                                    name=boundary.name,
                                    type=boundary.type,
                                    is_access_boundary=boundary.is_access_boundary,
                                    is_visual_boundary=boundary.is_visual_boundary,
                                    geometry=Geometry.from_topology(face),
                                    base_item=boundary.base_item,
                                    )

                                new_boundary.inherit_relationships_from(boundary)

                                healed_boundaries.append(new_boundary)
                    else:
                        print("Boundary ID not found in all boundaries:", boundary_id)
                else:
                    print(f"Face {face} does not have an id, skipping.")


            # put the new boundaries in the model
            self.boundaries = {b.id: b for b in healed_boundaries}
            return healed_boundaries



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

        max_workers = min(multiprocessing.cpu_count(), int(os.getenv("WORKERS", 4)))  # Don't overwhelm the system

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


        # Safety check for empty groups
        if not groups:
            return []
            
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
                        boundary.geometry._opencascade_shape = healed_faces[i]
                        
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



    def infer_bounds(self):
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
            # Avoid division by zero when all decks are at the same height
            if wall_span == 0:
                wall_height_ratio = 1.0  # Assume full height wall if no span
            else:
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
                                             type=boundary.type.lower(),
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
                                             type=boundary.type.lower(),
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
              

                rel = Creates(wall.id, boundary.id)
                boundary.relationships.append(rel)
                self.relationships[boundary.id].append(rel)

       
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

            # add relatinoship between the deck and the boundary
            rel = Creates(deck.id, wall.id)
            boundary.relationships.append(rel)
            self.relationships[boundary.id].append(rel)
            # self.building_graph.add_edge(deck.id, boundary.id, "OBJECT_CREATES_BOUNDARY", from_label='Object', to_label='Boundary')
            # Don't process decks
                


        # Now lets heal the boundaries by finding intersections and extending them
        # self.heal_boundaries(dimentions=dimentions)
        healed_boundary_dict, healed_boundaries = self.heal_boundaries(tolerance=25.0,
                                             version='occ'
                                             )

        self.boundaries = healed_boundary_dict

        # Add boundary info to the boundary graph and the building graph
        for i, boundary in enumerate(self.boundaries.values()):
            self.boundary_graph.add_node(boundary.id,
                                        type=boundary.type.lower(),
                                        geometry=boundary.geometry,
                                        is_access_boundary=boundary.is_access_boundary,
                                        is_visual_boundary=boundary.is_visual_boundary,
                                        base_item=boundary.base_item,
                                        height=boundary.height,
                                        normal_vector=boundary.normal_vector,
                                        centroid_x=boundary.base_item.get_centroid().x,
                                        centroid_y=boundary.base_item.get_centroid().y,
                                        centroid_z=boundary.base_item.get_centroid().z)

           

            features = {
                'boundary_id': boundary.id,
                'type': boundary.type,
                'is_access_boundary': boundary.is_access_boundary,
                'is_visual_boundary': boundary.is_visual_boundary,
                'centroid_x': boundary.base_item.get_centroid().x,
                'centroid_y': boundary.base_item.get_centroid().y,
                'centroid_z': boundary.base_item.get_centroid().z
            }

            self.building_graph.add_node('Boundary', node_id=boundary.id, features=features)
            self.building_graph.add_edge(boundary.base_item.id, boundary.id, "OBJECT_CREATES_BOUNDARY", from_label='Object', to_label='Boundary')


        # test_healing_validation(self.boundaries, occ_faces)

        # Now lets create the adjacency relationships between boundaries
        # for boundary_id, boundary in self.boundaries.items():
        #     # Find adjacent boundaries based on their geometry
        #     for other_boundary_id, other_boundary in self.boundaries.items():
        #         if boundary_id == other_boundary_id:
        #             continue
        #         # Check if the boundaries intersect this needs to be a mesh intersect
        #         # Skip if either boundary has no geometry
        #         if boundary.geometry is None or other_boundary.geometry is None:
        #             continue
        #         if boundary.geometry.mesh_intersects(other_boundary.geometry):
        #             # Create adjacency relationship
        #             rel = AdjacentTo(boundary_id, other_boundary_id)
        #             boundary.relationships.append(rel)

        #             self.relationships[boundary_id].append(rel)
        #             self.boundary_graph.add_edge(boundary_id, other_boundary_id, relationship=rel.type)

        #             # Add the other boundary to the adjacent spaces list
        #             boundary.adjacent_spaces.append(other_boundary_id)
        #             other_boundary.adjacent_spaces.append(boundary_id)

        #             # Add to the building graph
        #             self.building_graph.add_edge(boundary_id, other_boundary_id, 'BOUNDARY_ADJACENT_TO', from_label='Boundary', to_label='Boundary')

        print(f"Boundaries inferred: {len(self.boundaries)}")



    def infer_spaces(self, 
                    existing_spaces: Optional[List[Space]] = None,
                    ) -> List[Space]:
        """
        Infer spaces by finding boundary cycles in the boundary graph (networkx) whose edges form closed loops.
        This method will create Space objects from the boundaries and their relationships.
        Returns:
            List[Space]: A list of Space objects representing the inferred spaces.
        """

        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cell import Cell
        from itertools import combinations

        def build_cellcomplex_dropping_bad_faces(faces, tolerance=0.0001, silent=False):
            """
            Build a CellComplex by progressively adding faces, dropping any that fail to merge.
            
            Returns:
            - cellcomplex: The resulting CellComplex (or None if failed)
            - good_faces: List of faces that were successfully included
            - bad_faces: List of faces that were dropped
            """
            from topologicpy.CellComplex import CellComplex
            from topologicpy.Topology import Topology

            if not faces:
                return None, [], []

            good_faces = []
            bad_faces = []

            # Start with the first valid face
            cellcomplex = None
            for i, face in enumerate(faces):
                if Topology.IsInstance(face, "Face"):
                    cellcomplex = face
                    good_faces.append(face)
                    start_index = i + 1
                    break

            if not cellcomplex:
                if not silent:
                    print("No valid starting face found")
                return None, [], faces

            # Try to merge each remaining face
            for i in range(start_index, len(faces)):
                face = faces[i]
                if not Topology.IsInstance(face, "Face"):
                    bad_faces.append(face)
                    continue

                try:
                    # Attempt to merge the face
                    new_cellcomplex = cellcomplex.Merge(face, False, tolerance)

                    if new_cellcomplex and Topology.IsInstance(new_cellcomplex, "Topology"):
                        cellcomplex = new_cellcomplex
                        good_faces.append(face)
                    else:
                        bad_faces.append(face)
                        if not silent:
                            print(f"Dropped face #{i} - merge returned invalid topology")
                except Exception as e:
                    bad_faces.append(face)
                    if not silent:
                        print(f"Dropped face #{i} - merge failed with error")

            # Check if we actually got a CellComplex
            if Topology.Type(cellcomplex) != Topology.TypeID("CellComplex"):
                if not silent:
                    print(f"Warning: Result is {Topology.TypeAsString(cellcomplex)}, not a CellComplex")

            return cellcomplex, good_faces, bad_faces


        space_counter = 0
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
            from OCC.Core.Message import Message_ProgressRange
            # generate temp file
            temp_path = f'temp_{uuid4()}.brep'
            # write occ to brep file - progress range is optional
            breptools.Write(occ_shape, temp_path)

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
            if boundary.geometry.topologic is None:
                topologic_face = occ_to_topologic(boundary.geometry._opencascade_shape)
            else:
                topologic_face = boundary.geometry.topologic
            if topologic_face:
                face_dict = Topology.Dictionary(topologic_face)
                face_dict = Dictionary.SetValueAtKey(face_dict, 'boundary_id', boundary.id)
                topologic_face = Topology.SetDictionary(topologic_face, face_dict)


                topologic_faces.append(topologic_face)

        # find all combinations of 4+ faces that form a closed volume
        cc = CellComplex.ByFaces(topologic_faces, tolerance=0.01, silent=True)


        if cc is None:
            cc, g_faces, b_faces = build_cellcomplex_dropping_bad_faces(topologic_faces)
            raise ValueError("No valid cell complex could be created.")

        cells = Topology.Cells(cc)

        ## TODO: When load_from_ifc cells is empty [] 

        for cell in cells:
            geometry = Geometry.from_topology(cell)

            faces = Topology.Faces(cell)
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

            for face in faces:
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


            # Check if the space overlaps with an existing space and apply the existing space's attributes
            existing_space = next((s for s in existing_spaces if s.geometry.bbox_intersects(space.geometry, return_overlap_percent=True) > 0.8), None)

            if existing_space:
                print(f"Applying existing space attributes to inferred space: {existing_space.name}")
                space.name = existing_space.name
                

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

    def apply_ontologies(self):
        """
        Apply the provided ontology to the spaces in the model.
        This will categorize spaces based on the ontology definitions.
        
        Args:
            ontology (dict): A dictionary representing the ontology with space types and their definitions.
        """
        from hierarchical.ontologies.space import OmniclassSpaceOntology

        labeled_spaces = []
        for space in self.spaces.values():
            try:
                # Apply the Omniclass Space Ontology to each space
                ontology = OmniclassSpaceOntology()
                space, ontology_attrs = ontology.apply_ontology(space)

                # update the space in the building graph with ontology attributes
                self.building_graph.update_node(
                    space.id,
                    'Space',
                    attributes=ontology_attrs
                )
                
                labeled_spaces.append(space)
            except Exception as e:
                print(f"Error applying ontology to space '{space.name}': {e}")
        self.spaces = {space.id: space for space in labeled_spaces}

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
                    # Check if item has geometry 
                    if not hasattr(item, 'geometry'):
                        continue

                    # Use mesh representation for vertex/face extraction
                    try:
                        mesh = item.geometry.mesh
                        item_vertices = mesh.get("vertices", [])
                        item_faces = mesh.get("faces", [])
                        
                        # Add vertices with offset
                        offset = len(vertices)
                        vertices.extend(item_vertices)
                        
                        # Add faces with vertex index offset
                        for face in item_faces:
                            if len(face) >= 3:
                                faces.append(tuple(idx + offset for idx in face))
                        continue
                    except:
                        # Fallback: no geometry data available
                        continue

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

    def show_spaces_graph(self, by=None):
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

    def show_spaces(self, by=None):
        """
        Display the inferred spaces in the model using Plotly.
        Each space is represented as a 3D mesh with its boundaries.
        Colors spaces based on the value stored in space.attributes[by] if provided.
        """
        import plotly.graph_objects as go
        import plotly.express as px
        import numpy as np

        fig = go.Figure()

        if not self.spaces:
            print("No spaces to display.")
            return

        # Collect color values if 'by' parameter is provided
        color_values = []
        space_list = list(self.spaces.values())
        
        if by:
            for space in space_list:
                if hasattr(space, 'attributes') and isinstance(space.attributes, dict):
                    color_value = space.attributes.get(by, None)
                    color_values.append(color_value)
                else:
                    color_values.append(None)
        
        # Determine coloring strategy
        use_coloring = by and any(cv is not None for cv in color_values)
        color_map = {}
        colorscale = None
        
        if use_coloring:
            # Filter out None values for analysis
            valid_values = [cv for cv in color_values if cv is not None]
            
            if not valid_values:
                use_coloring = False
            else:
                # Check if values are numeric
                try:
                    numeric_values = [float(v) for v in valid_values]
                    is_numeric = True
                    
                    # Normalize values for colorscale (0-1 range)
                    min_val, max_val = min(numeric_values), max(numeric_values)
                    if max_val > min_val:
                        colorscale = px.colors.sequential.Viridis
                    else:
                        is_numeric = False
                except (ValueError, TypeError):
                    is_numeric = False
                
                if not is_numeric:
                    # Categorical coloring
                    unique_values = list(set(valid_values))
                    colors_palette = px.colors.qualitative.Set1
                    if len(unique_values) > len(colors_palette):
                        colors_palette = px.colors.qualitative.Light24
                    
                    for i, val in enumerate(unique_values):
                        color_map[val] = colors_palette[i % len(colors_palette)]

        for i, space in enumerate(space_list):
            # Extract geometry data
            vertices = space.geometry.get_vertices()
            faces = space.geometry.get_faces()

            if not vertices or not faces:
                continue

            x, y, z = zip(*vertices)
            i_faces, j_faces, k_faces = zip(*faces)

            # Determine color for this space
            space_color = 'lightblue'  # Default color
            
            if use_coloring and i < len(color_values) and color_values[i] is not None:
                if by in ['numeric'] or (colorscale and isinstance(color_values[i], (int, float))):
                    # Numeric coloring - convert to colorscale index
                    try:
                        normalized_val = (float(color_values[i]) - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                        color_idx = int(normalized_val * (len(colorscale) - 1))
                        space_color = colorscale[color_idx]
                    except:
                        space_color = 'lightblue'
                else:
                    # Categorical coloring
                    space_color = color_map.get(color_values[i], 'lightblue')

            # Create name with attribute value if coloring
            space_name = f"Space: {space.id}"
            if use_coloring and i < len(color_values) and color_values[i] is not None:
                space_name += f" ({by}: {color_values[i]})"

            # Create mesh3d trace for the space
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i_faces, j=j_faces, k=k_faces,
                opacity=0.7,
                color=space_color,
                name=space_name,
                hoverinfo='name',
                showlegend=True
            ))

        # Update layout
        title = "Inferred Spaces"
        if use_coloring:
            title += f" - Colored by {by}"
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            showlegend=True
        )

        # Add colorbar for numeric values
        if use_coloring and colorscale and 'min_val' in locals() and 'max_val' in locals():
            # Create a dummy scatter trace for the colorbar
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(
                    size=0.1,
                    color=[min_val, max_val],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title=by,
                        x=1.02
                    )
                ),
                showlegend=False,
                hoverinfo='skip'
            ))

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
    
    def get_openai_client(self):
        from openai import OpenAI
        client = OpenAI(api_key=OPEN_AI_API_KEY)
        return client

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
        
        client = self.get_openai_client()

        # Step 1: Get schema context
        node_types = self.building_graph.get_node_types_to_string()
        rel_types = self.building_graph.get_relationship_types_to_string()
        connections = self.building_graph.get_connection_schema_string()

        # Step 2: Construct system prompt
        system_prompt = f"""
    You are an expert in developing Cypher queries for kuzu to query a building model stored in a graph database.

    Here is the graph schema:
    - Node Types: 
        structured: [n.type, label(n)]
        {node_types}
    - Relationship Types: {rel_types}
    - Connection Patterns:\n{connections}

    Tips:
    - "has" or "have" questions about two types of items can be answered by looking for OBJECT_EMBEDS or CONTAINS relationships.
    - "in" questions about two types of items can be answered by looking for OBJECT_EMBEDDED_IN or PART_OF relationships.


    Examples:
    
    What walls have doors? -> MATCH (o1:Object)-[e:OBJECT_EMBEDS]-(o2:Object) WHERE o1.type = 'Wall' AND o2.type = 'Door' RETURN o1
    

    Given a user's question, return the Cypher query that can be run on this graph to answer it.
    Always use the available types and relationships, and return only the query its self, nothing else.
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
