from hierarchical.items import Element, Component, Wall, Deck, Window, Door, Object, BaseItem
from hierarchical.relationships import AdjacentTo, Relationship
from hierarchical.geometry import Geometry
from collections import defaultdict
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from copy import deepcopy
from hierarchical.utils import generate_id, plot_shapely_lines, plot_topologic_objects, plot_items
import matplotlib.pyplot as plt
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Vertex import Vertex
from topologicpy.Cell import Cell



# Abstractions are things like spaces and zones. They are not elements but are 
# defined by elements. They are used to group elements together in ways that are meaningful 
# and speak to how people experience the building. For example, a room is a space that is defined by
# walls, floors, and ceilings. (Maybe boundary abstractions)

# Idea 1: We create boundaries out of our walls and floors -> We make spaces out of boundaries?

# Idea 2: We create spaces out of our walls and floors -> We make boundaries out of spaces?

@dataclass
class Boundary:
    """Represents a space boundary with its properties and geometry."""
    id: str
    type: str  # 'full', 'partial', 'open'
    geometry: 'Geometry'
    is_access_boundary: bool
    is_visual_boundary: bool
    base_item: 'BaseItem'  # Wall elements that form this boundary
    start_point_bottom: Tuple[float, float, float] # left most vertex of the bottom edge, if equal in the x direction then the top most vertex in the y
    end_point_bottom: Tuple[float, float, float] # right most vertex of the bottom edge, if equal in the x direction then the bottom most vertex in the y
    height: float
    normal_vector: Tuple[float, float, float]
    adjacent_spaces: List[str]  # IDs of spaces this boundary separates
    
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
    
@dataclass
class Space:
    """
    Represents a space in the building model, defined by its boundaries and properties.
    """
    id: str
    name: str
    geometry: Geometry
    boundaries: List[Boundary] = field(default_factory=list)
    volume: float = 0.0
    area: float = 0.0

    relationships: Dict[str, List[Relationship]] = field(default_factory=lambda: defaultdict(list))


    

class Model:
    """
    A class representing a building model, which can
    - contain objects and their components and elements.
    - contain spaces and zones.
    - contain relationships between all types of items
    """
    def __init__(self):
        self.elements = {}
        self.components = {}
        self.objects = {}
        self.spaces = {}
        self.zones = {}
        self.relationships = defaultdict(list)
        self.boundaries = {}
        self.building_graph = nx.DiGraph()
        # Boundary Graph - Might collapse with building_graph in future
        self.boundary_graph = nx.Graph()  

    @classmethod
    def from_objects(cls, objects):
        """
        Create a model from a list of objects.
        """
        model = cls()
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
                "length": getattr(obj, "length", None),
                "width": getattr(obj, "width", None),
                "height": getattr(obj, "height", None),
                "volume": getattr(obj, "volume", None),
                "centroid_x": obj.get_centroid().x,
                "centroid_y": obj.get_centroid().y,
                "centroid_z": obj.get_centroid().z,
                "object": obj  # preserve full object for further use
            }

            model.building_graph.add_node(obj_id, **features)


        return model

    def create_adjacency_relationships(self, tolerance=0.01):
        """
        Add adjacency relationships between objects in the model to the building_graph.
        """
        for obj in self.objects.values():
            adjacent_items = obj.find_adjacent_items(self.objects.values(), tolerance=tolerance)
            for adjacent_item in adjacent_items:
                rel = AdjacentTo(obj.id, adjacent_item.id)
                self.relationships[obj.id].append(rel)

                # Add edge to graph
                self.building_graph.add_edge(obj.id, adjacent_item.id, relationship=rel.type)


    ### Bounds
    def heal_boundaries(self, max_extension: float = 2.0, extension_step: float = 0.01, max_iterations: int = 200):
        """
        Heal boundaries by interpreting them as lines and calculating direct intersections.
        Much more efficient than iterative extension.
        
        Args:
            max_extension: Maximum distance to extend any boundary (in meters)
            extension_step: Not used in direct intersection approach
            max_iterations: Not used in direct intersection approach
        """
        print(f"Starting boundary healing process with {len(self.boundaries)} boundaries...")
        
        healed_pairs = set()  # Track which boundary pairs have been healed
        
        for boundary_id, boundary in self.boundaries.items():
            if not hasattr(boundary, 'geometry') or not boundary.geometry:
                continue
                
            # find the boundaries that refer the the adjacent items to the base_item of the boundary
            base_item_relationships = self.relationships[boundary.base_item.id]
            adjacent_boundaries = []
            for r in base_item_relationships:
                if isinstance(r, AdjacentTo):
                    target_item = r.target
                    # check if there is a boundary for this target item
                    try:
                        target_object = self.objects[target_item]
                        if not hasattr(target_object, 'boundary_id'):
                            continue  # Skip if no boundary_id exists
                        target_boundary = target_object.boundary_id
                        target_boundary = self.boundaries[target_boundary]
                        adjacent_boundaries.append(target_boundary)
                    except KeyError:
                        # If no boundary exists for this target item, we can skip it
                        continue
                    
            for adj_boundary in adjacent_boundaries:
                pair_key = tuple(sorted([boundary_id, adj_boundary.id]))
                if pair_key in healed_pairs:
                    continue  # Already healed this pair
                
                # Check current intersection status
                intersects = boundary.geometry.bbox_intersects(adj_boundary.geometry)
                
                if intersects:
                    print(f"Boundaries {boundary_id} and {adj_boundary.id} already intersect")
                    healed_pairs.add(pair_key)
                    continue

                # get bottom edge of the boundary
                start_point = boundary.start_point_bottom
                end_point = boundary.end_point_bottom

                # line function y = mx + b
                m = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0]) if end_point[0] != start_point[0] else float('inf')
                b = start_point[1] - m * start_point[0] if m != float('inf') else start_point[0]

                # line function adj y = m_adj * x + b_adj
                adj_start_point = adj_boundary.start_point_bottom
                adj_end_point = adj_boundary.end_point_bottom
                m_adj = (adj_end_point[1] - adj_start_point[1]) / (adj_end_point[0] - adj_start_point[0]) if adj_end_point[0] != adj_start_point[0] else float('inf')
                b_adj = adj_start_point[1] - m_adj * adj_start_point[0] if m_adj != float('inf') else adj_start_point[0]

                # Calculate intersection point
                if m == m_adj:
                    print(f"Boundaries {boundary_id} and {adj_boundary.id} are parallel, skipping healing")
                    continue
                if m == float('inf'):
                    # Boundary is vertical
                    intersection_x = start_point[0]
                    intersection_y = m_adj * intersection_x + b_adj
                elif m_adj == float('inf'):
                    # Adjacent boundary is vertical
                    intersection_x = adj_start_point[0]
                    intersection_y = m * intersection_x + b
                else:
                    # Solve for intersection
                    intersection_x = (b_adj - b) / (m - m_adj)
                    intersection_y = m * intersection_x + b

                intersection_point = (intersection_x, intersection_y, start_point[2])  # Assuming z-coordinate is the same
                # NEW APPROACH: Extend boundary face toward intersection, not edge collapse
                intersection_x, intersection_y, intersection_z = intersection_point

                # Get current boundary bounds
                vertices_array = np.array(boundary.geometry.get_vertices())
                min_x, max_x = vertices_array[:, 0].min(), vertices_array[:, 0].max()
                min_y, max_y = vertices_array[:, 1].min(), vertices_array[:, 1].max()

                tolerance = 0.01  # 1cm tolerance

                # Create updated vertices list
                vertices_list = list(boundary.geometry.get_vertices())

                print(f"Boundary {boundary_id} bounds: X[{min_x:.3f}, {max_x:.3f}], Y[{min_y:.3f}, {max_y:.3f}]")
                print(f"Intersection: ({intersection_x:.3f}, {intersection_y:.3f}, {intersection_z:.3f})")

                # Extend vertices toward intersection point while maintaining wall structure
                vertices_modified = False
                for i, vertex in enumerate(vertices_list):
                    x, y, z = vertex
                    new_vertex = [x, y, z]
                    original_vertex = new_vertex.copy()
                    
                    # Extend in X direction if intersection is outside current bounds
                    if intersection_x > max_x and abs(x - max_x) < tolerance:
                        new_vertex[0] = intersection_x  # Extend max-X face
                        vertices_modified = True
                        print(f"Extended vertex {i} in +X direction: {vertex} -> {tuple(new_vertex)}")
                    elif intersection_x < min_x and abs(x - min_x) < tolerance:
                        new_vertex[0] = intersection_x  # Extend min-X face  
                        vertices_modified = True
                        print(f"Extended vertex {i} in -X direction: {vertex} -> {tuple(new_vertex)}")
                        
                    # Extend in Y direction if intersection is outside current bounds
                    if intersection_y > max_y and abs(y - max_y) < tolerance:
                        new_vertex[1] = intersection_y  # Extend max-Y face
                        vertices_modified = True
                        print(f"Extended vertex {i} in +Y direction: {vertex} -> {tuple(new_vertex)}")
                    elif intersection_y < min_y and abs(y - min_y) < tolerance:
                        new_vertex[1] = intersection_y  # Extend min-Y face
                        vertices_modified = True
                        print(f"Extended vertex {i} in -Y direction: {vertex} -> {tuple(new_vertex)}")
                    
                    vertices_list[i] = tuple(new_vertex)

                if vertices_modified:
                    print(f"Updated boundary geometry for {boundary_id} with new vertices: {vertices_list}")
                else:
                    print(f"No vertices modified for {boundary_id} - intersection may be within bounds")

                # Update boundary geometry
                original_faces = boundary.geometry.get_faces()
                new_geometry = Geometry()
                new_geometry.mesh_data['vertices'] = vertices_list  
                new_geometry.mesh_data['faces'] = original_faces
                new_geometry._generate_brep_from_mesh()
                boundary.geometry = new_geometry

                # SAME APPROACH for adjacent boundary
                adj_vertices_array = np.array(adj_boundary.geometry.get_vertices())
                adj_min_x, adj_max_x = adj_vertices_array[:, 0].min(), adj_vertices_array[:, 0].max()
                adj_min_y, adj_max_y = adj_vertices_array[:, 1].min(), adj_vertices_array[:, 1].max()

                print(f"Adjacent boundary {adj_boundary.id} bounds: X[{adj_min_x:.3f}, {adj_max_x:.3f}], Y[{adj_min_y:.3f}, {adj_max_y:.3f}]")

                adj_vertices_list = list(adj_boundary.geometry.get_vertices())
                adj_vertices_modified = False

                for i, vertex in enumerate(adj_vertices_list):
                    x, y, z = vertex
                    new_vertex = [x, y, z]
                    
                    if intersection_x > adj_max_x and abs(x - adj_max_x) < tolerance:
                        new_vertex[0] = intersection_x
                        adj_vertices_modified = True
                        print(f"Extended adjacent vertex {i} in +X direction: {vertex} -> {tuple(new_vertex)}")
                    elif intersection_x < adj_min_x and abs(x - adj_min_x) < tolerance:
                        new_vertex[0] = intersection_x
                        adj_vertices_modified = True
                        print(f"Extended adjacent vertex {i} in -X direction: {vertex} -> {tuple(new_vertex)}")
                        
                    if intersection_y > adj_max_y and abs(y - adj_max_y) < tolerance:
                        new_vertex[1] = intersection_y
                        adj_vertices_modified = True
                        print(f"Extended adjacent vertex {i} in +Y direction: {vertex} -> {tuple(new_vertex)}")
                    elif intersection_y < adj_min_y and abs(y - adj_min_y) < tolerance:
                        new_vertex[1] = intersection_y
                        adj_vertices_modified = True
                        print(f"Extended adjacent vertex {i} in -Y direction: {vertex} -> {tuple(new_vertex)}")
                    
                    adj_vertices_list[i] = tuple(new_vertex)

                if adj_vertices_modified:
                    print(f"Updated adjacent boundary geometry for {adj_boundary.id} with new vertices: {adj_vertices_list}")
                else:
                    print(f"No vertices modified for adjacent boundary {adj_boundary.id}")

                # Update adjacent boundary geometry
                adj_original_faces = adj_boundary.geometry.get_faces()
                new_adj_geometry = Geometry()
                new_adj_geometry.mesh_data['vertices'] = adj_vertices_list
                new_adj_geometry.mesh_data['faces'] = adj_original_faces  
                new_adj_geometry._generate_brep_from_mesh()
                adj_boundary.geometry = new_adj_geometry

                # Add healed pair to set
                self.boundaries[boundary.id] = boundary
                self.boundaries[adj_boundary.id] = adj_boundary

        print(f"Boundary healing process completed. Healed {len(healed_pairs)} pairs of boundaries.")
                    

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
        # Find wall objects in the model that are of class Wall or inharit from Wall

        # for each wall determine if its a full height wall based on its adjacency relationships to decks at the top and the bottom


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
                    id=generate_id('boundary'),
                    type='full',
                    geometry=wall.get_centerplane_geometry(),
                    is_access_boundary=True,
                    is_visual_boundary=True,
                    base_item=wall,
                    start_point_bottom=wall.get_centerplane_start_point_bottom(),
                    end_point_bottom=wall.get_centerplane_end_point_bottom(),
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
                                             start_point_bottom=boundary.start_point_bottom,
                                             end_point_bottom=boundary.end_point_bottom,
                                             height=boundary.height,
                                             normal_vector=boundary.normal_vector, 
                                             centroid_x=wall.get_centroid().x,
                                             centroid_y=wall.get_centroid().y,
                                             centroid_z=wall.get_centroid().z)
                
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
                    start_point_bottom=wall.get_centerplane_start_point_bottom(),
                    end_point_bottom=wall.get_centerplane_end_point_bottom(),
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
                                             start_point_bottom=boundary.start_point_bottom,
                                             end_point_bottom=boundary.end_point_bottom,
                                             height=boundary.height,
                                             normal_vector=boundary.normal_vector, 
                                             centroid_x=wall.get_centroid().x,
                                             centroid_y=wall.get_centroid().y,
                                             centroid_z=wall.get_centroid().z)
                
                # add boundary_id to the walls boundary_id attribute
                wall.boundary_id = boundary.id

                                             

            else:
                boundary = Boundary(
                    id=generate_id('boundary'),
                    type='open',
                    geometry=wall.get_centerplane_geometry(),
                    is_access_boundary=False,
                    is_visual_boundary=False,
                    base_item=wall,
                    start_point_bottom=wall.get_centerplane_start_point_bottom(),
                    end_point_bottom=wall.get_centerplane_end_point_bottom(),
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
                                            start_point_bottom=boundary.start_point_bottom,
                                            end_point_bottom=boundary.end_point_bottom,
                                            height=boundary.height,
                                            normal_vector=boundary.normal_vector,
                                            centroid_x=wall.get_centroid().x,
                                            centroid_y=wall.get_centroid().y,
                                            centroid_z=wall.get_centroid().z)
                
                # add boundary_id to the walls boundary_id attribute
                wall.boundary_id = boundary.id
                
        # Now lets heal the boundaries by finding intersections and extending them
        self.heal_boundaries()

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
                    self.relationships[boundary_id].append(rel)
                    self.boundary_graph.add_edge(boundary_id, other_boundary_id, relationship=rel.type)

                    # Add the other boundary to the adjacent spaces list
                    boundary.adjacent_spaces.append(other_boundary_id)
                    other_boundary.adjacent_spaces.append(boundary_id)
                    
        print(f"Boundaries inferred: {len(self.boundaries)}")
                
        

    def infer_spaces(self):
        """
        Infer spaces by finding boundary cycles in the boundary graph (networkx) whose edges form closed loops.
        This method will create Space objects from the boundaries and their relationships.
        Returns:
            List[Space]: A list of Space objects representing the inferred spaces.
        """
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

                space_id = generate_id('space')
                cell = Cell.ByThickenedFace(face, thickness=max([boundary.height for boundary in cycle_boundaries]))
                # Create geometry from the cell
                geometry = Geometry.from_topology(cell)
                space = Space(
                    id=space_id,
                    name=f"Space {space_id}",
                    geometry=geometry,
                    boundaries=cycle_boundaries,
                    area=Face.Area(face) # Assuming area as a proxy for area in 2D
                )
                
                self.spaces[space_id] = space

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
                
                face = Face.ByEdges(topologic_edges, tolerance=0.01)
                if face:
                    space_height = max([boundary.height for boundary in cycle_boundaries])
                    cell = Cell.ByThickenedFace(face, thickness=space_height)
                    
                    # Create geometry from the cell
                    geometry = Geometry.from_topology(cell)
                    space_id = generate_id('space')
                    space = Space(
                        id=space_id,
                        name=f"Space {space_id}",
                        boundaries=cycle_boundaries,
                        area=Face.Area(face),
                        geometry=geometry  # Assuming area as a proxy for area in 2D
                    )
                    
                    self.spaces[space_id] = space
                

                


            




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
        # simple nx show
        """
        Display the boundaries graph using NetworkX and Matplotlib.
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.boundary_graph, seed=42)
        nx.draw(self.boundary_graph, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_color='black', edge_color='gray')

        # show the node attributes as labels
        node_labels = {node: f"{node}\n{data['type']}" for node, data in self.boundary_graph.nodes(data=True)}
        nx.draw_networkx_labels(self.boundary_graph, pos, labels=node_labels, font_size=8, font_color='black')
        plt.title("Boundaries Graph")
        plt.show()



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
    
