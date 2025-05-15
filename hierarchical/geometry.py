from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path


@dataclass(slots=True)
class Vector3D:
    """A 3D vector representation"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """Return vector as tuple"""
        return (self.x, self.y, self.z)
    
    def as_array(self) -> np.ndarray:
        """Return vector as numpy array"""
        return np.array([self.x, self.y, self.z])


@dataclass(slots=True)
class Geometry:
    """
    A class that defines an item's geometry.

    Stores representations in 3 ways:
    1) A set of sub-item geometries (referenced by item IDs)
    2) A mesh representation of the union of sub-geometries
    3) A B-rep representation of the union of sub-geometries
    """

    # 1) Hierarchical sub-geometries (by item ID reference)
    sub_geometries: Tuple[str, ...] = field(default_factory=tuple)

    # 2) Compact mesh representation (e.g., vertex and face data)
    # Highly compressed format for fast rendering or physics
    mesh_data: Dict[str, Any] = field(default_factory=dict)
    # Example keys: {"vertices": [...], "faces": [...]}

    # 3) Exact B-rep representation
    # Precise boundary surfaces + topology for CAD or fabrication
    brep_data: Dict[str, Any] = field(default_factory=dict)
    # Example keys: {"surfaces": [...], "edges": [...], "vertices": [...]}
    
    # Origin point of the geometry
    origin: Vector3D = field(default_factory=Vector3D)
    
    # Transformation matrix for local coordinate system
    transform: Optional[np.ndarray] = None

    def __repr__(self):
        verts = len(self.mesh_data.get("vertices", [])) if self.mesh_data else 0
        faces = len(self.mesh_data.get("faces", [])) if self.mesh_data else 0
        return f"Geometry(vertices={verts}, faces={faces})"

    @classmethod
    def from_obj(cls, obj_path: Union[str, Path]) -> None:
        """
        Create mesh and brep_data from an OBJ file
        
        Args:
            obj_path: Path to the OBJ file
        """
        vertices = []
        faces = []
        
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):  # Vertex
                    parts = line.split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append((x, y, z))
                elif line.startswith('f '):  # Face
                    parts = line.split()
                    face_indices = []
                    for p in parts[1:]:
                        # OBJ indices start at 1, so subtract 1
                        idx = int(p.split('/')[0]) - 1
                        face_indices.append(idx)
                    faces.append(tuple(face_indices))
        
        geom = cls()
        geom.mesh_data = {"vertices": vertices, "faces": faces}
        geom._generate_brep_from_mesh()

        return geom

    @classmethod
    def from_stl(cls, stl_path: Union[str, Path]) -> None:
        """
        Create mesh data from an STL file
        
        Args:
            stl_path: Path to the STL file
        """
        # This is a simplified implementation
        # In a real system, you'd use a library like numpy-stl
        vertices = []
        faces = []
        
        # Simple STL parsing logic
        # (In production code, use a proper STL parser library)
        
        geom = cls()
        geom.mesh_data = {"vertices": vertices, "faces": faces}
        geom._generate_brep_from_mesh()

        return geom

    @classmethod
    def from_primitive(cls, primitive_type: str, dimensions: Dict[str, float]) -> None:
        """
        Create geometry from primitive shapes
        
        Args:
            primitive_type: One of 'box', 'cylinder', 'sphere', etc.
            dimensions: Dictionary with dimensions (varies by primitive type)
        """
        if primitive_type == 'box':
            width = dimensions.get('width', 1.0)
            depth = dimensions.get('depth', 1.0)
            height = dimensions.get('height', 1.0)
            
            # Create vertices for a box
            vertices = [
                (0, 0, 0), (width, 0, 0), (width, depth, 0), (0, depth, 0),
                (0, 0, height), (width, 0, height), (width, depth, height), (0, depth, height)
            ]
            
            # Create faces (using triangles)
            faces = [
                # Bottom face
                (0, 1, 2), (0, 2, 3),
                # Top face
                (4, 6, 5), (4, 7, 6),
                # Side faces
                (0, 4, 1), (1, 4, 5),
                (1, 5, 2), (2, 5, 6),
                (2, 6, 3), (3, 6, 7),
                (3, 7, 0), (0, 7, 4)
            ]
            
            
            
        elif primitive_type == 'cylinder':
            # Implementation for cylinder
            radius = dimensions.get('radius', 0.5)
            height = dimensions.get('height', 1.0)
            segments = dimensions.get('segments', 16)
            
            # Create vertices and faces for a cylinder
            # (simplified implementation)
            vertices = []
            faces = []
            
            # Add bottom and top center points
            vertices.append((0, 0, 0))  # Bottom center
            vertices.append((0, 0, height))  # Top center
            
            # Add circular points
            for i in range(segments):
                angle = 2 * np.pi * i / segments
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                # Add bottom and top rim points
                vertices.append((x, y, 0))
                vertices.append((x, y, height))
            
            # Create faces
            for i in range(segments):
                bottom_idx = 2 + i * 2
                top_idx = 3 + i * 2
                next_bottom_idx = 2 + ((i + 1) % segments) * 2
                next_top_idx = 3 + ((i + 1) % segments) * 2
                
                # Bottom face triangle
                faces.append((0, bottom_idx, next_bottom_idx))
                
                # Top face triangle
                faces.append((1, next_top_idx, top_idx))
                
                # Side face (two triangles)
                faces.append((bottom_idx, top_idx, next_bottom_idx))
                faces.append((next_bottom_idx, top_idx, next_top_idx))
            
        
        elif primitive_type == 'sphere':
            # Implementation for sphere
            # (simplified)
            pass

        geom = cls()
        geom.mesh_data = {"vertices": vertices, "faces": faces}
        geom._generate_brep_from_mesh()

        return geom

    @classmethod
    def from_prism(cls, base_points: List[Tuple[float, float]], height: float) -> None:
        """
        Create a vertical prism from a base polygon.

        Args:
            base_points: Ordered (x, y) tuples.
            height: Extrusion height in Z direction.
        """
        num = len(base_points)
        vertices = []

        # Bottom face
        vertices += [(x, y, 0.0) for (x, y) in base_points]

        # Top face
        vertices += [(x, y, height) for (x, y) in base_points]

        faces = []

        # Bottom face triangles (assumes simple convex)
        for i in range(1, num - 1):
            faces.append((0, i, i + 1))

        # Top face triangles
        for i in range(1, num - 1):
            faces.append((num, num + i, num + i + 1))

        # Side faces
        for i in range(num):
            next_i = (i + 1) % num
            faces.append((i, next_i, num + i))
            faces.append((num + i, next_i, num + next_i))

        geom = cls()
        geom.mesh_data = {"vertices": vertices, "faces": faces}
        geom._generate_brep_from_mesh()

        return geom

    def from_surface(cls, points: List[Tuple[float, float, float]]) -> None:
        """
        Create a mesh directly from unordered 3D points.

        Args:
            points: List of 3D points (x, y, z). Convex hull will be used to generate mesh faces.
        """
        from scipy.spatial import ConvexHull
        import numpy as np

        points_array = np.array(points)
        hull = ConvexHull(points_array)

        vertices = [tuple(p) for p in points_array]
        faces = [tuple(face) for face in hull.simplices]

        geom = cls()
        geom.mesh_data = {"vertices": vertices, "faces": faces}
        geom._generate_brep_from_mesh()

        return geom
    
    def _generate_brep_from_mesh(self) -> None:
        """
        Generate a simple B-rep representation from the mesh data
        This is a simplified version - real B-rep creation is more complex
        """
        if not self.mesh_data:
            return
            
        vertices = self.mesh_data.get("vertices", [])
        faces = self.mesh_data.get("faces", [])
        
        # Convert triangular mesh to B-rep surfaces
        surfaces = []
        edges = set()
        
        for face in faces:
            # Create a surface from the face
            surface = {
                "type": "planar",
                "vertices": [vertices[idx] for idx in face]
            }
            surfaces.append(surface)
            
            # Create edges from face boundaries
            for i in range(len(face)):
                edge = tuple(sorted([face[i], face[(i+1) % len(face)]]))
                edges.add(edge)
        
        self.brep_data = {
            "vertices": vertices,
            "edges": list(edges),
            "surfaces": surfaces
        }
    
    def transform_geometry(self, matrix: np.ndarray) -> None:
        """
        Apply a transformation matrix to the geometry
        
        Args:
            matrix: 4x4 transformation matrix
        """
        # Store the transformation
        if self.transform is None:
            self.transform = matrix
        else:
            self.transform = np.matmul(matrix, self.transform)
            
        # Transform vertices in mesh data
        if self.mesh_data and "vertices" in self.mesh_data:
            transformed_vertices = []
            for vertex in self.mesh_data["vertices"]:
                # Convert to homogeneous coordinates
                v = np.array([vertex[0], vertex[1], vertex[2], 1.0])
                # Apply transformation
                v_transformed = np.matmul(matrix, v)
                # Convert back to 3D
                transformed_vertices.append((
                    v_transformed[0]/v_transformed[3],
                    v_transformed[1]/v_transformed[3],
                    v_transformed[2]/v_transformed[3]
                ))
            self.mesh_data["vertices"] = transformed_vertices
            
        # Also transform B-rep vertices
        if self.brep_data and "vertices" in self.brep_data:
            transformed_brep_vertices = []
            for vertex in self.brep_data["vertices"]:
                v = np.array([vertex[0], vertex[1], vertex[2], 1.0])  # homogeneous
                v_transformed = np.matmul(matrix, v)
                transformed_brep_vertices.append((
                    v_transformed[0] / v_transformed[3],
                    v_transformed[1] / v_transformed[3],
                    v_transformed[2] / v_transformed[3]
                ))
            self.brep_data["vertices"] = transformed_brep_vertices

            # Update surface references too
            if "surfaces" in self.brep_data:
                for surface in self.brep_data["surfaces"]:
                    if "vertices" in surface:
                        surface["vertices"] = [
                            (
                                np.matmul(matrix, np.array([vx, vy, vz, 1.0]))[:3] /
                                np.matmul(matrix, np.array([vx, vy, vz, 1.0]))[3]
                            )
                            for (vx, vy, vz) in surface["vertices"]
                        ]

    def transform_all_geometry(self, matrix: np.ndarray):
        """
        Recursively apply a transformation matrix to this item and all sub-items.

        Args:
            matrix: A 4x4 transformation matrix
        """
        self.transform_geometry(matrix)

        for sub in getattr(self, "sub_geometries", []):
            if isinstance(sub, Geometry):  # We now store objects, not just IDs
                sub.transform_all_geometry(matrix)
    
    def right(self, dx: float) -> "Geometry":
        return self._translate(dx, 0, 0)

    def left(self, dx: float) -> "Geometry":
        return self._translate(-dx, 0, 0)

    def forward(self, dy: float) -> "Geometry":
        return self._translate(0, dy, 0)

    def back(self, dy: float) -> "Geometry":
        return self._translate(0, -dy, 0)

    def up(self, dz: float) -> "Geometry":
        return self._translate(0, 0, dz)

    def down(self, dz: float) -> "Geometry":
        return self._translate(0, 0, -dz)

    def _translate(self, dx: float, dy: float, dz: float) -> "Geometry":
        matrix = np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]
        ])
        self.transform_geometry(matrix)
        return self
        
    def rotate_z(self, angle_rad: float) -> "Geometry":
        """
        Rotate geometry around Z-axis by angle_rad (in radians).
        Returns self to allow chaining.
        """
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        rot_matrix = np.array([
            [cos_theta, -sin_theta, 0.0, 0.0],
            [sin_theta,  cos_theta, 0.0, 0.0],
            [0.0,        0.0,       1.0, 0.0],
            [0.0,        0.0,       0.0, 1.0]
        ])
        self.transform_geometry(rot_matrix)
        return self

    def get_bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box as (min_point, max_point)."""
        if self.mesh_data and "vertices" in self.mesh_data:
            vertices = np.array(self.mesh_data["vertices"])
            min_point = vertices.min(axis=0)
            max_point = vertices.max(axis=0)
            return min_point, max_point
        return np.zeros(3), np.zeros(3)
    
    def bbox_intersects(self, other: "Geometry", return_overlap_percent: bool = False) -> Union[bool, float]:
        """
        Check if bounding boxes intersect.
        
        Args:
            other: Another Geometry object to check intersection with
            return_overlap_percent: If True, return the overlap percentage instead of boolean
            
        Returns:
            If return_overlap_percent is False: bool indicating intersection
            If return_overlap_percent is True: float representing overlap percentage (0.0 to 100.0)
        """
        min1, max1 = self.get_bbox()
        min2, max2 = other.get_bbox()
        
        # Calculate the intersection box
        intersection_min = np.maximum(min1, min2)
        intersection_max = np.minimum(max1, max2)
        
        # Check if there's an intersection
        if np.any(intersection_max < intersection_min):
            return 0.0 if return_overlap_percent else False
        
        if not return_overlap_percent:
            return True
        
        # Calculate volumes
        intersection_dims = intersection_max - intersection_min
        intersection_volume = np.prod(intersection_dims)
        
        # Calculate volumes of original boxes
        box1_dims = max1 - min1
        box1_volume = np.prod(box1_dims)
        
        box2_dims = max2 - min2
        box2_volume = np.prod(box2_dims)
        
        # Calculate overlap percentage (can use different methods)
        # Method 1: Percentage of smaller box that overlaps
        min_volume = min(box1_volume, box2_volume)
        if min_volume == 0:
            return 0.0
        overlap_percent = (intersection_volume / min_volume) * 100.0
        
        return overlap_percent

    
    def distance_to(self, other: 'Geometry') -> float:
        """Calculate minimum distance between two geometries."""
        # Simple implementation using bounding box centers
        # For more accuracy, implement point-to-mesh distance
        center1 = (self.get_bbox()[0] + self.get_bbox()[1]) / 2
        center2 = (other.get_bbox()[0] + other.get_bbox()[1]) / 2
        return np.linalg.norm(center1 - center2)
    
    
    def mesh_intersects(self, other: 'Geometry', return_overlap_percent: bool = False) -> bool:
        """Check if the actual meshes intersect (more precise than bbox)."""
        ## TODO: Implement actual mesh intersection logic
        
        # This is a placeholder for a more complex mesh intersection algorithm
        # You would typically use a library like trimesh or CGAL for this
        return self.bbox_intersects(other, return_overlap_percent)  # Simplified for now
    
    def compute_volume(self) -> float:
        """
        Compute the volume of the geometry
        Note: This is an approximation for closed meshes
        """
        if not self.mesh_data or "vertices" not in self.mesh_data or "faces" not in self.mesh_data:
            return 0.0
            
        vertices = self.mesh_data["vertices"]
        faces = self.mesh_data["faces"]
        
        # Compute volume using the divergence theorem
        # For each triangular face
        volume = 0.0
        for face in faces:
            if len(face) >= 3:  # Make sure it's at least a triangle
                v1 = np.array(vertices[face[0]])
                v2 = np.array(vertices[face[1]])
                v3 = np.array(vertices[face[2]])
                
                # Compute the signed volume of the tetrahedron
                volume += np.dot(v1, np.cross(v2, v3)) / 6.0
                
        return abs(volume)  # Take absolute value as orientation might be reversed

