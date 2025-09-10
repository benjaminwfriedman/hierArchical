from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import math
import time
import uuid
import warnings
from topologicpy.Topology import Topology
from topologicpy.Cell import Cell
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Face import Face



@dataclass(slots=True)
class Vector3D:
    """A 3D vector representation"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __iter__(self):
        """Allow unpacking or iteration over the vector"""
        yield self.x
        yield self.y
        yield self.z
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """Return vector as tuple"""
        return (self.x, self.y, self.z)
    
    def as_array(self) -> np.ndarray:
        """Return vector as numpy array"""
        return np.array([self.x, self.y, self.z])


@dataclass(slots=True)
class Geometry:
    """
    A class that defines an item's geometry with triple representation:
    1) Mesh - lightweight triangular mesh for basic operations
    2) OpenCascade - pythonocc-core shapes for precise CAD operations  
    3) TopologicPy - topology objects for advanced geometric analysis

    Representations are generated lazily with fallback chain:
    TopologicPy → OpenCascade → Mesh
    """

    # Hierarchical sub-geometries (by item ID reference)
    sub_geometries: Tuple[str, ...] = field(default_factory=tuple)

    # Legacy field for backward compatibility (DEPRECATED - use .mesh property)
    mesh_data: Dict[str, Any] = field(default_factory=dict)

    # Private representation storage (use properties for access)
    _mesh_data: Optional[Dict[str, Any]] = field(default=None, init=False)
    _opencascade_shape: Optional[Any] = field(default=None, init=False)  # TopoDS_Shape
    _topologic_topology: Optional[Topology] = field(default=None, init=False)

    # Generation flags for lazy loading
    _mesh_generated: bool = field(default=False, init=False)
    _occ_generated: bool = field(default=False, init=False)
    _topologic_generated: bool = field(default=False, init=False)

    # Metadata
    geometry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    # Origin point of the geometry
    origin: Vector3D = field(default_factory=Vector3D)
    
    # Transformation matrix for local coordinate system
    transform: Optional[np.ndarray] = None

    def __repr__(self):
        # Use new mesh property for cleaner access
        mesh = self.mesh if hasattr(self, '_mesh_data') else self.mesh_data
        verts = len(mesh.get("vertices", [])) if mesh else 0
        faces = len(mesh.get("faces", [])) if mesh else 0
        return f"Geometry(vertices={verts}, faces={faces}, id={self.geometry_id[:8]})"

    # Property-based accessors for triple representation
    @property 
    def mesh(self) -> Dict[str, Any]:
        """Always available mesh representation - fallback for all operations"""
        if not self._mesh_generated:
            self._generate_mesh()
        return self._mesh_data or {}

    @property
    def opencascade(self):
        """OpenCascade shape representation - generated from topologic or mesh"""
        if not self._occ_generated:
            self._generate_opencascade()
        return self._opencascade_shape
        
    @property  
    def topologic(self) -> Optional[Topology]:
        """TopologicPy topology representation - preferred when available"""
        if not self._topologic_generated:
            self._generate_topologic()
        return self._topologic_topology

    # Backward compatibility for mesh_data access
    def _get_mesh_data(self) -> Dict[str, Any]:
        """Backward compatibility getter with deprecation warning"""
        if self._mesh_data is not None or self._mesh_generated:
            return self.mesh
        return self.mesh_data

    def _set_mesh_data(self, value: Dict[str, Any]):
        """Backward compatibility setter with deprecation warning"""
        warnings.warn(
            "Direct mesh_data assignment is deprecated. Use geometry.mesh property or factory methods.",
            DeprecationWarning,
            stacklevel=2
        )
        self._mesh_data = value
        self._mesh_generated = True
        # Invalidate other representations when mesh changes
        self._occ_generated = False
        self._topologic_generated = False

    # Conversion methods with fallback logic
    def _generate_mesh(self):
        """Generate mesh from best available source"""
        if self._topologic_topology:
            self._mesh_data = self._topologic_to_mesh(self._topologic_topology)
        elif self._opencascade_shape:
            self._mesh_data = self._opencascade_to_mesh(self._opencascade_shape)
        elif self.mesh_data:  # Legacy compatibility
            self._mesh_data = self.mesh_data.copy()
        else:
            # Create empty mesh as fallback
            self._mesh_data = {"vertices": [], "faces": []}
        self._mesh_generated = True

    def _generate_opencascade(self):
        """Generate OpenCascade shape from best available source"""
        if self._topologic_topology:
            self._opencascade_shape = self._topologic_to_opencascade(self._topologic_topology)
        elif self._mesh_data or self.mesh_data:
            mesh = self._mesh_data or self.mesh_data
            self._opencascade_shape = self._mesh_to_opencascade(mesh)
        else:
            raise ValueError("No geometry data available for OpenCascade conversion")
        self._occ_generated = True

    def _generate_topologic(self):
        """Generate TopologicPy topology from best available source"""
        if self._topologic_topology:
            return
        elif self._opencascade_shape:
            self._topologic_topology = self._opencascade_to_topologic(self._opencascade_shape)
        elif self._mesh_data or self.mesh_data:
            mesh = self._mesh_data or self.mesh_data
            self._topologic_topology = self._mesh_to_topologic(mesh)
        else:
            raise ValueError("No geometry data available for TopologicPy conversion")
        self._topologic_generated = True

    def _topologic_to_mesh(self, topology: Topology) -> Dict[str, Any]:
        """Convert TopologicPy topology to mesh representation"""
        try:
            from topologicpy.Topology import Topology
            
            # Use the MeshData static method to convert topology to mesh
            mesh_data = Topology.MeshData(topology, mode=0, transferDictionaries=False, mantissa=6, silent=True)
            
            # Convert to the expected format
            vertices = [tuple(vertex) for vertex in mesh_data['vertices']]
            faces = [tuple(face) for face in mesh_data['faces']]
            
            return {"vertices": vertices, "faces": faces}
        except Exception as e:
            raise ValueError(f"Failed to convert TopologicPy to mesh: {e}")

    def _topologic_to_opencascade(self, topology):
      """Convert TopologicPy topology to OpenCascade shape"""
      try:
          # Use existing conversion if available in abstractions
          from .abstractions import shape_from_topology_brep
          return shape_from_topology_brep(topology)
      except ImportError:
          # Fallback: Use BREP string as intermediate format
          from OCC.Core.BRepTools import breptools
          from OCC.Core.BRep import BRep_Builder
          from OCC.Core.TopoDS import TopoDS_Shape
          from OCC.Core.BRepTools import breptools_Read
          from topologicpy.Topology import Topology

          try:
              # Get BREP string from topologic topology
              brep_string = Topology.BREPString(topology)
              
              if not brep_string:
                  return None

              # Create a shape and builder
              shape = TopoDS_Shape()
              builder = BRep_Builder()

              # Read the BREP string into the shape
              success = breptools.Read(shape, brep_string, builder)

              if success:
                  return shape
              else:
                  return None

          except Exception as e:
              print(f"Error converting topology to OpenCascade shape: {e}")
              return None

    def _opencascade_to_mesh(self, shape) -> Dict[str, Any]:
        """Convert OpenCascade shape to mesh representation"""
        try:
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TopLoc import TopLoc_Location
            
            # Triangulate the shape
            mesh = BRepMesh_IncrementalMesh(shape, 0.1)
            mesh.Perform()
            
            vertices = []
            faces = []
            vertex_map = {}
            
            # Extract triangulation from each face
            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while explorer.More():
                face = explorer.Current()
                location = TopLoc_Location()
                triangulation = BRep_Tool.Triangulation(face, location)
                
                if triangulation:
                    # Get transformation if location exists
                    if not location.IsIdentity():
                        trsf = location.Transformation()
                    else:
                        trsf = None

                    # Map local to global vertex indices for this face
                    face_vertex_map = {}
                    
                    # Extract vertices
                    for i in range(1, triangulation.NbNodes() + 1):
                        node = triangulation.Node(i)
                        
                        # Apply transformation if needed
                        if trsf:
                            node.Transform(trsf)
                        
                        vertex_key = (round(node.X(), 6), round(node.Y(), 6), round(node.Z(), 6))
                        
                        if vertex_key not in vertex_map:
                            global_idx = len(vertices)  # Use length as index
                            vertices.append([node.X(), node.Y(), node.Z()])
                            vertex_map[vertex_key] = global_idx
                            face_vertex_map[i] = global_idx
                        else:
                            face_vertex_map[i] = vertex_map[vertex_key]
                    
                    # Extract triangles
                    for i in range(1, triangulation.NbTriangles() + 1):
                        triangle = triangulation.Triangle(i)
                        n1, n2, n3 = triangle.Get()
                        
                        # Convert to global vertex indices
                        if n1 in face_vertex_map and n2 in face_vertex_map and n3 in face_vertex_map:
                            faces.append([face_vertex_map[n1], face_vertex_map[n2], face_vertex_map[n3]])
                
                explorer.Next()
            
            return {"vertices": vertices, "faces": faces}
            
        except Exception as e:
            print(f"Encountered Error: {e}")
            return {
                "vertices": [],
                "faces": []
            }
            
            
       

    def _mesh_to_opencascade(self, mesh: Dict[str, Any]):
        """Convert mesh to OpenCascade shape (limited precision)"""
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
            from OCC.Core.gp import gp_Pnt
            
            vertices = mesh.get("vertices", [])
            faces = mesh.get("faces", [])
            
            if not vertices or not faces:
                return None
            
            sewing = BRepBuilderAPI_Sewing()
            
            # Convert each face to OpenCascade face
            for face in faces:
                if len(face) >= 3:
                    points = []
                    for idx in face:
                        if idx < len(vertices):
                            x, y, z = vertices[idx]
                            points.append(gp_Pnt(x, y, z))
                    
                    # Simplified face creation - real implementation would be more robust
                    if len(points) >= 3:
                        # For now, skip complex face creation
                        pass
            
            return None  # Return None to indicate conversion not yet implemented
            
        except Exception:
            return None

    def _mesh_to_topologic(self, mesh: Dict[str, Any]) -> Optional[Topology]:
        """Convert mesh to TopologicPy topology"""
        try:
            from topologicpy.Vertex import Vertex
            from topologicpy.Face import Face
            from topologicpy.Shell import Shell
            from topologicpy.Cell import Cell
            
            vertices = mesh.get("vertices", [])
            faces = mesh.get("faces", [])
            
            if not vertices or not faces:
                return None
            
            # Create topologic vertices
            topo_vertices = []
            for x, y, z in vertices:
                vertex = Vertex.ByCoordinates(x, y, z)
                topo_vertices.append(vertex)
            
            # Create topologic faces
            topo_faces = []
            for face in faces:
                if len(face) >= 3:
                    face_vertices = []
                    for idx in face:
                        if idx < len(topo_vertices):
                            face_vertices.append(topo_vertices[idx])
                    
                    if len(face_vertices) >= 3:
                        topo_face = Face.ByVertices(face_vertices)
                        if topo_face:
                            topo_faces.append(topo_face)
            
            # Create shell or cell from faces
            if topo_faces:
                try:
                    shell = Shell.ByFaces(topo_faces)
                    if shell:
                        # Try to create a cell if it's a closed shell
                        cell = Cell.ByShell(shell)
                        return cell if cell else shell
                    else:
                        # Return first face if shell creation fails
                        return topo_faces[0] if topo_faces else None
                except:
                    return topo_faces[0] if topo_faces else None
            
            return None
            
        except Exception as e:
            print(f"Warning: Failed to convert mesh to TopologicPy: {e}")
            return None

    def _opencascade_to_topologic(self, shape) -> Optional[Topology]:
        """Convert OpenCascade shape to TopologicPy topology"""
        from topologicpy.Topology import Topology
        from OCC.Core.BRepTools import breptools
        from OCC.Core.Message import Message_ProgressRange
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.TopoDS import TopoDS_Shape
        from topologicpy.Topology import Topology
        import tempfile
        import os
        from uuid import uuid4


        temp_file = None
        try:
        
            # generate temp file
            temp_path = f'temp_{uuid4()}.brep'
            # write occ to brep file - progress range is optional
            breptools.Write(shape, temp_path)

            topology = Topology.ByBREPPath(temp_path)

            # Clean up the temp file
            os.remove(temp_path)

            # Return the topology
            return topology
        
        except Exception as e:
            print(f"Experianced Error: {e}")
                        
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass  # Ignore cleanup errors

    @classmethod
    def from_obj(cls, obj_path: Union[str, Path]) -> "Geometry":
        """
        Create geometry from an OBJ file
        
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
        geom._mesh_data = {"vertices": vertices, "faces": faces}
        geom._mesh_generated = True
        return geom
    
    @classmethod
    def from_topology(cls, topology:Topology):
        """
        Create geometry from a Topology object
        
        Args:
            topology: A Topology object containing faces and vertices
        """
        import numpy as np
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.CellComplex import CellComplex
        
        tolerance = 1e-6

        def triangulate_face_indices(face_indices, triangulation_method="fan"):
            """
            Triangulate a face given its vertex indices.
            
            Args:
                face_indices: List of vertex indices forming the face
                triangulation_method: "fan" (simple) or "center" (better for convex faces)
            
            Returns:
                List of triangles (each triangle is a list of 3 vertex indices)
            """
            if len(face_indices) < 3:
                return []
            elif len(face_indices) == 3:
                return [face_indices]
            
            triangles = []
            
            if triangulation_method == "fan":
                # Fan triangulation - connects all vertices to first vertex
                for i in range(1, len(face_indices) - 1):
                    triangle = [face_indices[0], face_indices[i], face_indices[i + 1]]
                    triangles.append(triangle)
            
            elif triangulation_method == "center":
                # This would require calculating centroid - simplified version
                # For now, fall back to fan triangulation
                for i in range(1, len(face_indices) - 1):
                    triangle = [face_indices[0], face_indices[i], face_indices[i + 1]]
                    triangles.append(triangle)
            
            return triangles

        # Create geometry with topologic as primary representation
        geom = cls()
        geom._topologic_topology = topology
        geom._topologic_generated = True

        # create mesh data from topology
        vertices = []
        faces = []
        if Topology.IsInstance(topology, "Face"):
            face_vertices = Topology.Vertices(topology)
            if len(face_vertices) < 3:
                return None
            face_indices = []
            for vertex in face_vertices:
                x, y, z = Vertex.Coordinates(vertex)
                # Check if vertex already exists
                if (x, y, z) not in vertices:
                    vertices.append((x, y, z))
                face_indices.append(vertices.index((x, y, z)))
            # Triangulate the face
            triangles = triangulate_face_indices(face_indices, triangulation_method="fan")
            for triangle in triangles:
                faces.append(triangle)

        elif Topology.IsInstance(topology, "Cell") or Topology.IsInstance(topology, "CellComplex"):
            for face in Topology.Faces(topology):
                face_vertices = Topology.Vertices(face)
                if len(face_vertices) < 3:
                    continue

                face_indices = []
                for vertex in face_vertices:
                    x, y, z = Vertex.Coordinates(vertex)
                    # Check if vertex already exists
                    if (x, y, z) not in vertices:
                        vertices.append((x, y, z))
                    face_indices.append(vertices.index((x, y, z)))
                # Triangulate the face  
                triangles = triangulate_face_indices(face_indices, triangulation_method="fan")
                for triangle in triangles:
                    faces.append(triangle)

        geom._mesh_data = {"vertices": vertices, "faces": faces}
        geom._mesh_generated = True

        occ_shape = geom._topologic_to_opencascade(topology)

        geom._opencascade_shape = occ_shape
        geom._occ_generated = True

        return geom
    
    @classmethod
    def from_occ(cls, shape):

        geom = cls()
        
        mesh_data = geom._opencascade_to_mesh(shape)

        #TODO: Implement e_opencascade_to_topologic
        topology = geom._opencascade_to_topologic(shape)

        geom._mesh_data = mesh_data
        geom.mesh_data = mesh_data
        geom._mesh_generated = True
        geom._opencascade_shape = shape
        geom._topologic_topology = topology
        geom._topologic_generated = True


        return geom

    from specklepy.objects.geometry import Mesh
    @classmethod
    def from_speckle_mesh(cls, speckle_mesh:Mesh) -> "Geometry":
        """
        Create geometry from a Speckle mesh dictionary
        
        Args:
            speckle_mesh: A Speckle Mesh object
        """
        verticies = speckle_mesh.vertices
        # vertices from speckle mesh ar a flat list, convert to tuples
        verticies = [tuple(verticies[i:i+3]) for i in range(0, len(verticies), 3)]

        raw_faces = speckle_mesh.faces

        # Parse faces that start with vertex count (like [3, 0, 1, 2, 3, 5, 6, 3, ...])
        faces = []
        i = 0
        while i < len(raw_faces):
            vertex_count = raw_faces[i]
            if vertex_count == 3:  # Triangle
                face_tuple = tuple(raw_faces[i+1:i+4])  # Get next 3 indices
                faces.append(face_tuple)
                i += 4  # Move past count + 3 vertices
            elif vertex_count == 4:  # Quad (if you need to handle them)
                # Convert quad to two triangles
                v0, v1, v2, v3 = raw_faces[i+1:i+5]
                faces.append((v0, v1, v2))
                faces.append((v0, v2, v3))
                i += 5  # Move past count + 4 vertices
            else:
                # Handle other polygon types or skip
                i += vertex_count + 1

        if not verticies or not faces:
            raise ValueError("Speckle mesh must have vertices and faces")
        
        geom = cls()
        geom._mesh_data = {"vertices": verticies, "faces": faces}
        geom._mesh_generated = True
        geom._mesh_to_opencascade(geom._mesh_data)
        geom._mesh_to_topologic(geom._mesh_data)

        return geom
        

    @classmethod
    def from_stl(cls, stl_path: Union[str, Path]) -> "Geometry":
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
        geom._mesh_data = {"vertices": vertices, "faces": faces}
        geom._mesh_generated = True
        return geom

    @classmethod
    def from_primitive(cls, primitive_type: str, dimensions: Dict[str, float]) -> "Geometry":
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
        geom._mesh_data = {"vertices": vertices, "faces": faces}
        geom._mesh_generated = True
        return geom

    @classmethod
    def from_prism(cls, base_points: List[Tuple[float, float]], height: float) -> "Geometry":
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
        geom._mesh_data = {"vertices": vertices, "faces": faces}
        geom._mesh_generated = True
        return geom

    @classmethod
    def from_surface(cls, points: List[Tuple[float, float, float]]) -> "Geometry":
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
        geom._mesh_data = {"vertices": vertices, "faces": faces}
        geom._mesh_generated = True
        return geom
    
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
        
        # Helper function to transform vertices
        def transform_vertices(vertices):
            transformed = []
            for vertex in vertices:
                v = np.array([vertex[0], vertex[1], vertex[2], 1.0])
                v_transformed = np.matmul(matrix, v)
                transformed.append((
                    v_transformed[0]/v_transformed[3],
                    v_transformed[1]/v_transformed[3],
                    v_transformed[2]/v_transformed[3]
                ))
            return transformed
            
        # Transform mesh data (legacy and new)
        if self.mesh_data and "vertices" in self.mesh_data:
            self.mesh_data["vertices"] = transform_vertices(self.mesh_data["vertices"])
            
        if self._mesh_data and "vertices" in self._mesh_data:
            self._mesh_data["vertices"] = transform_vertices(self._mesh_data["vertices"])
        
        # Invalidate other representations since geometry has changed
        # TODO: Proper transformation should be applied to OpenCascade and TopologicPy objects
        # For now, invalidate them so they'll be regenerated from the transformed mesh
        self._occ_generated = False
        self._topologic_generated = False

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
        
    def rotate_z(self, angle_rad: float, rotation_point: Optional[np.ndarray] = None) -> "Geometry":
        """
        Rotate geometry around Z-axis by angle_rad (in radians).
        
        Args:
            angle_rad: Rotation angle in radians
            rotation_point: 3D point [x, y, z] to rotate around. If None, rotates around origin.
        
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
        
        if rotation_point is not None:
            # Translate to rotation point, rotate, then translate back
            translate_to_origin = np.array([
                [1.0, 0.0, 0.0, -rotation_point[0]],
                [0.0, 1.0, 0.0, -rotation_point[1]],
                [0.0, 0.0, 1.0, -rotation_point[2]],
                [0.0, 0.0, 0.0, 1.0]
            ])
            
            translate_back = np.array([
                [1.0, 0.0, 0.0, rotation_point[0]],
                [0.0, 1.0, 0.0, rotation_point[1]],
                [0.0, 0.0, 1.0, rotation_point[2]],
                [0.0, 0.0, 0.0, 1.0]
            ])
            
            # Combine transformations: translate back * rotate * translate to origin
            combined_matrix = translate_back @ rot_matrix @ translate_to_origin
            self.transform_geometry(combined_matrix)
        else:
            self.transform_geometry(rot_matrix)
        
        return self
    
    def rotate_x(self, angle_rad: float, rotation_point: Optional[np.ndarray] = None) -> "Geometry":
        """
        Rotate geometry around X-axis by angle_rad (in radians).
        
        Args:
            angle_rad: Rotation angle in radians
            rotation_point: 3D point [x, y, z] to rotate around. If None, rotates around origin.
        
        Returns self to allow chaining.
        """
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        rot_matrix = np.array([
            [1.0, 0.0,        0.0,       0.0],
            [0.0, cos_theta, -sin_theta, 0.0],
            [0.0, sin_theta,  cos_theta, 0.0],
            [0.0, 0.0,        0.0,       1.0]
        ])
        
        if rotation_point is not None:
            # Translate to rotation point, rotate, then translate back
            translate_to_origin = np.array([
                [1.0, 0.0, 0.0, -rotation_point[0]],
                [0.0, 1.0, 0.0, -rotation_point[1]],
                [0.0, 0.0, 1.0, -rotation_point[2]],
                [0.0, 0.0, 0.0, 1.0]
            ])
            
            translate_back = np.array([
                [1.0, 0.0, 0.0, rotation_point[0]],
                [0.0, 1.0, 0.0, rotation_point[1]],
                [0.0, 0.0, 1.0, rotation_point[2]],
                [0.0, 0.0, 0.0, 1.0]
            ])
            
            # Combine transformations: translate back * rotate * translate to origin
            combined_matrix = translate_back @ rot_matrix @ translate_to_origin
            self.transform_geometry(combined_matrix)
        else:
            self.transform_geometry(rot_matrix)
        
        return self

    def rotate_y(self, angle_rad: float, rotation_point: Optional[np.ndarray] = None) -> "Geometry":
        """
        Rotate geometry around Y-axis by angle_rad (in radians).
        
        Args:
            angle_rad: Rotation angle in radians
            rotation_point: 3D point [x, y, z] to rotate around. If None, rotates around origin.
        
        Returns self to allow chaining.
        """
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        rot_matrix = np.array([
            [cos_theta, 0.0, sin_theta, 0.0],
            [0.0,       1.0, 0.0,       0.0],
            [-sin_theta, 0.0, cos_theta, 0.0],
            [0.0,       0.0, 0.0,       1.0]
        ])
        
        if rotation_point is not None:
            # Translate to rotation point, rotate, then translate back
            translate_to_origin = np.array([
                [1.0, 0.0, 0.0, -rotation_point[0]],
                [0.0, 1.0, 0.0, -rotation_point[1]],
                [0.0, 0.0, 1.0, -rotation_point[2]],
                [0.0, 0.0, 0.0, 1.0]
            ])
            
            translate_back = np.array([
                [1.0, 0.0, 0.0, rotation_point[0]],
                [0.0, 1.0, 0.0, rotation_point[1]],
                [0.0, 0.0, 1.0, rotation_point[2]],
                [0.0, 0.0, 0.0, 1.0]
            ])
            
            # Combine transformations: translate back * rotate * translate to origin
            combined_matrix = translate_back @ rot_matrix @ translate_to_origin
            self.transform_geometry(combined_matrix)
        else:
            self.transform_geometry(rot_matrix)
        
        return self
    
    def get_centroid(self) -> Vector3D:
        """Get the centroid of the geometry."""
        mesh = self.mesh
        if mesh and "vertices" in mesh:
            vertices = np.array(mesh["vertices"])
            centroid = vertices.mean(axis=0)
            return Vector3D(*centroid)
        return Vector3D()
    
    def get_vertices(self) -> List[Tuple[float, float, float]]:
        """Get the vertices of the geometry."""
        mesh = self.mesh
        if mesh and "vertices" in mesh:
            return [tuple(v) for v in mesh["vertices"]]
        return []
    def get_faces(self) -> List[Tuple[int, ...]]:
        """Get the faces of the geometry."""
        mesh = self.mesh
        if mesh and "faces" in mesh:
            return [tuple(f) for f in mesh["faces"]]
        return []
    
    def get_height(self) -> float:
        """Get the height of the geometry."""
        mesh = self.mesh
        if mesh and "vertices" in mesh:
            vertices = np.array(mesh["vertices"])
            min_z = vertices[:, 2].min()
            max_z = vertices[:, 2].max()
            return max_z - min_z
        return 0.0

    def get_bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box as (min_point, max_point)."""
        mesh = self.mesh
        if mesh and "vertices" in mesh:
            vertices = np.array(mesh["vertices"])
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

    def open_cascade_intersects(self, other: 'Geometry', return_overlap_percent: bool = False) -> Union[bool, float]:
        """
        Check if OpenCascade shapes intersect.
        
        Args:
            other: Another Geometry object to check intersection with
            return_overlap_percent: If True, return the overlap percentage instead of boolean
            
        Returns:
            If return_overlap_percent is False: bool indicating intersection
            If return_overlap_percent is True: float representing overlap percentage (0.0 to 100.0)
        """
        shape1 = self.opencascade
        shape2 = other.opencascade
        
        if not shape1 or not shape2:
            # If either shape is not available, fallback to bounding box intersection
            return self.bbox_intersects(other, return_overlap_percent)
        
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
        
        common_shape = BRepAlgoAPI_Common(shape1, shape2)
        
        if common_shape.IsDone():
            if common_shape.Shape().IsNull():
                return 0.0 if return_overlap_percent else False
            else:
                if return_overlap_percent:
                    # Calculate overlap volume percentage
                    common_volume = Topology.Volume(common_shape.Shape())
                    volume1 = Topology.Volume(shape1)
                    
                    percentage = (common_volume / volume1) * 100.0 if volume1 > 0 else 0.0
                    return percentage
                else:
                    return True
        else:
            return 0.0 if return_overlap_percent else False
    
    
    def topologic_intersects(self, other: 'Geometry', return_overlap_percent: bool = False) -> Union[bool, float]:
        from topologicpy.Topology import Topology
        topology1 = self.topologic
        topology2 = other.topologic

        if not topology1 or not topology2:
            # If either topology is not available, fallback to bounding box intersection
            return self.bbox_intersects(other, return_overlap_percent)
        
        intersect = Topology.Intersect(topology1, topology2, tolerance=0.01)
        if not intersect:
            return 0.0 if return_overlap_percent else False
        else:
            if return_overlap_percent:
                intersect_volume = Topology.Volume(intersect)
                volume1 = Topology.Volume(topology1)

                percentage = (intersect_volume / volume1) * 100.0 if volume1 > 0 else 0.0
                return percentage
            else:
                return True
    
    
    def mesh_intersects(self, other: 'Geometry', return_overlap_percent: bool = False) -> Union[bool, float]:
        """Check if the actual meshes intersect using trimesh."""
        try:
            import trimesh
        except ImportError:
            print("trimesh not installed, falling back to bbox intersection")
            return self.bbox_intersects(other, return_overlap_percent)
        
        # Convert both geometries to trimesh objects
        mesh1 = self._to_trimesh()
        mesh2 = other._to_trimesh()
        
        # If either mesh conversion failed, fallback to bounding box intersection
        if mesh1 is None or mesh2 is None:
            return self.bbox_intersects(other, return_overlap_percent)
        
        else:
            try:
                # Check for intersection using trimesh
                intersection = mesh1.intersection(mesh2)
                if not intersection:
                    return 0.0 if return_overlap_percent else False
                else:
                    # Intersection found
                    if return_overlap_percent:
                        num_samples = 50_000
                        if mesh1.is_watertight:
                            points = mesh1.sample_volume(num_samples)
                        else:
                            # For non-watertight meshes, sample the surface and nearby volume
                            surface_points, _ = mesh1.sample(num_samples // 2)
                            
                            # Add some points slightly inside/outside the surface
                            normals = mesh1.face_normals[mesh1.nearest.on_surface(surface_points)[2]]
                            offset_distance = mesh1.scale / 100  # Small offset
                            
                            inside_points = surface_points - normals * offset_distance
                            outside_points = surface_points + normals * offset_distance
                            
                            points = np.vstack([surface_points, inside_points, outside_points])
                        
                        # Check which points are inside mesh2
                        inside_mask = mesh2.contains(points)
                        
                        # Calculate percentage
                        percentage = (np.sum(inside_mask) / len(points)) * 100
                        return percentage
                    else:
                        # If we just want to know if they intersect
                        return True
            except Exception:
                return self.bbox_intersects(other, return_overlap_percent)
            
    def average_vertex_distance(self, geom_b):
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        topology_a = self.topologic
        topology_b = geom_b.topologic

        verts_a = Topology.Vertices(topology_a)
        verts_b = Topology.Vertices(topology_b)

        vertex_distances = []
        for v_a in verts_a:
            for v_b in verts_b:
                vertex_distances.append(Vertex.Distance(v_a, v_b))

        return sum(vertex_distances) / len(vertex_distances)

    def is_coplanar(self, geom_b, tolerance=0.5, mantissa=6, angle_tolerance=5.0):
      """
      Check if two geometries are coplanar with better tolerance handling.
      
      Parameters:
      -----------
      geom_b : Geometry
          The second geometry to compare
      tolerance : float
          Distance tolerance for coplanarity (default 0.5)
      mantissa : int
          Decimal precision (default 6)
      angle_tolerance : float
          Maximum angle difference in degrees for normals (default 5.0)
      """
      from topologicpy.Topology import Topology
      from topologicpy.Face import Face
      from topologicpy.Vertex import Vertex
      import math

      topology_a = self.topologic
      topology_b = geom_b.topologic

      if not (Topology.IsInstance(topology_a, "Face") and Topology.IsInstance(topology_b, "Face")):
          raise ValueError("Can only run is_coplanar on geometry that can be represented as Topologic Faces")

      # Get normals
      normal_a = Face.Normal(topology_a, mantissa=mantissa)
      normal_b = Face.Normal(topology_b, mantissa=mantissa)

      # Normalize the normals (they should already be, but let's ensure)
      def normalize(v):
          mag = math.sqrt(sum(x**2 for x in v))
          return [x/mag for x in v] if mag > 0 else v

      normal_a = normalize(normal_a)
      normal_b = normalize(normal_b)

      # Check if normals are parallel (either same or opposite direction)
      dot_product = sum(a * b for a, b in zip(normal_a, normal_b))

      # Normals are parallel if dot product is close to 1 or -1
      if abs(abs(dot_product) - 1.0) > math.radians(angle_tolerance):
          return False  # Normals not parallel enough

      # Now check if the faces are on the same plane by testing point-to-plane distance
      # Get center points from both faces
      center_a = Face.VertexByParameters(topology_a, 0.5, 0.5)
      coords_a = [Vertex.X(center_a), Vertex.Y(center_a), Vertex.Z(center_a)]

      # Get plane equation for face B
      plane_b = Face.PlaneEquation(topology_b, mantissa=mantissa)

      # Calculate signed distance from point A to plane B
      # Distance = |ax + by + cz + d| / sqrt(a² + b² + c²)
      numerator = abs(
          plane_b['a'] * coords_a[0] +
          plane_b['b'] * coords_a[1] +
          plane_b['c'] * coords_a[2] +
          plane_b['d']
      )

      denominator = math.sqrt(
          plane_b['a']**2 +
          plane_b['b']**2 +
          plane_b['c']**2
      )

      if denominator < 1e-10:  # Degenerate plane
          return False

      distance = numerator / denominator

      # Return true if distance is within tolerance
      return distance <= tolerance
    
            

    def _to_trimesh(self) -> Optional['trimesh.Trimesh']:
        """Convert this geometry to a trimesh object."""
        try:
            import trimesh
        except ImportError:
            return None
        
        mesh = self.mesh
        if not mesh or "vertices" not in mesh or "faces" not in mesh:
            return None
        
        vertices = mesh["vertices"]
        faces = mesh["faces"]
        
        if not vertices or not faces:
            return None
        
        try:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            return mesh
        except Exception as e:
            print(f"Error creating trimesh: {e}")
            return None
    
    def compute_volume(self) -> float:
        """
        Compute the volume of a closed triangular mesh using the centroid-shifted divergence theorem.
        """
        mesh = self.mesh
        if not mesh or "vertices" not in mesh or "faces" not in mesh:
            return 0.0

        vertices = np.array(mesh["vertices"])
        faces = mesh["faces"]

        # Compute centroid of the mesh
        centroid = np.mean(vertices, axis=0)

        volume = 0.0
        for face in faces:
            if len(face) >= 3:
                v1 = vertices[face[0]] - centroid
                v2 = vertices[face[1]] - centroid
                v3 = vertices[face[2]] - centroid
                volume += np.dot(v1, np.cross(v2, v3)) / 6.0

        return abs(volume)

    def order_vertices_by_angle(vertices):
        """Order vertices by angle from centroid - works for convex and most star-shaped polygons."""
    
        # Convert to numpy array and extract x,y coordinates (ignore z)
        points = np.array([(float(v[0]), float(v[1])) for v in vertices])
        
        # Calculate centroid
        centroid = np.mean(points, axis=0)
        
        # Calculate angle from centroid to each point
        def angle_from_centroid(point):
            return math.atan2(point[1] - centroid[1], point[0] - centroid[0])
        
        # Sort points by angle
        points_with_angles = [(point, angle_from_centroid(point)) for point in points]
        points_with_angles.sort(key=lambda x: x[1])
        
        # Extract ordered points and convert back to original format
        ordered_points = [point for point, angle in points_with_angles]
        
        # Convert back to 3D with original z-coordinate
        z_coord = vertices[0][2]  # Use z from first vertex
        ordered_vertices = [(x, y, z_coord) for x, y in ordered_points]
        
        return ordered_vertices
    


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
    

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

class IssueType(Enum):
    INVALID_ORIENTATION = "invalid_orientation"
    SELF_INTERSECTION = "self_intersection"
    DEGENERATE = "degenerate"
    INVALID_WIRES = "invalid_wires"
    HIGH_TOLERANCE = "high_tolerance"
    POOR_CONTINUITY = "poor_continuity"
    GAPS = "gaps"
    UNKNOWN_ERROR = "unknown_error"

class IssueSeverity(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

@dataclass
class FaceIssue:
    issue_type: IssueType
    severity: IssueSeverity
    description: str
    suggested_fix: str
    details: Dict = None

@dataclass
class FaceValidationResult:
    face_id: int
    is_valid: bool
    issues: List[FaceIssue]
    properties: Dict
    
class FaceValidator:
    """Comprehensive face validation and analysis manager"""
    
    def __init__(self, config=None):
        """
        Initialize validator with configuration options
        
        Args:
            config: Dictionary with validation thresholds
        """
        self.config = config or self._default_config()
        self.results = {}
        
    def _default_config(self):
        """Default validation configuration"""
        return {
            'min_area_threshold': 1e-6,
            'max_tolerance': 1e-3,
            'min_continuity': 0,  # GeomAbs_C0
            'gap_tolerance': 1e-6,
            'enable_detailed_analysis': True,
            'auto_fix_attempts': True
        }
    
    def validate_face(self, face, face_id=None) -> FaceValidationResult:
        """
        Run comprehensive validation on a single face
        
        Args:
            face: TopoDS_Face to validate
            face_id: Optional identifier for the face
            
        Returns:
            FaceValidationResult with all issues found
        """
        if face_id is None:
            face_id = id(face)
            
        issues = []
        properties = {}
        
        # Run all validation checks
        issues.extend(self._check_basic_validity(face, properties))
        issues.extend(self._check_geometry_issues(face, properties))
        issues.extend(self._check_topology_issues(face, properties))
        issues.extend(self._check_tolerance_issues(face, properties))
        issues.extend(self._check_surface_quality(face, properties))
        
        # Determine overall validity
        critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        is_valid = len(critical_issues) == 0
        
        result = FaceValidationResult(
            face_id=face_id,
            is_valid=is_valid,
            issues=issues,
            properties=properties
        )
        
        self.results[face_id] = result
        return result
    
    def _check_basic_validity(self, face, properties) -> List[FaceIssue]:
        """Check basic face validity"""
        issues = []
        
        try:
            from OCC.Core.BRepCheck import BRepCheck_Face
            from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
            
            checker = BRepCheck_Face(face)
            properties['basic_validity'] = checker.IsValid()
            
            if not checker.IsValid():
                issues.append(FaceIssue(
                    issue_type=IssueType.INVALID_ORIENTATION,
                    severity=IssueSeverity.CRITICAL,
                    description="Face has invalid basic topology",
                    suggested_fix="Use ShapeFix_Face to repair topology"
                ))
            
            # Check orientation
            orientation = face.Orientation()
            properties['orientation'] = "forward" if orientation == TopAbs_FORWARD else "reversed"
            
        except Exception as e:
            issues.append(FaceIssue(
                issue_type=IssueType.UNKNOWN_ERROR,
                severity=IssueSeverity.WARNING,
                description=f"Error checking basic validity: {e}",
                suggested_fix="Manual inspection required"
            ))
            
        return issues
    
    def _check_geometry_issues(self, face, properties) -> List[FaceIssue]:
        """Check for geometric issues like self-intersection and degeneracy"""
        issues = []
        
        try:
            from OCC.Core.BRepCheck import BRepCheck_Analyzer
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
            
            # Self-intersection check
            analyzer = BRepCheck_Analyzer(face)
            properties['has_geometric_issues'] = not analyzer.IsValid()
            
            if not analyzer.IsValid():
                issues.append(FaceIssue(
                    issue_type=IssueType.SELF_INTERSECTION,
                    severity=IssueSeverity.CRITICAL,
                    description="Face may have self-intersections or other geometric issues",
                    suggested_fix="Use BRepAlgoAPI_Defeaturing or rebuild face"
                ))
            
            # Degeneracy check
            props = GProp_GProps()
            brepgprop_SurfaceProperties(face, props)
            area = props.Mass()
            properties['area'] = area
            
            if area < self.config['min_area_threshold']:
                issues.append(FaceIssue(
                    issue_type=IssueType.DEGENERATE,
                    severity=IssueSeverity.CRITICAL,
                    description=f"Degenerate face with area {area} < {self.config['min_area_threshold']}",
                    suggested_fix="Remove face or merge with adjacent faces"
                ))
            
        except Exception as e:
            issues.append(FaceIssue(
                issue_type=IssueType.UNKNOWN_ERROR,
                severity=IssueSeverity.WARNING,
                description=f"Error checking geometry: {e}",
                suggested_fix="Manual inspection required"
            ))
            
        return issues
    
    def _check_topology_issues(self, face, properties) -> List[FaceIssue]:
        """Check wire structure and topology"""
        issues = []
        
        try:
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
            from OCC.Core.BRepCheck import BRepCheck_Wire
            from OCC.Core.BRep import BRep_Tool
            
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
            wire_count = 0
            invalid_wires = 0
            
            while wire_explorer.More():
                wire = wire_explorer.Current()
                wire_count += 1
                
                # Check wire validity
                wire_checker = BRepCheck_Wire(wire)
                if not wire_checker.IsValid():
                    invalid_wires += 1
                
                # Check if wire is closed
                if not BRep_Tool.IsClosed(wire):
                    invalid_wires += 1
                
                wire_explorer.Next()
            
            properties['wire_count'] = wire_count
            properties['invalid_wires'] = invalid_wires
            
            if invalid_wires > 0:
                issues.append(FaceIssue(
                    issue_type=IssueType.INVALID_WIRES,
                    severity=IssueSeverity.CRITICAL,
                    description=f"{invalid_wires} invalid or open wires out of {wire_count}",
                    suggested_fix="Use ShapeFix_Wire to repair wire topology"
                ))
                
        except Exception as e:
            issues.append(FaceIssue(
                issue_type=IssueType.UNKNOWN_ERROR,
                severity=IssueSeverity.WARNING,
                description=f"Error checking topology: {e}",
                suggested_fix="Manual inspection required"
            ))
            
        return issues
    
    def _check_tolerance_issues(self, face, properties) -> List[FaceIssue]:
        """Check tolerance values"""
        issues = []
        
        try:
            from OCC.Core.BRep import BRep_Tool
            
            tolerance = BRep_Tool.Tolerance(face)
            properties['tolerance'] = tolerance
            
            if tolerance > self.config['max_tolerance']:
                issues.append(FaceIssue(
                    issue_type=IssueType.HIGH_TOLERANCE,
                    severity=IssueSeverity.WARNING,
                    description=f"High tolerance {tolerance} > {self.config['max_tolerance']}",
                    suggested_fix="Consider rebuilding face with tighter tolerance"
                ))
                
        except Exception as e:
            issues.append(FaceIssue(
                issue_type=IssueType.UNKNOWN_ERROR,
                severity=IssueSeverity.WARNING,
                description=f"Error checking tolerance: {e}",
                suggested_fix="Manual inspection required"
            ))
            
        return issues
    
    def _check_surface_quality(self, face, properties) -> List[FaceIssue]:
        """Check surface continuity and quality"""
        issues = []
        
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C1, GeomAbs_C2
            
            adaptor = BRepAdaptor_Surface(face)
            
            u_continuity = adaptor.UContinuity()
            v_continuity = adaptor.VContinuity()
            
            properties['u_continuity'] = int(u_continuity)
            properties['v_continuity'] = int(v_continuity)
            
            # Compare with configured minimum continuity
            min_continuity = min(int(u_continuity), int(v_continuity))
            min_required = self.config.get('min_continuity', 0)
            
            if min_continuity < min_required:
                issues.append(FaceIssue(
                    issue_type=IssueType.POOR_CONTINUITY,
                    severity=IssueSeverity.INFO,
                    description=f"Low surface continuity: {min_continuity} < {min_required}",
                    suggested_fix="Consider surface smoothing if needed for your application"
                ))
                
        except Exception as e:
            issues.append(FaceIssue(
                issue_type=IssueType.UNKNOWN_ERROR,
                severity=IssueSeverity.WARNING,
                description=f"Error checking surface quality: {e}",
                suggested_fix="Manual inspection required"
            ))
            
        return issues
    
    def validate_solid(self, solid) -> Dict[int, FaceValidationResult]:
        """Validate all faces in a solid"""
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        
        results = {}
        face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
        face_count = 0
        
        while face_explorer.More():
            face = face_explorer.Current()
            face_count += 1
            
            result = self.validate_face(face, face_count)
            results[face_count] = result
            
            face_explorer.Next()
        
        return results
    
    def get_summary_report(self) -> str:
        """Generate a summary report of all validations"""
        if not self.results:
            return "No faces validated yet."
        
        total_faces = len(self.results)
        valid_faces = sum(1 for r in self.results.values() if r.is_valid)
        invalid_faces = total_faces - valid_faces
        
        # Count issues by type
        issue_counts = {}
        for result in self.results.values():
            for issue in result.issues:
                issue_type = issue.issue_type.value
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        report = f"""
Face Validation Summary
======================
Total faces validated: {total_faces}
Valid faces: {valid_faces}
Invalid faces: {invalid_faces}
Success rate: {(valid_faces/total_faces)*100:.1f}%

Issue Breakdown:
"""
        for issue_type, count in sorted(issue_counts.items()):
            report += f"  {issue_type.replace('_', ' ').title()}: {count}\n"
        
        return report
    
    def get_invalid_faces(self) -> List[Tuple[int, FaceValidationResult]]:
        """Get list of faces that failed validation"""
        return [(face_id, result) for face_id, result in self.results.items() 
                if not result.is_valid]
    
    def get_faces_with_issue(self, issue_type: IssueType) -> List[Tuple[int, FaceValidationResult]]:
        """Get faces that have a specific type of issue"""
        matching_faces = []
        for face_id, result in self.results.items():
            if any(issue.issue_type == issue_type for issue in result.issues):
                matching_faces.append((face_id, result))
        return matching_faces
    
    def suggest_fixes(self, face_id: int) -> List[str]:
        """Get suggested fixes for a specific face"""
        if face_id not in self.results:
            return ["Face not validated yet"]
        
        result = self.results[face_id]
        return [issue.suggested_fix for issue in result.issues]

# Usage example
def example_usage():
    """Example of how to use the FaceValidator"""
    
    # Create validator with custom config
    config = {
        'min_area_threshold': 1e-5,
        'max_tolerance': 5e-4,
        'enable_detailed_analysis': True
    }
    validator = FaceValidator(config)
    
    # Validate a single face
    # result = validator.validate_face(some_face)
    # print(f"Face valid: {result.is_valid}")
    # print(f"Issues found: {len(result.issues)}")
    
    # Validate all faces in a solid
    # results = validator.validate_solid(some_solid)
    
    # Get summary report
    # print(validator.get_summary_report())
    
    # Get faces with specific issues
    # degenerate_faces = validator.get_faces_with_issue(IssueType.DEGENERATE)
    
    # Get suggested fixes
    # fixes = validator.suggest_fixes(face_id=1)
    
    pass

# Usage example
def example_usage():
    """Example of how to use the FaceValidator"""
    
    # Create validator with custom config
    config = {
        'min_area_threshold': 1e-5,
        'max_tolerance': 5e-4,
        'enable_detailed_analysis': True
    }
    validator = FaceValidator(config)
    
    # Validate a single face
    # result = validator.validate_face(some_face)
    # print(f"Face valid: {result.is_valid}")
    # print(f"Issues found: {len(result.issues)}")
    
    # Validate all faces in a solid
    # results = validator.validate_solid(some_solid)
    
    # Get summary report
    # print(validator.get_summary_report())
    
    # Get faces with specific issues
    # degenerate_faces = validator.get_faces_with_issue(IssueType.DEGENERATE)
    
    # Get suggested fixes
    # fixes = validator.suggest_fixes(face_id=1)
    
    pass