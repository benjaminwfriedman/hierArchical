from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List, Union
from .geometry import Geometry
import numpy as np
from .helpers import generate_id, normalize_ifc_enum
from copy import deepcopy
from pathlib import Path
from .units import UnitSystem, UnitSystems, UNIT_TO_METER
from .relationships import EmbeddedIn, Embeds, PassesThrough, HasPassingThrough, Relationship, AdjacentTo
from uuid import uuid4




@dataclass(slots=True, repr=False)
class BaseItem:
    """Shared base class for all entities"""

    # A human-readable name for the item
    name: str
    
    # The geometry object describing the item's shape
    geometry: Geometry

    # A given name
    name: str

    # A defined type (e.g., 'wall', 'pipe', 'deck')
    type: str

    # An optional collection of sub-item IDs (hierarchical structure)
    sub_items: Tuple[str, ...] = field(default_factory=tuple)

    # add relationships to the item
    relationships: List[Relationship] = field(default_factory=list)
    # A set of relationships the item has — well suited for generating graphs
    # relationships: Tuple[Relationship, ...] = field(default_factory=tuple)

    # Arbitrary measurable or inferred properties (e.g., area, ghg_emissions)
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Ontological tags and values (e.g., {'structural': True, 'zone': 'care'})
    ontologies: Dict[str, Any] = field(default_factory=dict)

    materials: Dict[str, Dict[str, float]] = field(default_factory=dict)

    color: Optional[Tuple[float, float, float]] = None

    unit_system: UnitSystem = UnitSystem.METER

    # A unique UUID
    id: str = field(default_factory=lambda: str(uuid4()))

    def __repr__(self):
        """Lightweight representation for debugger"""
        return f"{self.__class__.__name__}(id={self.id}, name='{self.name}', type='{self.type}')"
    
    def __str__(self):
        return self.__repr__()
    

    @staticmethod
    def combine_geometries(geometries: Tuple[Geometry, ...]) -> Geometry:
        """
        Combines multiple Geometry instances into a single Geometry.
        """
        combined = Geometry()

        all_vertices = []
        all_faces = []
        vertex_offset = 0

        for geom in geometries:
            if not geom.mesh_data:
                continue

            verts = geom.mesh_data.get("vertices", [])
            faces = geom.mesh_data.get("faces", [])

            all_vertices.extend(verts)

            # Adjust face indices
            adjusted_faces = [
                tuple(idx + vertex_offset for idx in face) for face in faces
            ]
            all_faces.extend(adjusted_faces)

            vertex_offset += len(verts)

        combined.mesh_data = {
            "vertices": all_vertices,
            "faces": all_faces,
        }

        combined._generate_brep_from_mesh()

        combined.sub_geometries = deepcopy(geometries)
        return combined
    
    # ----- Geometry Methods -----
    def get_height(self) -> float:
        """
        Get the height of the item geometry.
        """
        return self.geometry.get_height()

    # ----- Transformation Methods -----

    def move(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "BaseItem":
        """
        Moves the item geometry by applying a single transformation matrix that
        shifts the object by (dx, dy, dz).
        """
        translation = np.eye(4)
        translation[:3, 3] = [dx, dy, dz]
        self.geometry.transform_geometry(translation)

        # Apply the translation to all sub-items
        for sub_item in self.sub_items:
            sub_item.move(dx, dy, dz)
        return self

    def right(self, dx: float = 1.0) -> "BaseItem":
        self.geometry.right(dx)

        # Apply the rightward movement to all sub-items
        for sub_item in self.sub_items:
            sub_item.right(dx)
        return self

    def forward(self, dy: float = 1.0) -> "BaseItem":
        self.geometry.forward(dy)

        # Apply the forward movement to all sub-items
        for sub_item in self.sub_items:
            sub_item.forward(dy)
        return self

    def up(self, dz: float = 1.0) -> "BaseItem":
        self.geometry.up(dz)
        # Apply the upward movement to all sub-items
        for sub_item in self.sub_items:
            sub_item.up(dz)
        return self

    def back(self, dy: float = 1.0) -> "BaseItem":
        self.geometry.back(dy)
        # Apply the backward movement to all sub-items
        for sub_item in self.sub_items:
            sub_item.back(dy)
        return self

    def left(self, dx: float = 1.0) -> "BaseItem":
        self.geometry.left(dx)

        # Apply the leftward movement to all sub-items
        for sub_item in self.sub_items:
            sub_item.left(dx)
        return self

    def down(self, dz: float = 1.0) -> "BaseItem":
        self.geometry.down(dz)

        # Apply the downward movement to all sub-items
        for sub_item in self.sub_items:
            sub_item.down(dz)
        return self

    def rotate_z(self, angle_rad: float) -> "BaseItem":
        """
        Rotate the item around the Z-axis. Returns self.
        """
        self.geometry.rotate_z(angle_rad)
        # Apply the rotation to all sub-items
        for sub_item in self.sub_items:
            sub_item.rotate_z(angle_rad)
        return self 


    @classmethod
    def geometry_from_sub_items(cls, sub_items: Tuple['BaseItem', ...]) -> Geometry:
        """
        Create combined geometry from the geometries of sub-items.
        """
        geometries = tuple(item.geometry for item in sub_items)
        return cls.combine_geometries(geometries)

    def copy(self, dx=0.0, dy=0.0, dz=0.0):

        geom_copy = deepcopy(self.geometry)
        geom_copy.right(dx).forward(dy).up(dz)

        return type(self)(
            name=self.name + " (copy)",
            type=self.type,
            geometry=geom_copy,
            sub_items=deepcopy(self.sub_items),
            attributes=deepcopy(self.attributes),
            ontologies=deepcopy(self.ontologies),
            materials=deepcopy(self.materials),
            **({"material": self.material} if hasattr(self, "material") else {})
        )
    
    def intersects_with(self, other: 'BaseItem', return_overlap_percent: bool = False) -> Union[bool, float]:
        """
        Check if this item's geometry intersects with another item's geometry.
        
        Args:
            other: Another BaseItem to check intersection with
            return_overlap_percent: If True, return the overlap percentage instead of boolean
            
        Returns:
            If return_overlap_percent is False: bool indicating intersection
            If return_overlap_percent is True: float representing overlap percentage (0.0 to 100.0)
        """
        return self.geometry.mesh_intersects(other.geometry, return_overlap_percent=return_overlap_percent)
    
    def is_adjacent_to(self, other: 'BaseItem', tolerance: float = 0.1) -> bool:
        """
        Check if this item is adjacent (touching or very close) to another item.
        
        Args:
            other: The other BaseItem to check against
            tolerance: Maximum distance to consider as "adjacent" (in model units)
        
        Returns:
            True if items are adjacent, False otherwise
        """
        # First check if they intersect (touching)
        if self.intersects_with(other):
            return True
        
        # Then check if they're within tolerance distance
        distance = self.geometry.distance_to(other.geometry)
        return distance <= tolerance
    
    def is_next_to_or_intersecting(self, other: 'BaseItem', tolerance: float = 0.1) -> bool:
        """
        Combined check for intersection or adjacency.
        
        This is a convenience method that combines intersects_with and is_adjacent_to.
        """
        return self.is_adjacent_to(other, tolerance)
    
    def find_adjacent_items(self, items: List['BaseItem'], tolerance: float = 0.1) -> List['BaseItem']:
        """
        Find all items from a list that are adjacent to this item.
        
        Args:
            items: List of BaseItems to check
            tolerance: Maximum distance to consider as "adjacent"
        
        Returns:
            List of adjacent items
        """
        return [item for item in items if item != self and self.is_adjacent_to(item, tolerance)]
    
    def find_intersecting_items(self, items: List['BaseItem'], threshold: float) -> List['BaseItem']:
        """
        Find all items from a list that intersect with this item above a threshold.
        
        Args:
            items: List of BaseItems to check
            threshold: Minimum overlap percentage to consider as intersecting (0.0 to 100.0)
        
        Returns:
            List of intersecting items that meet the threshold
        """
        intersecting_items = []

        for item in items:
            if item == self:
                continue
                
            # Get the overlap percentage using mesh_intersects
            overlap_percent = self.intersects_with(item, return_overlap_percent=True)
            
            # Add to list if it meets the threshold
            if overlap_percent > threshold:
                intersecting_items.append(item)

        return intersecting_items
    
    def add_embedded_in_relationship(self, other: 'BaseItem'):
        """
        Add an 'embedded_in' relationship to another item.
        
        Args:
            other: The other BaseItem to establish the relationship with
        """
        if not hasattr(self, "relationships"):
            self.relationships = []
        
        self.relationships.append(
            EmbeddedIn(
                source=self,
                target=other,
                attributes={
                    "overlap_percent": self.intersects_with(other, return_overlap_percent=True)
                }

            )
        )

        # apply the reciprocal relationship to the other item
        if not hasattr(other, "relationships"):
            other.relationships = []
        
        other.relationships.append(
            Embeds(
                source=other,
                target=self,
                attributes={
                    "overlap_percent": self.intersects_with(other, return_overlap_percent=True)
                }
            )
        )

        return 
    
    def add_adjacent_to_relationship(self, other: 'BaseItem'):
        """
        Add an 'adjacent_to' relationship to another item.
        
        Args:
            other: The other BaseItem to establish the relationship with
            tolerance: Maximum distance to consider as adjacent (default: 0.1)
        """
        if not hasattr(self, "relationships"):
            self.relationships = []

        # check that the relationship doesn't already exist
        for rel in self.relationships:
            if rel.type == "adjacent_to" and rel.target == other.id:
                return  # relationship already exists

                
        # Add relationship to self
        self.relationships.append(
            AdjacentTo(
                source=self,
                target=other,
                attributes={
                    "overlap_percent": self.intersects_with(other, return_overlap_percent=True)
                }
            )
        )
        
        # Apply the reciprocal relationship to the other item
        if not hasattr(other, "relationships"):
            other.relationships = []
        
        other.relationships.append(
            AdjacentTo(
                source=other,
                target=self,
                attributes={
                    "overlap_percent": other.intersects_with(self, return_overlap_percent=True)
                }
            )
        )
        
        return

    
    
    def convert_units(self, target_unit: UnitSystem, in_place: bool = True) -> Optional['BaseItem']:
        """
        Convert the item's geometry and attributes to a different unit system.
        
        Args:
            target_unit: The target unit system
            in_place: If True, modify this object. If False, return a new object.
            
        Returns:
            None if in_place=True, new BaseItem if in_place=False
        """
        if self.unit_system == target_unit:
            return None if in_place else self.copy()
        
        # Calculate conversion factor
        factor = self._get_conversion_factor(self.unit_system, target_unit)
        
        # Create a copy if not in-place
        item = self if in_place else self.copy()
        
        # Convert geometry
        item._convert_geometry(factor)
        
        # Convert relevant attributes
        item._convert_attributes(factor)
        
        # Convert sub-items recursively
        if item.sub_items:
            converted_sub_items = []
            for sub_item in item.sub_items:
                converted_sub = sub_item.convert_units(target_unit, in_place=False)
                converted_sub_items.append(converted_sub)
            item.sub_items = tuple(converted_sub_items)
        
        # Update unit system
        item.unit_system = target_unit
        
        return None if in_place else item
    
    def _get_conversion_factor(self, from_unit: UnitSystem, to_unit: UnitSystem) -> float:
        """Calculate conversion factor between two unit systems."""
        from_meters = UNIT_TO_METER[from_unit]
        to_meters = UNIT_TO_METER[to_unit]
        return from_meters / to_meters
    
    def _convert_geometry(self, factor: float):
        """Convert geometry by scaling factor."""
        if self.geometry:
            # Scale the geometry
            scale_matrix = np.eye(4)
            scale_matrix[0, 0] = factor
            scale_matrix[1, 1] = factor
            scale_matrix[2, 2] = factor
            self.geometry.transform_geometry(scale_matrix)
    
    def _convert_attributes(self, factor: float):
        """Convert unit-dependent attributes."""
        # Define which attributes need conversion and their dimension
        unit_conversions = {
            # Linear dimensions (multiply by factor)
            "length": 1,
            "width": 1,
            "height": 1,
            "thickness": 1,
            "diameter": 1,
            "radius": 1,
            "perimeter": 1,
            
            # Area dimensions (multiply by factor²)
            "area": 2,
            "surface_area": 2,
            "floor_area": 2,
            
            # Volume dimensions (multiply by factor³)
            "volume": 3,
            
            # Other specialized conversions
            "flow_rate": 3,  # volume/time
            "linear_density": -1,  # mass/length
        }
        
        for attr_name, dimension in unit_conversions.items():
            if attr_name in self.attributes:
                self.attributes[attr_name] *= (factor ** dimension)
        
        # Convert material volumes
        if self.materials:
            for material, data in self.materials.items():
                if "volume" in data:
                    data["volume"] *= (factor ** 3)

    def convert_to_metric(self, target_unit: UnitSystem = UnitSystem.METER, in_place: bool = True) -> Optional['BaseItem']:
        """
        Convenience method to convert to a metric unit system.
        
        Args:
            target_unit: Target metric unit (default: meters)
            in_place: If True, modify this object. If False, return a new object.
        """
        if target_unit not in UnitSystems.METRIC:
            raise ValueError(f"{target_unit} is not a metric unit")
        return self.convert_units(target_unit, in_place)
    
    def convert_to_imperial(self, target_unit: UnitSystem = UnitSystem.FOOT, in_place: bool = True) -> Optional['BaseItem']:
        """
        Convenience method to convert to an imperial unit system.
        
        Args:
            target_unit: Target imperial unit (default: feet)
            in_place: If True, modify this object. If False, return a new object.
        """
        if target_unit not in UnitSystems.IMPERIAL:
            raise ValueError(f"{target_unit} is not an imperial unit")
        return self.convert_units(target_unit, in_place)
    
    def get_dimension_in_units(self, dimension: str, unit: UnitSystem) -> float:
        """
        Get a specific dimension converted to specified units without changing the object.
        
        Args:
            dimension: The dimension name (e.g., 'length', 'area', 'volume')
            unit: The unit system to convert to
            
        Returns:
            The dimension value in the specified units
        """
        if dimension not in self.attributes:
            raise ValueError(f"Dimension '{dimension}' not found in attributes")
        
        value = self.attributes[dimension]
        
        if self.unit_system == unit:
            return value
        
        # Get conversion factor
        factor = self._get_conversion_factor(self.unit_system, unit)
        
        # Determine dimension type
        dimension_types = {
            "length": 1, "width": 1, "height": 1, "thickness": 1,
            "area": 2, "surface_area": 2,
            "volume": 3
        }
        
        dim_power = dimension_types.get(dimension, 1)
        return value * (factor ** dim_power)
    
    def ensure_unit_system(self, required_unit: UnitSystem):
        """
        Ensure the item is in the required unit system, converting if necessary.
        
        Args:
            required_unit: The required unit system
        """
        if self.unit_system != required_unit:
            self.convert_units(required_unit, in_place=True)
    
    def _format_dimension(self, value: float, dimension_type: str = "length") -> str:
        """Format a dimension value with appropriate units."""
        unit_str = self.unit_system.value
        
        # Format based on unit system and value magnitude
        if self.unit_system in UnitSystems.METRIC:
            if abs(value) < 0.01 and self.unit_system == UnitSystem.METER:
                # Convert to mm for small values
                value *= 1000
                unit_str = "mm"
            elif abs(value) > 1000 and self.unit_system == UnitSystem.METER:
                # Convert to km for large values
                value /= 1000
                unit_str = "km"
        
        if dimension_type == "area":
            unit_str += "²"
        elif dimension_type == "volume":
            unit_str += "³"
        
        return f"{value:.3f} {unit_str}"
    
    def get_dimensions_summary(self) -> Dict[str, str]:
        """Get a formatted summary of all dimensional attributes."""
        summary = {
            "unit_system": self.unit_system.value
        }
        
        dimension_types = {
            "length": "length", "width": "length", "height": "length",
            "thickness": "length", "diameter": "length", "radius": "length",
            "area": "area", "surface_area": "area", "floor_area": "area",
            "volume": "volume"
        }
        
        for attr, dim_type in dimension_types.items():
            if attr in self.attributes:
                summary[attr] = self._format_dimension(self.attributes[attr], dim_type)
        
        return summary
    
    def get_centroid(self):
        """
        Get the centroid of the geometry.
        
        Returns:
            A dictionary with x, y, z coordinates of the centroid
        """
        return self.geometry.get_centroid()


@dataclass(slots=True)
class Element(BaseItem):
    material: str = ""

    def __post_init__(self):
        
        vol = self.geometry.compute_volume()
        self.materials = {
            self.material: {
                "volume": vol,
                "percent": 1.0  # It's the only material
            }
        }


@dataclass(slots=True)
class Component(BaseItem):
    @classmethod
    def from_elements(cls, elements: Tuple[Element, ...], name: str, type: str = "component", **kwargs) -> "Component":
        geometry = BaseItem.geometry_from_sub_items(elements)

        material_volumes: Dict[str, float] = {}
        for e in elements:
            for mat, data in e.materials.items():
                material_volumes[mat] = material_volumes.get(mat, 0.0) + data.get("volume", 0.0)

        total_volume = sum(material_volumes.values())
        materials = {
            mat: {
                "volume": vol,
                "percent": vol / total_volume if total_volume > 0 else 0.0
            }
            for mat, vol in material_volumes.items()
        }

        return cls(
            name=name,
            type=type,
            sub_items=elements,
            geometry=geometry,
            materials=materials,
            **kwargs
        )
    
        
@dataclass(slots=True)
class Object(BaseItem):
    @classmethod
    def from_components(
        cls,
        components: Tuple[Component, ...],
        name: str,
        type: str = "object",
        **kwargs
    ) -> "Object":
        geometry = BaseItem.geometry_from_sub_items(components)

        # Aggregate materials from sub-components
        material_volumes: Dict[str, float] = {}

        for comp in components:
            for mat, data in comp.materials.items():
                material_volumes[mat] = material_volumes.get(mat, 0.0) + data.get("volume", 0.0)

        total_volume = sum(material_volumes.values())
        materials = {
            mat: {
                "volume": vol,
                "percent": vol / total_volume if total_volume > 0 else 0.0
            }
            for mat, vol in material_volumes.items()
        }

        return cls(
            name=name,
            type=type,
            sub_items=components,
            geometry=geometry,
            materials=materials,
            **kwargs
        )
    
    
    ##TODO: Add functionality to load IFC files in a way that determines the sub components as well.
    ## Currently it only loads the geometry of the object itself as one object.
    @classmethod
    def from_ifc(cls, ifc_path: Union[str, Path], object_type: str = "IfcDoor") -> "Object":
        """
        Load and convert a single IFC entity into an Object instance.

        Args:
            ifc_path: Path to the IFC file.
            object_type: IFC entity type to extract, e.g., "IfcDoor".

        Returns:
            An Object instance.
        """
        import ifcopenshell
        import ifcopenshell.geom

        ifc_file = ifcopenshell.open(str(ifc_path))
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)

        entity = ifc_file.by_type(object_type)[0]  # Assume only one target object
        name = entity.Name or object_type
        geometry = Geometry()

        try:
            shape = ifcopenshell.geom.create_shape(settings, entity)
            verts = list(zip(*[iter(shape.geometry.verts)] * 3))
            faces = list(zip(*[iter(shape.geometry.faces)] * 3))
            geometry.mesh_data = {
                "vertices": verts,
                "faces": faces
            }
            geometry._generate_brep_from_mesh()
        except Exception as e:
            print(f"[IFC] Geometry extraction failed for {name}: {e}")

        sub_items = []
        for rel in getattr(entity, "IsDecomposedBy", []):
            for part in getattr(rel, "RelatedObjects", []):
                sub_geom = Geometry()
                sub_items.append(
                    Element(
                        name=part.Name or "Sub Part",
                        type=part.is_a(),
                        geometry=sub_geom,
                        material="unknown"
                    )
                )

        return cls(
            name=name,
            type=object_type.lower(),
            geometry=geometry,
            sub_items=tuple(sub_items),
            materials={}
        )

@dataclass(slots=True)
class Wall(Object):

    boundary_id: Optional[str] = None


    @classmethod
    def from_components(
        cls,
        components: Tuple[Component, ...],
        name: str,
        **kwargs
    ) -> "Wall":
        return super(Wall, cls).from_components(
            components=components,
            name=name,
            type="wall",
            **kwargs
        )
    
    def vertices(self) -> Optional['Geometry']:
        """
        Get a Geometry object that represents the center plane face of the wall.
        
        Returns:
            Geometry object containing the center plane as a mesh, or None if no center plane found.
        """
        center_plane = self.get_center_plane()
        if not center_plane or not center_plane.get('vertices'):
            return None
        
        plane_vertices = center_plane['vertices']
        
        if len(plane_vertices) < 3:
            return None
        
        # Create triangulated faces from the planar vertices
        faces = []
        
        if len(plane_vertices) == 3:
            # Already a triangle
            faces = [(0, 1, 2)]
        elif len(plane_vertices) == 4:
            # Quadrilateral - create two triangles
            faces = [(0, 1, 2), (0, 2, 3)]
        else:
            # For polygons with more than 4 vertices, triangulate from the first vertex
            # This works for convex polygons
            for i in range(1, len(plane_vertices) - 1):
                faces.append((0, i, i + 1))
        
        # Create the geometry object
        # Note: Adjust import paths based on your project structure
        try:
            from .geometry import Geometry, Vector3D
        except ImportError:
            # Fallback imports
            from geometry import Geometry, Vector3D
        
        geometry = Geometry()
        geometry.mesh_data = {
            "vertices": plane_vertices,
            "faces": faces
        }
        
        # Generate B-rep data from the mesh
        geometry._generate_brep_from_mesh()
        
        # Set the origin to the center point of the plane
        center_point = center_plane['point']
        geometry.origin = Vector3D(center_point[0], center_point[1], center_point[2])
        
        return geometry
    
    def get_center_plane(self) -> Optional[Dict]:
        """
        Find the center plane of the wall that runs through the middle of its thickness.
        Uses PCA to find the true orientation for non-axis aligned walls.
        
        Returns:
            Dictionary containing 'normal', 'point', 'thickness_direction', and 'vertices' that define the center plane.
        """
        geometry = self.geometry
        if not geometry.mesh_data:
            return None
        
        vertices = geometry.mesh_data.get("vertices", [])
        if not vertices:
            return None
        
        vertices = np.array(vertices)
        
        if len(vertices) < 4:
            return None
        
        # Step 1: Use PCA to find the principal directions of the wall
        # Center the vertices
        centroid = np.mean(vertices, axis=0)
        centered_vertices = vertices - centroid
        
        # Compute covariance matrix and eigenvalues/eigenvectors
        cov_matrix = np.cov(centered_vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (largest to smallest)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # The eigenvector with the smallest eigenvalue is the thickness direction (normal to wall)
        thickness_direction = eigenvectors[:, 2]  # Smallest eigenvalue
        wall_length_direction = eigenvectors[:, 0]  # Largest eigenvalue
        wall_height_direction = eigenvectors[:, 1]  # Middle eigenvalue
        
        # Ensure consistent orientation (normal should point in consistent direction)
        if thickness_direction[2] < 0:  # Prefer normal pointing up if possible
            thickness_direction = -thickness_direction
        
        # Step 2: Find the center plane by projecting vertices onto the thickness direction
        # and finding the center along that direction
        projections = np.dot(centered_vertices, thickness_direction)
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        center_proj = (min_proj + max_proj) / 2
        
        # Step 3: Find the center point on the plane
        center_point = centroid + center_proj * thickness_direction
        
        # Step 4: Generate vertices that define the center plane
        # Project all original vertices onto the center plane
        plane_vertices = []
        
        for vertex in vertices:
            # Vector from center point to vertex
            to_vertex = vertex - center_point
            
            # Remove the component along the thickness direction (project onto plane)
            projection_onto_normal = np.dot(to_vertex, thickness_direction)
            vertex_on_plane = vertex - projection_onto_normal * thickness_direction
            
            plane_vertices.append(vertex_on_plane)
        
        plane_vertices = np.array(plane_vertices)
        
        # Step 5: Find the boundary vertices of the center plane
        # We'll find the convex hull of the projected vertices
        try:
            from scipy.spatial import ConvexHull
            
            # Project plane vertices to 2D for convex hull calculation
            # Use the wall length and height directions as the 2D basis
            coords_2d = np.column_stack([
                np.dot(plane_vertices - center_point, wall_length_direction),
                np.dot(plane_vertices - center_point, wall_height_direction)
            ])
            
            # Find convex hull
            hull = ConvexHull(coords_2d)
            boundary_vertices = plane_vertices[hull.vertices]
            
        except ImportError:
            # Fallback: use simplified boundary detection
            # Find extreme points in each direction
            projections_length = np.dot(plane_vertices - center_point, wall_length_direction)
            projections_height = np.dot(plane_vertices - center_point, wall_height_direction)
            
            # Find indices of extreme points
            extreme_indices = [
                np.argmin(projections_length),  # Min length
                np.argmax(projections_length),  # Max length
                np.argmin(projections_height),  # Min height
                np.argmax(projections_height),  # Max height
            ]
            
            # Remove duplicates and get unique boundary vertices
            extreme_indices = list(set(extreme_indices))
            boundary_vertices = plane_vertices[extreme_indices]
        
        # Step 6: Sort boundary vertices in a consistent order (counterclockwise)
        if len(boundary_vertices) > 2:
            # Calculate angles from center point to sort vertices
            vectors_to_boundary = boundary_vertices - center_point
            angles = []
            
            for vec in vectors_to_boundary:
                length_component = np.dot(vec, wall_length_direction)
                height_component = np.dot(vec, wall_height_direction)
                angle = np.arctan2(height_component, length_component)
                angles.append(angle)
            
            # Sort by angle
            sorted_indices = np.argsort(angles)
            boundary_vertices = boundary_vertices[sorted_indices]
        
        return {
            'normal': tuple(thickness_direction),
            'point': tuple(center_point),
            'thickness_direction': tuple(thickness_direction),
            'length_direction': tuple(wall_length_direction),
            'height_direction': tuple(wall_height_direction),
            'vertices': [tuple(v) for v in boundary_vertices],
            'thickness': float(max_proj - min_proj),
        }
    
    def get_centerplane_geometry(self) -> Optional[Geometry]:
        """
        Get the center plane geometry of the wall.
        
        Returns:
            Geometry object representing the center plane, or None if not found.
        """
        center_plane = self.get_center_plane()
        if not center_plane or not center_plane.get('vertices'):
            return None
        
        vertices = center_plane['vertices']
        
        # Create triangulated faces from the planar vertices
        faces = []
        
        if len(vertices) == 3:
            # Already a triangle
            faces = [(0, 1, 2)]
        elif len(vertices) == 4:
            # Quadrilateral - create two triangles
            faces = [(0, 1, 2), (0, 2, 3)]
        else:
            # For polygons with more than 4 vertices, triangulate from the first vertex
            for i in range(1, len(vertices) - 1):
                faces.append((0, i, i + 1))
        
        geometry = Geometry()
        geometry.mesh_data = {
            "vertices": vertices,
            "faces": faces,
        }
        
        # Generate B-rep data from the mesh
        geometry._generate_brep_from_mesh()
        
        
        return geometry


    def get_centerplane_top_edge(self) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """
        Find the top edge of the wall's center plane (highest Z coordinates).
        
        Returns:
            Dictionary with 'start_point', 'end_point', and 'edge_type'.
        """
        center_plane = self.get_center_plane()
        if not center_plane or not center_plane.get('vertices'):
            return None
        
        vertices = np.array(center_plane['vertices'])
        
        if len(vertices) < 2:
            return None
        
        # Find vertices at or near the maximum Z
        z_coords = vertices[:, 2]
        max_z = np.max(z_coords)
        z_tolerance = 0.1  # 10cm tolerance
        top_mask = z_coords >= (max_z - z_tolerance)
        top_vertices = vertices[top_mask]
        
        if len(top_vertices) < 2:
            return None
        
        # Use the wall's length direction to determine left/right
        length_direction = np.array(center_plane['length_direction'])
        center_point = np.array(center_plane['point'])
        
        # Project top vertices along the length direction to find extremes
        projections = np.dot(top_vertices - center_point, length_direction)
        leftmost_idx = np.argmin(projections)
        rightmost_idx = np.argmax(projections)
        
        return {
            'start_point': tuple(top_vertices[leftmost_idx]),
            'end_point': tuple(top_vertices[rightmost_idx]),
            'edge_type': 'top'
        }

    def get_centerplane_bottom_edge(self) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """
        Find the bottom edge of the wall's center plane (lowest Z coordinates).
        
        Returns:
            Dictionary with 'start_point', 'end_point', and 'edge_type'.
        """
        center_plane = self.get_center_plane()
        if not center_plane or not center_plane.get('vertices'):
            return None
        
        vertices = np.array(center_plane['vertices'])
        
        if len(vertices) < 2:
            return None
        
        # Find vertices at or near the minimum Z
        z_coords = vertices[:, 2]
        min_z = np.min(z_coords)
        z_tolerance = 0.1  # 10cm tolerance
        bottom_mask = z_coords <= (min_z + z_tolerance)
        bottom_vertices = vertices[bottom_mask]
        
        if len(bottom_vertices) < 2:
            return None
        
        # Use the wall's length direction to determine left/right
        length_direction = np.array(center_plane['length_direction'])
        center_point = np.array(center_plane['point'])
        
        # Project bottom vertices along the length direction to find extremes
        projections = np.dot(bottom_vertices - center_point, length_direction)
        leftmost_idx = np.argmin(projections)
        rightmost_idx = np.argmax(projections)
        
        return {
            'start_point': tuple(bottom_vertices[leftmost_idx]),
            'end_point': tuple(bottom_vertices[rightmost_idx]),
            'edge_type': 'bottom'
        }

    def get_centerplane_left_edge(self) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """
        Find the left edge of the wall's center plane (in the negative length direction).
        
        Returns:
            Dictionary with 'start_point', 'end_point', and 'edge_type'.
        """
        center_plane = self.get_center_plane()
        if not center_plane or not center_plane.get('vertices'):
            return None
        
        vertices = np.array(center_plane['vertices'])
        
        if len(vertices) < 2:
            return None
        
        # Use the wall's length direction to find the left edge
        length_direction = np.array(center_plane['length_direction'])
        center_point = np.array(center_plane['point'])
        
        # Project vertices along the length direction
        projections = np.dot(vertices - center_point, length_direction)
        min_proj = np.min(projections)
        proj_tolerance = 0.1  # 10cm tolerance
        
        left_mask = projections <= (min_proj + proj_tolerance)
        left_vertices = vertices[left_mask]
        
        if len(left_vertices) < 2:
            return None
        
        # Find the bottom and top points on the left side (use Z coordinates)
        z_coords = left_vertices[:, 2]
        bottom_idx = np.argmin(z_coords)  # Min Z
        top_idx = np.argmax(z_coords)     # Max Z
        
        return {
            'start_point': tuple(left_vertices[bottom_idx]),
            'end_point': tuple(left_vertices[top_idx]),
            'edge_type': 'left'
        }

    def get_centerplane_right_edge(self) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """
        Find the right edge of the wall's center plane (in the positive length direction).
        
        Returns:
            Dictionary with 'start_point', 'end_point', and 'edge_type'.
        """
        center_plane = self.get_center_plane()
        if not center_plane or not center_plane.get('vertices'):
            return None
        
        vertices = np.array(center_plane['vertices'])
        
        if len(vertices) < 2:
            return None
        
        # Use the wall's length direction to find the right edge
        length_direction = np.array(center_plane['length_direction'])
        center_point = np.array(center_plane['point'])
        
        # Project vertices along the length direction
        projections = np.dot(vertices - center_point, length_direction)
        max_proj = np.max(projections)
        proj_tolerance = 0.1  # 10cm tolerance
        
        right_mask = projections >= (max_proj - proj_tolerance)
        right_vertices = vertices[right_mask]
        
        if len(right_vertices) < 2:
            return None
        
        # Find the bottom and top points on the right side (use Z coordinates)
        z_coords = right_vertices[:, 2]
        bottom_idx = np.argmin(z_coords)  # Min Z
        top_idx = np.argmax(z_coords)     # Max Z
        
        return {
            'start_point': tuple(right_vertices[bottom_idx]),
            'end_point': tuple(right_vertices[top_idx]),
            'edge_type': 'right'
        }

    def get_centerplane_all_edges(self) -> Dict[str, Optional[Dict[str, Tuple[float, float, float]]]]:
        """
        Get all four edges of the wall's center plane.
        
        Returns:
            Dictionary containing all four edges: 'top', 'bottom', 'left', 'right'
        """
        return {
            'top': self.get_centerplane_top_edge(),
            'bottom': self.get_centerplane_bottom_edge(),
            'left': self.get_centerplane_left_edge(),
            'right': self.get_centerplane_right_edge()
        }

    def get_centerplane_start_point_bottom(self) -> Tuple[float, float, float]:
        """
        Get the start point of the wall at the bottom along the bottom edge.
        This finds the center line of the wall and uses the leftmost point as the start point.
        """
        bottom_edge = self.get_centerplane_bottom_edge()
        if bottom_edge:
            return bottom_edge['start_point']
        
        # Fallback to center plane vertices
        center_vertices = self.get_center_plane_vertices()
        if center_vertices:
            vertices = np.array(center_vertices)
            z_coords = vertices[:, 2]
            min_z = np.min(z_coords)
            z_tolerance = 0.1
            
            bottom_mask = z_coords <= (min_z + z_tolerance)
            bottom_vertices = vertices[bottom_mask]
            
            if len(bottom_vertices) > 0:
                # Find leftmost point (minimum X+Y)
                leftmost_idx = np.argmin(bottom_vertices[:, 0] + bottom_vertices[:, 1])
                return tuple(bottom_vertices[leftmost_idx])
        
        return (0.0, 0.0, 0.0)

    def get_centerplane_end_point_bottom(self) -> Tuple[float, float, float]:
        """
        Get the end point of the wall at the bottom along the bottom edge.
        This finds the center line of the wall and uses the rightmost point as the end point.
        """
        bottom_edge = self.get_centerplane_bottom_edge()
        if bottom_edge:
            return bottom_edge['end_point']
        
        # Fallback to center plane vertices
        center_vertices = self.get_center_plane_vertices()
        if center_vertices:
            vertices = np.array(center_vertices)
            z_coords = vertices[:, 2]
            min_z = np.min(z_coords)
            z_tolerance = 0.1
            
            bottom_mask = z_coords <= (min_z + z_tolerance)
            bottom_vertices = vertices[bottom_mask]
            
            if len(bottom_vertices) > 0:
                # Find rightmost point (maximum X+Y)
                rightmost_idx = np.argmax(bottom_vertices[:, 0] + bottom_vertices[:, 1])
                return tuple(bottom_vertices[rightmost_idx])
        
        return (0.0, 0.0, 0.0)

    def get_centerplane_normal_vector(self) -> Tuple[float, float, float]:
        """
        Get the normal vector of the wall's center plane.
        
        Returns:
            A tuple representing the normal vector (x, y, z).
        """
        center_plane = self.get_center_plane()
        if center_plane and 'normal' in center_plane:
            return tuple(center_plane['normal'])
        
        return None
    
    
                
    
@dataclass(slots=True)
class Door(Object):
    swing_direction: Optional[str] = None
    panel_position: Optional[str] = None

    
    @classmethod
    def from_components(
        cls,
        components: Tuple[Component, ...],
        name: str,
        **kwargs
    ) -> "Door":
        return super(Door, cls).from_components(
            components=components,
            name=name,
            type="Door",
            **kwargs
        )
    @classmethod
    def from_ifc(cls, ifc_path: Union[str, Path], object_type: str = "IfcDoor") -> "Door":
        """
        Load a single IFC IfcDoor entity into a Door object.
        """
        import ifcopenshell
        import ifcopenshell.geom

        ifc_file = ifcopenshell.open(str(ifc_path))
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)

        entity = ifc_file.by_type(object_type)[0]  # Assumes one door
        name = entity.Name or object_type
        geometry = Geometry()

        # Load mesh geometry
        try:
            shape = ifcopenshell.geom.create_shape(settings, entity)
            verts = list(zip(*[iter(shape.geometry.verts)] * 3))
            faces = list(zip(*[iter(shape.geometry.faces)] * 3))
            geometry.mesh_data = {
                "vertices": verts,
                "faces": faces
            }
            geometry._generate_brep_from_mesh()
        except Exception as e:
            print(f"[IFC] Geometry extraction failed for {name}: {e}")

        # Door swing + panel orientation
        swing_direction = None
        panel_position = None

        style_rel = next(
            (rel for rel in ifc_file.by_type("IfcRelDefinesByType") if rel.RelatedObjects[0] == entity),
            None
        )

        if style_rel and hasattr(style_rel, "RelatingType"):
            style = style_rel.RelatingType
            if style.is_a("IfcDoorStyle"):
                swing_direction = style.OperationType
            if hasattr(style, "HasPropertySets"):
                for prop in style.HasPropertySets:
                    if prop.is_a("IfcDoorPanelProperties"):
                        panel_position = prop.PanelPosition

        # Material
        material_name = None
        for rel in ifc_file.by_type("IfcRelAssociatesMaterial"):
            if entity in rel.RelatedObjects:
                mat = rel.RelatingMaterial
                if hasattr(mat, "Name"):
                    material_name = mat.Name
                elif hasattr(mat, "MaterialLayers"):
                    material_name = ", ".join(layer.Material.Name for layer in mat.MaterialLayers if layer.Material)

        # Color
        color_rgb = None
        try:
            reps = entity.Representation.Representations
            for rep in reps:
                for item in rep.Items:
                    if hasattr(item, "StyledByItem") and item.StyledByItem:
                        styled = item.StyledByItem[0]
                        if styled.Styles:
                            style = styled.Styles[0]
                            if hasattr(style, "SurfaceColour"):
                                c = style.SurfaceColour
                                color_rgb = (round(c.Red, 3), round(c.Green, 3), round(c.Blue, 3))
        except:
            pass

        # Optional: collect decomposed sub-elements
        sub_items = []
        for rel in getattr(entity, "IsDecomposedBy", []):
            for part in getattr(rel, "RelatedObjects", []):
                sub_geom = Geometry()
                sub_items.append(
                    Element(
                        name=part.Name or "Sub Part",
                        type=part.is_a(),
                        geometry=sub_geom,
                        material="unknown"
                    )
                )

        return cls(
            name=name,
            type="door",
            geometry=geometry,
            sub_items=tuple(sub_items),
            materials={material_name: {"volume": geometry.compute_volume(), "percent": 1.0}} if material_name else {},
            swing_direction=normalize_ifc_enum(swing_direction),
            panel_position=normalize_ifc_enum(panel_position),
            color=color_rgb
        )


@dataclass(slots=True)
class Window(Object):
    @classmethod
    def from_components(
        cls,
        components: Tuple[Component, ...],
        name: str,
        **kwargs
    ) -> "Window":
        return super(Window, cls).from_components(
            components=components,
            name=name,
            type="window",
            **kwargs
        )
    

@dataclass(slots=True)
class Deck(Object):

    boundary_id: Optional[str] = None

    
    @classmethod
    def from_components(
        cls,
        components: Tuple[Component, ...],
        name: str,
        **kwargs
    ) -> "Deck":
        return super(Deck, cls).from_components(
            components=components,
            name=name,
            type="deck",
            **kwargs
        )

    def get_centerplane_geometry(self) -> Optional['Geometry']:
        """
        Get a Geometry object that represents the center plane face of the wall.
        
        Returns:
            Geometry object containing the center plane as a mesh, or None if no center plane found.
        """
        center_plane = self.get_center_plane()
        if not center_plane or not center_plane.get('vertices'):
            return None
        
        plane_vertices = center_plane['vertices']
        
        if len(plane_vertices) < 3:
            return None
        
        # Create triangulated faces from the planar vertices
        faces = []
        
        if len(plane_vertices) == 3:
            # Already a triangle
            faces = [(0, 1, 2)]
        elif len(plane_vertices) == 4:
            # Quadrilateral - create two triangles
            faces = [(0, 1, 2), (0, 2, 3)]
        else:
            # For polygons with more than 4 vertices, triangulate from the first vertex
            # This works for convex polygons
            for i in range(1, len(plane_vertices) - 1):
                faces.append((0, i, i + 1))
        
        # Create the geometry object
        # Note: Adjust import paths based on your project structure
        try:
            from .geometry import Geometry, Vector3D
        except ImportError:
            # Fallback imports
            from geometry import Geometry, Vector3D
        
        geometry = Geometry()
        geometry.mesh_data = {
            "vertices": plane_vertices,
            "faces": faces
        }
        
        # Generate B-rep data from the mesh
        geometry._generate_brep_from_mesh()
        
        # Set the origin to the center point of the plane
        center_point = center_plane['point']
        geometry.origin = Vector3D(center_point[0], center_point[1], center_point[2])
        
        return geometry
    
    def get_center_plane(self) -> Optional[Dict]:
        """
        Find the center plane of the wall that runs through the middle of its thickness.
        Uses PCA to find the true orientation for non-axis aligned walls.
        
        Returns:
            Dictionary containing 'normal', 'point', 'thickness_direction', and 'vertices' that define the center plane.
        """
        geometry = self.geometry
        if not geometry.mesh_data:
            return None
        
        vertices = geometry.mesh_data.get("vertices", [])
        if not vertices:
            return None
        
        vertices = np.array(vertices)
        
        if len(vertices) < 4:
            return None
        
        # Step 1: Use PCA to find the principal directions of the wall
        # Center the vertices
        centroid = np.mean(vertices, axis=0)
        centered_vertices = vertices - centroid
        
        # Compute covariance matrix and eigenvalues/eigenvectors
        cov_matrix = np.cov(centered_vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (largest to smallest)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # The eigenvector with the smallest eigenvalue is the thickness direction (normal to wall)
        thickness_direction = eigenvectors[:, 2]  # Smallest eigenvalue
        wall_length_direction = eigenvectors[:, 0]  # Largest eigenvalue
        wall_height_direction = eigenvectors[:, 1]  # Middle eigenvalue
        
        # Ensure consistent orientation (normal should point in consistent direction)
        if thickness_direction[2] < 0:  # Prefer normal pointing up if possible
            thickness_direction = -thickness_direction
        
        # Step 2: Find the center plane by projecting vertices onto the thickness direction
        # and finding the center along that direction
        projections = np.dot(centered_vertices, thickness_direction)
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        center_proj = (min_proj + max_proj) / 2
        
        # Step 3: Find the center point on the plane
        center_point = centroid + center_proj * thickness_direction
        
        # Step 4: Generate vertices that define the center plane
        # Project all original vertices onto the center plane
        plane_vertices = []
        
        for vertex in vertices:
            # Vector from center point to vertex
            to_vertex = vertex - center_point
            
            # Remove the component along the thickness direction (project onto plane)
            projection_onto_normal = np.dot(to_vertex, thickness_direction)
            vertex_on_plane = vertex - projection_onto_normal * thickness_direction
            
            plane_vertices.append(vertex_on_plane)
        
        plane_vertices = np.array(plane_vertices)
        
        # Step 5: Find the boundary vertices of the center plane
        # We'll find the convex hull of the projected vertices
        try:
            from scipy.spatial import ConvexHull
            
            # Project plane vertices to 2D for convex hull calculation
            # Use the wall length and height directions as the 2D basis
            coords_2d = np.column_stack([
                np.dot(plane_vertices - center_point, wall_length_direction),
                np.dot(plane_vertices - center_point, wall_height_direction)
            ])
            
            # Find convex hull
            hull = ConvexHull(coords_2d)
            boundary_vertices = plane_vertices[hull.vertices]
            
        except ImportError:
            # Fallback: use simplified boundary detection
            # Find extreme points in each direction
            projections_length = np.dot(plane_vertices - center_point, wall_length_direction)
            projections_height = np.dot(plane_vertices - center_point, wall_height_direction)
            
            # Find indices of extreme points
            extreme_indices = [
                np.argmin(projections_length),  # Min length
                np.argmax(projections_length),  # Max length
                np.argmin(projections_height),  # Min height
                np.argmax(projections_height),  # Max height
            ]
            
            # Remove duplicates and get unique boundary vertices
            extreme_indices = list(set(extreme_indices))
            boundary_vertices = plane_vertices[extreme_indices]
        
        # Step 6: Sort boundary vertices in a consistent order (counterclockwise)
        if len(boundary_vertices) > 2:
            # Calculate angles from center point to sort vertices
            vectors_to_boundary = boundary_vertices - center_point
            angles = []
            
            for vec in vectors_to_boundary:
                length_component = np.dot(vec, wall_length_direction)
                height_component = np.dot(vec, wall_height_direction)
                angle = np.arctan2(height_component, length_component)
                angles.append(angle)
            
            # Sort by angle
            sorted_indices = np.argsort(angles)
            boundary_vertices = boundary_vertices[sorted_indices]
        
        return {
            'normal': tuple(thickness_direction),
            'point': tuple(center_point),
            'thickness_direction': tuple(thickness_direction),
            'length_direction': tuple(wall_length_direction),
            'height_direction': tuple(wall_height_direction),
            'vertices': [tuple(v) for v in boundary_vertices],
            'thickness': float(max_proj - min_proj),
        }
    def get_centerplane_normal_vector(self) -> Tuple[float, float, float]:
        """
        Get the normal vector of the wall's center plane.
        
        Returns:
            A tuple representing the normal vector (x, y, z).
        """
        center_plane = self.get_center_plane()
        if center_plane and 'normal' in center_plane:
            return tuple(center_plane['normal'])
        
        return None

    

if __name__ == "__main__":
    pass