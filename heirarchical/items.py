from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List, Union
from .geometry import Geometry
import numpy as np
from .helpers import generate_id, normalize_ifc_enum
from copy import deepcopy
from pathlib import Path




@dataclass(slots=True)
class BaseItem:
    """Shared base class for all entities"""

    # A unique UUID
    id: str
    
    # The geometry object describing the item's shape
    geometry: Geometry

    # A given name
    name: str

    # A defined type (e.g., 'wall', 'pipe', 'deck')
    type: str

    # An optional collection of sub-item IDs (hierarchical structure)
    sub_items: Tuple[str, ...] = field(default_factory=tuple)


    # A set of relationships the item has — well suited for generating graphs
    # relationships: Tuple[Relationship, ...] = field(default_factory=tuple)

    # Arbitrary measurable or inferred properties (e.g., area, ghg_emissions)
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Ontological tags and values (e.g., {'structural': True, 'zone': 'care'})
    ontologies: Dict[str, Any] = field(default_factory=dict)

    materials: Dict[str, Dict[str, float]] = field(default_factory=dict)

    color: Optional[Tuple[float, float, float]] = None


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
            id=generate_id(self.type),
            name=self.name + " (copy)",
            type=self.type,
            geometry=geom_copy,
            sub_items=deepcopy(self.sub_items),
            attributes=deepcopy(self.attributes),
            ontologies=deepcopy(self.ontologies),
            materials=deepcopy(self.materials),
            **({"material": self.material} if hasattr(self, "material") else {})
        )


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
            id=kwargs.get("id", f"component_{name.lower().replace(' ', '_')}"),
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
            id=kwargs.get("id", f"object_{name.lower().replace(' ', '_')}"),
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
                        id=generate_id("element"),
                        name=part.Name or "Sub Part",
                        type=part.is_a(),
                        geometry=sub_geom,
                        material="unknown"
                    )
                )

        return cls(
            id=generate_id(object_type),
            name=name,
            type=object_type.lower(),
            geometry=geometry,
            sub_items=tuple(sub_items),
            materials={}
        )

@dataclass(slots=True)
class Wall(Object):
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
            type="door",
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
                        id=generate_id("element"),
                        name=part.Name or "Sub Part",
                        type=part.is_a(),
                        geometry=sub_geom,
                        material="unknown"
                    )
                )

        return cls(
            id=generate_id(object_type),
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
class Deck(Object):
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

    

if __name__ == "__main__":
    pass