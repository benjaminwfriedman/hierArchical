"""
Sample geometry fixtures for testing hierArchical components.

This module provides a collection of pre-defined geometries for consistent
testing across the test suite. All geometries use lightweight test doubles
to avoid heavy dependency requirements.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class TestGeometryData:
    """Container for test geometry data."""
    name: str
    vertices: List[Tuple[float, float, float]]
    faces: List[Tuple[int, ...]]
    expected_volume: float
    expected_centroid: Tuple[float, float, float]
    description: str


# =============================================================================
# BASIC GEOMETRIC SHAPES
# =============================================================================

UNIT_CUBE = TestGeometryData(
    name="unit_cube",
    vertices=[
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),  # bottom face
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)   # top face
    ],
    faces=[
        # Bottom face (z=0)
        (0, 1, 2), (0, 2, 3),
        # Top face (z=1)
        (4, 6, 5), (4, 7, 6),
        # Side faces
        (0, 4, 1), (1, 4, 5),  # front
        (1, 5, 2), (2, 5, 6),  # right
        (2, 6, 3), (3, 6, 7),  # back
        (3, 7, 0), (0, 7, 4)   # left
    ],
    expected_volume=1.0,
    expected_centroid=(0.5, 0.5, 0.5),
    description="Unit cube from (0,0,0) to (1,1,1)"
)

SIMPLE_TRIANGLE = TestGeometryData(
    name="simple_triangle",
    vertices=[
        (0, 0, 0),
        (1, 0, 0),
        (0.5, 1, 0)
    ],
    faces=[
        (0, 1, 2)
    ],
    expected_volume=0.0,  # 2D triangle has no volume
    expected_centroid=(0.5, 0.333333, 0.0),
    description="Simple triangle in XY plane"
)

TETRAHEDRON = TestGeometryData(
    name="tetrahedron",
    vertices=[
        (0, 0, 0),      # base vertex 1
        (1, 0, 0),      # base vertex 2
        (0.5, 1, 0),    # base vertex 3
        (0.5, 0.5, 1)   # apex
    ],
    faces=[
        (0, 1, 2),      # base
        (0, 1, 3),      # side 1
        (1, 2, 3),      # side 2
        (2, 0, 3)       # side 3
    ],
    expected_volume=0.166667,  # 1/6 for this tetrahedron
    expected_centroid=(0.5, 0.375, 0.25),
    description="Simple tetrahedron"
)

RECTANGULAR_PRISM = TestGeometryData(
    name="rectangular_prism",
    vertices=[
        # Bottom face (z=0)
        (0, 0, 0), (2, 0, 0), (2, 3, 0), (0, 3, 0),
        # Top face (z=1)
        (0, 0, 1), (2, 0, 1), (2, 3, 1), (0, 3, 1)
    ],
    faces=[
        # Bottom and top faces
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        # Side faces
        (0, 4, 1), (1, 4, 5),
        (1, 5, 2), (2, 5, 6),
        (2, 6, 3), (3, 6, 7),
        (3, 7, 0), (0, 7, 4)
    ],
    expected_volume=6.0,  # 2 * 3 * 1
    expected_centroid=(1.0, 1.5, 0.5),
    description="Rectangular prism 2x3x1"
)


# =============================================================================
# ARCHITECTURAL SHAPES
# =============================================================================

WALL_SEGMENT = TestGeometryData(
    name="wall_segment",
    vertices=[
        # Wall running along X-axis, thickness in Y direction
        (0, 0, 0), (10, 0, 0), (10, 0.2, 0), (0, 0.2, 0),    # bottom
        (0, 0, 3), (10, 0, 3), (10, 0.2, 3), (0, 0.2, 3)     # top
    ],
    faces=[
        # Bottom and top
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        # Sides
        (0, 4, 1), (1, 4, 5),  # front face
        (1, 5, 2), (2, 5, 6),  # right end
        (2, 6, 3), (3, 6, 7),  # back face
        (3, 7, 0), (0, 7, 4)   # left end
    ],
    expected_volume=6.0,  # 10 * 0.2 * 3
    expected_centroid=(5.0, 0.1, 1.5),
    description="Wall segment 10m long, 0.2m thick, 3m high"
)

DOOR_OPENING = TestGeometryData(
    name="door_opening",
    vertices=[
        # Door frame (simplified as rectangular opening)
        (0, 0, 0), (0.9, 0, 0), (0.9, 0.2, 0), (0, 0.2, 0),     # bottom
        (0, 0, 2.1), (0.9, 0, 2.1), (0.9, 0.2, 2.1), (0, 0.2, 2.1)  # top
    ],
    faces=[
        # Bottom and top
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        # Sides
        (0, 4, 1), (1, 4, 5),
        (1, 5, 2), (2, 5, 6),
        (2, 6, 3), (3, 6, 7),
        (3, 7, 0), (0, 7, 4)
    ],
    expected_volume=0.378,  # 0.9 * 0.2 * 2.1
    expected_centroid=(0.45, 0.1, 1.05),
    description="Standard door opening 0.9m x 2.1m x 0.2m thick"
)


# =============================================================================
# EDGE CASES AND SPECIAL GEOMETRIES
# =============================================================================

EMPTY_GEOMETRY = TestGeometryData(
    name="empty_geometry",
    vertices=[],
    faces=[],
    expected_volume=0.0,
    expected_centroid=(0.0, 0.0, 0.0),
    description="Empty geometry with no vertices or faces"
)

SINGLE_POINT = TestGeometryData(
    name="single_point",
    vertices=[(0, 0, 0)],
    faces=[],
    expected_volume=0.0,
    expected_centroid=(0.0, 0.0, 0.0),
    description="Single point geometry"
)

DEGENERATE_TRIANGLE = TestGeometryData(
    name="degenerate_triangle",
    vertices=[
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0)  # Collinear points
    ],
    faces=[(0, 1, 2)],
    expected_volume=0.0,
    expected_centroid=(1.0, 0.0, 0.0),
    description="Degenerate triangle with collinear vertices"
)

COMPLEX_POLYHEDRON = TestGeometryData(
    name="complex_polyhedron",
    vertices=[
        # Base octagon (approximation)
        (1, 0, 0), (0.7, 0.7, 0), (0, 1, 0), (-0.7, 0.7, 0),
        (-1, 0, 0), (-0.7, -0.7, 0), (0, -1, 0), (0.7, -0.7, 0),
        # Top center point
        (0, 0, 2)
    ],
    faces=[
        # Base faces (triangulated octagon)
        (0, 1, 8), (1, 2, 8), (2, 3, 8), (3, 4, 8),
        (4, 5, 8), (5, 6, 8), (6, 7, 8), (7, 0, 8),
        # Bottom faces
        (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 4),
        (0, 6, 5), (0, 7, 6), (7, 0, 6)
    ],
    expected_volume=4.0,  # Approximate
    expected_centroid=(0.0, 0.0, 0.5),
    description="Complex polyhedron with octagonal base"
)


# =============================================================================
# GEOMETRY COLLECTIONS
# =============================================================================

ALL_BASIC_SHAPES = [
    UNIT_CUBE,
    SIMPLE_TRIANGLE,
    TETRAHEDRON,
    RECTANGULAR_PRISM
]

ALL_ARCHITECTURAL_SHAPES = [
    WALL_SEGMENT,
    DOOR_OPENING
]

ALL_EDGE_CASES = [
    EMPTY_GEOMETRY,
    SINGLE_POINT,
    DEGENERATE_TRIANGLE
]

ALL_COMPLEX_SHAPES = [
    COMPLEX_POLYHEDRON
]

ALL_TEST_GEOMETRIES = (
    ALL_BASIC_SHAPES + 
    ALL_ARCHITECTURAL_SHAPES + 
    ALL_EDGE_CASES + 
    ALL_COMPLEX_SHAPES
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_geometry_by_name(name: str) -> TestGeometryData:
    """Get test geometry data by name."""
    for geom in ALL_TEST_GEOMETRIES:
        if geom.name == name:
            return geom
    raise ValueError(f"No test geometry found with name: {name}")


def get_geometries_by_type(geometry_type: str) -> List[TestGeometryData]:
    """Get test geometries by type category."""
    type_map = {
        "basic": ALL_BASIC_SHAPES,
        "architectural": ALL_ARCHITECTURAL_SHAPES,
        "edge_cases": ALL_EDGE_CASES,
        "complex": ALL_COMPLEX_SHAPES
    }
    
    if geometry_type not in type_map:
        raise ValueError(f"Unknown geometry type: {geometry_type}")
    
    return type_map[geometry_type]


def create_mock_geometry_from_data(data: TestGeometryData):
    """Create a MockGeometry instance from TestGeometryData."""
    from tests.conftest import MockGeometry
    return MockGeometry(data.vertices, data.faces)


# =============================================================================
# TRANSFORMATION TEST DATA
# =============================================================================

TRANSFORMATION_TESTS = {
    "translation": {
        "transformations": [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, -1)],
        "should_preserve": ["volume", "shape"],
        "description": "Translation should preserve volume and shape"
    },
    "rotation_z": {
        "transformations": [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
        "should_preserve": ["volume", "shape", "distance_from_origin"],
        "description": "Z-rotation should preserve volume and shape"
    },
    "uniform_scaling": {
        "transformations": [0.5, 1.0, 2.0, 10.0],
        "should_preserve": ["shape_ratios"],
        "volume_scaling": "cubic",  # Volume scales as scale_factor^3
        "description": "Uniform scaling preserves shape ratios"
    }
}


# =============================================================================
# MATERIAL TEST DATA
# =============================================================================

MATERIAL_TEST_DATA = {
    "concrete": {
        "density": 2400,  # kg/m³
        "properties": {"compressive_strength": 30, "fire_rating": "4hr"}
    },
    "steel": {
        "density": 7850,  # kg/m³
        "properties": {"yield_strength": 250, "fire_rating": "2hr"}
    },
    "wood": {
        "density": 600,   # kg/m³
        "properties": {"species": "douglas_fir", "fire_rating": "1hr"}
    },
    "aluminum": {
        "density": 2700,  # kg/m³
        "properties": {"alloy": "6061-T6", "corrosion_resistance": "excellent"}
    }
}


# =============================================================================
# SPATIAL RELATIONSHIP TEST DATA
# =============================================================================

SPATIAL_RELATIONSHIP_TESTS = {
    "intersection": {
        "intersecting_pairs": [
            (UNIT_CUBE, RECTANGULAR_PRISM),  # Should intersect
        ],
        "non_intersecting_pairs": [
            (UNIT_CUBE, SIMPLE_TRIANGLE),   # Different planes
        ],
        "tolerance": 1e-6
    },
    "adjacency": {
        "adjacent_pairs": [],  # Define adjacent geometries
        "non_adjacent_pairs": [],
        "tolerance": 0.1
    },
    "containment": {
        "container_contained_pairs": [],  # Define containment relationships
        "tolerance": 1e-6
    }
}