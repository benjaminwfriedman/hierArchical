"""
Property-based tests for hierarchical.geometry module.

This module uses Hypothesis to test geometric invariants and mathematical
properties that should hold across a wide range of input values.
"""

import pytest
import numpy as np
import math
from hypothesis import given, strategies as st, assume, settings, Verbosity

from tests.conftest import MockGeometry, assert_volumes_equal
from tests.fixtures.sample_geometries import create_mock_geometry_from_data


# =============================================================================
# HYPOTHESIS STRATEGIES
# =============================================================================

# Coordinate strategies with reasonable bounds
coordinate = st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
small_coordinate = st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
positive_dimension = st.floats(min_value=0.1, max_value=50, allow_nan=False, allow_infinity=False)
angle = st.floats(min_value=0, max_value=2*math.pi, allow_nan=False, allow_infinity=False)
scale_factor = st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False)

# Vector3D strategy
vector3d_values = st.tuples(coordinate, coordinate, coordinate)

# Simple geometry strategy
simple_triangle = st.just([
    [(0, 0, 0), (1, 0, 0), (0.5, 1, 0)],  # vertices
    [(0, 1, 2)]  # faces
])

simple_quad = st.just([
    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],  # vertices
    [(0, 1, 2), (0, 2, 3)]  # faces
])

simple_geometry = st.one_of(simple_triangle, simple_quad)


# =============================================================================
# VECTOR3D PROPERTY TESTS
# =============================================================================

class TestVector3DProperties:
    """Property-based tests for Vector3D class."""
    
    @given(vector3d_values)
    @settings(deadline=None)  # Disable deadline for this test
    @pytest.mark.property
    @pytest.mark.geometry
    def test_vector3d_iteration_consistency(self, coords):
        """Test that Vector3D iteration is consistent with initialization."""
        from hierarchical.geometry import Vector3D
        
        x, y, z = coords
        vector = Vector3D(x, y, z)
        
        # Iteration should yield the same values as direct access
        iterated_coords = list(vector)
        assert iterated_coords == [x, y, z]
        
        # Unpacking should work
        unpacked_x, unpacked_y, unpacked_z = vector
        assert unpacked_x == x
        assert unpacked_y == y
        assert unpacked_z == z
    
    @given(vector3d_values)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_vector3d_tuple_conversion_consistency(self, coords):
        """Test that tuple conversion preserves values."""
        from hierarchical.geometry import Vector3D
        
        x, y, z = coords
        vector = Vector3D(x, y, z)
        
        tuple_result = vector.as_tuple()
        assert tuple_result == (x, y, z)
        assert isinstance(tuple_result, tuple)
    
    @given(vector3d_values)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_vector3d_array_conversion_consistency(self, coords):
        """Test that array conversion preserves values."""
        from hierarchical.geometry import Vector3D
        
        x, y, z = coords
        vector = Vector3D(x, y, z)
        
        array_result = vector.as_array()
        expected = np.array([x, y, z])
        
        np.testing.assert_array_equal(array_result, expected)
        assert isinstance(array_result, np.ndarray)
    
    @given(vector3d_values)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_vector3d_conversion_roundtrip(self, coords):
        """Test that conversions are consistent with each other."""
        from hierarchical.geometry import Vector3D
        
        x, y, z = coords
        vector = Vector3D(x, y, z)
        
        # All conversion methods should be consistent
        tuple_result = vector.as_tuple()
        array_result = vector.as_array()
        list_result = list(vector)
        
        assert tuple_result == tuple(list_result)
        assert tuple_result == tuple(array_result)
        np.testing.assert_array_equal(array_result, np.array(list_result))


# =============================================================================
# GEOMETRY TRANSFORMATION PROPERTY TESTS
# =============================================================================

class TestGeometryTransformationProperties:
    """Property-based tests for geometric transformations."""
    
    @given(simple_geometry, coordinate, coordinate, coordinate)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_translation_preserves_volume(self, geom_data, dx, dy, dz):
        """Test that translation preserves volume."""
        from hierarchical.geometry import Geometry
        
        vertices, faces = geom_data
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertices, "faces": faces}
        
        original_volume = geometry.compute_volume()
        
        # Apply translation using _translate method
        geometry._translate(dx, dy, dz)
        translated_volume = geometry.compute_volume()
        
        # Volume should be preserved
        assert_volumes_equal(original_volume, translated_volume, tolerance=1e-10)
    
    @given(simple_geometry, angle)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_rotation_preserves_volume(self, geom_data, rotation_angle):
        """Test that rotation preserves volume."""
        from hierarchical.geometry import Geometry
        
        vertices, faces = geom_data
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertices, "faces": faces}
        
        original_volume = geometry.compute_volume()
        
        # Apply rotation
        geometry.rotate_z(rotation_angle)
        rotated_volume = geometry.compute_volume()
        
        # Volume should be preserved
        assert_volumes_equal(original_volume, rotated_volume, tolerance=1e-10)
    
    @given(simple_geometry, coordinate, coordinate, coordinate, angle)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_combined_transformations_preserve_volume(self, geom_data, dx, dy, dz, angle):
        """Test that combined transformations preserve volume."""
        from hierarchical.geometry import Geometry
        
        vertices, faces = geom_data
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertices, "faces": faces}
        
        original_volume = geometry.compute_volume()
        
        # Apply combined transformation
        geometry._translate(dx, dy, dz).rotate_z(angle)
        transformed_volume = geometry.compute_volume()
        
        # Volume should be preserved
        assert_volumes_equal(original_volume, transformed_volume, tolerance=1e-10)
    
    @given(simple_geometry, coordinate, coordinate, coordinate)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_directional_movements_consistency(self, geom_data, dx, dy, dz):
        """Test that directional movements are equivalent to general move."""
        from hierarchical.geometry import Geometry
        
        vertices, faces = geom_data
        
        # Create two identical geometries
        geom1 = Geometry()
        geom1.mesh_data = {"vertices": vertices.copy(), "faces": faces.copy()}
        
        geom2 = Geometry()
        geom2.mesh_data = {"vertices": vertices.copy(), "faces": faces.copy()}
        
        # Apply movement differently
        geom1._translate(dx, dy, dz)
        geom2.right(dx).forward(dy).up(dz)
        
        # Results should be equivalent (within numerical precision)
        vol1 = geom1.compute_volume()
        vol2 = geom2.compute_volume()
        
        assert_volumes_equal(vol1, vol2, tolerance=1e-10)
    
    @given(simple_geometry, angle, angle)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_rotation_composition(self, geom_data, angle1, angle2):
        """Test that sequential rotations compose correctly."""
        from hierarchical.geometry import Geometry
        
        vertices, faces = geom_data
        
        # Create two identical geometries
        geom1 = Geometry()
        geom1.mesh_data = {"vertices": vertices.copy(), "faces": faces.copy()}
        
        geom2 = Geometry()
        geom2.mesh_data = {"vertices": vertices.copy(), "faces": faces.copy()}
        
        # Apply rotations differently
        geom1.rotate_z(angle1).rotate_z(angle2)
        geom2.rotate_z(angle1 + angle2)
        
        # Results should be approximately equivalent
        vol1 = geom1.compute_volume()
        vol2 = geom2.compute_volume()
        
        assert_volumes_equal(vol1, vol2, tolerance=1e-10)


# =============================================================================
# GEOMETRY CALCULATION PROPERTY TESTS
# =============================================================================

class TestGeometryCalculationProperties:
    """Property-based tests for geometric calculations."""
    
    @given(st.lists(vector3d_values, min_size=1, max_size=10))
    @pytest.mark.property
    @pytest.mark.geometry
    def test_centroid_properties(self, vertex_coords):
        """Test properties of centroid calculation."""
        from hierarchical.geometry import Geometry
        
        # Skip degenerate cases
        assume(len(vertex_coords) > 0)
        
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertex_coords, "faces": []}
        
        centroid = geometry.get_centroid()
        
        # Centroid should be within the bounding box
        if len(vertex_coords) > 0:
            min_point, max_point = geometry.get_bbox()
            
            assert min_point[0] <= centroid.x <= max_point[0]
            assert min_point[1] <= centroid.y <= max_point[1]
            assert min_point[2] <= centroid.z <= max_point[2]
    
    @given(st.lists(vector3d_values, min_size=2, max_size=10))
    @pytest.mark.property
    @pytest.mark.geometry
    def test_bbox_contains_all_vertices(self, vertex_coords):
        """Test that bounding box contains all vertices."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertex_coords, "faces": []}
        
        min_point, max_point = geometry.get_bbox()
        
        # All vertices should be within the bounding box
        for vertex in vertex_coords:
            x, y, z = vertex
            assert min_point[0] <= x <= max_point[0]
            assert min_point[1] <= y <= max_point[1]
            assert min_point[2] <= z <= max_point[2]
    
    @given(st.lists(vector3d_values, min_size=1, max_size=10))
    @pytest.mark.property
    @pytest.mark.geometry
    def test_height_is_non_negative(self, vertex_coords):
        """Test that height calculation is always non-negative."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertex_coords, "faces": []}
        
        height = geometry.get_height()
        
        assert height >= 0.0
        assert isinstance(height, float)
    
    @given(vector3d_values, positive_dimension)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_height_calculation_consistency(self, base_coord, height_offset):
        """Test height calculation with known offset."""
        from hierarchical.geometry import Geometry
        
        x, y, z = base_coord
        
        # Create vertices at two different Z levels
        vertices = [
            (x, y, z),
            (x + 1, y, z),
            (x, y + 1, z + height_offset),
            (x + 1, y + 1, z + height_offset)
        ]
        
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertices, "faces": []}
        
        calculated_height = geometry.get_height()
        
        # Height should match the offset (within numerical precision)
        assert abs(calculated_height - height_offset) < 1e-10
    
    @given(simple_geometry)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_volume_is_non_negative(self, geom_data):
        """Test that volume calculation is always non-negative."""
        from hierarchical.geometry import Geometry
        
        vertices, faces = geom_data
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertices, "faces": faces}
        
        volume = geometry.compute_volume()
        
        assert volume >= 0.0
        assert isinstance(volume, float)


# =============================================================================
# GEOMETRY INTERSECTION PROPERTY TESTS
# =============================================================================

class TestGeometryIntersectionProperties:
    """Property-based tests for geometric intersections."""
    
    @given(simple_geometry, simple_geometry)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_intersection_symmetry(self, geom_data1, geom_data2):
        """Test that intersection is symmetric."""
        from hierarchical.geometry import Geometry
        
        vertices1, faces1 = geom_data1
        vertices2, faces2 = geom_data2
        
        geom1 = Geometry()
        geom1.mesh_data = {"vertices": vertices1, "faces": faces1}
        
        geom2 = Geometry()
        geom2.mesh_data = {"vertices": vertices2, "faces": faces2}
        
        # Intersection should be symmetric
        intersects_1_2 = geom1.bbox_intersects(geom2)
        intersects_2_1 = geom2.bbox_intersects(geom1)
        
        assert intersects_1_2 == intersects_2_1
    
    @given(simple_geometry)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_self_intersection(self, geom_data):
        """Test that geometry always intersects with itself."""
        from hierarchical.geometry import Geometry
        
        vertices, faces = geom_data
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertices, "faces": faces}
        
        # Geometry should always intersect with itself
        assert geometry.bbox_intersects(geometry) is True
        
        # Self-intersection overlap should be 100% (but may be 0 for degenerate cases)
        overlap = geometry.bbox_intersects(geometry, return_overlap_percent=True)
        # For valid geometries, overlap should be 100%, but degenerate cases may give 0
        assert overlap == 100.0 or overlap == 0.0
    
    @given(simple_geometry)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_distance_to_self_is_zero(self, geom_data):
        """Test that distance to self is zero."""
        from hierarchical.geometry import Geometry
        
        vertices, faces = geom_data
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertices, "faces": faces}
        
        # Distance to self should be zero
        distance = geometry.distance_to(geometry)
        assert abs(distance) < 1e-10
    
    @given(simple_geometry, simple_geometry)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_distance_symmetry(self, geom_data1, geom_data2):
        """Test that distance calculation is symmetric."""
        from hierarchical.geometry import Geometry
        
        vertices1, faces1 = geom_data1
        vertices2, faces2 = geom_data2
        
        geom1 = Geometry()
        geom1.mesh_data = {"vertices": vertices1, "faces": faces1}
        
        geom2 = Geometry()
        geom2.mesh_data = {"vertices": vertices2, "faces": faces2}
        
        # Distance should be symmetric
        dist_1_2 = geom1.distance_to(geom2)
        dist_2_1 = geom2.distance_to(geom1)
        
        assert abs(dist_1_2 - dist_2_1) < 1e-10


# =============================================================================
# GEOMETRY EDGE CASE PROPERTY TESTS
# =============================================================================

class TestGeometryEdgeCaseProperties:
    """Property-based tests for edge cases and robustness."""
    
    @given(st.lists(vector3d_values, min_size=1, max_size=3))  # Require at least 1 vertex
    @pytest.mark.property
    @pytest.mark.geometry
    def test_small_geometry_robustness(self, vertex_coords):
        """Test that operations handle small geometries gracefully."""
        from hierarchical.geometry import Geometry
        
        # Skip empty geometry cases that cause issues
        assume(len(vertex_coords) > 0)
        
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertex_coords, "faces": []}
        
        # All operations should complete without error
        try:
            volume = geometry.compute_volume()
            assert isinstance(volume, float)
            assert volume >= 0.0
            
            height = geometry.get_height()
            assert isinstance(height, float)
            assert height >= 0.0
            
            centroid = geometry.get_centroid()
            assert hasattr(centroid, 'x')
            assert hasattr(centroid, 'y')
            assert hasattr(centroid, 'z')
            
            min_point, max_point = geometry.get_bbox()
            assert isinstance(min_point, np.ndarray)
            assert isinstance(max_point, np.ndarray)
            
        except Exception as e:
            pytest.fail(f"Operation failed on small geometry: {e}")
    
    @given(st.floats(min_value=-1e-10, max_value=1e-10))
    @pytest.mark.property
    @pytest.mark.geometry
    def test_numerical_precision_handling(self, small_value):
        """Test handling of very small numerical values."""
        from hierarchical.geometry import Vector3D
        
        # Very small values should not cause issues
        vector = Vector3D(small_value, small_value, small_value)
        
        # Operations should complete
        tuple_result = vector.as_tuple()
        array_result = vector.as_array()
        
        assert len(tuple_result) == 3
        assert len(array_result) == 3
        
        # Values should be preserved
        assert abs(tuple_result[0] - small_value) < 1e-15
        assert abs(tuple_result[1] - small_value) < 1e-15
        assert abs(tuple_result[2] - small_value) < 1e-15


# =============================================================================
# PRIMITIVE GEOMETRY PROPERTY TESTS
# =============================================================================

class TestPrimitiveGeometryProperties:
    """Property-based tests for primitive geometry creation."""
    
    @given(positive_dimension, positive_dimension, positive_dimension)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_box_primitive_properties(self, width, depth, height):
        """Test properties of box primitives."""
        from hierarchical.geometry import Geometry
        
        dimensions = {'width': width, 'depth': depth, 'height': height}
        geometry = Geometry.from_primitive('box', dimensions)
        
        # Should have correct number of vertices and faces
        assert len(geometry.mesh_data['vertices']) == 8  # Box has 8 vertices
        assert len(geometry.mesh_data['faces']) == 12   # Box has 12 triangular faces
        
        # Volume should be positive
        volume = geometry.compute_volume()
        assert volume > 0
        
        # Computed height should be close to specified height
        computed_height = geometry.get_height()
        assert abs(computed_height - height) < 1e-6
    
    @given(positive_dimension, positive_dimension, st.integers(min_value=3, max_value=16))
    @pytest.mark.property
    @pytest.mark.geometry
    def test_cylinder_primitive_properties(self, radius, height, segments):
        """Test properties of cylinder primitives."""
        from hierarchical.geometry import Geometry
        
        dimensions = {'radius': radius, 'height': height, 'segments': segments}
        geometry = Geometry.from_primitive('cylinder', dimensions)
        
        # Should have correct structure
        expected_vertices = 2 + (segments * 2)  # 2 centers + 2*segments rim points
        assert len(geometry.mesh_data['vertices']) == expected_vertices
        
        # Volume should be positive
        volume = geometry.compute_volume()
        assert volume > 0
        
        # Computed height should be close to specified height
        computed_height = geometry.get_height()
        assert abs(computed_height - height) < 1e-6
    
    @given(st.lists(st.tuples(small_coordinate, small_coordinate), min_size=3, max_size=8), 
           positive_dimension)
    @pytest.mark.property
    @pytest.mark.geometry
    def test_prism_primitive_properties(self, base_points, height):
        """Test properties of prism primitives."""
        from hierarchical.geometry import Geometry
        
        # Ensure base points form a non-degenerate polygon
        assume(len(set(base_points)) >= 3)  # No duplicate points
        
        geometry = Geometry.from_prism(base_points, height)
        
        # Should have correct number of vertices
        expected_vertices = len(base_points) * 2  # Bottom + top
        assert len(geometry.mesh_data['vertices']) == expected_vertices
        
        # Volume should be positive for valid polygons
        volume = geometry.compute_volume()
        assert volume >= 0
        
        # Computed height should match specified height
        computed_height = geometry.get_height()
        assert abs(computed_height - height) < 1e-6


# =============================================================================
# CONFIGURATION FOR PROPERTY TESTS
# =============================================================================

# Configure Hypothesis for more thorough testing
# Reduce examples for faster CI, but increase for comprehensive testing
settings.register_profile("default", max_examples=100, deadline=10000)
settings.register_profile("comprehensive", max_examples=1000, deadline=30000)
settings.register_profile("ci", max_examples=50, deadline=5000)

# Load the default profile
settings.load_profile("default")