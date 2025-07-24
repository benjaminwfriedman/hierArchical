"""
Unit tests for hierarchical.geometry module.

This module tests the Vector3D and Geometry classes with comprehensive
coverage of all geometric operations and calculations.
"""

import pytest
import numpy as np
import math
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st

from tests.conftest import (
    MockGeometry, MockVertex, MockFace,
    assert_geometries_equal, assert_volumes_equal
)
from tests.fixtures.sample_geometries import (
    UNIT_CUBE, SIMPLE_TRIANGLE, TETRAHEDRON, RECTANGULAR_PRISM,
    get_geometry_by_name, create_mock_geometry_from_data
)


class TestVector3D:
    """Test the Vector3D class functionality."""
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_vector3d_initialization_default(self):
        """Test Vector3D initialization with default values."""
        # Import here to avoid dependency issues during collection
        from hierarchical.geometry import Vector3D
        
        vector = Vector3D()
        assert vector.x == 0.0
        assert vector.y == 0.0
        assert vector.z == 0.0
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_vector3d_initialization_with_values(self):
        """Test Vector3D initialization with specific values."""
        from hierarchical.geometry import Vector3D
        
        vector = Vector3D(1.5, -2.3, 4.7)
        assert vector.x == 1.5
        assert vector.y == -2.3
        assert vector.z == 4.7
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_vector3d_iteration(self):
        """Test Vector3D iteration and unpacking."""
        from hierarchical.geometry import Vector3D
        
        vector = Vector3D(1.0, 2.0, 3.0)
        x, y, z = vector
        
        assert x == 1.0
        assert y == 2.0
        assert z == 3.0
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_vector3d_iteration_with_list(self):
        """Test Vector3D iteration with list conversion."""
        from hierarchical.geometry import Vector3D
        
        vector = Vector3D(4.5, -1.2, 0.8)
        coords = list(vector)
        
        assert coords == [4.5, -1.2, 0.8]
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_vector3d_as_tuple(self):
        """Test Vector3D as_tuple method."""
        from hierarchical.geometry import Vector3D
        
        vector = Vector3D(10.0, 20.0, 30.0)
        tuple_result = vector.as_tuple()
        
        assert tuple_result == (10.0, 20.0, 30.0)
        assert isinstance(tuple_result, tuple)
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_vector3d_as_array(self):
        """Test Vector3D as_array method."""
        from hierarchical.geometry import Vector3D
        
        vector = Vector3D(5.5, -3.3, 7.7)
        array_result = vector.as_array()
        
        expected = np.array([5.5, -3.3, 7.7])
        np.testing.assert_array_equal(array_result, expected)
        assert isinstance(array_result, np.ndarray)
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_vector3d_edge_cases(self):
        """Test Vector3D with edge case values."""
        from hierarchical.geometry import Vector3D
        
        # Test with zero values
        zero_vector = Vector3D(0.0, 0.0, 0.0)
        assert zero_vector.as_tuple() == (0.0, 0.0, 0.0)
        
        # Test with negative values
        neg_vector = Vector3D(-1.0, -2.0, -3.0)
        assert neg_vector.as_tuple() == (-1.0, -2.0, -3.0)
        
        # Test with very small values
        small_vector = Vector3D(1e-10, 1e-10, 1e-10)
        assert abs(small_vector.x - 1e-10) < 1e-15
        assert abs(small_vector.y - 1e-10) < 1e-15
        assert abs(small_vector.z - 1e-10) < 1e-15


class TestGeometryInitialization:
    """Test Geometry class initialization and basic properties."""
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_geometry_default_initialization(self):
        """Test Geometry initialization with default values."""
        from hierarchical.geometry import Geometry, Vector3D
        
        geometry = Geometry()
        
        assert geometry.sub_geometries == ()
        assert geometry.mesh_data == {}
        assert geometry.brep_data == {}
        assert geometry._opencascade_shape is None
        assert isinstance(geometry.origin, Vector3D)
        assert geometry.origin.as_tuple() == (0.0, 0.0, 0.0)
        assert geometry.transform is None
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_geometry_initialization_with_data(self):
        """Test Geometry initialization with mesh data."""
        from hierarchical.geometry import Geometry
        
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        faces = [(0, 1, 2)]
        mesh_data = {"vertices": vertices, "faces": faces}
        
        geometry = Geometry(mesh_data=mesh_data)
        
        assert geometry.mesh_data["vertices"] == vertices
        assert geometry.mesh_data["faces"] == faces
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_geometry_repr(self):
        """Test Geometry string representation."""
        from hierarchical.geometry import Geometry
        
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        faces = [(0, 1, 2)]
        mesh_data = {"vertices": vertices, "faces": faces}
        
        geometry = Geometry(mesh_data=mesh_data)
        repr_str = repr(geometry)
        
        assert "vertices=3" in repr_str
        assert "faces=1" in repr_str
        
        # Test empty geometry representation
        empty_geometry = Geometry()
        empty_repr = repr(empty_geometry)
        assert "vertices=0" in empty_repr
        assert "faces=0" in empty_repr


class TestGeometryFactoryMethods:
    """Test Geometry factory methods with mocked dependencies."""
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_from_primitive_box(self):
        """Test creating box geometry from primitive."""
        from hierarchical.geometry import Geometry
        
        dimensions = {'width': 2.0, 'depth': 3.0, 'height': 1.5}
        geometry = Geometry.from_primitive('box', dimensions)
        
        assert geometry is not None
        assert 'vertices' in geometry.mesh_data
        assert 'faces' in geometry.mesh_data
        
        # Should have 8 vertices for a box
        assert len(geometry.mesh_data['vertices']) == 8
        
        # Should have 12 triangular faces (6 rectangular faces × 2 triangles each)
        assert len(geometry.mesh_data['faces']) == 12
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_from_primitive_cylinder(self):
        """Test creating cylinder geometry from primitive."""
        from hierarchical.geometry import Geometry
        
        dimensions = {'radius': 1.0, 'height': 2.0, 'segments': 8}
        geometry = Geometry.from_primitive('cylinder', dimensions)
        
        assert geometry is not None
        assert 'vertices' in geometry.mesh_data
        assert 'faces' in geometry.mesh_data
        
        # Should have vertices: 2 centers + 2*segments rim points
        expected_vertices = 2 + (8 * 2)  # 18 vertices
        assert len(geometry.mesh_data['vertices']) == expected_vertices
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_from_prism(self):
        """Test creating prism geometry from base points."""
        from hierarchical.geometry import Geometry
        
        # Square base
        base_points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        height = 2.0
        
        geometry = Geometry.from_prism(base_points, height)
        
        assert geometry is not None
        assert 'vertices' in geometry.mesh_data
        assert 'faces' in geometry.mesh_data
        
        # Should have 8 vertices (4 bottom + 4 top)
        assert len(geometry.mesh_data['vertices']) == 8
        
        # Check that bottom and top vertices are correctly positioned
        vertices = geometry.mesh_data['vertices']
        
        # Bottom vertices should have z=0
        for i in range(4):
            assert vertices[i][2] == 0.0
        
        # Top vertices should have z=height
        for i in range(4, 8):
            assert vertices[i][2] == height
    
    @pytest.mark.unit
    @pytest.mark.geometry 
    def test_from_prism_triangle_base(self):
        """Test creating prism with triangular base."""
        from hierarchical.geometry import Geometry
        
        # Triangle base
        base_points = [(0, 0), (1, 0), (0.5, 1)]
        height = 1.5
        
        geometry = Geometry.from_prism(base_points, height)
        
        assert geometry is not None
        # Should have 6 vertices (3 bottom + 3 top)
        assert len(geometry.mesh_data['vertices']) == 6


class TestGeometryTransformations:
    """Test geometric transformations."""
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_transform_geometry_translation(self, unit_box_geometry):
        """Test applying translation transformation."""
        from hierarchical.geometry import Geometry
        
        # Create real geometry for transformation testing
        geometry = Geometry()
        vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        geometry.mesh_data = {"vertices": vertices, "faces": []}
        
        # Create translation matrix
        translation_matrix = np.array([
            [1, 0, 0, 2],
            [0, 1, 0, 3],
            [0, 0, 1, 1],
            [0, 0, 0, 1]
        ])
        
        geometry.transform_geometry(translation_matrix)
        
        # Check that transform matrix is stored
        np.testing.assert_array_equal(geometry.transform, translation_matrix)
        
        # Check that vertices are transformed
        transformed_vertices = geometry.mesh_data["vertices"]
        expected = [(2, 3, 1), (3, 3, 1), (3, 4, 1), (2, 4, 1)]
        
        for actual, expected_vertex in zip(transformed_vertices, expected):
            assert abs(actual[0] - expected_vertex[0]) < 1e-10
            assert abs(actual[1] - expected_vertex[1]) < 1e-10
            assert abs(actual[2] - expected_vertex[2]) < 1e-10
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_transform_geometry_rotation(self):
        """Test applying rotation transformation."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        vertices = [(1, 0, 0), (0, 1, 0)]
        geometry.mesh_data = {"vertices": vertices, "faces": []}
        
        # 90-degree rotation around Z-axis
        angle = np.pi / 2
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle),  np.cos(angle), 0, 0],
            [0,              0,             1, 0],
            [0,              0,             0, 1]
        ])
        
        geometry.transform_geometry(rotation_matrix)
        
        # First vertex (1,0,0) should become approximately (0,1,0)
        # Second vertex (0,1,0) should become approximately (-1,0,0)
        transformed = geometry.mesh_data["vertices"]
        
        assert abs(transformed[0][0] - 0.0) < 1e-10  # x close to 0
        assert abs(transformed[0][1] - 1.0) < 1e-10  # y close to 1
        assert abs(transformed[1][0] - (-1.0)) < 1e-10  # x close to -1
        assert abs(transformed[1][1] - 0.0) < 1e-10  # y close to 0
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_directional_movements(self):
        """Test directional movement methods."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        vertices = [(0, 0, 0), (1, 1, 1)]
        geometry.mesh_data = {"vertices": vertices, "faces": []}
        
        # Test right movement
        geometry.right(2.0)
        assert geometry.transform is not None
        
        # Reset and test forward movement
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertices, "faces": []}
        geometry.forward(1.5)
        assert geometry.transform is not None
        
        # Reset and test up movement
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertices, "faces": []}
        geometry.up(3.0)
        assert geometry.transform is not None
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_rotation_z(self):
        """Test Z-axis rotation method."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        vertices = [(1, 0, 0), (0, 1, 0)]
        geometry.mesh_data = {"vertices": vertices, "faces": []}
        
        # Rotate 90 degrees
        result = geometry.rotate_z(np.pi / 2)
        
        # Should return self for chaining
        assert result is geometry
        
        # Should have transform applied
        assert geometry.transform is not None
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_chained_transformations(self):
        """Test chaining multiple transformations."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        vertices = [(0, 0, 0)]
        geometry.mesh_data = {"vertices": vertices, "faces": []}
        
        # Chain multiple operations
        result = geometry.right(1.0).forward(2.0).up(3.0).rotate_z(np.pi/4)
        
        # Should return self for chaining
        assert result is geometry
        
        # Should have composed transformation
        assert geometry.transform is not None


class TestGeometryCalculations:
    """Test geometric calculations and properties."""
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_get_centroid_simple(self):
        """Test centroid calculation for simple geometry."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        vertices = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 2, 0)]
        geometry.mesh_data = {"vertices": vertices, "faces": []}
        
        centroid = geometry.get_centroid()
        
        # Centroid should be at (1, 1, 0)
        assert abs(centroid.x - 1.0) < 1e-10
        assert abs(centroid.y - 1.0) < 1e-10
        assert abs(centroid.z - 0.0) < 1e-10
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_get_centroid_empty_geometry(self):
        """Test centroid calculation for empty geometry."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        centroid = geometry.get_centroid()
        
        # Should return zero vector
        assert centroid.x == 0.0
        assert centroid.y == 0.0
        assert centroid.z == 0.0
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_get_vertices(self):
        """Test getting vertices from geometry."""
        from hierarchical.geometry import Geometry
        
        vertices = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertices, "faces": []}
        
        result_vertices = geometry.get_vertices()
        
        assert result_vertices == vertices
        assert len(result_vertices) == 3
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_get_faces(self):
        """Test getting faces from geometry."""
        from hierarchical.geometry import Geometry
        
        faces = [(0, 1, 2), (0, 2, 3), (1, 2, 3)]
        geometry = Geometry()
        geometry.mesh_data = {"vertices": [], "faces": faces}
        
        result_faces = geometry.get_faces()
        
        assert result_faces == faces
        assert len(result_faces) == 3
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_get_height(self):
        """Test height calculation."""
        from hierarchical.geometry import Geometry
        
        # Create geometry with vertices at different Z levels
        vertices = [(0, 0, 1), (1, 0, 1), (0, 1, 5), (1, 1, 5)]
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertices, "faces": []}
        
        height = geometry.get_height()
        
        # Height should be max_z - min_z = 5 - 1 = 4
        assert abs(height - 4.0) < 1e-10
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_get_bbox(self):
        """Test bounding box calculation."""
        from hierarchical.geometry import Geometry
        
        vertices = [(-1, -2, -3), (2, 3, 4), (0, 1, 0)]
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertices, "faces": []}
        
        min_point, max_point = geometry.get_bbox()
        
        expected_min = np.array([-1, -2, -3])
        expected_max = np.array([2, 3, 4])
        
        np.testing.assert_array_equal(min_point, expected_min)
        np.testing.assert_array_equal(max_point, expected_max)
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_compute_volume_simple(self):
        """Test volume computation for simple geometry."""
        from hierarchical.geometry import Geometry
        
        # Create a simple tetrahedron
        vertices = [(0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0.5, 0.5, 1)]
        faces = [(0, 1, 2), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
        
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertices, "faces": faces}
        
        volume = geometry.compute_volume()
        
        # Should return a positive volume
        assert volume > 0
        assert isinstance(volume, float)
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_compute_volume_empty(self):
        """Test volume computation for empty geometry."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        volume = geometry.compute_volume()
        
        assert volume == 0.0


class TestGeometryIntersections:
    """Test geometric intersection calculations with mocked dependencies."""
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_bbox_intersects_overlapping(self):
        """Test bounding box intersection for overlapping boxes."""
        from hierarchical.geometry import Geometry
        
        # Create two overlapping geometries
        geom1 = Geometry()
        geom1.mesh_data = {"vertices": [(0, 0, 0), (2, 2, 2)], "faces": []}
        
        geom2 = Geometry()  
        geom2.mesh_data = {"vertices": [(1, 1, 1), (3, 3, 3)], "faces": []}
        
        # Should intersect
        assert geom1.bbox_intersects(geom2) is True
        
        # Test with overlap percentage
        overlap_percent = geom1.bbox_intersects(geom2, return_overlap_percent=True)
        assert overlap_percent > 0
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_bbox_intersects_non_overlapping(self):
        """Test bounding box intersection for non-overlapping boxes."""
        from hierarchical.geometry import Geometry
        
        # Create two non-overlapping geometries
        geom1 = Geometry()
        geom1.mesh_data = {"vertices": [(0, 0, 0), (1, 1, 1)], "faces": []}
        
        geom2 = Geometry()
        geom2.mesh_data = {"vertices": [(2, 2, 2), (3, 3, 3)], "faces": []}
        
        # Should not intersect
        assert geom1.bbox_intersects(geom2) is False
        
        # Test with overlap percentage
        overlap_percent = geom1.bbox_intersects(geom2, return_overlap_percent=True)
        assert overlap_percent == 0.0
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_distance_to(self):
        """Test distance calculation between geometries."""
        from hierarchical.geometry import Geometry
        
        # Create two geometries with known separation
        geom1 = Geometry()
        geom1.mesh_data = {"vertices": [(0, 0, 0), (1, 1, 1)], "faces": []}
        
        geom2 = Geometry()
        geom2.mesh_data = {"vertices": [(3, 3, 3), (4, 4, 4)], "faces": []}
        
        distance = geom1.distance_to(geom2)
        
        # Should return a positive distance
        assert distance > 0
        assert isinstance(distance, float)
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_mesh_intersects_with_mock_trimesh(self, mock_trimesh):
        """Test mesh intersection with mocked trimesh."""
        from hierarchical.geometry import Geometry
        
        # Create geometries
        geom1 = Geometry()
        geom1.mesh_data = {"vertices": [(0, 0, 0), (1, 0, 0), (0, 1, 0)], "faces": [(0, 1, 2)]}
        
        geom2 = Geometry()
        geom2.mesh_data = {"vertices": [(0.5, 0.5, 0), (1.5, 0.5, 0), (0.5, 1.5, 0)], "faces": [(0, 1, 2)]}
        
        # Mock trimesh import inside the method
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: mock_trimesh if name == 'trimesh' else __import__(name, *args, **kwargs)):
            result = geom1.mesh_intersects(geom2)
            
            # Should return the mocked result
            assert result is True
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_mesh_intersects_fallback_to_bbox(self):
        """Test mesh intersection fallback when trimesh is not available."""
        from hierarchical.geometry import Geometry
        
        geom1 = Geometry()
        geom1.mesh_data = {"vertices": [(0, 0, 0), (2, 2, 2)], "faces": []}
        
        geom2 = Geometry()
        geom2.mesh_data = {"vertices": [(1, 1, 1), (3, 3, 3)], "faces": []}
        
        # Mock trimesh import to fail
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: exec('raise ImportError()') if name == 'trimesh' else __import__(name, *args, **kwargs)):
            result = geom1.mesh_intersects(geom2)
            
            # Should fall back to bbox intersection
            assert isinstance(result, bool)


class TestGeometryPrivateMethods:
    """Test private/internal geometry methods."""
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_generate_brep_from_mesh(self):
        """Test B-rep generation from mesh data."""
        from hierarchical.geometry import Geometry
        
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        faces = [(0, 1, 2)]
        
        geometry = Geometry()
        geometry.mesh_data = {"vertices": vertices, "faces": faces}
        geometry._generate_brep_from_mesh()
        
        # Should populate brep_data
        assert 'vertices' in geometry.brep_data
        assert 'edges' in geometry.brep_data
        assert 'surfaces' in geometry.brep_data
        
        # Should have correct data
        assert geometry.brep_data['vertices'] == vertices
        assert len(geometry.brep_data['surfaces']) == 1
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_to_trimesh_success(self, mock_trimesh):
        """Test successful trimesh conversion."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        geometry.mesh_data = {
            "vertices": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            "faces": [(0, 1, 2)]
        }
        
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: mock_trimesh if name == 'trimesh' else __import__(name, *args, **kwargs)):
            result = geometry._to_trimesh()
            
            # Should return the mocked trimesh object
            assert result is not None
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_to_trimesh_import_error(self):
        """Test trimesh conversion when trimesh is not available."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        geometry.mesh_data = {
            "vertices": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            "faces": [(0, 1, 2)]
        }
        
        def mock_import(name, *args, **kwargs):
            if name == 'trimesh':
                raise ImportError("No module named 'trimesh'")
            return __import__(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = geometry._to_trimesh()
            
            # Should return None when trimesh is not available
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_to_trimesh_empty_data(self):
        """Test trimesh conversion with empty geometry data."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()  # Empty geometry
        result = geometry._to_trimesh()
        
        # Should return None for empty geometry
        assert result is None


class TestGeometryEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_empty_geometry_operations(self):
        """Test operations on empty geometry."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        
        # All operations should handle empty geometry gracefully
        assert geometry.compute_volume() == 0.0
        assert geometry.get_height() == 0.0
        
        centroid = geometry.get_centroid()
        assert centroid.as_tuple() == (0.0, 0.0, 0.0)
        
        min_point, max_point = geometry.get_bbox()
        np.testing.assert_array_equal(min_point, np.zeros(3))
        np.testing.assert_array_equal(max_point, np.zeros(3))
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_single_vertex_geometry(self):
        """Test geometry with only one vertex."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        geometry.mesh_data = {"vertices": [(1, 2, 3)], "faces": []}
        
        centroid = geometry.get_centroid()
        assert centroid.as_tuple() == (1.0, 2.0, 3.0)
        
        assert geometry.get_height() == 0.0  # No height with single point
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_malformed_mesh_data(self):
        """Test handling of malformed mesh data."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        
        # Missing vertices key
        geometry.mesh_data = {"faces": [(0, 1, 2)]}
        assert geometry.compute_volume() == 0.0
        
        # Missing faces key
        geometry.mesh_data = {"vertices": [(0, 0, 0)]}
        assert geometry.compute_volume() == 0.0
        
        # Empty lists
        geometry.mesh_data = {"vertices": [], "faces": []}
        assert geometry.compute_volume() == 0.0
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_invalid_face_indices(self):
        """Test handling of invalid face indices."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        # Face indices that exceed vertex count
        geometry.mesh_data = {
            "vertices": [(0, 0, 0), (1, 0, 0)],
            "faces": [(0, 1, 5)]  # Index 5 doesn't exist
        }
        
        # Should raise IndexError for invalid indices (this is expected behavior)
        # The geometry module doesn't currently handle this gracefully
        with pytest.raises(IndexError):
            geometry.compute_volume()
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_degenerate_faces(self):
        """Test handling of degenerate faces."""
        from hierarchical.geometry import Geometry
        
        geometry = Geometry()
        # Degenerate triangle (all points are the same)
        geometry.mesh_data = {
            "vertices": [(0, 0, 0), (0, 0, 0), (0, 0, 0)],
            "faces": [(0, 1, 2)]
        }
        
        volume = geometry.compute_volume()
        assert volume >= 0.0  # Should not be negative