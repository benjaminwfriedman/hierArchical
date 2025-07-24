"""
Test Infrastructure Validation

This module contains tests to verify that the testing infrastructure
itself is working correctly, including mocking, fixtures, and test doubles.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from tests.conftest import (
    MockGeometry, MockVertex, MockFace,
    create_test_element, create_test_component,
    assert_geometries_equal, assert_volumes_equal
)
from tests.fixtures.sample_geometries import (
    UNIT_CUBE, SIMPLE_TRIANGLE, get_geometry_by_name,
    create_mock_geometry_from_data
)


class TestMockingFramework:
    """Test that all mocking is working correctly."""
    
    @pytest.mark.unit
    def test_heavy_dependencies_are_mocked(self, mock_heavy_dependencies):
        """Verify all heavy dependencies are properly mocked."""
        mocks = mock_heavy_dependencies
        
        # Verify OpenCascade mocks
        assert 'topology' in mocks
        assert 'cell' in mocks
        assert 'vertex' in mocks
        assert 'edge' in mocks
        assert 'face' in mocks
        
        # Verify visualization mocks
        assert 'plotly_go' in mocks
        assert 'plotly_px' in mocks
        
        # Verify other dependency mocks
        assert 'trimesh' in mocks
        assert 'ifcopenshell' in mocks
        assert 'matplotlib' in mocks
    
    @pytest.mark.unit
    def test_plotly_mock_functionality(self, mock_plotly):
        """Test that plotly mocking works correctly."""
        fig = mock_plotly.Figure()
        assert fig is not None
        
        mesh = mock_plotly.Mesh3d()
        assert mesh is not None
        
        scatter = mock_plotly.Scatter3d()
        assert scatter is not None
    
    @pytest.mark.unit
    def test_trimesh_mock_functionality(self, mock_trimesh):
        """Test that trimesh mocking works correctly."""
        mesh = mock_trimesh.Trimesh(vertices=[], faces=[])
        
        assert mesh.volume == 1.0
        assert mesh.is_watertight is True
        assert mesh.intersects_mesh(mesh) is True
    
    @pytest.mark.unit
    def test_opencascade_mock_functionality(self, mock_opencascade):
        """Test that OpenCascade mocking works correctly."""
        analyzer = mock_opencascade['analyzer']()
        assert analyzer.IsValid() is True
        
        sewing = mock_opencascade['sewing']()
        shape = sewing.SewedShape()
        assert shape is not None


class TestLightweightGeometryDoubles:
    """Test that geometry test doubles work correctly."""
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_mock_vertex_creation(self):
        """Test MockVertex functionality."""
        vertex = MockVertex(1.0, 2.0, 3.0)
        
        assert vertex.x == 1.0
        assert vertex.y == 2.0
        assert vertex.z == 3.0
        assert vertex.as_tuple() == (1.0, 2.0, 3.0)
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_mock_face_creation(self):
        """Test MockFace functionality."""
        face = MockFace((0, 1, 2))
        
        assert face.indices == (0, 1, 2)
        assert len(face) == 3
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_mock_geometry_creation(self):
        """Test MockGeometry basic functionality."""
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        faces = [(0, 1, 2)]
        
        geometry = MockGeometry(vertices, faces)
        
        assert geometry.mesh_data["vertices"] == vertices
        assert geometry.mesh_data["faces"] == faces
        # Test that new representation fields exist and are initialized properly
        assert hasattr(geometry, '_mesh_data')
        assert hasattr(geometry, '_opencascade_shape')
        assert hasattr(geometry, '_topologic_topology')
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_mock_geometry_volume_calculation(self, unit_box_geometry):
        """Test volume calculation in MockGeometry."""
        volume = unit_box_geometry.compute_volume()
        
        # Volume should be based on vertex count for testing
        expected_volume = len(unit_box_geometry.mesh_data["vertices"]) * 0.1
        assert volume == expected_volume
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_mock_geometry_centroid_calculation(self, triangle_geometry):
        """Test centroid calculation in MockGeometry."""
        centroid = triangle_geometry.get_centroid()
        
        # Should calculate actual centroid
        expected_x = (0 + 1 + 0.5) / 3
        expected_y = (0 + 0 + 1) / 3
        expected_z = 0
        
        assert abs(centroid.x - expected_x) < 1e-6
        assert abs(centroid.y - expected_y) < 1e-6
        assert abs(centroid.z - expected_z) < 1e-6
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_mock_geometry_bbox_calculation(self, unit_box_geometry):
        """Test bounding box calculation in MockGeometry."""
        min_point, max_point = unit_box_geometry.get_bbox()
        
        expected_min = np.array([0, 0, 0])
        expected_max = np.array([1, 1, 1])
        
        np.testing.assert_array_equal(min_point, expected_min)
        np.testing.assert_array_equal(max_point, expected_max)
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_mock_geometry_transformation(self, unit_box_geometry):
        """Test transformation functionality in MockGeometry."""
        # Test translation
        unit_box_geometry.move(1.0, 2.0, 3.0)
        
        # Should have transform matrix set
        assert unit_box_geometry.transform is not None
        
        # Translation matrix should be correctly formed
        expected_translation = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ])
        
        np.testing.assert_array_equal(unit_box_geometry.transform, expected_translation)
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_mock_geometry_rotation(self, unit_box_geometry):
        """Test rotation functionality in MockGeometry."""
        angle = np.pi / 4  # 45 degrees
        unit_box_geometry.rotate_z(angle)
        
        # Should have transform matrix set
        assert unit_box_geometry.transform is not None
        
        # Check that rotation matrix has correct structure
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        expected_rotation = np.array([
            [cos_theta, -sin_theta, 0, 0],
            [sin_theta,  cos_theta, 0, 0],
            [0,          0,         1, 0],
            [0,          0,         0, 1]
        ])
        
        np.testing.assert_array_almost_equal(unit_box_geometry.transform, expected_rotation)


class TestFixturesAndTestData:
    """Test that fixtures and test data are working correctly."""
    
    @pytest.mark.unit
    def test_standard_fixtures_available(self, unit_box_geometry, triangle_geometry, 
                                       empty_geometry, complex_geometry):
        """Test that standard geometry fixtures are available."""
        assert unit_box_geometry is not None
        assert triangle_geometry is not None
        assert empty_geometry is not None
        assert complex_geometry is not None
    
    @pytest.mark.unit
    def test_tolerance_fixtures_available(self, geometric_tolerance, volume_tolerance, 
                                        angle_tolerance):
        """Test that tolerance fixtures are available."""
        assert geometric_tolerance == 1e-6
        assert volume_tolerance == 1e-10
        assert angle_tolerance == 1e-8
    
    @pytest.mark.unit
    def test_material_fixtures_available(self, sample_material_data):
        """Test that material fixtures are available."""
        assert "concrete" in sample_material_data
        assert "steel" in sample_material_data
        assert "wood" in sample_material_data
        
        # Check structure
        for material, data in sample_material_data.items():
            assert "volume" in data
            assert "percent" in data
    
    @pytest.mark.unit
    def test_sample_geometry_data_loading(self):
        """Test that sample geometry data loads correctly."""
        unit_cube = get_geometry_by_name("unit_cube")
        
        assert unit_cube.name == "unit_cube"
        assert len(unit_cube.vertices) == 8  # Cube has 8 vertices
        assert unit_cube.expected_volume == 1.0
        assert unit_cube.expected_centroid == (0.5, 0.5, 0.5)
    
    @pytest.mark.unit
    def test_mock_geometry_creation_from_data(self):
        """Test creating MockGeometry from test data."""
        unit_cube_data = get_geometry_by_name("unit_cube")
        mock_geometry = create_mock_geometry_from_data(unit_cube_data)
        
        assert len(mock_geometry.mesh_data["vertices"]) == 8
        assert len(mock_geometry.mesh_data["faces"]) == 12  # 12 triangular faces


class TestUtilityFunctions:
    """Test utility functions for testing."""
    
    @pytest.mark.unit
    def test_create_test_element(self):
        """Test creation of test elements."""
        element = create_test_element("test_element", "wood")
        
        assert element.name == "test_element"
        assert element.material == "wood"
        assert element.geometry is not None
    
    @pytest.mark.unit
    def test_create_test_component(self):
        """Test creation of test components."""
        component = create_test_component("test_component", 3)
        
        assert component.name == "test_component"
        assert len(component.sub_items) == 3
    
    @pytest.mark.unit
    def test_assert_geometries_equal(self):
        """Test geometry equality assertion."""
        vertices1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        faces1 = [(0, 1, 2)]
        
        vertices2 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        faces2 = [(0, 1, 2)]
        
        geo1 = MockGeometry(vertices1, faces1)
        geo2 = MockGeometry(vertices2, faces2)
        
        # Should not raise
        assert_geometries_equal(geo1, geo2)
    
    @pytest.mark.unit
    def test_assert_volumes_equal(self):
        """Test volume equality assertion."""
        vol1 = 1.0
        vol2 = 1.0000000001  # Within tolerance
        
        # Should not raise
        assert_volumes_equal(vol1, vol2, tolerance=1e-8)
        
        # Should raise
        with pytest.raises(AssertionError):
            assert_volumes_equal(vol1, 2.0, tolerance=1e-8)


class TestHypothesisStrategies:
    """Test that Hypothesis strategies are working."""
    
    @pytest.mark.unit
    @pytest.mark.property
    def test_coordinate_strategy_available(self, coordinate_strategy):
        """Test that coordinate strategy is available."""
        assert coordinate_strategy is not None
        
        # Test generating a value
        from hypothesis import strategies as st
        example = coordinate_strategy.example()
        assert isinstance(example, float)
        assert -1000 <= example <= 1000
    
    @pytest.mark.unit
    @pytest.mark.property
    def test_dimension_strategy_available(self, dimension_strategy):
        """Test that dimension strategy is available."""
        assert dimension_strategy is not None
        
        example = dimension_strategy.example()
        assert isinstance(example, float)
        assert 0.1 <= example <= 1000
    
    @pytest.mark.unit
    @pytest.mark.property
    def test_angle_strategy_available(self, angle_strategy):
        """Test that angle strategy is available."""
        assert angle_strategy is not None
        
        example = angle_strategy.example()
        assert isinstance(example, float)
        assert 0 <= example <= 2 * np.pi


class TestTestConfiguration:
    """Test that pytest configuration is working correctly."""
    
    @pytest.mark.unit
    def test_markers_are_working(self, request):
        """Test that custom markers are working."""
        # This test has the 'unit' marker
        assert any(mark.name == 'unit' for mark in request.node.iter_markers())
    
    @pytest.mark.unit
    @pytest.mark.geometry
    def test_multiple_markers_work(self, request):
        """Test that multiple markers work on the same test."""
        marker_names = [mark.name for mark in request.node.iter_markers()]
        assert 'unit' in marker_names
        assert 'geometry' in marker_names
    
    @pytest.mark.unit
    def test_pytest_plugins_loaded(self):
        """Test that required pytest plugins are loaded."""
        import pytest_mock
        import pytest_cov
        
        # If we can import them, they're available
        assert pytest_mock is not None
        assert pytest_cov is not None