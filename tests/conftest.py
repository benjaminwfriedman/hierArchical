"""
Shared pytest fixtures and configuration for hierArchical testing.

This module provides comprehensive mocking for all heavy dependencies
and creates lightweight test doubles for fast, isolated unit testing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import math


# =============================================================================
# COMPREHENSIVE MOCKING FIXTURES
# =============================================================================

@pytest.fixture
def mock_heavy_dependencies():
    """
    Mock heavy dependencies for tests that need them.
    This fixture is NOT auto-used, so tests must explicitly request it.
    """
    mocks = {}
    
    # Create simple mock objects instead of trying to patch non-existent modules
    mocks['topology'] = Mock()
    mocks['cell'] = Mock()
    mocks['vertex'] = Mock()
    mocks['edge'] = Mock()
    mocks['face'] = Mock()
    mocks['plotly_go'] = Mock()
    mocks['plotly_px'] = Mock()
    mocks['plotly_pio'] = Mock()
    mocks['trimesh'] = Mock()
    mocks['ifcopenshell'] = Mock()
    mocks['matplotlib'] = Mock()
    
    yield mocks


@pytest.fixture
def mock_plotly():
    """Mock plotly for visualization tests."""
    mock_go = Mock()
    mock_fig = Mock()
    mock_go.Figure.return_value = mock_fig
    mock_go.Mesh3d.return_value = Mock()
    mock_go.Scatter3d.return_value = Mock()
    return mock_go


@pytest.fixture
def mock_trimesh():
    """Mock trimesh for geometry intersection tests."""
    mock = Mock()
    mock_mesh = Mock()
    mock_mesh.volume = 1.0
    mock_mesh.is_watertight = True
    # Mock trimesh intersection methods don't exist in real trimesh
    mock_mesh.intersection.return_value = mock_mesh
    mock.Trimesh.return_value = mock_mesh
    return mock


@pytest.fixture
def mock_opencascade():
    """Mock OpenCascade components for geometry tests."""
    mocks = {}
    
    # Create mock objects that behave like OpenCascade components
    mock_analyzer = Mock()
    mock_analyzer.return_value.IsValid.return_value = True
    mocks['analyzer'] = mock_analyzer
    
    mock_sewing = Mock()
    mock_sewing.return_value.SewedShape.return_value = Mock()
    mocks['sewing'] = mock_sewing
    
    return mocks


# =============================================================================
# LIGHTWEIGHT GEOMETRY TEST DOUBLES
# =============================================================================

@dataclass
class MockVertex:
    """Lightweight vertex for testing."""
    x: float
    y: float
    z: float
    
    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class MockFace:
    """Lightweight face for testing."""
    indices: Tuple[int, ...]
    
    def __len__(self):
        return len(self.indices)


class MockGeometry:
    """Lightweight geometry double for fast testing."""
    
    def __init__(self, vertices: List[Tuple[float, float, float]] = None, 
                 faces: List[Tuple[int, ...]] = None):
        self.mesh_data = {
            "vertices": vertices or [],
            "faces": faces or []
        }
        # Initialize new representation fields for mock
        self._mesh_data = None
        self._opencascade_shape = None  
        self._topologic_topology = None
        self._mesh_generated = False
        self._occ_generated = False
        self._topologic_generated = False
        self.origin = MockVertex(0, 0, 0)
        self.transform = None
        self.sub_geometries = ()
    
    @property
    def mesh(self):
        """Mock mesh property for compatibility with new geometry system"""
        return self.mesh_data
    
    @property  
    def opencascade(self):
        """Mock opencascade property"""
        return None
        
    @property
    def topologic(self):
        """Mock topologic property"""
        return None
    
    def compute_volume(self) -> float:
        """Simple volume calculation for testing."""
        if not self.mesh_data.get("vertices") or not self.mesh_data.get("faces"):
            return 0.0
        # Return a deterministic volume based on vertex count for testing
        return float(len(self.mesh_data["vertices"]) * 0.1)
    
    def get_centroid(self) -> MockVertex:
        """Calculate centroid for testing."""
        vertices = self.mesh_data.get("vertices", [])
        if not vertices:
            return MockVertex(0, 0, 0)
        
        x = sum(v[0] for v in vertices) / len(vertices)
        y = sum(v[1] for v in vertices) / len(vertices)
        z = sum(v[2] for v in vertices) / len(vertices)
        return MockVertex(x, y, z)
    
    def get_bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box for testing."""
        vertices = self.mesh_data.get("vertices", [])
        if not vertices:
            return np.zeros(3), np.zeros(3)
        
        vertices_array = np.array(vertices)
        return vertices_array.min(axis=0), vertices_array.max(axis=0)
    
    def transform_geometry(self, matrix: np.ndarray):
        """Mock transformation for testing."""
        if self.transform is None:
            self.transform = matrix.copy()
        else:
            self.transform = np.matmul(matrix, self.transform)
    
    def move(self, dx: float, dy: float, dz: float):
        """Mock movement for testing."""
        translation = np.eye(4)
        translation[:3, 3] = [dx, dy, dz]
        self.transform_geometry(translation)
        return self
    
    def rotate_z(self, angle_rad: float):
        """Mock rotation for testing."""
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


# =============================================================================
# GEOMETRY FIXTURES
# =============================================================================

@pytest.fixture
def unit_box_geometry():
    """Create a simple 1x1x1 box geometry for testing."""
    vertices = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),  # bottom
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)   # top
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),  # bottom
        (4, 6, 5), (4, 7, 6),  # top
        (0, 4, 1), (1, 4, 5),  # front
        (1, 5, 2), (2, 5, 6),  # right
        (2, 6, 3), (3, 6, 7),  # back
        (3, 7, 0), (0, 7, 4)   # left
    ]
    return MockGeometry(vertices, faces)


@pytest.fixture
def triangle_geometry():
    """Create a simple triangle geometry for testing."""
    vertices = [(0, 0, 0), (1, 0, 0), (0.5, 1, 0)]
    faces = [(0, 1, 2)]
    return MockGeometry(vertices, faces)


@pytest.fixture
def empty_geometry():
    """Create an empty geometry for edge case testing."""
    return MockGeometry()


@pytest.fixture
def complex_geometry():
    """Create a more complex geometry for advanced testing."""
    # Create a simple pyramid
    vertices = [
        (0, 0, 0), (1, 0, 0), (0.5, 1, 0),  # base triangle
        (0.5, 0.5, 1)  # apex
    ]
    faces = [
        (0, 1, 2),  # base
        (0, 1, 3),  # side 1
        (1, 2, 3),  # side 2
        (2, 0, 3)   # side 3
    ]
    return MockGeometry(vertices, faces)


# =============================================================================
# MATERIAL AND ITEM FIXTURES
# =============================================================================

@pytest.fixture
def sample_material_data():
    """Sample material data for testing."""
    return {
        "concrete": {"volume": 2.5, "percent": 50.0},
        "steel": {"volume": 1.0, "percent": 20.0},
        "wood": {"volume": 1.5, "percent": 30.0}
    }


@pytest.fixture
def sample_attributes():
    """Sample attributes for testing."""
    return {
        "length": 10.0,
        "width": 5.0,
        "height": 3.0,
        "area": 50.0,
        "volume": 150.0
    }


@pytest.fixture
def sample_ontologies():
    """Sample ontologies for testing."""
    return {
        "structural": True,
        "load_bearing": True,
        "fire_rating": "2hr",
        "zone": "residential"
    }


# =============================================================================
# TOLERANCE AND PRECISION FIXTURES
# =============================================================================

@pytest.fixture
def geometric_tolerance():
    """Standard geometric tolerance for tests."""
    return 1e-6


@pytest.fixture
def volume_tolerance():
    """Volume calculation tolerance for tests."""
    return 1e-10


@pytest.fixture
def angle_tolerance():
    """Angle tolerance in radians for tests."""
    return 1e-8


# =============================================================================
# HYPOTHESIS STRATEGIES
# =============================================================================

@pytest.fixture
def coordinate_strategy():
    """Hypothesis strategy for coordinates."""
    from hypothesis import strategies as st
    return st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)


@pytest.fixture
def dimension_strategy():
    """Hypothesis strategy for positive dimensions."""
    from hypothesis import strategies as st
    return st.floats(min_value=0.1, max_value=1000, allow_nan=False, allow_infinity=False)


@pytest.fixture
def angle_strategy():
    """Hypothesis strategy for angles in radians."""
    from hypothesis import strategies as st
    return st.floats(min_value=0, max_value=2*math.pi, allow_nan=False, allow_infinity=False)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_test_element(name: str = "test_element", material: str = "test_material"):
    """Create a test element with mocked dependencies."""
    # Create with minimal geometry to avoid OpenCascade calls
    geometry = MockGeometry([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
    
    # Create a mock element instead of importing actual Element
    mock_element = Mock()
    mock_element.name = name
    mock_element.material = material
    mock_element.geometry = geometry
    mock_element.type = "test_element"
    
    return mock_element


def create_test_component(name: str = "test_component", num_elements: int = 3):
    """Create a test component with mocked elements."""
    elements = tuple(
        create_test_element(f"element_{i}", f"material_{i}") 
        for i in range(num_elements)
    )
    
    # Create a mock component instead of importing actual Component
    mock_component = Mock()
    mock_component.name = name
    mock_component.sub_items = elements
    mock_component.type = "test_component"
    
    return mock_component


def assert_geometries_equal(geo1: MockGeometry, geo2: MockGeometry, tolerance: float = 1e-6):
    """Assert that two geometries are equal within tolerance."""
    assert len(geo1.mesh_data["vertices"]) == len(geo2.mesh_data["vertices"])
    assert len(geo1.mesh_data["faces"]) == len(geo2.mesh_data["faces"])
    
    for v1, v2 in zip(geo1.mesh_data["vertices"], geo2.mesh_data["vertices"]):
        for c1, c2 in zip(v1, v2):
            assert abs(c1 - c2) < tolerance


def assert_volumes_equal(volume1: float, volume2: float, tolerance: float = 1e-10):
    """Assert that two volumes are equal within tolerance."""
    assert abs(volume1 - volume2) < tolerance, f"Volumes {volume1} and {volume2} differ by more than {tolerance}"


# =============================================================================
# TEST MARKERS AND CONFIGURATION
# =============================================================================

# Add custom markers for test categorization
pytest_plugins = ["pytest_mock"]

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "property: Property-based tests") 
    config.addinivalue_line("markers", "geometry: Geometry-related tests")
    config.addinivalue_line("markers", "relationships: Relationship system tests")
    config.addinivalue_line("markers", "materials: Material and unit system tests")
    config.addinivalue_line("markers", "helpers: Helper function tests")