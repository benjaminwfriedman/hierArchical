# High-Quality Unit Testing Work Plan for hierArchical

## Project Overview
This document outlines a focused plan for implementing high-quality **unit tests and property-based tests** for the hierArchical architectural data modeling framework. The testing strategy emphasizes thorough coverage of core functionality, maintainable code, and robust geometric validation through isolated testing.

**Scope:** This plan focuses exclusively on unit and property tests. Integration tests and performance testing will be addressed in a separate phase once the unit test foundation is established.

## Current Status Analysis
- **No existing test structure** - starting from scratch
- **Complex geometry operations** requiring specialized test approaches  
- **Heavy dependencies** (OpenCascade, trimesh, plotly) need comprehensive mocking
- **Hierarchical data structures** require systematic fixture management

## Phase 1: Test Infrastructure Setup (Week 1)

### 1.1 Framework Selection & Configuration
**Recommended Testing Stack:**
- `pytest`: Advanced test discovery, fixtures, parametrization
- `pytest-cov`: Coverage reporting
- `pytest-mock`: Mocking capabilities  
- `hypothesis`: Property-based testing for geometry
- `hypothesis`: Property-based testing for geometric invariants

**Installation:**
```bash
pip install pytest pytest-cov pytest-mock hypothesis
```

### 1.2 Test Directory Structure
```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── fixtures/                   # Test data and sample geometries
│   ├── sample_geometries.py   # Geometry test fixtures
│   └── reference_data.json    # Expected results
├── unit/                      # Unit tests by module
│   ├── test_geometry.py
│   ├── test_items.py
│   ├── test_relationships.py
│   ├── test_units.py
│   └── test_helpers.py
└── property/                  # Property-based tests
    ├── test_geometry_properties.py
    ├── test_transformation_invariants.py
    └── test_material_properties.py
```

### 1.3 Dependency Management & Mocking Strategy
- **Mock all heavy dependencies** (OpenCascade, trimesh, plotly, ifcopenshell) for unit tests
- **Create lightweight geometry test doubles** for fast, isolated testing
- **Focus on testing logic, not external library behavior**

**Example Mock Strategy:**
```python
# Mock expensive operations
@pytest.fixture
def mock_opencascade():
    with patch('hierarchical.geometry.OpenCascade') as mock:
        yield mock

# Use lightweight test doubles
@pytest.fixture
def simple_geometry():
    return SimpleGeometry(vertices=[(0,0,0), (1,0,0), (1,1,0)], 
                         faces=[(0,1,2)])
```

## Phase 2: Core Geometry Tests (Week 2)

### 2.1 Vector3D Class Tests (`test_geometry.py`)
```python
class TestVector3D:
    def test_initialization()
    def test_iteration_unpacking()
    def test_as_tuple_conversion()
    def test_as_array_conversion()
    def test_arithmetic_operations()  # if implemented
```

### 2.2 Geometry Class Tests
```python
class TestGeometry:
    def test_initialization()
    def test_from_obj_import()
    def test_from_primitive_creation()
    def test_from_prism_creation()
    def test_mesh_data_generation()
    def test_brep_data_generation()
    def test_transformation_matrices()
    def test_directional_movements()
    def test_rotation_operations()
    def test_centroid_calculation()
    def test_bbox_calculations()
    def test_volume_computation()
    def test_intersection_detection()
    def test_distance_calculations()
```

### 2.3 Property-Based Testing
Use Hypothesis for testing geometric invariants:
- **Transformation invariants**: Volume preservation, centroid behavior
- **Geometric constraints**: Bounding box containment, face count consistency
- **Numerical stability**: Float precision handling

**Example Property Test:**
```python
from hypothesis import given, strategies as st

@given(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
def test_translation_preserves_volume(dx, dy, dz):
    geometry = create_test_box()
    original_volume = geometry.compute_volume()
    geometry.move(dx, dy, dz)
    assert abs(geometry.compute_volume() - original_volume) < 1e-10

@given(st.floats(min_value=0, max_value=2*math.pi, allow_nan=False))
def test_rotation_preserves_volume(angle):
    geometry = create_test_box()
    original_volume = geometry.compute_volume()
    geometry.rotate_z(angle)
    assert abs(geometry.compute_volume() - original_volume) < 1e-10
```

## Phase 3: Item Hierarchy Tests (Week 3)

### 3.1 BaseItem Tests (`test_items.py`)
```python
class TestBaseItem:
    def test_initialization()
    def test_geometry_combination()
    def test_transformation_propagation()
    def test_unit_conversion()
    def test_spatial_relationship_methods()
    def test_copy_operations()
    def test_material_aggregation()
```

### 3.2 Element, Component, Object Tests
```python
class TestElement:
    def test_material_assignment()
    def test_volume_calculation()

class TestComponent:
    def test_from_elements_creation()
    def test_material_aggregation()

class TestObject:
    def test_from_components_creation()
    # Note: IFC import testing will be covered in integration tests
```

### 3.3 Specialized Object Tests
```python
class TestWall:
    def test_center_plane_calculation()
    def test_edge_detection()
    def test_normal_vector_computation()
    def test_pca_analysis()

class TestDoor:
    # Note: IFC-related tests will be covered in integration tests
    def test_swing_direction_assignment()
    def test_panel_position_assignment()
```

## Phase 4: Relationship System Tests (Week 4)

### 4.1 Relationship Classes (`test_relationships.py`)
```python
class TestRelationships:
    def test_relationship_creation()
    def test_bidirectional_relationships()
    def test_relationship_attributes()
    def test_graph_conversion()
```

### 4.2 Spatial Analysis Tests (Unit Level)
```python
class TestSpatialAnalysis:
    def test_intersection_detection_with_mocked_geometry()
    def test_adjacency_calculation_with_known_coordinates()
    def test_embedding_relationships_logic()
    def test_tolerance_handling_edge_cases()
```

## Phase 5: Unit System & Material Tests (Week 5)

### 5.1 Unit Conversion Tests (`test_units.py`)
```python
class TestUnitConversion:
    def test_metric_conversions()
    def test_imperial_conversions()
    def test_mixed_unit_operations()
    def test_dimension_scaling()
    def test_volume_area_conversions()
```

### 5.2 Material System Tests
```python
class TestMaterials:
    def test_material_assignment()
    def test_volume_aggregation()
    def test_percentage_calculations()
    def test_bom_generation()
```

## Phase 6: Utility & Visualization Tests (Week 6)

### 6.1 Helper Function Tests (`test_helpers.py`)
```python
class TestHelpers:
    def test_id_generation()
    def test_color_generation()
    def test_ifc_normalization()
    def test_boundary_healing_validation()
```

### 6.2 Visualization Tests (`test_utils.py`)
```python
class TestVisualization:
    def test_plot_items_data_preparation()  # Mock plotly
    def test_bom_calculation()
    def test_element_collection()
    def test_shapely_conversion()
```

## Phase 7: Property-Based Test Expansion (Week 7)

### 7.1 Advanced Property Tests
```python
class TestGeometricInvariants:
    def test_transformation_composition()
    def test_material_conservation()
    def test_hierarchical_volume_consistency()
    def test_unit_conversion_invariants()
```

### 7.2 Edge Cases & Error Handling (Unit Level)
```python
class TestEdgeCases:
    def test_empty_geometry_handling()
    def test_zero_dimension_handling()
    def test_numerical_precision_limits()
    def test_invalid_parameters()
    def test_circular_relationships()
```

## Quality Metrics & Future Testing Phases

### Unit Test Quality Metrics
**Target Metrics:**
- Unit test coverage: >90% for core modules
- Property test coverage: 1000+ examples per property
- Test execution time: <30 seconds for unit tests only
- Zero integration dependencies in unit tests

### Future Testing Phases (Not in Scope)
**Integration Testing Phase** (Future):
- End-to-end workflows with real dependencies
- IFC file import/export testing
- OpenCascade integration testing
- Cross-module interaction testing

**Performance Testing Phase** (Future):
- Large geometry operation benchmarks
- Memory usage profiling
- Scalability testing
- Regression testing for performance

## Implementation Guidelines

### Testing Best Practices
1. **Test isolation**: Each test independent, no shared state
2. **Descriptive names**: `test_wall_center_plane_calculation_for_angled_wall()`
3. **Arrange-Act-Assert pattern**: Clear test structure
4. **Parameterized tests**: Cover multiple scenarios efficiently
5. **Property-based testing**: For geometric invariants

### Fixture Management
```python
# Hierarchical fixtures
@pytest.fixture
def sample_element():
    return Element(name="test_stud", geometry=box_geometry(), material="wood")

@pytest.fixture
def sample_wall(sample_element):
    return Wall.from_elements([sample_element], name="test_wall")
```

### Continuous Integration Setup
```yaml
# GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov pytest-mock hypothesis
      - name: Run tests
        run: pytest tests/ --cov=hierarchical --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## Test Configuration Files

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = --strict-markers --disable-warnings --tb=short
markers =
    unit: Unit tests (fast, isolated)
    property: Property-based tests
    geometry: Geometry-related tests
    relationships: Relationship system tests
    materials: Material and unit system tests
```

### conftest.py Template
```python
import pytest
import numpy as np
from unittest.mock import Mock, patch
from hierarchical.geometry import Geometry, Vector3D
from hierarchical.items import Element, Component, Object

@pytest.fixture
def mock_plotly():
    with patch('hierarchical.utils.go') as mock:
        yield mock

@pytest.fixture
def sample_box_geometry():
    """Create a simple box geometry for testing"""
    return Geometry.from_primitive('box', {
        'width': 1.0, 'depth': 1.0, 'height': 1.0
    })

@pytest.fixture
def tolerance():
    """Standard geometric tolerance for tests"""
    return 1e-6
```

## Expected Deliverables

### Week 1-2: Foundation
- [ ] Complete test infrastructure setup with comprehensive mocking
- [ ] Core geometry unit tests (Vector3D, basic Geometry)
- [ ] Test fixtures and lightweight geometry doubles
- [ ] Property-based testing framework setup

### Week 3-4: Core Functionality
- [ ] Item hierarchy unit tests (Element, Component, Object)
- [ ] Relationship system unit tests with mocked dependencies
- [ ] Specialized object tests (Wall, Door) - unit level only

### Week 5-6: Supporting Systems
- [ ] Unit conversion and material unit tests
- [ ] Helper function comprehensive tests
- [ ] Utility function tests with mocked visualization

### Week 7: Property Testing & Edge Cases
- [ ] Advanced property-based tests for geometric invariants
- [ ] Edge case handling and error conditions
- [ ] Test suite optimization and cleanup

## Success Criteria
- **Coverage**: >90% unit test coverage for core modules (geometry, items, relationships, units)
- **Speed**: Unit test suite runs in <30 seconds
- **Isolation**: Zero dependencies on external services or heavy libraries in unit tests
- **Property Coverage**: All geometric and mathematical properties tested with 1000+ examples
- **Reliability**: All tests pass consistently across environments
- **Maintainability**: Clear, well-documented test code with comprehensive fixtures

## Next Steps After Unit Testing
Once this unit testing foundation is complete, the following phases should be addressed:
1. **Integration Testing**: End-to-end workflows with real dependencies
2. **Performance Testing**: Benchmarking and scalability analysis
3. **System Testing**: Full application testing with real-world data

This focused unit testing plan will establish a solid foundation for code quality, enabling confident refactoring and feature development while maintaining fast feedback loops during development.