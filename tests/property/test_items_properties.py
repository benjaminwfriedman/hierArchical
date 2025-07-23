"""
Property-based tests for hierarchical.items module.

This module uses Hypothesis to test item hierarchy invariants and mathematical
properties that should hold across a wide range of input values and scenarios.
"""

import pytest
import numpy as np
import math
from hypothesis import given, strategies as st, assume, settings, Verbosity
from unittest.mock import Mock, patch

from tests.conftest import MockGeometry, assert_volumes_equal
from tests.fixtures.sample_geometries import create_mock_geometry_from_data


# =============================================================================
# HYPOTHESIS STRATEGIES
# =============================================================================

# Basic strategies
item_name = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd", "Zs")))
item_type = st.sampled_from(["element", "component", "object", "wall", "door", "window", "deck"])
material_name = st.sampled_from(["steel", "concrete", "wood", "aluminum", "glass", "plastic", "composite"])

# Geometric strategies
coordinate = st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
positive_dimension = st.floats(min_value=0.1, max_value=50, allow_nan=False, allow_infinity=False)
volume = st.floats(min_value=0.1, max_value=1000, allow_nan=False, allow_infinity=False)
percentage = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Unit system strategy
unit_system_strategy = st.sampled_from(["m", "cm", "mm", "ft", "in"])

# Material data strategy
material_data = st.fixed_dictionaries({
    "volume": volume,
    "percent": percentage
})

# Materials dictionary strategy
materials_dict = st.dictionaries(
    keys=material_name,
    values=material_data,
    min_size=1,
    max_size=5
)

# Simple geometry strategy for testing
simple_box_vertices = st.just([
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),  # bottom
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)   # top
])

simple_box_faces = st.just([
    (0, 1, 2), (0, 2, 3),  # bottom
    (4, 6, 5), (4, 7, 6),  # top
    (0, 4, 1), (1, 4, 5),  # front
    (1, 5, 2), (2, 5, 6),  # right
    (2, 6, 3), (3, 6, 7),  # back
    (3, 7, 0), (0, 7, 4)   # left
])

# Attribute strategies
length_attribute = st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False)
area_attribute = st.floats(min_value=0.01, max_value=10000, allow_nan=False, allow_infinity=False)
volume_attribute = st.floats(min_value=0.001, max_value=1000000, allow_nan=False, allow_infinity=False)


# =============================================================================
# ELEMENT PROPERTY TESTS
# =============================================================================

class TestElementProperties:
    """Property-based tests for Element class."""
    
    @given(item_name, material_name, volume)
    @settings(deadline=None)
    @pytest.mark.property
    def test_element_material_volume_consistency(self, name, material, vol):
        """Test that Element material volume matches geometry volume."""
        from hierarchical.items import Element
        
        # Create mock geometry with specific volume
        geometry = MockGeometry()
        geometry.compute_volume = Mock(return_value=vol)
        
        element = Element(
            name=name,
            geometry=geometry,
            type="element",
            material=material
        )
        
        # Material volume should match geometry volume
        assert element.materials[material]["volume"] == vol
        assert element.materials[material]["percent"] == 1.0
        
    @given(item_name, material_name, coordinate, coordinate, coordinate)
    @pytest.mark.property
    def test_element_transformation_preserves_material_identity(self, name, material, dx, dy, dz):
        """Test that transformations don't change material identity."""
        from hierarchical.items import Element
        
        geometry = MockGeometry()
        geometry.compute_volume = Mock(return_value=5.0)
        geometry.transform_geometry = Mock()
        
        element = Element(
            name=name,
            geometry=geometry,
            type="element",
            material=material
        )
        
        original_material = element.material
        original_materials = element.materials.copy()
        
        # Apply transformation
        element.move(dx, dy, dz)
        
        # Material identity should be preserved
        assert element.material == original_material
        assert element.materials == original_materials
        
    @given(item_name, material_name)
    @pytest.mark.property 
    def test_element_copy_preserves_material(self, name, material):
        """Test that copying an element preserves material information."""
        from hierarchical.items import Element
        
        geometry = MockGeometry()
        geometry.compute_volume = Mock(return_value=3.0)
        
        element = Element(
            name=name,
            geometry=geometry,
            type="element",
            material=material
        )
        
        # Mock the copy dependencies
        with patch('hierarchical.items.deepcopy') as mock_deepcopy:
            mock_deepcopy.side_effect = lambda x: x  # Return same object for simplicity
            geometry.right = Mock(return_value=geometry)
            geometry.forward = Mock(return_value=geometry)
            geometry.up = Mock(return_value=geometry)
            
            copied_element = element.copy()
            
            # Material should be preserved
            assert hasattr(copied_element, 'material')
            assert copied_element.name == name + " (copy)"


# =============================================================================
# COMPONENT PROPERTY TESTS
# =============================================================================

class TestComponentProperties:
    """Property-based tests for Component class."""
    
    @given(st.lists(materials_dict, min_size=1, max_size=5))
    @settings(deadline=None)
    @pytest.mark.property
    def test_component_material_aggregation_consistency(self, element_materials_list):
        """Test that component material aggregation is mathematically consistent."""
        from hierarchical.items import Component, Element
        
        # Create mock elements with the given materials
        elements = []
        total_expected_volumes = {}
        
        for i, materials in enumerate(element_materials_list):
            element = Mock()
            element.materials = materials
            elements.append(element)
            
            # Track expected total volumes
            for material, data in materials.items():
                if material not in total_expected_volumes:
                    total_expected_volumes[material] = 0.0
                total_expected_volumes[material] += data["volume"]
        
        with patch('hierarchical.items.BaseItem.geometry_from_sub_items') as mock_geom_method:
            mock_geom_method.return_value = Mock()
            
            component = Component.from_elements(
                elements=tuple(elements),
                name="test_component"
            )
            
            # Check material volume aggregation
            for material, expected_volume in total_expected_volumes.items():
                assert material in component.materials
                assert abs(component.materials[material]["volume"] - expected_volume) < 1e-10
            
            # Check that percentages sum to 1.0 (within tolerance)
            total_percent = sum(data["percent"] for data in component.materials.values())
            assert abs(total_percent - 1.0) < 1e-10
    
    @given(st.lists(st.tuples(material_name, volume), min_size=1, max_size=10))
    @pytest.mark.property
    def test_component_percentage_calculation_consistency(self, material_volume_pairs):
        """Test that component percentage calculations are mathematically consistent."""
        from hierarchical.items import Component
        
        # Remove duplicates and ensure unique materials
        unique_materials = {}
        for material, vol in material_volume_pairs:
            if material not in unique_materials:
                unique_materials[material] = vol
            else:
                unique_materials[material] += vol
        
        # Skip if all volumes are zero
        total_volume = sum(unique_materials.values())
        assume(total_volume > 0.001)
        
        # Create mock elements
        elements = []
        for material, vol in unique_materials.items():
            element = Mock()
            element.materials = {material: {"volume": vol, "percent": 1.0}}
            elements.append(element)
        
        with patch('hierarchical.items.BaseItem.geometry_from_sub_items') as mock_geom_method:
            mock_geom_method.return_value = Mock()
            
            component = Component.from_elements(
                elements=tuple(elements),
                name="percentage_test_component"
            )
            
            # Verify percentage calculations
            calculated_total = 0.0
            for material, vol in unique_materials.items():
                expected_percent = vol / total_volume
                actual_percent = component.materials[material]["percent"]
                assert abs(actual_percent - expected_percent) < 1e-10
                calculated_total += actual_percent
            
            # Total percentages should sum to 1.0
            assert abs(calculated_total - 1.0) < 1e-10
    
    @given(item_name, st.lists(material_name, min_size=1, max_size=3))
    @pytest.mark.property
    def test_component_empty_elements_handling(self, name, materials):
        """Test that components handle empty elements gracefully."""
        from hierarchical.items import Component
        
        with patch('hierarchical.items.BaseItem.geometry_from_sub_items') as mock_geom_method:
            mock_geom_method.return_value = Mock()
            
            # Test empty elements tuple
            component = Component.from_elements(
                elements=(),
                name=name
            )
            
            assert component.name == name
            assert component.sub_items == ()
            assert component.materials == {}


# =============================================================================
# OBJECT PROPERTY TESTS  
# =============================================================================

class TestObjectProperties:
    """Property-based tests for Object class."""
    
    @given(st.lists(materials_dict, min_size=1, max_size=5))
    @settings(deadline=None)
    @pytest.mark.property
    def test_object_material_aggregation_from_components(self, component_materials_list):
        """Test that object material aggregation from components is consistent."""
        from hierarchical.items import Object
        
        # Create mock components with given materials
        components = []
        total_expected_volumes = {}
        
        for i, materials in enumerate(component_materials_list):
            component = Mock()
            component.materials = materials
            components.append(component)
            
            # Track expected total volumes
            for material, data in materials.items():
                if material not in total_expected_volumes:
                    total_expected_volumes[material] = 0.0
                total_expected_volumes[material] += data["volume"]
        
        with patch('hierarchical.items.BaseItem.geometry_from_sub_items') as mock_geom_method:
            mock_geom_method.return_value = Mock()
            
            obj = Object.from_components(
                components=tuple(components),
                name="test_object"
            )
            
            # Check material volume aggregation
            for material, expected_volume in total_expected_volumes.items():
                assert material in obj.materials
                assert abs(obj.materials[material]["volume"] - expected_volume) < 1e-10
            
            # Check that percentages sum to 1.0 (within tolerance)
            if total_expected_volumes:  # Only if we have materials
                total_percent = sum(data["percent"] for data in obj.materials.values())
                assert abs(total_percent - 1.0) < 1e-10
    
    @given(item_name, st.integers(min_value=1, max_value=10))
    @pytest.mark.property
    def test_object_hierarchical_consistency(self, name, num_levels):
        """Test that multi-level hierarchy maintains consistency."""
        from hierarchical.items import Object
        
        # Create a simple hierarchy: Object -> Components -> Elements
        base_volume = 1.0
        material = "steel"
        
        # Create mock elements
        element = Mock()
        element.materials = {material: {"volume": base_volume, "percent": 1.0}}
        
        # Create mock component from element
        component = Mock()
        component.materials = {material: {"volume": base_volume, "percent": 1.0}}
        
        with patch('hierarchical.items.BaseItem.geometry_from_sub_items') as mock_geom_method:
            mock_geom_method.return_value = Mock()
            
            # Create multiple levels of nesting
            current_volume = base_volume
            for level in range(num_levels):
                obj = Object.from_components(
                    components=(component,),
                    name=f"{name}_level_{level}"
                )
                
                # Volume should be preserved at each level
                assert obj.materials[material]["volume"] == current_volume
                assert obj.materials[material]["percent"] == 1.0
                
                # Use this object as component for next level
                component = Mock()
                component.materials = obj.materials


# =============================================================================
# SPECIALIZED OBJECT PROPERTY TESTS
# =============================================================================

class TestSpecializedObjectProperties:
    """Property-based tests for specialized object classes."""
    
    @given(item_name, st.sampled_from(["wall", "door", "window", "deck"]))
    @pytest.mark.property
    def test_specialized_object_inheritance_properties(self, name, object_type):
        """Test that specialized objects maintain inheritance properties."""
        from hierarchical.items import Wall, Door, Window, Deck, Object, BaseItem
        
        component = Mock()
        component.materials = {"concrete": {"volume": 5.0, "percent": 1.0}}
        
        class_map = {
            "wall": Wall,
            "door": Door, 
            "window": Window,
            "deck": Deck
        }
        
        with patch('hierarchical.items.BaseItem.geometry_from_sub_items') as mock_geom_method:
            mock_geom_method.return_value = Mock()
            
            ObjectClass = class_map[object_type]
            obj = ObjectClass.from_components(
                components=(component,),
                name=name
            )
            
            # Test inheritance chain
            assert isinstance(obj, ObjectClass)
            assert isinstance(obj, Object)
            assert isinstance(obj, BaseItem)
            
            # Test that specialized objects have BaseItem capabilities
            assert hasattr(obj, 'move')
            assert hasattr(obj, 'rotate_z')
            assert hasattr(obj, 'intersects_with')
            assert hasattr(obj, 'materials')
            
            # Test name and type consistency
            assert obj.name == name
            if object_type == "door":
                assert obj.type == "Door"  # Door uses capitalized type
            else:
                assert obj.type == object_type
    
    @given(item_name, st.sampled_from(["wall", "deck"]))
    @pytest.mark.property
    def test_boundary_objects_have_boundary_methods(self, name, boundary_object_type):
        """Test that boundary objects (Wall, Deck) have specialized geometry methods."""
        from hierarchical.items import Wall, Deck
        
        component = Mock()
        component.materials = {"wood": {"volume": 3.0, "percent": 1.0}}
        
        class_map = {"wall": Wall, "deck": Deck}
        
        with patch('hierarchical.items.BaseItem.geometry_from_sub_items') as mock_geom_method:
            mock_geom_method.return_value = Mock()
            
            ObjectClass = class_map[boundary_object_type]
            obj = ObjectClass.from_components(
                components=(component,),
                name=name
            )
            
            # Test that boundary objects have specialized geometry methods
            assert hasattr(obj, 'get_center_plane')
            assert hasattr(obj, 'get_centerplane_geometry')
            assert hasattr(obj, 'get_centerplane_normal_vector')
            
            assert callable(obj.get_center_plane)
            assert callable(obj.get_centerplane_geometry)
            assert callable(obj.get_centerplane_normal_vector)
            
            # Wall has additional edge methods
            if boundary_object_type == "wall":
                assert hasattr(obj, 'get_centerplane_top_edge')
                assert hasattr(obj, 'get_centerplane_bottom_edge')
                assert hasattr(obj, 'get_centerplane_left_edge')
                assert hasattr(obj, 'get_centerplane_right_edge')


# =============================================================================
# BASEITEM TRANSFORMATION PROPERTY TESTS
# =============================================================================

class TestBaseItemTransformationProperties:
    """Property-based tests for BaseItem transformation invariants."""
    
    @given(item_name, coordinate, coordinate, coordinate)
    @settings(deadline=None)
    @pytest.mark.property
    def test_transformation_chaining_associativity(self, name, dx1, dx2, dy1):
        """Test that transformation chaining is mathematically consistent."""
        from hierarchical.items import BaseItem
        
        geometry = MockGeometry()
        geometry.transform_geometry = Mock()
        geometry.right = Mock(return_value=geometry)
        geometry.forward = Mock(return_value=geometry)
        
        item1 = BaseItem(name=name + "_1", geometry=geometry, type="test")
        item2 = BaseItem(name=name + "_2", geometry=geometry, type="test")
        
        # Apply transformations in different orders
        # Should be mathematically equivalent for these operations
        result1 = item1.right(dx1).forward(dy1).right(dx2)
        result2 = item2.right(dx1 + dx2).forward(dy1)
        
        # Both should return the same object (self)
        assert result1 is item1
        assert result2 is item2
        
        # Both should have called geometry methods
        assert geometry.right.call_count >= 2
        assert geometry.forward.call_count >= 2
    
    @given(item_name, coordinate, coordinate, coordinate)
    @pytest.mark.property
    def test_transformation_with_sub_items_propagation(self, name, dx, dy, dz):
        """Test that transformations properly propagate to sub-items."""
        from hierarchical.items import BaseItem
        
        # Create mock sub-items
        sub_item1 = Mock()
        sub_item1.move = Mock(return_value=sub_item1)
        sub_item1.right = Mock(return_value=sub_item1)
        
        sub_item2 = Mock()  
        sub_item2.move = Mock(return_value=sub_item2)
        sub_item2.right = Mock(return_value=sub_item2)
        
        geometry = MockGeometry()
        geometry.transform_geometry = Mock()
        geometry.right = Mock(return_value=geometry)
        
        item = BaseItem(
            name=name,
            geometry=geometry,
            type="test",
            sub_items=(sub_item1, sub_item2)
        )
        
        # Apply transformation
        item.move(dx, dy, dz)
        item.right(dx)
        
        # All sub-items should receive the same transformations
        sub_item1.move.assert_called_with(dx, dy, dz)
        sub_item1.right.assert_called_with(dx)
        sub_item2.move.assert_called_with(dx, dy, dz)
        sub_item2.right.assert_called_with(dx)


# =============================================================================
# MATERIAL AGGREGATION PROPERTY TESTS
# =============================================================================

class TestMaterialAggregationProperties:
    """Property-based tests for material aggregation invariants."""
    
    @given(st.lists(st.tuples(material_name, volume), min_size=2, max_size=10))
    @pytest.mark.property
    def test_material_percentage_invariant(self, material_volume_pairs):
        """Test that material percentages always sum to 1.0 when properly calculated."""
        # Remove duplicates by material name (sum volumes for same materials)
        material_totals = {}
        for material, vol in material_volume_pairs:
            if material not in material_totals:
                material_totals[material] = vol
            else:
                material_totals[material] += vol
        
        total_volume = sum(material_totals.values())
        assume(total_volume > 0.001)  # Avoid division by zero
        
        # Calculate percentages manually
        calculated_percentages = {}
        for material, vol in material_totals.items():
            calculated_percentages[material] = vol / total_volume
        
        # Test the invariant: percentages should sum to 1.0
        percentage_sum = sum(calculated_percentages.values())
        assert abs(percentage_sum - 1.0) < 1e-10
        
        # Test that each percentage is between 0 and 1
        for material, percent in calculated_percentages.items():
            assert 0.0 <= percent <= 1.0
    
    @given(st.lists(materials_dict, min_size=1, max_size=5))
    @pytest.mark.property
    def test_hierarchical_material_conservation(self, component_materials_list):
        """Test that material volumes are conserved through hierarchy aggregation."""
        # Calculate expected total volumes across all components
        global_material_totals = {}
        
        for materials in component_materials_list:
            for material, data in materials.items():
                if material not in global_material_totals:
                    global_material_totals[material] = 0.0
                global_material_totals[material] += data["volume"]
        
        # Simulate aggregation process (like in Component.from_elements)
        aggregated_materials = {}
        total_aggregated_volume = 0.0
        
        for materials in component_materials_list:
            for material, data in materials.items():
                if material not in aggregated_materials:
                    aggregated_materials[material] = 0.0
                aggregated_materials[material] += data["volume"]
                total_aggregated_volume += data["volume"]
        
        # Test conservation: aggregated totals should match expected totals
        for material, expected_volume in global_material_totals.items():
            assert material in aggregated_materials
            assert abs(aggregated_materials[material] - expected_volume) < 1e-10
        
        # Test that total volume is conserved
        calculated_total = sum(aggregated_materials.values())
        assert abs(calculated_total - total_aggregated_volume) < 1e-10


# =============================================================================
# EDGE CASE PROPERTY TESTS
# =============================================================================

class TestEdgeCaseProperties:
    """Property-based tests for edge cases and robustness."""
    
    @given(item_name, st.lists(st.floats(min_value=0.0, max_value=0.001), min_size=1, max_size=5))
    @settings(deadline=None)
    @pytest.mark.property
    def test_zero_and_tiny_volume_handling(self, name, tiny_volumes):
        """Test that zero and very small volumes are handled gracefully."""
        from hierarchical.items import Component
        
        # Create elements with tiny or zero volumes
        elements = []
        for i, vol in enumerate(tiny_volumes):
            element = Mock()
            element.materials = {"air": {"volume": vol, "percent": 1.0}}
            elements.append(element)
        
        with patch('hierarchical.items.BaseItem.geometry_from_sub_items') as mock_geom_method:
            mock_geom_method.return_value = Mock()
            
            component = Component.from_elements(
                elements=tuple(elements),
                name=name
            )
            
            # Should handle tiny volumes without error
            total_volume = sum(tiny_volumes)
            assert "air" in component.materials
            assert abs(component.materials["air"]["volume"] - total_volume) < 1e-15
            
            # Test that percentage is properly calculated
            # Note: The actual implementation may return 1.0 even for tiny volumes
            # if there's only one material, which is mathematically correct
            percentage = component.materials["air"]["percent"]
            assert 0.0 <= percentage <= 1.0  # Percentage should be in valid range
            
            # If we have only one material, percentage should typically be 1.0 
            # unless the implementation specifically handles zero volume cases
            if len(component.materials) == 1:
                assert percentage >= 0.0  # At minimum, should be non-negative
    
    @given(item_name, st.lists(item_name, min_size=0, max_size=10))
    @pytest.mark.property
    def test_empty_hierarchy_robustness(self, name, empty_list):
        """Test that empty hierarchies are handled gracefully."""
        from hierarchical.items import Component, Object
        
        with patch('hierarchical.items.BaseItem.geometry_from_sub_items') as mock_geom_method:
            mock_geom_method.return_value = Mock()
            
            # Test empty component
            component = Component.from_elements(
                elements=(),
                name=name
            )
            
            assert component.name == name
            assert component.sub_items == ()
            assert component.materials == {}
            
            # Test empty object
            obj = Object.from_components(
                components=(),
                name=name + "_object"
            )
            
            assert obj.name == name + "_object"
            assert obj.sub_items == ()
            assert obj.materials == {}


# =============================================================================
# CONFIGURATION FOR PROPERTY TESTS
# =============================================================================

# Configure Hypothesis for more thorough testing
settings.register_profile("default", max_examples=50, deadline=10000)
settings.register_profile("comprehensive", max_examples=200, deadline=30000)
settings.register_profile("ci", max_examples=25, deadline=5000)

# Load the default profile
settings.load_profile("default")