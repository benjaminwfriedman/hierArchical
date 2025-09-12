#!/usr/bin/env python3
"""
Test script for the parametric catalog system.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hierarchical.catalog.elements.lumber import Lumber2X4, Lumber2X6, Lumber4X4


def test_lumber_creation():
    """Test creating lumber elements with different parameters."""
    
    print("Testing Parametric Lumber Catalog")
    print("=" * 40)
    
    # Test default 2x4
    print("\n1. Creating default 8-foot 2x4...")
    stud = Lumber2X4()
    print(f"   Name: {stud.name}")
    print(f"   Type: {stud.type}")
    print(f"   Material: {stud.materials}")
    print(f"   Parameters: {stud.params}")
    print(f"   Height: {stud.get_height():.3f} ft")
    
    # Test custom length 2x4
    print("\n2. Creating 12-foot 2x4 with Douglas Fir...")
    long_stud = Lumber2X4(length=12.0, species="Douglas Fir")
    print(f"   Name: {long_stud.name}")
    print(f"   Parameters: {long_stud.params}")
    print(f"   Height: {long_stud.get_height():.3f} ft")
    
    # Test 2x6
    print("\n3. Creating 10-foot 2x6...")
    joist = Lumber2X6(length=10.0, species="Pine")
    print(f"   Name: {joist.name}")
    print(f"   Parameters: {joist.params}")
    print(f"   Height: {joist.get_height():.3f} ft")
    
    # Test 4x4 post
    print("\n4. Creating 8-foot 4x4 post...")
    post = Lumber4X4(length=8.0)
    print(f"   Name: {post.name}")
    print(f"   Parameters: {post.params}")
    print(f"   Height: {post.get_height():.3f} ft")
    
    # Test parameter validation
    print("\n5. Testing parameter validation...")
    try:
        bad_stud = Lumber2X4(length=-1.0)  # Should fail
        print("   ERROR: Should have failed with negative length!")
    except ValueError as e:
        print(f"   ✓ Correctly caught error: {e}")
    
    try:
        bad_stud = Lumber2X4(length=25.0)  # Should fail (>20ft max)
        print("   ERROR: Should have failed with length > 20ft!")
    except ValueError as e:
        print(f"   ✓ Correctly caught error: {e}")
    
    # Test that they are actual Elements
    print("\n6. Testing Element inheritance...")
    print(f"   2x4 is Element: {isinstance(stud, type(stud).__bases__[0])}")
    print(f"   Has get_centroid(): {hasattr(stud, 'get_centroid')}")
    print(f"   Has move(): {hasattr(stud, 'move')}")
    print(f"   Has intersects_with(): {hasattr(stud, 'intersects_with')}")
    
    # Test geometric operations
    print("\n7. Testing geometric operations...")
    centroid = stud.get_centroid()
    print(f"   2x4 centroid: ({centroid.x:.3f}, {centroid.y:.3f}, {centroid.z:.3f})")
    
    # Move the stud
    stud.move(dx=1.0, dy=2.0, dz=0.0)
    new_centroid = stud.get_centroid()
    print(f"   After move(1,2,0): ({new_centroid.x:.3f}, {new_centroid.y:.3f}, {new_centroid.z:.3f})")
    
    print("\n✓ All tests completed successfully!")


def test_parameter_info():
    """Test parameter information retrieval."""
    
    print("\n\nParameter Information")
    print("=" * 40)
    
    # Get parameter definitions
    params = Lumber2X4.get_parameters()
    print(f"\nLumber2X4 parameters:")
    for name, param in params.items():
        print(f"  {name}:")
        print(f"    Type: {param.type.__name__}")
        print(f"    Default: {param.default}")
        print(f"    Min: {param.min_value}")
        print(f"    Max: {param.max_value}")
        print(f"    Unit: {param.unit}")
        print(f"    Description: {param.description}")
    
    print(f"\nMaterial type: {Lumber2X4.get_material_type()}")


if __name__ == "__main__":
    test_lumber_creation()
    test_parameter_info()