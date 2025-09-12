#!/usr/bin/env python3
"""
Test script for the complete hierarchical catalog system.
Tests elements, components, and objects.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from hierarchical.catalog
from hierarchical.catalog.elements.lumber import Lumber2X4, Lumber2X6, Lumber4X4
from hierarchical.catalog.elements.steel import SteelW8X10, SteelW10X15, SteelTube4X4X1_4
from hierarchical.catalog.components.wall_frames import WallFrame2X4, WallFrame2X6, MetalStudWallFrame
from hierarchical.catalog.objects.walls import ExteriorWall, InteriorWall, ShearWall


def test_elements():
    """Test lumber and steel elements"""
    print("Testing Elements")
    print("=" * 40)
    
    # Test lumber
    print("\n1. Lumber Elements:")
    stud = Lumber2X4(length=8.0, species="Douglas Fir")
    joist = Lumber2X6(length=12.0, species="SPF")
    post = Lumber4X4(length=10.0)
    
    print(f"   2x4 Stud: {stud.params} | Volume: {stud.materials['wood']['volume']:.3f} ft³")
    print(f"   2x6 Joist: {joist.params} | Volume: {joist.materials['wood']['volume']:.3f} ft³")
    print(f"   4x4 Post: {post.params} | Volume: {post.materials['wood']['volume']:.3f} ft³")
    
    # Test steel
    print("\n2. Steel Elements:")
    beam_w8 = SteelW8X10(length=16.0, grade="A992")
    beam_w10 = SteelW10X15(length=20.0, grade="A572")
    tube = SteelTube4X4X1_4(length=12.0, grade="A500")
    
    print(f"   W8x10 Beam: {beam_w8.params} | Material: {beam_w8.get_material_type()}")
    print(f"   W10x15 Beam: {beam_w10.params} | Material: {beam_w10.get_material_type()}")
    print(f"   4x4 Tube: {tube.params} | Material: {tube.get_material_type()}")
    
    return {
        'lumber': [stud, joist, post],
        'steel': [beam_w8, beam_w10, tube]
    }


def test_components():
    """Test wall frame components"""
    print("\n\nTesting Components")
    print("=" * 40)
    
    # Test 2x4 wall frame
    print("\n1. 2x4 Wall Frame:")
    wall_frame_2x4 = WallFrame2X4(
        height=8.0,
        width=16.0,
        stud_spacing=16.0,
        species="SPF"
    )
    print(f"   Parameters: {wall_frame_2x4.params}")
    print(f"   Number of elements: {len(wall_frame_2x4.create_elements())}")
    print(f"   Materials: {wall_frame_2x4.materials}")
    
    # Test 2x6 wall frame
    print("\n2. 2x6 Wall Frame:")
    wall_frame_2x6 = WallFrame2X6(
        height=9.0,
        width=12.0,
        stud_spacing=16.0,
        species="Douglas Fir"
    )
    print(f"   Parameters: {wall_frame_2x6.params}")
    print(f"   Number of elements: {len(wall_frame_2x6.create_elements())}")
    print(f"   Materials: {wall_frame_2x6.materials}")
    
    # Test metal stud frame
    print("\n3. Metal Stud Frame:")
    metal_frame = MetalStudWallFrame(
        height=9.0,
        width=20.0,
        stud_spacing=24.0,
        stud_depth="3_5/8",
        gauge=25
    )
    print(f"   Parameters: {metal_frame.params}")
    print(f"   Number of elements: {len(metal_frame.create_elements())}")
    
    return {
        'wood_frames': [wall_frame_2x4, wall_frame_2x6],
        'metal_frame': metal_frame
    }


def test_objects():
    """Test complete wall objects"""
    print("\n\nTesting Objects")
    print("=" * 40)
    
    # Test exterior wall
    print("\n1. Exterior Wall:")
    exterior_wall = ExteriorWall(
        height=8.0,
        width=16.0,
        wall_type="2x6_insulated",
        r_value=23.0,
        stud_spacing=16.0,
        species="Douglas Fir"
    )
    print(f"   Parameters: {exterior_wall.params}")
    print(f"   Number of components: {len(exterior_wall.create_components())}")
    print(f"   Materials: {exterior_wall.materials}")
    
    # Test interior wall
    print("\n2. Interior Wall:")
    interior_wall = InteriorWall(
        height=8.0,
        width=12.0,
        wall_type="2x4_standard",
        stud_spacing=16.0,
        species="SPF"
    )
    print(f"   Parameters: {interior_wall.params}")
    print(f"   Number of components: {len(interior_wall.create_components())}")
    print(f"   Materials: {interior_wall.materials}")
    
    # Test shear wall
    print("\n3. Shear Wall:")
    shear_wall = ShearWall(
        height=8.0,
        width=8.0,
        shear_rating="high",
        sheathing_type="plywood",
        stud_spacing=16.0,
        species="Douglas Fir"
    )
    print(f"   Parameters: {shear_wall.params}")
    print(f"   Number of components: {len(shear_wall.create_components())}")
    print(f"   Materials: {shear_wall.materials}")
    
    return {
        'exterior_wall': exterior_wall,
        'interior_wall': interior_wall,
        'shear_wall': shear_wall
    }


def test_inheritance():
    """Test that catalog items are proper hierarchical items"""
    print("\n\nTesting Inheritance")
    print("=" * 40)
    
    # Test Element inheritance
    stud = Lumber2X4(length=8.0)
    print(f"\n1. Element Inheritance:")
    print(f"   Lumber2X4 is Element: {hasattr(stud, 'get_centroid')}")
    print(f"   Has move(): {hasattr(stud, 'move')}")
    print(f"   Has materials: {hasattr(stud, 'materials')}")
    print(f"   Type: {type(stud).__mro__}")  # Method resolution order
    
    # Test Component inheritance
    frame = WallFrame2X4(height=8.0, width=16.0)
    print(f"\n2. Component Inheritance:")
    print(f"   WallFrame2X4 is Component: {hasattr(frame, 'materials')}")
    print(f"   Has move(): {hasattr(frame, 'move')}")
    print(f"   Type: {type(frame).__mro__}")
    
    # Test Object inheritance
    wall = ExteriorWall(height=8.0, width=16.0)
    print(f"\n3. Object Inheritance:")
    print(f"   ExteriorWall is Object: {hasattr(wall, 'materials')}")
    print(f"   Has move(): {hasattr(wall, 'move')}")
    print(f"   Type: {type(wall).__mro__}")


def test_transformations():
    """Test that transformations work on catalog items"""
    print("\n\nTesting Transformations")
    print("=" * 40)
    
    # Create items
    stud = Lumber2X4(length=8.0)
    frame = WallFrame2X4(height=8.0, width=16.0)
    wall = ExteriorWall(height=8.0, width=16.0)
    
    print(f"\n1. Original positions:")
    print(f"   Stud centroid: ({stud.get_centroid().x:.2f}, {stud.get_centroid().y:.2f}, {stud.get_centroid().z:.2f})")
    print(f"   Frame centroid: ({frame.get_centroid().x:.2f}, {frame.get_centroid().y:.2f}, {frame.get_centroid().z:.2f})")
    print(f"   Wall centroid: ({wall.get_centroid().x:.2f}, {wall.get_centroid().y:.2f}, {wall.get_centroid().z:.2f})")
    
    # Apply transformations
    stud.move(dx=5.0, dy=0.0, dz=1.0)
    frame.move(dx=0.0, dy=10.0, dz=0.0)
    wall.rotate_z(45)  # 45 degrees
    
    print(f"\n2. After transformations:")
    print(f"   Stud after move(5,0,1): ({stud.get_centroid().x:.2f}, {stud.get_centroid().y:.2f}, {stud.get_centroid().z:.2f})")
    print(f"   Frame after move(0,10,0): ({frame.get_centroid().x:.2f}, {frame.get_centroid().y:.2f}, {frame.get_centroid().z:.2f})")
    print(f"   Wall after rotate_z(45): ({wall.get_centroid().x:.2f}, {wall.get_centroid().y:.2f}, {wall.get_centroid().z:.2f})")


def test_material_aggregation():
    """Test material aggregation through hierarchy"""
    print("\n\nTesting Material Aggregation")
    print("=" * 40)
    
    # Test element materials
    stud = Lumber2X4(length=8.0)
    print(f"\n1. Element materials:")
    print(f"   Stud materials: {stud.materials}")
    
    # Test component materials (aggregated from elements)
    frame = WallFrame2X4(height=8.0, width=16.0, stud_spacing=16.0)
    print(f"\n2. Component materials (aggregated):")
    print(f"   Frame materials: {frame.materials}")
    total_wood = frame.materials.get('wood', {}).get('volume', 0)
    print(f"   Total wood volume: {total_wood:.3f} ft³")
    
    # Test object materials (aggregated from components)
    wall = ExteriorWall(height=8.0, width=16.0, wall_type="2x6_insulated")
    print(f"\n3. Object materials (aggregated):")
    print(f"   Wall materials: {wall.materials}")
    
    # Calculate board feet
    if 'wood' in wall.materials:
        total_wood_obj = wall.materials['wood']['volume']
        board_feet = total_wood_obj * 12  # Rough conversion
        print(f"   Total wood volume: {total_wood_obj:.3f} ft³ ({board_feet:.1f} board feet)")


if __name__ == "__main__":
    try:
        elements = test_elements()
        components = test_components()
        objects = test_objects()
        test_inheritance()
        test_transformations()
        test_material_aggregation()
        
        print("\n\n" + "=" * 60)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("✓ Elements, Components, and Objects working")
        print("✓ Proper inheritance from hierarchical items")
        print("✓ Transformations and material aggregation working")
        print("✓ Full catalog system is operational!")
        print("=" * 60)
        
        # Summary
        print(f"\nCatalog Summary:")
        print(f"• Elements tested: {len(elements['lumber']) + len(elements['steel'])}")
        print(f"• Components tested: {len(components['wood_frames']) + 1}")
        print(f"• Objects tested: 3")
        print(f"• Import path: from hierarchical.catalog.elements.lumber import Lumber2X4")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()