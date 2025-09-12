#!/usr/bin/env python3
"""
Example usage of the parametric catalog system.

This demonstrates the clean import structure and usage patterns.
"""

# Import lumber elements directly
from hierarchical.catalog.elements.lumber import Lumber2X4, Lumber2X6, Lumber2X8, Lumber4X4

def simple_wall_frame_example():
    """Create a simple wall frame using parametric lumber."""
    
    print("Building a Simple Wall Frame")
    print("=" * 30)
    
    # Wall dimensions
    wall_height = 8.0  # feet
    wall_width = 16.0  # feet
    stud_spacing = 16.0 / 12  # 16" on center, converted to feet
    
    # Create bottom plate
    bottom_plate = Lumber2X4(length=wall_width, species="Douglas Fir")
    print(f"Bottom plate: {bottom_plate.params}")
    
    # Create top plates (double top plate)
    top_plate_1 = Lumber2X4(length=wall_width, species="Douglas Fir")
    top_plate_2 = Lumber2X4(length=wall_width, species="Douglas Fir")
    print(f"Top plates: 2x {top_plate_1.params}")
    
    # Calculate number of studs needed
    # For 16' wall with 16" spacing: studs at 0", 16", 32", 48", 64", 80", 96", 112", 128", 144", 160" + end stud
    num_studs = int(wall_width / stud_spacing) + 1
    
    # Create studs
    studs = []
    for i in range(num_studs):
        stud = Lumber2X4(length=wall_height - (1.5 * 2 / 12), species="SPF")  # Subtract plate thickness
        studs.append(stud)
    
    print(f"Created {len(studs)} studs at {wall_height - (1.5 * 2 / 12):.2f}' long")
    
    # Create king studs for opening (if any)
    king_stud = Lumber2X4(length=wall_height - (1.5 * 2 / 12), species="SPF")
    
    # Create header for opening
    header = Lumber2X8(length=4.0, species="Douglas Fir")  # 4' opening
    print(f"Header: {header.params}")
    
    # Position elements (example positioning)
    print(f"\nPositioning elements...")
    
    # Bottom plate stays at origin
    print(f"Bottom plate at: (0, 0, 0)")
    
    # Position studs
    for i, stud in enumerate(studs):
        x_pos = i * stud_spacing
        stud.move(dx=x_pos, dy=0, dz=1.5/12)  # Move up by bottom plate thickness
        centroid = stud.get_centroid()
        print(f"Stud {i+1} at: ({centroid.x:.1f}, {centroid.y:.1f}, {centroid.z:.1f})")
    
    # Position top plates
    top_plate_1.move(dx=0, dy=0, dz=wall_height - 1.5/12)
    top_plate_2.move(dx=0, dy=0, dz=wall_height)
    
    print(f"\nWall frame complete!")
    print(f"Total lumber pieces: {1 + 2 + len(studs) + 1 + 1} pieces")
    
    return {
        'bottom_plate': bottom_plate,
        'top_plates': [top_plate_1, top_plate_2],
        'studs': studs,
        'king_stud': king_stud,
        'header': header
    }


def material_calculation_example():
    """Example of automatic material calculation."""
    
    print("\n\nMaterial Calculation Example")
    print("=" * 30)
    
    # Create different lumber pieces
    pieces = [
        Lumber2X4(length=8.0),    # Standard stud
        Lumber2X4(length=12.0),   # Long piece
        Lumber2X6(length=10.0),   # Joist
        Lumber4X4(length=8.0),    # Post
    ]
    
    total_volume = 0
    print("Individual piece volumes:")
    
    for i, piece in enumerate(pieces):
        volume = piece.materials['wood']['volume']
        total_volume += volume
        print(f"  Piece {i+1} ({piece.__class__.__name__}): {volume:.4f} cubic feet")
    
    print(f"\nTotal wood volume: {total_volume:.4f} cubic feet")
    print(f"Total board feet (approx): {total_volume * 12:.2f} bf")  # Rough conversion


def parameter_info_example():
    """Show parameter information for different lumber types."""
    
    print("\n\nParameter Information Example")
    print("=" * 30)
    
    lumber_types = [Lumber2X4, Lumber2X6, Lumber2X8, Lumber4X4]
    
    for lumber_class in lumber_types:
        print(f"\n{lumber_class.__name__}:")
        print(f"  Actual dimensions: {lumber_class.ACTUAL_WIDTH * 12:.1f}\" x {lumber_class.ACTUAL_HEIGHT * 12:.1f}\"")
        print(f"  Material type: {lumber_class.get_material_type()}")
        
        params = lumber_class.get_parameters()
        for param_name, param in params.items():
            print(f"  {param_name}: {param.description}")
            print(f"    Default: {param.default} {param.unit}")
            if param.min_value:
                print(f"    Range: {param.min_value} - {param.max_value} {param.unit}")


def element_methods_example():
    """Demonstrate that catalog items have all Element methods."""
    
    print("\n\nElement Methods Example")
    print("=" * 30)
    
    # Create a 2x4
    stud = Lumber2X4(length=10.0)
    
    print(f"Created: {stud.__class__.__name__}")
    print(f"Type: {type(stud)}")
    print(f"Is Element: {hasattr(stud, 'get_centroid')}")
    
    # Test geometric properties
    height = stud.get_height()
    centroid = stud.get_centroid()
    min_point, max_point = stud.geometry.get_bbox()
    
    print(f"\nGeometric properties:")
    print(f"  Height: {height:.3f} ft")
    print(f"  Centroid: ({centroid.x:.3f}, {centroid.y:.3f}, {centroid.z:.3f})")
    print(f"  Bounding box min: ({min_point[0]:.3f}, {min_point[1]:.3f}, {min_point[2]:.3f})")
    print(f"  Bounding box max: ({max_point[0]:.3f}, {max_point[1]:.3f}, {max_point[2]:.3f})")
    
    # Test transformations
    print(f"\nTesting transformations:")
    original_centroid = stud.get_centroid()
    print(f"  Original centroid: ({original_centroid.x:.3f}, {original_centroid.y:.3f}, {original_centroid.z:.3f})")
    
    # Move and rotate
    stud.move(dx=2.0, dy=1.0, dz=0.5)
    stud.rotate_z(45)  # 45 degrees
    
    new_centroid = stud.get_centroid()
    print(f"  After move & rotate: ({new_centroid.x:.3f}, {new_centroid.y:.3f}, {new_centroid.z:.3f})")
    
    # Test materials
    print(f"\nMaterials: {stud.materials}")


if __name__ == "__main__":
    # Run all examples
    wall_frame = simple_wall_frame_example()
    material_calculation_example()
    parameter_info_example()
    element_methods_example()
    
    print("\n\n✓ All examples completed successfully!")
    print("\nThe parametric catalog system is working and ready for use!")