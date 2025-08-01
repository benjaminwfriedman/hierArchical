"""
Parametric drywall elements with standard dimensions.
"""

from typing import Dict
from hierarchical.catalog.base import ParametricElement, Parameter
from hierarchical.geometry import Geometry

## STANDARDIZATION ##

# All items are positioned starting at 0,0,0 with:
# X axis = longest dimension (length)
# Y axis = second longest dimension (width)
# Z axis = shortest dimension (thickness - "up" when relevant)
# This ensures consistency in positioning, rotation, and alignment.

class BaseDrywall(ParametricElement):
    """Base class for drywall sheets"""
    
    ACTUAL_WIDTH = 4.0      # feet (Y axis - standard width)
    ACTUAL_LENGTH = 8.0     # feet (X axis - default length)
    THICKNESS = None        # inches (Z axis - to be set by subclasses)
    TYPE = None             # drywall type (Regular, Moisture Resistant, etc.)
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'length': Parameter(
                name='length',
                type=float,
                default=cls.ACTUAL_LENGTH,
                min_value=0.01,
                max_value=16.0,
                unit='ft',
                description="Length of the drywall sheet (X-axis - longest dimension)"
            ),
            'width': Parameter(
                name='width',
                type=float,
                default=cls.ACTUAL_WIDTH,
                min_value=0.01,
                max_value=4.5,
                unit='ft',
                description="Width of the drywall sheet (Y-axis - second longest dimension)"
            ),
            'type': Parameter(
                name='type',
                type=str,
                default=cls.TYPE,
                description="Drywall type (Regular, MR, Type X, etc.)"
            )
        }
    
    @classmethod
    def get_material_type(cls) -> str:
        return "gypsum board"
    
    def create_geometry(self) -> Geometry:
        # Following standardization: X=length, Y=width, Z=thickness
        base_points = [
            (0, 0),
            (self.params['length'], 0),
            (self.params['length'], self.params['width']),
            (0, self.params['width'])
        ]
        return Geometry.from_prism(base_points, self.THICKNESS / 12.0)  # Convert inches to feet for Z


def create_drywall_classes():
    """Generate all standard drywall classes"""
    
    # Standard drywall sizes: thickness (inches) and types
    drywall_specs = [
        (0.25, "Regular"),
        (0.375, "Regular"),
        (0.5, "Regular"),
        (0.5, "Moisture Resistant"),
        # handle Type X in the component creation
    ]
    
    classes = {}
    
    for thickness, dtype in drywall_specs:
        class_name = f"Drywall_{dtype.replace(' ', '')}_{str(thickness).replace('.', '_')}"
        
        # Create class dynamically
        cls = type(class_name, (BaseDrywall,), {
            'THICKNESS': thickness,
            'TYPE': dtype,
            '__doc__': f"{dtype} drywall sheet - {thickness}\" thick\n"
                       f"X=length(default 8ft), Y=width(4ft), Z=thickness({thickness}\")"
        })
        
        classes[class_name] = cls
        globals()[class_name] = cls
    
    return classes


def plot_all_drywall_3d_standardized(length: float = 8.0, y_spacing: float = 0.5, z_spacing: float = 0.1):
    """
    Create a 3D visualization of drywall sheets following standardization.
    X = length (longest), Y = width (second longest), Z = thickness (shortest/up)
    
    Args:
        length: Default length of drywall sheets in feet (X axis)
        y_spacing: Spacing between sheets along Y axis (feet)
        z_spacing: Spacing between sheets along Z axis (feet)
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np
    except ImportError:
        print("matplotlib required for 3D plotting. Install with: pip install matplotlib")
        return
    
    drywall_specs = [
        (0.25, "Regular"),
        (0.375, "Regular"),
        (0.5, "Regular"),
        (0.5, "Moisture Resistant"),
        # Handle Type X in the component creation
    ]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(drywall_specs)))
    
    current_y = 0
    current_z = 0
    
    for i, (thickness, dtype) in enumerate(drywall_specs):
        width_ft = 4.0
        length_ft = length
        thickness_ft = thickness / 12.0
        
        x = [0, length_ft, length_ft, 0, 0, length_ft, length_ft, 0]
        y = [current_y, current_y, current_y + width_ft, current_y + width_ft,
             current_y, current_y, current_y + width_ft, current_y + width_ft]
        z = [current_z, current_z, current_z, current_z,
             current_z + thickness_ft, current_z + thickness_ft, current_z + thickness_ft, current_z + thickness_ft]
        
        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [0, 3, 7, 4]
        ]
        
        vertices = [[[x[j], y[j], z[j]] for j in face] for face in faces]
        poly3d = Poly3DCollection(vertices, alpha=0.7, facecolor=colors[i], edgecolor='black', linewidth=0.5)
        ax.add_collection3d(poly3d)
        
        label_text = f"{dtype}\n{thickness}\""
        ax.text(length_ft / 2, current_y + width_ft / 2, current_z + thickness_ft + 0.02,
                label_text, fontsize=7, ha='center', va='bottom')
        
        current_y += width_ft + y_spacing
        if (i + 1) % 3 == 0:
            current_y = 0
            current_z += max([spec[0] / 12.0 for spec in drywall_specs]) + z_spacing
    
    ax.set_xlabel('X - Length (ft)')
    ax.set_ylabel('Y - Width (ft)')
    ax.set_zlabel('Z - Thickness (ft)')
    ax.set_title('Standard Drywall Sheets')
    
    ax.set_box_aspect([1, 1, 0.3])
    plt.tight_layout()
    plt.show()


def show_drywall_visualization():
    """Call this to display the drywall visualization following standardization"""
    print("Displaying 3D drywall visualization...")
    print("X = Length (longest), Y = Width (second longest), Z = Thickness (shortest/up)")
    plot_all_drywall_3d_standardized()


# Create all drywall classes
drywall_classes = create_drywall_classes()

if __name__ == "__main__":
    show_drywall_visualization()
