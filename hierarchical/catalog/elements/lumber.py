"""
Parametric lumber elements with standard dimensions.
"""

from typing import Dict
from hierarchical.catalog.base import ParametricElement, Parameter
from hierarchical.geometry import Geometry

## STANDARDIZATION ##

# All items, components, and objects are built starting at 0,0,0 with the X axis being the longest dimension, 
# y being the second longest, and z being the 3rd (or up when up is important)
# This allows for easy alignment and positioning of items in a 3D space.
# This also standardizes how objects must be moved, rotated and scaled to work together.


class BaseLumber(ParametricElement):
    """Base class for standard lumber"""
    
    ACTUAL_WIDTH = None    # To be set by subclasses (in feet) - shortest dimension (Z)
    ACTUAL_HEIGHT = None   # To be set by subclasses (in feet) - middle dimension (Y)
    LUMBER_TYPE = None     # To be set by subclasses
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'length': Parameter(
                name='length', 
                type=float, 
                default=8.0, 
                min_value=0.5, 
                max_value=20.0, 
                unit='ft',
                description="Length of the lumber piece (X-axis - longest dimension)"
            ),
            'species': Parameter(
                name='species', 
                type=str, 
                default='SPF', 
                description="Wood species (SPF, Douglas Fir, Pine, etc.)"
            )
        }
    
    @classmethod
    def get_material_type(cls) -> str:
        return "wood"
    
    def create_geometry(self) -> Geometry:
        # Following standardization: X=longest (length), Y=second longest (height), Z=shortest (width)
        base_points = [
            (0, 0), 
            (self.params['length'], 0),  # X = length (longest)
            (self.params['length'], self.ACTUAL_HEIGHT),  # Y = height (second longest)
            (0, self.ACTUAL_HEIGHT)
        ]
        return Geometry.from_prism(base_points, self.ACTUAL_WIDTH)  # Z = width (shortest)


class BaseSheet(ParametricElement):
    """Base class for wood sheets"""
    
    ACTUAL_WIDTH = 4.0     # feet (second longest dimension - Y axis)
    ACTUAL_LENGTH = 8.0    # feet (longest dimension - X axis) 
    THICKNESS = None       # To be set by subclasses (in inches) - shortest dimension (Z axis)
    MATERIAL = None        # To be set by subclasses
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'length': Parameter(
                name='length', 
                type=float, 
                default=8.0, 
                min_value=0.5, 
                max_value=20.0, 
                unit='ft',
                description="Length of the sheet (X-axis - longest dimension)"
            ),
            'width': Parameter(
                name='width',
                type=float,
                default=4.0,
                min_value=0.5,
                max_value=4.0,
                unit='ft',
                description="Width of the sheet (Y-axis - middle dimension)"
            ),
            'species': Parameter(
                name='species', 
                type=str, 
                default=cls.MATERIAL, 
                description="Wood species or type"
            )
        }
    
    @classmethod
    def get_material_type(cls) -> str:
        return "wood"
    
    def create_geometry(self) -> Geometry:
        # Following standardization: X=longest (length), Y=second longest (width), Z=shortest (thickness)
        base_points = [
            (0, 0), 
            (self.params['length'], 0),  # X = length (longest - 8ft default)
            (self.params['length'], self.params['width']),  # Y = width (second longest - 4ft)
            (0, self.params['width'])
        ]
        return Geometry.from_prism(base_points, self.THICKNESS / 12.0)  # Z = thickness (shortest)


def create_lumber_classes():
    """Generate all standard lumber classes"""
    
    # Standard lumber dimensions: (nominal_size, actual_width_inches, actual_height_inches)
    # Note: width=shortest, height=middle (length will be longest via parameter)
    lumber_specs = [
        ('2X4', 1.5, 3.5),
        ('2X6', 1.5, 5.5),
        ('2X8', 1.5, 7.25),
        ('2X10', 1.5, 9.25),
        ('2X12', 1.5, 11.25),
        ('4X4', 3.5, 3.5),
        ('6X6', 5.5, 5.5),
        ('8X8', 7.25, 7.25),
        ('10X10', 9.25, 9.25),
        ('12X12', 11.25, 11.25),
        ('2X14', 1.5, 13.25),
        ('2X16', 1.5, 15.25),
    ]
    
    classes = {}
    
    for nominal, width_inches, height_inches in lumber_specs:
        class_name = f"Lumber{nominal}"
        
        # Create the class dynamically
        cls = type(class_name, (BaseLumber,), {
            'ACTUAL_WIDTH': width_inches / 12.0,   # Z-axis (shortest)
            'ACTUAL_HEIGHT': height_inches / 12.0, # Y-axis (middle)
            'LUMBER_TYPE': nominal,
            '__doc__': f"Standard {nominal} lumber (actual: {width_inches}\" x {height_inches}\")\nX=length, Y=height({height_inches}\"), Z=width({width_inches}\")"
        })
        
        classes[class_name] = cls
        globals()[class_name] = cls
    
    return classes


def create_sheet_classes():
    """Generate all standard sheet classes"""
    
    classes = {}
    materials = ['Plywood', 'OSB', 'MDF']
    thicknesses = [0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.25, 1.5]
    
    for material in materials:
        for thickness in thicknesses:
            class_name = f"{material}Sheet_{str(thickness).replace('.', '_')}"
            
            # Create the class dynamically
            cls = type(class_name, (BaseSheet,), {
                'THICKNESS': thickness,
                'MATERIAL': material,
                '__doc__': f"{material} sheet - {thickness}\" thick\nX=length(8ft), Y=width(4ft), Z=thickness({thickness}\")"
            })
            
            classes[class_name] = cls
            globals()[class_name] = cls
    
    return classes


def plot_all_lumber_3d_standardized(length: float = 8.0, y_spacing: float = 2.0, z_spacing: float = 1.0):
    """
    Create a 3D visualization of all lumber types following standardization rules.
    X = length (longest), Y = height (second longest), Z = width (shortest)
    
    Args:
        length: Length of lumber pieces to display (feet) - X axis
        y_spacing: Spacing between lumber pieces along Y axis (feet)
        z_spacing: Spacing between lumber pieces along Z axis (feet)
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np
    except ImportError:
        print("matplotlib required for 3D plotting. Install with: pip install matplotlib")
        return
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get all lumber specs
    lumber_specs = [
        ('2X4', 1.5, 3.5),
        ('2X6', 1.5, 5.5),
        ('2X8', 1.5, 7.25),
        ('2X10', 1.5, 9.25),
        ('2X12', 1.5, 11.25),
        ('4X4', 3.5, 3.5),
        ('6X6', 5.5, 5.5),
        ('8X8', 7.25, 7.25),
        ('10X10', 9.25, 9.25),
        ('12X12', 11.25, 11.25),
    ]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(lumber_specs)))
    
    current_y = 0
    current_z = 0
    
    for i, (nominal, width_inches, height_inches) in enumerate(lumber_specs):
        # Convert to feet - following standardization
        width_ft = width_inches / 12.0   # Z dimension (shortest)
        height_ft = height_inches / 12.0 # Y dimension (middle)
        length_ft = length               # X dimension (longest)
        
        # Create lumber geometry following standardization: X=length, Y=height, Z=width
        x = [0, length_ft, length_ft, 0, 0, length_ft, length_ft, 0]
        y = [current_y, current_y, current_y + height_ft, current_y + height_ft, 
             current_y, current_y, current_y + height_ft, current_y + height_ft]
        z = [current_z, current_z, current_z, current_z, 
             current_z + width_ft, current_z + width_ft, current_z + width_ft, current_z + width_ft]
        
        # Define the 6 faces of the box
        faces = [
            [0, 1, 2, 3],  # bottom face (XY plane at z=current_z)
            [4, 5, 6, 7],  # top face (XY plane at z=current_z+width)
            [0, 1, 5, 4],  # front face (XZ plane at y=current_y)
            [2, 3, 7, 6],  # back face (XZ plane at y=current_y+height)
            [1, 2, 6, 5],  # right face (YZ plane at x=length)
            [0, 3, 7, 4]   # left face (YZ plane at x=0)
        ]
        
        # Create vertices for each face
        vertices = []
        for face in faces:
            face_vertices = [[x[j], y[j], z[j]] for j in face]
            vertices.append(face_vertices)
        
        # Add to plot
        poly3d = Poly3DCollection(vertices, alpha=0.7, facecolor=colors[i], edgecolor='black', linewidth=0.5)
        ax.add_collection3d(poly3d)
        
        # Add label showing dimensions
        label_text = f"{nominal}\n{width_inches}\"×{height_inches}\"×{length_ft}'"
        ax.text(length_ft/2, current_y + height_ft/2, current_z + width_ft + 0.05, 
                label_text, fontsize=8, ha='center', va='bottom')
        
        # Update position for next lumber piece
        current_y += height_ft + y_spacing
        if i == 4:  # After 2X12, start a new "row" for 4X4 and larger
            current_y = 0
            current_z += max([spec[1]/12.0 for spec in lumber_specs[:5]]) + z_spacing
    
    # Set labels and title following standardization
    ax.set_xlabel('X - Length (ft) - Longest Dimension')
    ax.set_ylabel('Y - Height (ft) - Second Longest Dimension')
    ax.set_zlabel('Z - Width (ft) - Shortest Dimension (Up)')
    ax.set_title('Standard Lumber Following Standardization Rules\nX=Length(longest), Y=Height(middle), Z=Width(shortest/up)')
    
    # Set limits
    ax.set_xlim(0, length_ft + 1)
    ax.set_ylim(0, current_y + max([spec[2]/12.0 for spec in lumber_specs]) + y_spacing)
    ax.set_zlim(0, current_z + max([spec[1]/12.0 for spec in lumber_specs[5:]]) + z_spacing)
    
    # Set view angle to better show the standardization
    ax.view_init(elev=25, azim=-45)
    
    # set aspect ratio
    # Set limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)

    ax.set_box_aspect([1, 1, 1])  # Equal aspect
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, label=spec[0]) 
                      for i, spec in enumerate(lumber_specs)]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    plt.tight_layout()
    plt.show()





def show_lumber_visualization():
    """Call this to display the lumber visualization following standardization"""
    print("Displaying 3D lumber visualization following standardization rules...")
    print("X = Length (longest), Y = Height (second longest), Z = Width (shortest/up)")
    plot_all_lumber_3d_standardized()
    

# Create all lumber and sheet classes
lumber_classes = create_lumber_classes()
sheet_classes = create_sheet_classes()

# Now available classes include:
# Lumber: Lumber2X4, Lumber2X6, Lumber2X8, Lumber2X10, Lumber2X12, Lumber4X4, Lumber6X6, etc.
# Sheets: PlywoodSheet_0_25, PlywoodSheet_0_375, OSBSheet_0_5, MDFSheet_0_75, etc.
#
# All follow standardization: X=longest, Y=second longest, Z=shortest (up)

# Uncomment to show visualization on import
if __name__ == "__main__":
    show_lumber_visualization()