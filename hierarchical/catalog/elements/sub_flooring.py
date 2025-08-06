"""
Parametric subflooring elements using existing sheet classes with subflooring-specific properties.
"""

from typing import Dict, List, Tuple
from hierarchical.catalog.base import ParametricElement, Parameter
from hierarchical.catalog.elements.lumber import (
    PlywoodSheet_0_5, PlywoodSheet_0_625, PlywoodSheet_0_75, PlywoodSheet_1_0, PlywoodSheet_1_25,
    OSBSheet_0_5, OSBSheet_0_625, OSBSheet_0_75, OSBSheet_1_0, OSBSheet_1_25
)
from hierarchical.geometry import Geometry

## STANDARDIZATION ##

# All items are positioned starting at 0,0,0 with:
# X axis = longest dimension (length)
# Y axis = second longest dimension (width)  
# Z axis = shortest dimension (thickness - "up" when relevant)
# This ensures consistency in positioning, rotation, and alignment.

class SubflooringProperties:
    """Mixin class to add subflooring-specific properties to existing sheet classes"""
    
    # Standard subflooring specifications
    SUBFLOORING_SPECS = {
        # Format: thickness -> (span_rating, typical_joist_spacing, edge_type_options)
        0.5: ("16/0", [12, 16], ["Square"]),
        0.625: ("20/0", [16, 19.2], ["Square", "T&G"]),
        0.75: ("24/16", [16, 19.2, 24], ["Square", "T&G"]),
        0.875: ("32/16", [19.2, 24], ["Square", "T&G"]),
        1.0: ("32/16", [19.2, 24], ["Square", "T&G"]),
        1.125: ("48/24", [24], ["T&G"]),
        1.25: ("48/24", [24], ["T&G"]),
    }
    
    @classmethod
    def get_subflooring_parameters(cls) -> Dict[str, Parameter]:
        """Add subflooring-specific parameters to base sheet parameters"""
        base_params = super().get_parameters()
        
        # Get specs for this thickness
        thickness = cls.THICKNESS
        span_rating, joist_spacings, edge_options = cls.SUBFLOORING_SPECS.get(thickness, ("24/16", [16], ["Square"]))
        
        subflooring_params = {
            'span_rating': Parameter(
                name='span_rating',
                type=str,
                default=span_rating,
                description=f"APA span rating for {thickness}\" subflooring"
            ),
            'edge_type': Parameter(
                name='edge_type',
                type=str,
                default=edge_options[0],
                description=f"Edge type: {', '.join(edge_options)}"
            ),
            'grade': Parameter(
                name='grade',
                type=str,
                default='Structural',
                description="Grade specification (Structural, Exposure 1, etc.)"
            ),
            'joist_spacing': Parameter(
                name='joist_spacing',
                type=float,
                default=min(joist_spacings),
                min_value=12.0,
                max_value=24.0,
                unit='in',
                description=f"Recommended joist spacing: {joist_spacings} inches"
            ),
            'glue_compatible': Parameter(
                name='glue_compatible',
                type=bool,
                default=True,
                description="Compatible with construction adhesive"
            )
        }
        
        # Merge with base parameters
        base_params.update(subflooring_params)
        return base_params


def create_subflooring_classes():
    """Create subflooring classes by adding properties to existing sheet classes"""
    
    # Map existing sheet classes to subflooring variants
    subflooring_mappings = [
        # (base_class, subflooring_name, thickness)
        (PlywoodSheet_0_5, "PlywoodSubflooring_0_5", 0.5),
        (PlywoodSheet_0_625, "PlywoodSubflooring_0_625", 0.625),
        (PlywoodSheet_0_75, "PlywoodSubflooring_0_75", 0.75),
        (PlywoodSheet_1_0, "PlywoodSubflooring_1_0", 1.0),
        (PlywoodSheet_1_25, "PlywoodSubflooring_1_25", 1.25),
        
        (OSBSheet_0_5, "OSBSubflooring_0_5", 0.5),
        (OSBSheet_0_625, "OSBSubflooring_0_625", 0.625),
        (OSBSheet_0_75, "OSBSubflooring_0_75", 0.75),
        (OSBSheet_1_0, "OSBSubflooring_1_0", 1.0),
        (OSBSheet_1_25, "OSBSubflooring_1_25", 1.25),
    ]
    
    classes = {}
    
    for base_class, class_name, thickness in subflooring_mappings:
        # Create new class that inherits from both the base sheet class and subflooring properties
        cls = type(class_name, (SubflooringProperties, base_class), {
            'THICKNESS': thickness,
            'IS_SUBFLOORING': True,
            '__doc__': f"{base_class.__name__} configured for subflooring use - {thickness}\" thick\n"
                       f"Includes span ratings, joist spacing, and edge type specifications\n"
                       f"X=length(8ft), Y=width(4ft), Z=thickness({thickness}\")",
            '__module__': __name__  # Fix pickle support
        })
        
        # Override get_parameters to include subflooring properties
        cls.get_parameters = classmethod(lambda cls: cls.get_subflooring_parameters())
        
        classes[class_name] = cls
        globals()[class_name] = cls
    
    return classes


# Advantech is specialty engineered lumber, so we'll create a dedicated class
class AdvantechSubflooring(SubflooringProperties, ParametricElement):
    """Advantech engineered subflooring with moisture resistance"""
    
    ACTUAL_WIDTH = 4.0      # feet (Y axis - standard width)
    ACTUAL_LENGTH = 8.0     # feet (X axis - standard length)
    MATERIAL_TYPE = "Advantech"
    EDGE_TYPE = "T&G"       # Always tongue and groove
    
    def __init__(self, thickness: float = 0.75, **kwargs):
        self.THICKNESS = thickness
        super().__init__(**kwargs)
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        base_params = {
            'length': Parameter(
                name='length',
                type=float,
                default=8.0,
                min_value=0.01,
                max_value=12.0,
                unit='ft',
                description="Length of Advantech sheet (X-axis - longest dimension)"
            ),
            'width': Parameter(
                name='width',
                type=float,
                default=4.0,
                min_value=0.01,
                max_value=4.5,
                unit='ft',
                description="Width of Advantech sheet (Y-axis - second longest dimension)"
            ),
            'thickness': Parameter(
                name='thickness',
                type=float,
                default=0.75,
                min_value=0.625,
                max_value=1.25,
                unit='in',
                description="Thickness of Advantech sheet"
            )
        }
        
        # Add subflooring properties
        cls_instance = cls.__new__(cls)
        cls_instance.THICKNESS = 0.75  # Default for property lookup
        subflooring_params = cls_instance.get_subflooring_parameters()
        
        # Override defaults for Advantech
        subflooring_params['edge_type'].default = "T&G"
        subflooring_params['grade'].default = "Moisture Resistant"
        
        base_params.update({k: v for k, v in subflooring_params.items() 
                           if k not in ['length', 'width']})
        return base_params
    
    @classmethod
    def get_material_type(cls) -> str:
        return "engineered wood"
    
    def create_geometry(self) -> Geometry:
        # Following standardization: X=length, Y=width, Z=thickness
        thickness = self.params.get('thickness', self.THICKNESS)
        base_points = [
            (0, 0),
            (self.params['length'], 0),
            (self.params['length'], self.params['width']),
            (0, self.params['width'])
        ]
        return Geometry.from_prism(base_points, thickness / 12.0)  # Convert inches to feet for Z


def show_subflooring_options():
    """Display available subflooring options and their specifications"""
    print("Available Subflooring Options:")
    print("=" * 60)
    
    for name, cls in globals().items():
        if (hasattr(cls, 'IS_SUBFLOORING') or 'Subflooring' in name) and hasattr(cls, 'THICKNESS'):
            thickness = cls.THICKNESS
            if thickness in SubflooringProperties.SUBFLOORING_SPECS:
                span_rating, joist_spacings, edge_options = SubflooringProperties.SUBFLOORING_SPECS[thickness]
                material = "Plywood" if "Plywood" in name else "OSB" if "OSB" in name else "Advantech"
                
                print(f"\n{name}:")
                print(f"  Material: {material}")
                print(f"  Thickness: {thickness}\"")
                print(f"  Span Rating: {span_rating}")
                print(f"  Joist Spacing: {joist_spacings} inches")
                print(f"  Edge Options: {', '.join(edge_options)}")


# Create all subflooring classes
subflooring_classes = create_subflooring_classes()

if __name__ == "__main__":
    print("Subflooring system using existing sheet classes with added properties")
    print("=" * 65)
    
    show_subflooring_options()
    
   