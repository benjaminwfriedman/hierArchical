"""
Parametric plaster lath elements for traditional plaster ceiling systems.
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

class BasePlasterLath(ParametricElement):
    """Base class for all plaster lath elements"""
    
    MATERIAL_TYPE = None        # To be set by subclasses
    TYPICAL_WIDTH = None        # Typical width for this lath type
    TYPICAL_LENGTH = None       # Typical length for this lath type
    THICKNESS = None            # Typical thickness in inches
    INSTALLATION_METHOD = None  # Installation method
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'length': Parameter(
                name='length',
                type=float,
                default=cls.TYPICAL_LENGTH or 4.0,
                min_value=1.0,
                max_value=8.0,
                unit='ft',
                description="Length of the lath piece (X-axis - longest dimension)"
            ),
            'width': Parameter(
                name='width',
                type=float,
                default=cls.TYPICAL_WIDTH or 0.25,
                min_value=0.1,
                max_value=2.0,
                unit='ft',
                description="Width of the lath piece (Y-axis - middle dimension)"
            ),
            'thickness': Parameter(
                name='thickness',
                type=float,
                default=cls.THICKNESS or 0.375,
                min_value=0.125,
                max_value=1.0,
                unit='in',
                description="Thickness of the lath piece"
            )
        }
    
    @classmethod
    def get_material_type(cls) -> str:
        return cls.MATERIAL_TYPE.lower() if cls.MATERIAL_TYPE else "plaster_lath"
    
    def create_geometry(self) -> Geometry:
        # Following standardization: X=length, Y=width, Z=thickness
        base_points = [
            (0, 0),
            (self.params['length'], 0),
            (self.params['length'], self.params['width']),
            (0, self.params['width'])
        ]
        return Geometry.from_prism(base_points, self.params['thickness'] / 12.0)  # Convert inches to feet


# WOOD LATH
class BaseWoodLath(BasePlasterLath):
    """Base class for traditional wood lath"""
    
    MATERIAL_TYPE = "Wood Lath"
    INSTALLATION_METHOD = "Nail"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'wood_species': Parameter(
                name='wood_species',
                type=str,
                default='Pine',
                description="Wood species (Pine, Fir, Chestnut)"
            ),
            'grade': Parameter(
                name='grade',
                type=str,
                default='Standard',
                description="Lath grade (Standard, Select)"
            ),
            'moisture_content': Parameter(
                name='moisture_content',
                type=float,
                default=12.0,
                min_value=8.0,
                max_value=20.0,
                unit='%',
                description="Moisture content percentage"
            ),
            'spacing_requirement': Parameter(
                name='spacing_requirement',
                type=float,
                default=0.375,
                min_value=0.25,
                max_value=0.5,
                unit='in',
                description="Required spacing between lath strips"
            )
        })
        return params


class WoodLath_Standard(BaseWoodLath):
    """Standard wood lath - 1/4" x 1-1/2" x 4'"""
    TYPICAL_WIDTH = 1.5 / 12.0  # 1.5" converted to feet
    TYPICAL_LENGTH = 4.0
    THICKNESS = 0.25


class WoodLath_Wide(BaseWoodLath):
    """Wide wood lath - 3/8" x 2" x 4'"""
    TYPICAL_WIDTH = 2.0 / 12.0  # 2" converted to feet
    TYPICAL_LENGTH = 4.0
    THICKNESS = 0.375


# METAL LATH
class BaseMetalLath(BasePlasterLath):
    """Base class for metal lath"""
    
    MATERIAL_TYPE = "Metal Lath"
    INSTALLATION_METHOD = "Nail/Staple"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'metal_type': Parameter(
                name='metal_type',
                type=str,
                default='Galvanized Steel',
                description="Metal type (Galvanized Steel, Stainless Steel)"
            ),
            'gauge': Parameter(
                name='gauge',
                type=int,
                default=27,
                min_value=24,
                max_value=30,
                description="Metal gauge (thickness)"
            ),
            'mesh_pattern': Parameter(
                name='mesh_pattern',
                type=str,
                default='Diamond',
                description="Mesh pattern (Diamond, Square, Expanded)"
            ),
            'coating': Parameter(
                name='coating',
                type=str,
                default='Galvanized',
                description="Protective coating (Galvanized, Painted, Stainless)"
            ),
            'weight': Parameter(
                name='weight',
                type=float,
                default=3.4,
                min_value=2.5,
                max_value=8.0,
                unit='lb/sq yd',
                description="Weight per square yard"
            )
        })
        return params


class MetalLath_Diamond_27ga(BaseMetalLath):
    """Diamond mesh metal lath - 27 gauge"""
    TYPICAL_WIDTH = 2.25  # 27" standard width
    TYPICAL_LENGTH = 8.0
    THICKNESS = 0.05  # Very thin


class MetalLath_Expanded_24ga(BaseMetalLath):
    """Expanded metal lath - 24 gauge (heavier duty)"""
    TYPICAL_WIDTH = 2.25
    TYPICAL_LENGTH = 8.0
    THICKNESS = 0.08
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params['gauge'].default = 24
        params['weight'].default = 8.0  # Heavier
        return params


class MetalLath_Self_Furring(BaseMetalLath):
    """Self-furring metal lath with built-in spacers"""
    TYPICAL_WIDTH = 2.25
    TYPICAL_LENGTH = 8.0
    THICKNESS = 0.25  # Thicker due to furring dimples
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'furring_height': Parameter(
                name='furring_height',
                type=float,
                default=0.125,
                min_value=0.0625,
                max_value=0.25,
                unit='in',
                description="Height of furring dimples"
            )
        })
        return params


# GYPSUM LATH
class BaseGypsumLath(BasePlasterLath):
    """Base class for gypsum lath (modern alternative to wood lath)"""
    
    MATERIAL_TYPE = "Gypsum Lath"
    INSTALLATION_METHOD = "Nail/Screw"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'core_type': Parameter(
                name='core_type',
                type=str,
                default='Standard Gypsum',
                description="Core material (Standard Gypsum, Fire-Resistant, Moisture-Resistant)"
            ),
            'face_paper': Parameter(
                name='face_paper',
                type=str,
                default='Perforated',
                description="Face paper type (Perforated, Solid, Special)"
            ),
            'edge_type': Parameter(
                name='edge_type',
                type=str,
                default='Square',
                description="Edge type (Square, Rounded, Beveled)"
            ),
            'fire_rating': Parameter(
                name='fire_rating',
                type=str,
                default='Standard',
                description="Fire rating (Standard, Type X)"
            )
        })
        return params


class GypsumLath_3_8x16x48(BaseGypsumLath):
    """3/8" x 16" x 48" gypsum lath"""
    TYPICAL_WIDTH = 16.0 / 12.0  # 16" converted to feet
    TYPICAL_LENGTH = 4.0  # 48" converted to feet
    THICKNESS = 0.375


class GypsumLath_1_2x16x48(BaseGypsumLath):
    """1/2" x 16" x 48" gypsum lath"""
    TYPICAL_WIDTH = 16.0 / 12.0
    TYPICAL_LENGTH = 4.0
    THICKNESS = 0.5


# ROCK LATH (similar to gypsum but different composition)
class BaseRockLath(BasePlasterLath):
    """Base class for rock lath"""
    
    MATERIAL_TYPE = "Rock Lath"
    INSTALLATION_METHOD = "Nail/Screw"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'aggregate_type': Parameter(
                name='aggregate_type',
                type=str,
                default='Perlite',
                description="Aggregate type (Perlite, Vermiculite)"
            ),
            'density': Parameter(
                name='density',
                type=float,
                default=50.0,
                min_value=35.0,
                max_value=65.0,
                unit='lb/cu ft',
                description="Material density"
            ),
            'fire_resistance': Parameter(
                name='fire_resistance',
                type=int,
                default=60,
                min_value=30,
                max_value=120,
                unit='minutes',
                description="Fire resistance rating in minutes"
            )
        })
        return params


class RockLath_1_2x16x48(BaseRockLath):
    """1/2" x 16" x 48" rock lath"""
    TYPICAL_WIDTH = 16.0 / 12.0
    TYPICAL_LENGTH = 4.0
    THICKNESS = 0.5


# WIRE LATH (for heavy-duty applications)
class BaseWireLath(BasePlasterLath):
    """Base class for wire lath (welded wire mesh)"""
    
    MATERIAL_TYPE = "Wire Lath"
    INSTALLATION_METHOD = "Tie/Clip"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'wire_gauge': Parameter(
                name='wire_gauge',
                type=int,
                default=16,
                min_value=12,
                max_value=20,
                description="Wire gauge (thickness)"
            ),
            'mesh_size': Parameter(
                name='mesh_size',
                type=str,
                default='2x2',
                description="Mesh size (2x2, 1x2, 1x1 inches)"
            ),
            'coating': Parameter(
                name='coating',
                type=str,
                default='Galvanized',
                description="Wire coating (Galvanized, PVC, Stainless)"
            ),
            'tensile_strength': Parameter(
                name='tensile_strength',
                type=int,
                default=70000,
                min_value=50000,
                max_value=90000,
                unit='psi',
                description="Wire tensile strength"
            )
        })
        return params


class WireLath_2x2_16ga(BaseWireLath):
    """2" x 2" mesh, 16 gauge welded wire lath"""
    TYPICAL_WIDTH = 4.0  # 48" wide rolls
    TYPICAL_LENGTH = 8.0  # 8' long sections
    THICKNESS = 0.125  # Minimal thickness - mostly open


def show_plaster_lath_options():
    """Display available plaster lath options grouped by material type"""
    
    material_groups = {
        'Wood Lath': [],
        'Metal Lath': [],
        'Gypsum Lath': [],
        'Rock Lath': [],
        'Wire Lath': []
    }
    
    # Group classes by material type
    for name, cls in globals().items():
        if hasattr(cls, 'MATERIAL_TYPE') and hasattr(cls, 'TYPICAL_WIDTH'):
            material = cls.MATERIAL_TYPE
            if material in material_groups:
                material_groups[material].append((name, cls))
    
    print("Available Plaster Lath Options:")
    print("=" * 40)
    
    for material, classes in material_groups.items():
        if classes:
            print(f"\n{material}:")
            print("-" * 20)
            for name, cls in sorted(classes):
                width_in = cls.TYPICAL_WIDTH * 12 if cls.TYPICAL_WIDTH else 0
                length_ft = cls.TYPICAL_LENGTH if cls.TYPICAL_LENGTH else 0
                thickness_in = cls.THICKNESS if cls.THICKNESS else 0
                install_method = cls.INSTALLATION_METHOD if hasattr(cls, 'INSTALLATION_METHOD') else 'N/A'
                
                print(f"  {name}:")
                print(f"    Size: {length_ft}' × {width_in:.1f}\" × {thickness_in}\"")
                print(f"    Installation: {install_method}")


if __name__ == "__main__":
    print("Plaster Lath Elements - Traditional and Modern Systems")
    print("=" * 54)
    
    show_plaster_lath_options()
    
    # Example usage
    print("\nExample Usage:")
    print("-" * 15)
    
    # Create sample lath
    wood_lath = WoodLath_Standard(wood_species='Chestnut', moisture_content=10.0)
    wood_lath.name = "Traditional Chestnut Wood Lath"
    
    metal_lath = MetalLath_Diamond_27ga(metal_type='Galvanized Steel', mesh_pattern='Diamond')
    metal_lath.name = "Diamond Mesh Metal Lath"
    
    gypsum_lath = GypsumLath_3_8x16x48(core_type='Fire-Resistant', face_paper='Perforated')
    gypsum_lath.name = "Fire-Resistant Gypsum Lath"
    
    print(f"Created {wood_lath.name}: {wood_lath.MATERIAL_TYPE}")
    print(f"Created {metal_lath.name}: {metal_lath.MATERIAL_TYPE}")
    print(f"Created {gypsum_lath.name}: {gypsum_lath.MATERIAL_TYPE}")
    
    # Visualize if available
    try:
        from hierarchical.utils import plot_items
        
        # Position lath for comparison  
        metal_lath.move(dx=5.0)    # Move 5' to the right
        gypsum_lath.move(dx=10.0)  # Move 10' to the right
        
        print("Visualizing plaster lath types...")
        plot_items([wood_lath, metal_lath, gypsum_lath])
    except ImportError:
        print("Visualization not available")