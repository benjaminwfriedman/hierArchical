"""
Parametric ceiling tile elements for suspended ceiling systems.
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

class BaseCeilingTile(ParametricElement):
    """Base class for all ceiling tile elements"""
    
    MATERIAL_TYPE = None        # To be set by subclasses
    TYPICAL_WIDTH = None        # Typical width for this tile type
    TYPICAL_LENGTH = None       # Typical length for this tile type
    THICKNESS = None            # Typical thickness in inches
    EDGE_TYPE = None            # Edge type (tegular, square, etc.)
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'length': Parameter(
                name='length',
                type=float,
                default=cls.TYPICAL_LENGTH or 2.0,
                min_value=0.5,
                max_value=4.0,
                unit='ft',
                description="Length of the ceiling tile (X-axis - longest dimension)"
            ),
            'width': Parameter(
                name='width',
                type=float,
                default=cls.TYPICAL_WIDTH or 2.0,
                min_value=0.5,
                max_value=4.0,
                unit='ft',
                description="Width of the ceiling tile (Y-axis - middle dimension)"
            ),
            'thickness': Parameter(
                name='thickness',
                type=float,
                default=cls.THICKNESS or 0.625,
                min_value=0.375,
                max_value=1.0,
                unit='in',
                description="Thickness of the ceiling tile"
            )
        }
    
    @classmethod
    def get_material_type(cls) -> str:
        return cls.MATERIAL_TYPE.lower() if cls.MATERIAL_TYPE else "ceiling_tile"
    
    def create_geometry(self) -> Geometry:
        # Following standardization: X=length, Y=width, Z=thickness
        base_points = [
            (0, 0),
            (self.params['length'], 0),
            (self.params['length'], self.params['width']),
            (0, self.params['width'])
        ]
        return Geometry.from_prism(base_points, self.params['thickness'] / 12.0)  # Convert inches to feet


# ACOUSTIC CEILING TILES
class BaseAcousticTile(BaseCeilingTile):
    """Base class for acoustic ceiling tiles"""
    
    MATERIAL_TYPE = "Acoustic Tile"
    EDGE_TYPE = "Tegular"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'nrc_rating': Parameter(
                name='nrc_rating',
                type=float,
                default=0.70,
                min_value=0.0,
                max_value=1.0,
                description="Noise Reduction Coefficient (0.0-1.0)"
            ),
            'cac_rating': Parameter(
                name='cac_rating',
                type=int,
                default=35,
                min_value=25,
                max_value=50,
                description="Ceiling Attenuation Class rating"
            ),
            'edge_type': Parameter(
                name='edge_type',
                type=str,
                default=cls.EDGE_TYPE,
                description="Edge type (Tegular, Square, Beveled)"
            ),
            'surface_texture': Parameter(
                name='surface_texture',
                type=str,
                default='Textured',
                description="Surface texture (Smooth, Textured, Fissured)"
            )
        })
        return params


class AcousticTile_2x2(BaseAcousticTile):
    """2' x 2' acoustic ceiling tile"""
    TYPICAL_WIDTH = 2.0
    TYPICAL_LENGTH = 2.0
    THICKNESS = 0.625  # 5/8" typical


class AcousticTile_2x4(BaseAcousticTile):
    """2' x 4' acoustic ceiling tile"""
    TYPICAL_WIDTH = 2.0
    TYPICAL_LENGTH = 4.0
    THICKNESS = 0.625


class AcousticTile_1x4(BaseAcousticTile):
    """1' x 4' acoustic ceiling tile (linear pattern)"""
    TYPICAL_WIDTH = 1.0
    TYPICAL_LENGTH = 4.0
    THICKNESS = 0.75


# MINERAL FIBER TILES
class BaseMineralFiberTile(BaseCeilingTile):
    """Base class for mineral fiber ceiling tiles"""
    
    MATERIAL_TYPE = "Mineral Fiber"
    EDGE_TYPE = "Square"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'fire_rating': Parameter(
                name='fire_rating',
                type=str,
                default='Class A',
                description="Fire rating (Class A, Class B, Class C)"
            ),
            'humidity_resistance': Parameter(
                name='humidity_resistance',
                type=str,
                default='Standard',
                description="Humidity resistance (Standard, High, Washable)"
            ),
            'sag_resistance': Parameter(
                name='sag_resistance',
                type=str,
                default='Standard',
                description="Sag resistance rating"
            )
        })
        return params


class MineralFiberTile_2x2(BaseMineralFiberTile):
    """2' x 2' mineral fiber ceiling tile"""
    TYPICAL_WIDTH = 2.0
    TYPICAL_LENGTH = 2.0
    THICKNESS = 0.75


class MineralFiberTile_2x4(BaseMineralFiberTile):
    """2' x 4' mineral fiber ceiling tile"""
    TYPICAL_WIDTH = 2.0
    TYPICAL_LENGTH = 4.0
    THICKNESS = 0.75


# METAL CEILING TILES
class BaseMetalTile(BaseCeilingTile):
    """Base class for metal ceiling tiles"""
    
    MATERIAL_TYPE = "Metal"
    EDGE_TYPE = "Tegular"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'metal_type': Parameter(
                name='metal_type',
                type=str,
                default='Aluminum',
                description="Metal type (Aluminum, Steel, Stainless Steel)"
            ),
            'finish': Parameter(
                name='finish',
                type=str,
                default='Painted',
                description="Finish type (Painted, Anodized, Mill, Powder Coated)"
            ),
            'perforation_pattern': Parameter(
                name='perforation_pattern',
                type=str,
                default='None',
                description="Perforation pattern (None, Round, Square, Decorative)"
            ),
            'gauge': Parameter(
                name='gauge',
                type=int,
                default=24,
                min_value=20,
                max_value=26,
                description="Metal gauge (thickness)"
            )
        })
        return params


class MetalTile_2x2(BaseMetalTile):
    """2' x 2' metal ceiling tile"""
    TYPICAL_WIDTH = 2.0
    TYPICAL_LENGTH = 2.0
    THICKNESS = 0.05  # Very thin


class MetalTile_2x4(BaseMetalTile):
    """2' x 4' metal ceiling tile"""
    TYPICAL_WIDTH = 2.0
    TYPICAL_LENGTH = 4.0
    THICKNESS = 0.05


# WOOD CEILING TILES
class BaseWoodTile(BaseCeilingTile):
    """Base class for wood ceiling tiles"""
    
    MATERIAL_TYPE = "Wood"
    EDGE_TYPE = "Square"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'wood_species': Parameter(
                name='wood_species',
                type=str,
                default='Pine',
                description="Wood species (Pine, Oak, Cherry, Maple)"
            ),
            'finish': Parameter(
                name='finish',
                type=str,
                default='Unfinished',
                description="Finish type (Unfinished, Stained, Painted, Clear Coat)"
            ),
            'grain_direction': Parameter(
                name='grain_direction',
                type=str,
                default='With Length',
                description="Grain direction (With Length, Across Width)"
            )
        })
        return params


class WoodTile_2x2(BaseWoodTile):
    """2' x 2' wood ceiling tile"""
    TYPICAL_WIDTH = 2.0
    TYPICAL_LENGTH = 2.0
    THICKNESS = 0.375


class WoodTile_2x4(BaseWoodTile):
    """2' x 4' wood ceiling tile"""
    TYPICAL_WIDTH = 2.0
    TYPICAL_LENGTH = 4.0
    THICKNESS = 0.375


# GYPSUM CEILING TILES
class BaseGypsumTile(BaseCeilingTile):
    """Base class for gypsum ceiling tiles"""
    
    MATERIAL_TYPE = "Gypsum"
    EDGE_TYPE = "Tegular"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'backing_type': Parameter(
                name='backing_type',
                type=str,
                default='Vinyl',
                description="Backing material (Vinyl, Aluminum, None)"
            ),
            'fire_rating': Parameter(
                name='fire_rating',
                type=str,
                default='Class A',
                description="Fire rating classification"
            ),
            'moisture_resistance': Parameter(
                name='moisture_resistance',
                type=str,
                default='Standard',
                description="Moisture resistance level"
            )
        })
        return params


class GypsumTile_2x2(BaseGypsumTile):
    """2' x 2' gypsum ceiling tile"""
    TYPICAL_WIDTH = 2.0
    TYPICAL_LENGTH = 2.0
    THICKNESS = 0.5


class GypsumTile_2x4(BaseGypsumTile):
    """2' x 4' gypsum ceiling tile"""
    TYPICAL_WIDTH = 2.0  
    TYPICAL_LENGTH = 4.0
    THICKNESS = 0.5


def show_ceiling_tile_options():
    """Display available ceiling tile options grouped by material type"""
    
    material_groups = {
        'Acoustic Tile': [],
        'Mineral Fiber': [],
        'Metal': [],
        'Wood': [],
        'Gypsum': []
    }
    
    # Group classes by material type
    for name, cls in globals().items():
        if hasattr(cls, 'MATERIAL_TYPE') and hasattr(cls, 'TYPICAL_WIDTH'):
            material = cls.MATERIAL_TYPE
            if material in material_groups:
                material_groups[material].append((name, cls))
    
    print("Available Ceiling Tile Options:")
    print("=" * 40)
    
    for material, classes in material_groups.items():
        if classes:
            print(f"\n{material}:")
            print("-" * 20)
            for name, cls in sorted(classes):
                width_ft = cls.TYPICAL_WIDTH
                length_ft = cls.TYPICAL_LENGTH
                thickness_in = cls.THICKNESS
                edge_type = cls.EDGE_TYPE if hasattr(cls, 'EDGE_TYPE') else 'N/A'
                
                print(f"  {name}:")
                print(f"    Size: {length_ft}' × {width_ft}' × {thickness_in}\"")
                print(f"    Edge: {edge_type}")


if __name__ == "__main__":
    print("Ceiling Tile Elements - Standard Commercial/Residential Sizes")
    print("=" * 60)
    
    show_ceiling_tile_options()
    
    # Example usage
    print("\nExample Usage:")
    print("-" * 15)
    
    # Create sample tiles
    acoustic_tile = AcousticTile_2x2(nrc_rating=0.85, surface_texture='Fissured')
    acoustic_tile.name = "High Performance Acoustic Tile"
    
    metal_tile = MetalTile_2x4(metal_type='Aluminum', finish='Anodized', perforation_pattern='Round')
    metal_tile.name = "Perforated Aluminum Tile"
    
    wood_tile = WoodTile_2x2(wood_species='Oak', finish='Stained')
    wood_tile.name = "Oak Stained Tile"
    
    print(f"Created {acoustic_tile.name}: {acoustic_tile.MATERIAL_TYPE}")
    print(f"Created {metal_tile.name}: {metal_tile.MATERIAL_TYPE}")
    print(f"Created {wood_tile.name}: {wood_tile.MATERIAL_TYPE}")
    
    # Visualize if available
    try:
        from hierarchical.utils import plot_items
        
        # Position tiles for comparison
        metal_tile.move(dx=3.0)  # Move 3' to the right
        wood_tile.move(dx=6.0)   # Move 6' to the right
        
        print("Visualizing ceiling tiles...")
        plot_items([acoustic_tile, metal_tile, wood_tile])
    except ImportError:
        print("Visualization not available")