"""
Parametric plaster elements - continuous coating systems applied over substrates.
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

class BasePlaster(ParametricElement):
    """Base class for all plaster coating elements"""
    
    MATERIAL_TYPE = None        # To be set by subclasses
    PLASTER_TYPE = None         # Lime, gypsum, clay, etc.
    APPLICATION_METHOD = None   # Hand trowel, spray, etc.
    TYPICAL_THICKNESS = None    # Total thickness in inches
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'length': Parameter(
                name='length',
                type=float,
                default=10.0,
                min_value=1.0,
                max_value=100.0,
                unit='ft',
                description="Length of plaster area (X-axis - longest dimension)"
            ),
            'width': Parameter(
                name='width',
                type=float,
                default=8.0,
                min_value=1.0,
                max_value=100.0,
                unit='ft',
                description="Width of plaster area (Y-axis - middle dimension)"
            ),
            'thickness': Parameter(
                name='thickness',
                type=float,
                default=cls.TYPICAL_THICKNESS or 0.75,
                min_value=0.25,
                max_value=2.0,
                unit='in',
                description="Total plaster thickness (Z-axis - shortest dimension)"
            )
        }
    
    @classmethod
    def get_material_type(cls) -> str:
        return cls.MATERIAL_TYPE.lower() if cls.MATERIAL_TYPE else "plaster"
    
    def create_geometry(self) -> Geometry:
        # Following standardization: X=length, Y=width, Z=thickness
        # Plaster is a continuous coating covering the entire area
        base_points = [
            (0, 0),
            (self.params['length'], 0),
            (self.params['length'], self.params['width']),
            (0, self.params['width'])
        ]
        return Geometry.from_prism(base_points, self.params['thickness'] / 12.0)  # Convert inches to feet


# TRADITIONAL LIME PLASTER
class BaseLimePlaster(BasePlaster):
    """Base class for traditional lime plaster systems"""
    
    MATERIAL_TYPE = "Lime Plaster"
    PLASTER_TYPE = "Lime"
    APPLICATION_METHOD = "Hand Trowel"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'lime_type': Parameter(
                name='lime_type',
                type=str,
                default='Hot Lime',
                description="Lime type (Hot Lime, Lime Putty, Hydraulic Lime)"
            ),
            'aggregate': Parameter(
                name='aggregate',
                type=str,
                default='Sharp Sand',
                description="Aggregate type (Sharp Sand, River Sand, Crushed Stone)"
            ),
            'hair_fiber': Parameter(
                name='hair_fiber',
                type=str,
                default='Goat Hair',
                description="Fiber reinforcement (Goat Hair, Horse Hair, None)"
            ),
            'mix_ratio': Parameter(
                name='mix_ratio',
                type=str,
                default='1:2.5',
                description="Lime to aggregate ratio (1:2.5, 1:3, etc.)"
            ),
            'num_coats': Parameter(
                name='num_coats',
                type=int,
                default=3,
                min_value=2,
                max_value=4,
                description="Number of plaster coats (scratch, brown, finish)"
            ),
            'cure_time': Parameter(
                name='cure_time',
                type=int,
                default=28,
                min_value=14,
                max_value=60,
                unit='days',
                description="Carbonation cure time"
            )
        })
        return params


class LimePlaster_ThreeCoat(BaseLimePlaster):
    """Traditional three-coat lime plaster system"""
    TYPICAL_THICKNESS = 0.75  # 3/4" total thickness
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'scratch_coat_thickness': Parameter(
                name='scratch_coat_thickness',
                type=float,
                default=0.375,
                min_value=0.25,
                max_value=0.5,
                unit='in',
                description="Scratch coat thickness"
            ),
            'brown_coat_thickness': Parameter(
                name='brown_coat_thickness',
                type=float,
                default=0.25,
                min_value=0.125,
                max_value=0.375,
                unit='in',
                description="Brown coat thickness"
            ),
            'finish_coat_thickness': Parameter(
                name='finish_coat_thickness',
                type=float,
                default=0.125,
                min_value=0.0625,
                max_value=0.25,
                unit='in',
                description="Finish coat thickness"
            )
        })
        return params


class LimePlaster_TwoCoat(BaseLimePlaster):
    """Two-coat lime plaster system"""
    TYPICAL_THICKNESS = 0.5  # 1/2" total thickness
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params['num_coats'].default = 2
        params.update({
            'base_coat_thickness': Parameter(
                name='base_coat_thickness',
                type=float,
                default=0.375,
                min_value=0.25,
                max_value=0.5,
                unit='in',
                description="Base coat thickness"
            ),
            'finish_coat_thickness': Parameter(
                name='finish_coat_thickness',
                type=float,
                default=0.125,
                min_value=0.0625,
                max_value=0.25,
                unit='in',
                description="Finish coat thickness"
            )
        })
        return params


# GYPSUM PLASTER
class BaseGypsumPlaster(BasePlaster):
    """Base class for gypsum plaster systems"""
    
    MATERIAL_TYPE = "Gypsum Plaster"
    PLASTER_TYPE = "Gypsum"
    APPLICATION_METHOD = "Hand Trowel"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'gypsum_type': Parameter(
                name='gypsum_type',
                type=str,
                default='Molding Plaster',
                description="Gypsum type (Molding Plaster, Gauging Plaster, Veneer Plaster)"
            ),
            'aggregate': Parameter(
                name='aggregate',
                type=str,
                default='Sand',
                description="Aggregate type (Sand, Perlite, Vermiculite)"
            ),
            'retarder': Parameter(
                name='retarder',
                type=str,
                default='Lime',
                description="Set retarder (Lime, Cream of Tartar, Commercial)"
            ),
            'water_ratio': Parameter(
                name='water_ratio',
                type=float,
                default=0.6,
                min_value=0.5,
                max_value=0.8,
                description="Water to plaster ratio by weight"
            ),
            'set_time': Parameter(
                name='set_time',
                type=int,
                default=45,
                min_value=20,
                max_value=120,
                unit='minutes',
                description="Working time before set"
            )
        })
        return params


class GypsumPlaster_ThreeCoat(BaseGypsumPlaster):
    """Traditional three-coat gypsum plaster system"""
    TYPICAL_THICKNESS = 0.875  # 7/8" total thickness


class GypsumPlaster_TwoCoat(BaseGypsumPlaster):
    """Two-coat gypsum plaster system"""
    TYPICAL_THICKNESS = 0.625  # 5/8" total thickness


class VeneerPlaster(BaseGypsumPlaster):
    """Thin veneer plaster system over gypsum base"""
    TYPICAL_THICKNESS = 0.125  # 1/8" thin coat
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params['gypsum_type'].default = 'Veneer Plaster'
        params.update({
            'base_type': Parameter(
                name='base_type',
                type=str,
                default='Blue Board',
                description="Base substrate (Blue Board, Gypsum Lath, Concrete)"
            ),
            'texture': Parameter(
                name='texture',
                type=str,
                default='Smooth',
                description="Finish texture (Smooth, Orange Peel, Knockdown, Skip Trowel)"
            )
        })
        return params


# CLAY PLASTER
class BaseClayPlaster(BasePlaster):
    """Base class for natural clay plaster systems"""
    
    MATERIAL_TYPE = "Clay Plaster"
    PLASTER_TYPE = "Clay"
    APPLICATION_METHOD = "Hand Trowel"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'clay_content': Parameter(
                name='clay_content',
                type=float,
                default=25.0,
                min_value=15.0,
                max_value=40.0,
                unit='%',
                description="Clay content percentage"
            ),
            'sand_type': Parameter(
                name='sand_type',
                type=str,
                default='Sharp Sand',
                description="Sand type (Sharp Sand, Fine Sand, Mica Sand)"
            ),
            'fiber': Parameter(
                name='fiber',
                type=str,
                default='Chopped Straw',
                description="Fiber reinforcement (Chopped Straw, Animal Hair, Synthetic)"
            ),
            'binder': Parameter(
                name='binder',
                type=str,
                default='Wheat Paste',
                description="Natural binder (Wheat Paste, Psyllium, Cactus Juice)"
            ),
            'pigment': Parameter(
                name='pigment',
                type=str,
                default='Natural Iron Oxide',
                description="Natural pigments for color"
            )
        })
        return params


class ClayPlaster_Alis(BaseClayPlaster):
    """Alis (finish coat) clay plaster"""
    TYPICAL_THICKNESS = 0.0625  # 1/16" very thin finish
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params['clay_content'].default = 35.0  # Higher clay content for smooth finish
        return params


class ClayPlaster_BaseCoat(BaseClayPlaster):
    """Base coat clay plaster"""
    TYPICAL_THICKNESS = 0.5  # 1/2" base coat


# SPECIALTY PLASTERS
class TadelaktPlaster(BaseLimePlaster):
    """Traditional Moroccan tadelakt lime plaster"""
    TYPICAL_THICKNESS = 0.25  # 1/4" polished finish
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'marble_powder': Parameter(
                name='marble_powder',
                type=bool,
                default=True,
                description="Include marble powder for smoothness"
            ),
            'soap_finish': Parameter(
                name='soap_finish',
                type=str,
                default='Black Soap',
                description="Soap type for sealing (Black Soap, Olive Oil Soap)"
            ),
            'polish_stone': Parameter(
                name='polish_stone',
                type=str,
                default='River Stone',
                description="Stone type for polishing"
            ),
            'wax_type': Parameter(
                name='wax_type',
                type=str,
                default='Beeswax',
                description="Final wax coating type"
            )
        })
        return params


class SGIPlaster(BaseGypsumPlaster):
    """Sgraffito decorative plaster technique"""
    TYPICAL_THICKNESS = 0.375  # 3/8" for layered decoration
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'base_color': Parameter(
                name='base_color',
                type=str,
                default='White',
                description="Base layer color"
            ),
            'top_color': Parameter(
                name='top_color',
                type=str,
                default='Ochre',
                description="Top layer color for scratching"
            ),
            'pattern_type': Parameter(
                name='pattern_type',
                type=str,
                default='Geometric',
                description="Sgraffito pattern (Geometric, Floral, Figurative)"
            )
        })
        return params


def show_plaster_options():
    """Display available plaster options grouped by material type"""
    
    material_groups = {
        'Lime Plaster': [],
        'Gypsum Plaster': [],
        'Clay Plaster': []
    }
    
    # Group classes by material type
    for name, cls in globals().items():
        if hasattr(cls, 'MATERIAL_TYPE') and hasattr(cls, 'TYPICAL_THICKNESS'):
            material = cls.MATERIAL_TYPE
            if material in material_groups:
                material_groups[material].append((name, cls))
    
    print("Available Plaster Coating Options:")
    print("=" * 42)
    
    for material, classes in material_groups.items():
        if classes:
            print(f"\n{material}:")
            print("-" * 20)
            for name, cls in sorted(classes):
                thickness_in = cls.TYPICAL_THICKNESS if cls.TYPICAL_THICKNESS else 0
                plaster_type = cls.PLASTER_TYPE if hasattr(cls, 'PLASTER_TYPE') else 'N/A'
                application = cls.APPLICATION_METHOD if hasattr(cls, 'APPLICATION_METHOD') else 'N/A'
                
                print(f"  {name}:")
                print(f"    Type: {plaster_type}")
                print(f"    Thickness: {thickness_in}\"")
                print(f"    Application: {application}")


if __name__ == "__main__":
    print("Plaster Elements - Continuous Coating Systems")
    print("=" * 47)
    
    show_plaster_options()
    
    # Example usage
    print("\nExample Usage:")
    print("-" * 15)
    
    # Create sample plaster coatings
    lime_plaster = LimePlaster_ThreeCoat(
        length=12.0, 
        width=10.0, 
        lime_type='Hot Lime', 
        hair_fiber='Goat Hair'
    )
    lime_plaster.name = "Traditional Lime Plaster Ceiling"
    
    clay_plaster = ClayPlaster_Alis(
        length=8.0, 
        width=8.0, 
        clay_content=35.0, 
        pigment='Natural Iron Oxide'
    )
    clay_plaster.name = "Clay Alis Finish"
    
    tadelakt = TadelaktPlaster(
        length=6.0, 
        width=6.0, 
        marble_powder=True, 
        soap_finish='Black Soap'
    )
    tadelakt.name = "Moroccan Tadelakt"
    
    print(f"Created {lime_plaster.name}: {lime_plaster.MATERIAL_TYPE}")
    print(f"Created {clay_plaster.name}: {clay_plaster.MATERIAL_TYPE}")
    print(f"Created {tadelakt.name}: {tadelakt.MATERIAL_TYPE}")
    
    # Visualize if available
    try:
        from hierarchical.utils import plot_items
        
        # Position plasters for comparison
        clay_plaster.move(dx=15.0)  # Move 15' to the right
        tadelakt.move(dx=25.0)      # Move 25' to the right
        
        print("Visualizing plaster coatings...")
        plot_items([lime_plaster, clay_plaster, tadelakt])
    except ImportError:
        print("Visualization not available")