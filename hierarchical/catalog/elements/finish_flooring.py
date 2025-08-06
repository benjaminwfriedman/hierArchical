"""
Parametric finish flooring elements with material-specific dimensions and properties.
"""

from typing import Dict, List
from hierarchical.catalog.base import ParametricElement, Parameter
from hierarchical.geometry import Geometry
from abc import ABC, abstractmethod

## STANDARDIZATION ##

# All items are positioned starting at 0,0,0 with:
# X axis = longest dimension (length)
# Y axis = second longest dimension (width)  
# Z axis = shortest dimension (thickness - "up" when relevant)
# This ensures consistency in positioning, rotation, and alignment.

class BaseFinishFlooring(ParametricElement, ABC):
    """Base class for all finish flooring elements"""
    
    MATERIAL_TYPE = None        # To be set by subclasses
    INSTALLATION_METHOD = None  # To be set by subclasses (nail, glue, float, etc.)
    TYPICAL_WIDTH = None        # Typical width for this flooring type
    TYPICAL_LENGTH = None       # Typical length for this flooring type
    THICKNESS = None            # Typical thickness in inches
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'length': Parameter(
                name='length',
                type=float,
                default=cls.TYPICAL_LENGTH or 4.0,
                min_value=0.1,
                max_value=cls._get_max_length(),
                unit='ft',
                description="Length of the flooring piece (X-axis - longest dimension)"
            ),
            'width': Parameter(
                name='width',
                type=float,
                default=cls.TYPICAL_WIDTH or 0.25,
                min_value=0.1,
                max_value=cls._get_max_width(),
                unit='ft',
                description="Width of the flooring piece (Y-axis - middle dimension)"
            ),
            'thickness': Parameter(
                name='thickness',
                type=float,
                default=cls.THICKNESS or 0.75,
                min_value=0.1,
                max_value=1.5,
                unit='in',
                description="Thickness of the flooring piece (Z-axis - shortest dimension)"
            )
        }
    
    @classmethod
    def _get_max_length(cls):
        return 12.0  # Default max length
    
    @classmethod
    def _get_max_width(cls):
        return 1.0   # Default max width
    
    @classmethod
    def get_material_type(cls) -> str:
        return cls.MATERIAL_TYPE.lower() if cls.MATERIAL_TYPE else "flooring"
    
    def create_geometry(self) -> Geometry:
        # Following standardization: X=length, Y=width, Z=thickness
        base_points = [
            (0, 0),
            (self.params['length'], 0),
            (self.params['length'], self.params['width']),
            (0, self.params['width'])
        ]
        return Geometry.from_prism(base_points, self.params['thickness'] / 12.0)  # Convert inches to feet


# HARDWOOD FLOORING CLASSES
class BaseHardwood(BaseFinishFlooring):
    """Base class for solid hardwood flooring"""
    
    MATERIAL_TYPE = "Hardwood"
    INSTALLATION_METHOD = "Nail"
    EDGE_TYPE = "Tongue & Groove"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'species': Parameter(
                name='species',
                type=str,
                default='Red Oak',
                description="Wood species (Red Oak, White Oak, Maple, etc.)"
            ),
            'grade': Parameter(
                name='grade',
                type=str,
                default='Select',
                description="NOFMA grade (Select, #1 Common, #2 Common)"
            ),
            'finish': Parameter(
                name='finish',
                type=str,
                default='Prefinished',
                description="Finish type (Prefinished, Unfinished)"
            )
        })
        return params
    
    @classmethod
    def _get_max_length(cls):
        return 8.0  # Hardwood typically comes up to 8' lengths
    
    @classmethod
    def _get_max_width(cls):
        return 0.5  # Hardwood up to 6" wide


class Hardwood_2_25(BaseHardwood):
    """2-1/4" wide solid hardwood flooring"""
    TYPICAL_WIDTH = 2.25 / 12.0  # Convert inches to feet
    TYPICAL_LENGTH = 4.0
    THICKNESS = 0.75


class Hardwood_3_0(BaseHardwood):
    """3" wide solid hardwood flooring"""
    TYPICAL_WIDTH = 3.0 / 12.0
    TYPICAL_LENGTH = 4.0
    THICKNESS = 0.75


class Hardwood_5_0(BaseHardwood):
    """5" wide solid hardwood flooring"""
    TYPICAL_WIDTH = 5.0 / 12.0
    TYPICAL_LENGTH = 4.0
    THICKNESS = 0.75


# ENGINEERED HARDWOOD CLASSES
class BaseEngineeredHardwood(BaseFinishFlooring):
    """Base class for engineered hardwood flooring"""
    
    MATERIAL_TYPE = "Engineered Hardwood"
    INSTALLATION_METHOD = "Float/Glue"
    EDGE_TYPE = "Click-Lock"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'species': Parameter(
                name='species',
                type=str,
                default='Oak',
                description="Wood species veneer (Oak, Maple, Hickory, etc.)"
            ),
            'core_material': Parameter(
                name='core_material',
                type=str,
                default='Plywood',
                description="Core material (Plywood, HDF, Softwood)"
            ),
            'wear_layer': Parameter(
                name='wear_layer',
                type=float,
                default=2.0,
                min_value=0.6,
                max_value=6.0,
                unit='mm',
                description="Wear layer thickness"
            )
        })
        return params
    
    @classmethod
    def _get_max_length(cls):
        return 7.0  # Engineered typically up to 7' planks
    
    @classmethod
    def _get_max_width(cls):
        return 0.75  # Up to 9" wide


class EngineeredHardwood_5_0(BaseEngineeredHardwood):
    """5" wide engineered hardwood"""
    TYPICAL_WIDTH = 5.0 / 12.0
    TYPICAL_LENGTH = 4.0
    THICKNESS = 0.5


class EngineeredHardwood_7_0(BaseEngineeredHardwood):
    """7" wide engineered hardwood"""
    TYPICAL_WIDTH = 7.0 / 12.0
    TYPICAL_LENGTH = 4.0
    THICKNESS = 0.5


# LVP/LVT CLASSES
class BaseLVP(BaseFinishFlooring):
    """Base class for Luxury Vinyl Plank flooring"""
    
    MATERIAL_TYPE = "LVP"
    INSTALLATION_METHOD = "Float"
    EDGE_TYPE = "Click-Lock"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'wear_layer': Parameter(
                name='wear_layer',
                type=float,
                default=12.0,
                min_value=6.0,
                max_value=28.0,
                unit='mil',
                description="Wear layer thickness in mils"
            ),
            'core_type': Parameter(
                name='core_type',
                type=str,
                default='SPC',
                description="Core type (SPC, WPC, Rigid)"
            ),
            'waterproof': Parameter(
                name='waterproof',
                type=bool,
                default=True,
                description="100% waterproof rating"
            )
        })
        return params
    
    @classmethod
    def _get_max_length(cls):
        return 6.0  # LVP planks typically up to 6'
    
    @classmethod
    def _get_max_width(cls):
        return 0.75  # Up to 9" wide


class LVP_6x48(BaseLVP):
    """6" x 48" LVP plank"""
    TYPICAL_WIDTH = 6.0 / 12.0
    TYPICAL_LENGTH = 48.0 / 12.0
    THICKNESS = 0.24  # 6mm typical


class LVP_7x48(BaseLVP):
    """7" x 48" LVP plank"""
    TYPICAL_WIDTH = 7.0 / 12.0
    TYPICAL_LENGTH = 48.0 / 12.0
    THICKNESS = 0.28  # 7mm typical


class LVP_9x60(BaseLVP):
    """9" x 60" LVP plank"""
    TYPICAL_WIDTH = 9.0 / 12.0
    TYPICAL_LENGTH = 60.0 / 12.0
    THICKNESS = 0.32  # 8mm typical


# TILE CLASSES
class BaseTile(BaseFinishFlooring):
    """Base class for ceramic and stone tile"""
    
    MATERIAL_TYPE = "Tile"
    INSTALLATION_METHOD = "Adhesive"
    EDGE_TYPE = "Straight"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'tile_type': Parameter(
                name='tile_type',
                type=str,
                default='Ceramic',
                description="Tile type (Ceramic, Porcelain, Natural Stone)"
            ),
            'finish': Parameter(
                name='finish',
                type=str,
                default='Glazed',
                description="Surface finish (Glazed, Unglazed, Polished, Textured)"
            ),
            'grout_width': Parameter(
                name='grout_width',
                type=float,
                default=0.125,
                min_value=0.0625,
                max_value=0.5,
                unit='in',
                description="Grout joint width"
            )
        })
        return params
    
    @classmethod
    def _get_max_length(cls):
        return 4.0  # Large format tiles up to 48"
    
    @classmethod
    def _get_max_width(cls):
        return 4.0  # Square tiles up to 48"


class Tile_12x12(BaseTile):
    """12" x 12" ceramic tile"""
    TYPICAL_WIDTH = 12.0 / 12.0
    TYPICAL_LENGTH = 12.0 / 12.0
    THICKNESS = 0.375  # 3/8" typical


class Tile_18x18(BaseTile):
    """18" x 18" ceramic tile"""
    TYPICAL_WIDTH = 18.0 / 12.0
    TYPICAL_LENGTH = 18.0 / 12.0
    THICKNESS = 0.5  # 1/2" typical


class Tile_12x24(BaseTile):
    """12" x 24" rectangular tile"""
    TYPICAL_WIDTH = 12.0 / 12.0
    TYPICAL_LENGTH = 24.0 / 12.0
    THICKNESS = 0.375


class Tile_6x36(BaseTile):
    """6" x 36" wood-look tile"""
    TYPICAL_WIDTH = 6.0 / 12.0
    TYPICAL_LENGTH = 36.0 / 12.0
    THICKNESS = 0.375


# CARPET CLASSES
class BaseCarpet(BaseFinishFlooring):
    """Base class for carpet flooring"""
    
    MATERIAL_TYPE = "Carpet"
    INSTALLATION_METHOD = "Stretch"
    EDGE_TYPE = "Cut"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'fiber_type': Parameter(
                name='fiber_type',
                type=str,
                default='Nylon',
                description="Fiber type (Nylon, Polyester, Wool, Polypropylene)"
            ),
            'pile_height': Parameter(
                name='pile_height',
                type=float,
                default=0.375,
                min_value=0.125,
                max_value=1.0,
                unit='in',
                description="Pile height"
            ),
            'backing_type': Parameter(
                name='backing_type',
                type=str,
                default='Action Bac',
                description="Backing type (Action Bac, Unitary, Woven)"
            )
        })
        return params
    
    @classmethod
    def _get_max_length(cls):
        return 100.0  # Carpet comes in very long rolls
    
    @classmethod
    def _get_max_width(cls):
        return 15.0  # Broadloom carpet up to 15' wide


class Carpet_12ft_Roll(BaseCarpet):
    """12' wide carpet roll"""
    TYPICAL_WIDTH = 12.0
    TYPICAL_LENGTH = 20.0  # Default 20' length
    THICKNESS = 0.375  # 3/8" typical with pad


class Carpet_15ft_Roll(BaseCarpet):
    """15' wide carpet roll"""
    TYPICAL_WIDTH = 15.0
    TYPICAL_LENGTH = 20.0
    THICKNESS = 0.375


# LAMINATE CLASSES
class BaseLaminate(BaseFinishFlooring):
    """Base class for laminate flooring"""
    
    MATERIAL_TYPE = "Laminate"
    INSTALLATION_METHOD = "Float"
    EDGE_TYPE = "Click-Lock"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'core_type': Parameter(
                name='core_type',
                type=str,
                default='HDF',
                description="Core material (HDF, MDF)"
            ),
            'ac_rating': Parameter(
                name='ac_rating',
                type=str,
                default='AC3',
                description="AC rating (AC1-AC5) for durability"
            ),
            'surface_texture': Parameter(
                name='surface_texture',
                type=str,
                default='Embossed',
                description="Surface texture (Smooth, Embossed, Hand-scraped)"
            )
        })
        return params
    
    @classmethod
    def _get_max_length(cls):
        return 5.0  # Laminate planks typically up to 5'
    
    @classmethod
    def _get_max_width(cls):
        return 0.67  # Up to 8" wide


class Laminate_5x47(BaseLaminate):
    """5" x 47" laminate plank"""
    TYPICAL_WIDTH = 5.0 / 12.0
    TYPICAL_LENGTH = 47.0 / 12.0
    THICKNESS = 0.315  # 8mm typical


class Laminate_8x47(BaseLaminate):
    """8" x 47" laminate plank"""
    TYPICAL_WIDTH = 8.0 / 12.0
    TYPICAL_LENGTH = 47.0 / 12.0
    THICKNESS = 0.472  # 12mm typical


def create_finish_flooring_classes():
    """Generate additional finish flooring classes with common sizes"""
    
    classes = {}
    
    # Additional hardwood sizes
    hardwood_sizes = [
        (2.25, 'Hardwood_2_25'),
        (3.25, 'Hardwood_3_25'),
        (4.0, 'Hardwood_4_0'),
        (6.0, 'Hardwood_6_0')
    ]
    
    for width_inches, class_name in hardwood_sizes:
        if class_name not in globals():
            cls = type(class_name, (BaseHardwood,), {
                'TYPICAL_WIDTH': width_inches / 12.0,
                'TYPICAL_LENGTH': 4.0,
                'THICKNESS': 0.75,
                '__doc__': f"{width_inches}\" wide solid hardwood flooring",
                '__module__': __name__  # Fix pickle support
            })
            classes[class_name] = cls
            globals()[class_name] = cls
    
    # Additional LVP sizes
    lvp_sizes = [
        (4, 36, 'LVP_4x36'),
        (5, 48, 'LVP_5x48'),
        (8, 48, 'LVP_8x48'),
        (9, 48, 'LVP_9x48')
    ]
    
    for width, length, class_name in lvp_sizes:
        if class_name not in globals():
            cls = type(class_name, (BaseLVP,), {
                'TYPICAL_WIDTH': width / 12.0,
                'TYPICAL_LENGTH': length / 12.0,
                'THICKNESS': 0.24,
                '__doc__': f"{width}\" x {length}\" LVP plank",
                '__module__': __name__  # Fix pickle support
            })
            classes[class_name] = cls
            globals()[class_name] = cls
    
    # Additional tile sizes
    tile_sizes = [
        (6, 6, 'Tile_6x6'),
        (8, 8, 'Tile_8x8'),
        (16, 16, 'Tile_16x16'),
        (24, 24, 'Tile_24x24'),
        (3, 12, 'Tile_3x12'),  # Subway tile
        (4, 16, 'Tile_4x16')   # Linear tile
    ]
    
    for width, length, class_name in tile_sizes:
        if class_name not in globals():
            cls = type(class_name, (BaseTile,), {
                'TYPICAL_WIDTH': width / 12.0,
                'TYPICAL_LENGTH': length / 12.0,
                'THICKNESS': 0.375,
                '__doc__': f"{width}\" x {length}\" ceramic tile",
                '__module__': __name__  # Fix pickle support
            })
            classes[class_name] = cls
            globals()[class_name] = cls
    
    return classes


def show_flooring_options_by_material():
    """Display available finish flooring options grouped by material type"""
    
    material_groups = {
        'Hardwood': [],
        'Engineered Hardwood': [],
        'LVP': [],
        'Tile': [],
        'Carpet': [],
        'Laminate': []
    }
    
    # Group classes by material type
    for name, cls in globals().items():
        if hasattr(cls, 'MATERIAL_TYPE') and hasattr(cls, 'TYPICAL_WIDTH'):
            material = cls.MATERIAL_TYPE
            if material in material_groups:
                material_groups[material].append((name, cls))
    
    print("Available Finish Flooring Options:")
    print("=" * 50)
    
    for material, classes in material_groups.items():
        if classes:
            print(f"\n{material}:")
            print("-" * 20)
            for name, cls in sorted(classes):
                width_in = cls.TYPICAL_WIDTH * 12 if cls.TYPICAL_WIDTH else 0
                length_in = cls.TYPICAL_LENGTH * 12 if cls.TYPICAL_LENGTH else 0
                thickness_in = cls.THICKNESS if cls.THICKNESS else 0
                install_method = cls.INSTALLATION_METHOD if hasattr(cls, 'INSTALLATION_METHOD') else 'N/A'
                
                print(f"  {name}:")
                print(f"    Size: {width_in:.1f}\" × {length_in:.1f}\" × {thickness_in:.3f}\"")
                print(f"    Installation: {install_method}")


def visualize_flooring_samples():
    """Create sample instances of each flooring type and visualize with plot_items"""
    try:
        from hierarchical.utils import plot_items
    except ImportError:
        print("plot_items not available - showing text output only")
        show_flooring_options_by_material()
        return
    
    # Create sample instances of each major flooring type
    flooring_samples = []
    
    # Spacing parameters
    x_spacing = 5.0  # feet between samples along X axis
    y_spacing = 3.0  # feet between samples along Y axis
    z_spacing = 0.2  # feet between samples along Z axis (stacking)
    
    current_x = 0
    current_y = 0
    current_z = 0
    samples_per_row = 3  # Number of samples per row
    sample_count = 0
    
    # Hardwood samples
    try:
        hardwood_sample = Hardwood_2_25(length=4.0, species='Red Oak')
        hardwood_sample.name = "Red Oak 2.25\" Hardwood"
        hardwood_sample.move(dx=current_x, dy=current_y, dz=current_z)
        flooring_samples.append(hardwood_sample)
        
        # Update position for next sample
        sample_count += 1
        current_x += x_spacing
        if sample_count % samples_per_row == 0:
            current_x = 0
            current_y += y_spacing
    except:
        pass
    
    try:
        hardwood_wide = Hardwood_5_0(length=4.0, species='White Oak')
        hardwood_wide.name = "White Oak 5\" Hardwood"
        hardwood_wide.move(dx=current_x, dy=current_y, dz=current_z)
        flooring_samples.append(hardwood_wide)
        
        sample_count += 1
        current_x += x_spacing
        if sample_count % samples_per_row == 0:
            current_x = 0
            current_y += y_spacing
    except:
        pass
    
    # Engineered hardwood samples
    try:
        eng_hardwood = EngineeredHardwood_5_0(length=4.0, species='Oak')
        eng_hardwood.name = "Oak 5\" Engineered"
        eng_hardwood.move(dx=current_x, dy=current_y, dz=current_z)
        flooring_samples.append(eng_hardwood)
        
        sample_count += 1
        current_x += x_spacing
        if sample_count % samples_per_row == 0:
            current_x = 0
            current_y += y_spacing
    except:
        pass
    
    # LVP samples
    try:
        lvp_sample = LVP_6x48(core_type='SPC')
        lvp_sample.name = "6×48 SPC LVP"
        lvp_sample.move(dx=current_x, dy=current_y, dz=current_z)
        flooring_samples.append(lvp_sample)
        
        sample_count += 1
        current_x += x_spacing
        if sample_count % samples_per_row == 0:
            current_x = 0
            current_y += y_spacing
    except:
        pass
    
    try:
        lvp_wide = LVP_9x60(core_type='WPC')
        lvp_wide.name = "9×60 WPC LVP"
        lvp_wide.move(dx=current_x, dy=current_y, dz=current_z)
        flooring_samples.append(lvp_wide)
        
        sample_count += 1
        current_x += x_spacing
        if sample_count % samples_per_row == 0:
            current_x = 0
            current_y += y_spacing
    except:
        pass
    
    # Tile samples
    try:
        tile_square = Tile_12x12(tile_type='Porcelain')
        tile_square.name = "12×12 Porcelain Tile"
        tile_square.move(dx=current_x, dy=current_y, dz=current_z)
        flooring_samples.append(tile_square)
        
        sample_count += 1
        current_x += x_spacing
        if sample_count % samples_per_row == 0:
            current_x = 0
            current_y += y_spacing
    except:
        pass
    
    try:
        tile_rect = Tile_12x24(tile_type='Ceramic')
        tile_rect.name = "12×24 Ceramic Tile"
        tile_rect.move(dx=current_x, dy=current_y, dz=current_z)
        flooring_samples.append(tile_rect)
        
        sample_count += 1
        current_x += x_spacing
        if sample_count % samples_per_row == 0:
            current_x = 0
            current_y += y_spacing
    except:
        pass
    
    try:
        tile_plank = Tile_6x36(tile_type='Porcelain')
        tile_plank.name = "6×36 Wood-Look Tile"
        tile_plank.move(dx=current_x, dy=current_y, dz=current_z)
        flooring_samples.append(tile_plank)
        
        sample_count += 1
        current_x += x_spacing
        if sample_count % samples_per_row == 0:
            current_x = 0
            current_y += y_spacing
    except:
        pass
    
    # Laminate samples
    try:
        laminate_sample = Laminate_5x47(ac_rating='AC3')
        laminate_sample.name = "5×47 AC3 Laminate"
        laminate_sample.move(dx=current_x, dy=current_y, dz=current_z)
        flooring_samples.append(laminate_sample)
        
        sample_count += 1
        current_x += x_spacing
        if sample_count % samples_per_row == 0:
            current_x = 0
            current_y += y_spacing
    except:
        pass
    
    # Carpet sample (smaller piece for visualization)
    try:
        carpet_sample = Carpet_12ft_Roll(length=6.0, width=6.0, fiber_type='Nylon')
        carpet_sample.name = "Nylon Carpet Sample"
        carpet_sample.move(dx=current_x, dy=current_y, dz=current_z)
        flooring_samples.append(carpet_sample)
        
        sample_count += 1
        current_x += x_spacing
        if sample_count % samples_per_row == 0:
            current_x = 0
            current_y += y_spacing
    except:
        pass
    
    if flooring_samples:
        print(f"Visualizing {len(flooring_samples)} finish flooring samples...")
        print("Each sample shows typical dimensions and proportions")
        print("Arranged in a 3×N grid with 5' spacing")
        print("X=Length, Y=Width, Z=Thickness")
        plot_items(flooring_samples)
    else:
        print("No flooring samples could be created - showing text output:")
        show_flooring_options_by_material()


# Create additional classes
additional_classes = create_finish_flooring_classes()

if __name__ == "__main__":
    print("Finish Flooring Elements - Material-Specific Dimensions")
    print("=" * 55)
    
    show_flooring_options_by_material()
    visualize_flooring_samples()
    
