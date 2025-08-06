"""
Parametric ceiling panel elements for various ceiling finishes.
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

class BaseCeilingPanel(ParametricElement):
    """Base class for all ceiling panel elements"""
    
    MATERIAL_TYPE = None        # To be set by subclasses
    TYPICAL_WIDTH = None        # Typical width for this panel type
    TYPICAL_LENGTH = None       # Typical length for this panel type
    THICKNESS = None            # Typical thickness in inches
    INSTALLATION_METHOD = None  # Installation method
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'length': Parameter(
                name='length',
                type=float,
                default=cls.TYPICAL_LENGTH or 8.0,
                min_value=1.0,
                max_value=16.0,
                unit='ft',
                description="Length of the ceiling panel (X-axis - longest dimension)"
            ),
            'width': Parameter(
                name='width',
                type=float,
                default=cls.TYPICAL_WIDTH or 4.0,
                min_value=0.5,
                max_value=8.0,
                unit='ft',
                description="Width of the ceiling panel (Y-axis - middle dimension)"
            ),
            'thickness': Parameter(
                name='thickness',
                type=float,
                default=cls.THICKNESS or 0.25,
                min_value=0.1,
                max_value=2.0,
                unit='in',
                description="Thickness of the ceiling panel"
            )
        }
    
    @classmethod
    def get_material_type(cls) -> str:
        return cls.MATERIAL_TYPE.lower() if cls.MATERIAL_TYPE else "ceiling_panel"
    
    def create_geometry(self) -> Geometry:
        # Following standardization: X=length, Y=width, Z=thickness
        base_points = [
            (0, 0),
            (self.params['length'], 0),
            (self.params['length'], self.params['width']),
            (0, self.params['width'])
        ]
        return Geometry.from_prism(base_points, self.params['thickness'] / 12.0)  # Convert inches to feet


# WOOD CEILING PANELS
class BaseWoodPanel(BaseCeilingPanel):
    """Base class for wood ceiling panels"""
    
    MATERIAL_TYPE = "Wood"
    INSTALLATION_METHOD = "Nail/Screw"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'wood_species': Parameter(
                name='wood_species',
                type=str,
                default='Pine',
                description="Wood species (Pine, Cedar, Oak, Maple, Cherry)"
            ),
            'grade': Parameter(
                name='grade',
                type=str,
                default='Select',
                description="Wood grade (Select, Common, Premium)"
            ),
            'finish': Parameter(
                name='finish',
                type=str,
                default='Unfinished',
                description="Finish type (Unfinished, Stained, Painted, Clear Coat)"
            ),
            'profile': Parameter(
                name='profile',
                type=str,
                default='Tongue & Groove',
                description="Edge profile (Tongue & Groove, Shiplap, V-Joint, Square)"
            ),
            'grain_direction': Parameter(
                name='grain_direction',
                type=str,
                default='With Length',
                description="Grain direction (With Length, Across Width)"
            )
        })
        return params


class WoodPanel_Plank_4x8(BaseWoodPanel):
    """4' x 8' wood ceiling panel"""
    TYPICAL_WIDTH = 4.0
    TYPICAL_LENGTH = 8.0
    THICKNESS = 0.75  # 3/4" typical


class WoodPanel_Plank_6x8(BaseWoodPanel):
    """6" x 8' wood plank panel"""
    TYPICAL_WIDTH = 0.5  # 6" wide planks
    TYPICAL_LENGTH = 8.0
    THICKNESS = 0.75


class WoodPanel_Bead_Board(BaseWoodPanel):
    """Beadboard wood ceiling panel"""
    TYPICAL_WIDTH = 4.0
    TYPICAL_LENGTH = 8.0
    THICKNESS = 0.375  # 3/8" typical for beadboard
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params['profile'].default = 'Beadboard'
        params.update({
            'bead_spacing': Parameter(
                name='bead_spacing',
                type=float,
                default=4.0,
                min_value=2.0,
                max_value=8.0,
                unit='in',
                description="Spacing between beads"
            )
        })
        return params


# METAL CEILING PANELS
class BaseMetalPanel(BaseCeilingPanel):
    """Base class for metal ceiling panels"""
    
    MATERIAL_TYPE = "Metal"
    INSTALLATION_METHOD = "Clip/Screw"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'metal_type': Parameter(
                name='metal_type',
                type=str,
                default='Aluminum',
                description="Metal type (Aluminum, Steel, Stainless Steel, Copper)"
            ),
            'finish': Parameter(
                name='finish',
                type=str,
                default='Mill',
                description="Finish type (Mill, Painted, Anodized, Powder Coated)"
            ),
            'gauge': Parameter(
                name='gauge',
                type=int,
                default=24,
                min_value=16,
                max_value=26,
                description="Metal gauge (thickness)"
            ),
            'profile': Parameter(
                name='profile',
                type=str,
                default='Linear',
                description="Panel profile (Linear, Corrugated, Standing Seam, Flat)"
            ),
            'perforation': Parameter(
                name='perforation',
                type=str,
                default='None',
                description="Perforation pattern (None, Round, Square, Decorative)"
            )
        })
        return params


class MetalPanel_Linear_12(BaseMetalPanel):
    """12" wide linear metal ceiling panel"""
    TYPICAL_WIDTH = 1.0  # 12" wide
    TYPICAL_LENGTH = 12.0  # Can be very long
    THICKNESS = 0.05  # Very thin


class MetalPanel_Linear_16(BaseMetalPanel):
    """16" wide linear metal ceiling panel"""
    TYPICAL_WIDTH = 1.33  # 16" wide
    TYPICAL_LENGTH = 12.0
    THICKNESS = 0.05


class MetalPanel_Linear_24(BaseMetalPanel):
    """24" wide linear metal ceiling panel"""
    TYPICAL_WIDTH = 2.0  # 24" wide
    TYPICAL_LENGTH = 12.0
    THICKNESS = 0.05


class MetalPanel_Sheet_4x8(BaseMetalPanel):
    """4' x 8' metal ceiling panel sheet"""
    TYPICAL_WIDTH = 4.0
    TYPICAL_LENGTH = 8.0
    THICKNESS = 0.05


# PVC/VINYL CEILING PANELS
class BasePVCPanel(BaseCeilingPanel):
    """Base class for PVC/vinyl ceiling panels"""
    
    MATERIAL_TYPE = "PVC"
    INSTALLATION_METHOD = "Clip/Nail"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'color': Parameter(
                name='color',
                type=str,
                default='White',
                description="Panel color"
            ),
            'texture': Parameter(
                name='texture',
                type=str,
                default='Smooth',
                description="Surface texture (Smooth, Woodgrain, Embossed)"
            ),
            'profile': Parameter(
                name='profile',
                type=str,
                default='Tongue & Groove',
                description="Edge profile (Tongue & Groove, Shiplap, Square)"
            ),
            'moisture_rating': Parameter(
                name='moisture_rating',
                type=str,
                default='High',
                description="Moisture resistance rating (Standard, High, Extreme)"
            )
        })
        return params


class PVCPanel_Plank_8(BaseMetalPanel):
    """8" wide PVC plank ceiling panel"""
    TYPICAL_WIDTH = 0.67  # 8" wide
    TYPICAL_LENGTH = 8.0
    THICKNESS = 0.375


class PVCPanel_Plank_12(BasePVCPanel):
    """12" wide PVC plank ceiling panel"""
    TYPICAL_WIDTH = 1.0  # 12" wide
    TYPICAL_LENGTH = 8.0
    THICKNESS = 0.375


class PVCPanel_Sheet_4x8(BasePVCPanel):
    """4' x 8' PVC ceiling panel sheet"""
    TYPICAL_WIDTH = 4.0
    TYPICAL_LENGTH = 8.0
    THICKNESS = 0.25


# COMPOSITE CEILING PANELS  
class BaseCompositePanel(BaseCeilingPanel):
    """Base class for composite ceiling panels (MDF, etc.)"""
    
    MATERIAL_TYPE = "Composite"
    INSTALLATION_METHOD = "Nail/Screw"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'core_material': Parameter(
                name='core_material',
                type=str,
                default='MDF',
                description="Core material (MDF, Particleboard, OSB)"
            ),
            'face_material': Parameter(
                name='face_material',
                type=str,
                default='Melamine',
                description="Face material (Melamine, Veneer, Laminate, Paint)"
            ),
            'moisture_resistance': Parameter(
                name='moisture_resistance',
                type=str,
                default='Standard',
                description="Moisture resistance (Standard, Enhanced, Marine Grade)"
            ),
            'fire_rating': Parameter(
                name='fire_rating',
                type=str,
                default='Class C',
                description="Fire rating (Class A, Class B, Class C)"
            )
        })
        return params


class CompositePanel_4x8(BaseCompositePanel):
    """4' x 8' composite ceiling panel"""
    TYPICAL_WIDTH = 4.0
    TYPICAL_LENGTH = 8.0
    THICKNESS = 0.75


class CompositePanel_Plank_6(BaseCompositePanel):
    """6" wide composite plank ceiling panel"""
    TYPICAL_WIDTH = 0.5  # 6" wide
    TYPICAL_LENGTH = 8.0
    THICKNESS = 0.5


# FABRIC CEILING PANELS
class BaseFabricPanel(BaseCeilingPanel):
    """Base class for fabric-wrapped ceiling panels"""
    
    MATERIAL_TYPE = "Fabric"
    INSTALLATION_METHOD = "Clip/Hook"
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        params = super().get_parameters()
        params.update({
            'fabric_type': Parameter(
                name='fabric_type',
                type=str,
                default='Acoustic Fabric',
                description="Fabric type (Acoustic Fabric, Canvas, Vinyl, Knit)"
            ),
            'core_material': Parameter(
                name='core_material',
                type=str,
                default='Fiberglass',
                description="Core material (Fiberglass, Foam, Rockwool)"
            ),
            'nrc_rating': Parameter(
                name='nrc_rating',
                type=float,
                default=0.85,
                min_value=0.0,
                max_value=1.0,
                description="Noise Reduction Coefficient"
            ),
            'fire_rating': Parameter(
                name='fire_rating',
                type=str,
                default='Class A',
                description="Fire rating classification"
            )
        })
        return params


class FabricPanel_2x4(BaseFabricPanel):
    """2' x 4' fabric-wrapped acoustic panel"""
    TYPICAL_WIDTH = 2.0
    TYPICAL_LENGTH = 4.0
    THICKNESS = 1.0  # Thicker for acoustic properties


class FabricPanel_2x2(BaseFabricPanel):
    """2' x 2' fabric-wrapped acoustic panel"""
    TYPICAL_WIDTH = 2.0
    TYPICAL_LENGTH = 2.0
    THICKNESS = 1.0


def show_ceiling_panel_options():
    """Display available ceiling panel options grouped by material type"""
    
    material_groups = {
        'Wood': [],
        'Metal': [],
        'PVC': [],
        'Composite': [],
        'Fabric': []
    }
    
    # Group classes by material type
    for name, cls in globals().items():
        if hasattr(cls, 'MATERIAL_TYPE') and hasattr(cls, 'TYPICAL_WIDTH'):
            material = cls.MATERIAL_TYPE
            if material in material_groups:
                material_groups[material].append((name, cls))
    
    print("Available Ceiling Panel Options:")
    print("=" * 42)
    
    for material, classes in material_groups.items():
        if classes:
            print(f"\n{material}:")
            print("-" * 20)
            for name, cls in sorted(classes):
                width_ft = cls.TYPICAL_WIDTH
                length_ft = cls.TYPICAL_LENGTH
                thickness_in = cls.THICKNESS
                install_method = cls.INSTALLATION_METHOD if hasattr(cls, 'INSTALLATION_METHOD') else 'N/A'
                
                print(f"  {name}:")
                print(f"    Size: {length_ft}' × {width_ft}' × {thickness_in}\"")
                print(f"    Installation: {install_method}")


if __name__ == "__main__":
    print("Ceiling Panel Elements - Various Materials and Profiles")
    print("=" * 58)
    
    show_ceiling_panel_options()
    
    # Example usage
    print("\nExample Usage:")
    print("-" * 15)
    
    # Create sample panels
    wood_panel = WoodPanel_Plank_4x8(wood_species='Cedar', finish='Clear Coat', profile='Tongue & Groove')
    wood_panel.name = "Cedar T&G Ceiling Panel"
    
    metal_panel = MetalPanel_Linear_16(metal_type='Aluminum', finish='Anodized', profile='Linear')
    metal_panel.name = "Linear Aluminum Panel"
    
    fabric_panel = FabricPanel_2x4(fabric_type='Acoustic Fabric', nrc_rating=0.90)
    fabric_panel.name = "High-Performance Acoustic Panel"
    
    print(f"Created {wood_panel.name}: {wood_panel.MATERIAL_TYPE}")
    print(f"Created {metal_panel.name}: {metal_panel.MATERIAL_TYPE}")
    print(f"Created {fabric_panel.name}: {fabric_panel.MATERIAL_TYPE}")
    
    # Visualize if available
    try:
        from hierarchical.utils import plot_items
        
        # Position panels for comparison
        metal_panel.move(dx=9.0)   # Move 9' to the right
        fabric_panel.move(dx=18.0)  # Move 18' to the right
        
        print("Visualizing ceiling panels...")
        plot_items([wood_panel, metal_panel, fabric_panel])
    except ImportError:
        print("Visualization not available")