"""
Ceiling assembly component classes following consistent assembly rules.
"""

from typing import Dict, List, Tuple
from hierarchical.catalog.base import ParametricComponent, Parameter
from hierarchical.catalog.elements.drywall import Drywall_Regular_0_25, Drywall_Regular_0_375, Drywall_Regular_0_5
from hierarchical.catalog.elements.ceiling_tiles import (
    AcousticTile_2x2, AcousticTile_2x4, MineralFiberTile_2x2, MineralFiberTile_2x4,
    MetalTile_2x2, MetalTile_2x4, WoodTile_2x2, WoodTile_2x4
)
from hierarchical.catalog.elements.ceiling_panels import (
    WoodPanel_Plank_4x8, MetalPanel_Linear_12, MetalPanel_Linear_16, PVCPanel_Sheet_4x8
)
from hierarchical.catalog.elements.plaster_lath import WoodLath_Standard, MetalLath_Diamond_27ga, GypsumLath_3_8x16x48
from hierarchical.catalog.elements.plaster import LimePlaster_ThreeCoat, GypsumPlaster_ThreeCoat
from hierarchical.catalog.elements.lumber import Lumber2X4  # For grid framework
from hierarchical.items import Element
import math

class BaseCeilingAssembly(ParametricComponent):
    """Base class for ceiling assemblies"""

    FINISH_ELEMENT_CLASS = None   # To be set by subclasses
    SUBSTRATE_ELEMENT_CLASS = None  # For lath, grid, etc.
    MATERIAL_TYPE = None          # To be set by subclasses
    ASSEMBLY_TYPE = None          # drywall, suspended, plaster, etc.
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'ceiling_width': Parameter(
                name='ceiling_width',
                type=float,
                default=12.0,
                min_value=4.0,
                max_value=math.inf,
                unit='ft',
                description="Width of the ceiling to assemble"
            ),
            'ceiling_length': Parameter(
                name='ceiling_length',
                type=float,
                default=16.0,
                min_value=4.0,
                max_value=math.inf,
                unit='ft',
                description="Length of the ceiling to assemble"
            ),
            'joist_direction': Parameter(
                name='joist_direction',
                type=str,
                default='parallel_to_length',
                description="Joist direction: 'parallel_to_length' or 'parallel_to_width'"
            ),
            'finish_orientation': Parameter(
                name='finish_orientation',
                type=str,
                default='perpendicular_to_joists',
                description="Finish element orientation relative to joists"
            ),
            'stagger_joints': Parameter(
                name='stagger_joints',
                type=bool,
                default=True,
                description="Stagger joints for structural integrity"
            ),
            'edge_gap': Parameter(
                name='edge_gap',
                type=float,
                default=0.125,
                min_value=0.0,
                max_value=0.25,
                unit='in',
                description="Gap between elements for expansion"
            ),
            'ceiling_frame': Parameter(
                name='ceiling_frame',
                type=object,
                default=None,
                description="Optional ceiling frame component to align elements with joist locations"
            )
        }
    
    def _apply_joint_staggering(self, col: int, row: int, sheet_width: float) -> Tuple[float, float]:
        """Apply joint staggering offset following building codes"""
        if not self.params['stagger_joints']:
            return 0.0, 0.0
        
        # Stagger every other row by half sheet width
        stagger_offset = 0.0
        if row % 2 == 1:  # Odd rows get staggered
            stagger_offset = sheet_width / 2.0  # Half sheet width stagger
        
        # Ensure stagger doesn't push elements beyond ceiling bounds
        current_x_without_stagger = col * self.SHEET_LENGTH
        if current_x_without_stagger + stagger_offset + self.SHEET_LENGTH > self.params['ceiling_length']:
            stagger_offset = 0.0  # Don't stagger if it would exceed bounds
            
        return stagger_offset, 0.0
    
    def create_elements(self) -> List[Element]:
        """Create ceiling assembly elements following consistent assembly rules:
        Create element → rotate element → move element into place → repeat
        """
        
        elements = []
        ceiling_frame = self.params.get('ceiling_frame')
        
        # Create finish ceiling elements (positioned at Z=0) - if applicable
        finish_elements = self._create_finish_elements()
        if finish_elements:
            elements.extend(finish_elements)
        
        # Create substrate elements if needed (grid, lath, etc.)
        if self.SUBSTRATE_ELEMENT_CLASS is not None:
            substrate_elements = self._create_substrate_elements()
            elements.extend(substrate_elements)
        
        return elements
    
    def _create_finish_elements(self) -> List[Element]:
        """Create finish ceiling elements - to be implemented by subclasses that need discrete finish elements"""
        return []  # Default: no discrete finish elements
    
    def _create_substrate_elements(self) -> List[Element]:
        """Create substrate elements - implemented by subclasses that need it"""
        return []


# DRYWALL CEILING ASSEMBLIES
class BaseDrywallCeilingAssembly(BaseCeilingAssembly):
    """Base class for drywall ceiling assemblies"""
    
    ASSEMBLY_TYPE = "drywall"
    
    # Standard drywall sheet dimensions
    SHEET_LENGTH = 8.0  # feet (X-axis - longest dimension)
    SHEET_WIDTH = 4.0   # feet (Y-axis - middle dimension)
    
    def _create_finish_elements(self) -> List[Element]:
        """Create drywall elements using sheet layout similar to floor assemblies"""
        elements = []
        ceiling_width = self.params['ceiling_width']
        ceiling_length = self.params['ceiling_length']
        orientation = self.params['finish_orientation']
        
        # Calculate sheet layout
        if orientation == 'perpendicular_to_joists':
            sheets_across = math.ceil(ceiling_length / self.SHEET_LENGTH)
            sheets_up = math.ceil(ceiling_width / self.SHEET_WIDTH)
        else:  # parallel_to_joists
            sheets_across = math.ceil(ceiling_length / self.SHEET_WIDTH)
            sheets_up = math.ceil(ceiling_width / self.SHEET_LENGTH)
        
        sheet_index = 0
        for row in range(sheets_up):
            for col in range(sheets_across):
                
                # Calculate sheet dimensions
                if orientation == 'perpendicular_to_joists':
                    if col == sheets_across - 1:  # Last column
                        sheet_length = ceiling_length - (col * self.SHEET_LENGTH)
                        sheet_length = min(sheet_length, self.SHEET_LENGTH)
                    else:
                        sheet_length = self.SHEET_LENGTH
                    
                    if row == sheets_up - 1:  # Last row
                        sheet_width = ceiling_width - (row * self.SHEET_WIDTH)
                        sheet_width = min(sheet_width, self.SHEET_WIDTH)
                    else:
                        sheet_width = self.SHEET_WIDTH
                else:  # parallel_to_joists
                    if col == sheets_across - 1:  # Last column
                        sheet_width = ceiling_length - (col * self.SHEET_WIDTH)
                        sheet_width = min(sheet_width, self.SHEET_WIDTH)
                    else:
                        sheet_width = self.SHEET_WIDTH
                    
                    if row == sheets_up - 1:  # Last row
                        sheet_length = ceiling_width - (row * self.SHEET_LENGTH)
                        sheet_length = min(sheet_length, self.SHEET_LENGTH)
                    else:
                        sheet_length = self.SHEET_LENGTH
                
                # CREATE drywall sheet element
                sheet = self.FINISH_ELEMENT_CLASS(length=sheet_length, width=sheet_width)
                sheet.name = f"{self.name}_drywall_{row+1}_{col+1}"
                
                # ROTATE: Orient sheet based on installation orientation
                if orientation == 'parallel_to_joists':
                    # Rotate 90° to run parallel to joists
                    sheet.rotate_z(math.pi/2, [0, 0, 0])
                
                # Calculate row stagger (applies to entire row)
                row_stagger_x = 0.0
                if self.params['stagger_joints'] and row % 2 == 1:
                    row_stagger_x = sheet_width / 2.0  # Half sheet width stagger
                    # Don't stagger if it would push the last sheet beyond bounds
                    if (sheets_across - 1) * self.SHEET_LENGTH + row_stagger_x + self.SHEET_LENGTH > ceiling_length:
                        row_stagger_x = 0.0
                
                # MOVE: Position sheet in final location (ceiling at Z=0)
                if orientation == 'perpendicular_to_joists':
                    x_pos = (col * self.SHEET_LENGTH) + row_stagger_x
                    y_pos = row * self.SHEET_WIDTH
                else:  # parallel_to_joists
                    x_pos = (col * self.SHEET_WIDTH) + row_stagger_x
                    y_pos = row * self.SHEET_LENGTH
                
                # Ensure sheets don't extend beyond ceiling bounds
                x_pos = min(x_pos, ceiling_length - sheet_length)
                y_pos = min(y_pos, ceiling_width - sheet_width)
                
                # Ensure compliance: fully positive coordinate space
                x_pos = max(0.0, x_pos)
                y_pos = max(0.0, y_pos)
                z_pos = 0.0  # Finish ceiling surface at Z=0
                
                sheet.move(dx=x_pos, dy=y_pos, dz=z_pos)
                elements.append(sheet)
                
                sheet_index += 1
        
        return elements


class DrywallCeilingAssembly_0_5(BaseDrywallCeilingAssembly):
    """1/2" drywall ceiling assembly"""
    FINISH_ELEMENT_CLASS = Drywall_Regular_0_5
    MATERIAL_TYPE = "1/2\" Drywall Ceiling"


class DrywallCeilingAssembly_0_625(BaseDrywallCeilingAssembly):
    """5/8" drywall ceiling assembly"""
    FINISH_ELEMENT_CLASS = Drywall_Regular_0_5  # Reusing 1/2" - would need 5/8" class
    MATERIAL_TYPE = "5/8\" Drywall Ceiling"


# SUSPENDED CEILING ASSEMBLIES
class BaseSuspendedCeilingAssembly(BaseCeilingAssembly):
    """Base class for suspended ceiling assemblies with grid and tiles"""
    
    ASSEMBLY_TYPE = "suspended"
    
    # Standard grid spacing
    MAIN_TEE_SPACING = 4.0    # 48" OC
    CROSS_TEE_SPACING = 2.0   # 24" OC
    
    def _create_finish_elements(self) -> List[Element]:
        """Create ceiling tiles"""
        elements = []
        ceiling_width = self.params['ceiling_width']
        ceiling_length = self.params['ceiling_length']
        
        # Get tile dimensions from the tile class
        tile_class = self.FINISH_ELEMENT_CLASS
        tile_length = getattr(tile_class, 'TYPICAL_LENGTH', 2.0)
        tile_width = getattr(tile_class, 'TYPICAL_WIDTH', 2.0)
        
        # Calculate tile layout
        tiles_across = math.ceil(ceiling_length / tile_length)
        tiles_up = math.ceil(ceiling_width / tile_width)
        
        for row in range(tiles_up):
            for col in range(tiles_across):
                
                # Calculate tile dimensions (may be partial at edges)
                if col == tiles_across - 1:  # Last column
                    actual_tile_length = ceiling_length - (col * tile_length)
                    actual_tile_length = min(actual_tile_length, tile_length)
                else:
                    actual_tile_length = tile_length
                
                if row == tiles_up - 1:  # Last row
                    actual_tile_width = ceiling_width - (row * tile_width)
                    actual_tile_width = min(actual_tile_width, tile_width)
                else:
                    actual_tile_width = tile_width
                
                # CREATE ceiling tile element
                tile = self.FINISH_ELEMENT_CLASS(length=actual_tile_length, width=actual_tile_width)
                tile.name = f"{self.name}_tile_{row+1}_{col+1}"
                
                # MOVE: Position tile in grid (no rotation needed for square/rectangular tiles)
                x_pos = col * tile_length
                y_pos = row * tile_width
                z_pos = 0.0  # Tiles sit in grid at Z=0
                
                tile.move(dx=x_pos, dy=y_pos, dz=z_pos)
                elements.append(tile)
        
        return elements
    
    def _create_substrate_elements(self) -> List[Element]:
        """Create suspended ceiling grid system aligned with tile dimensions"""
        elements = []
        ceiling_width = self.params['ceiling_width']
        ceiling_length = self.params['ceiling_length']
        
        # Get tile dimensions to align grid properly
        tile_class = self.FINISH_ELEMENT_CLASS
        tile_length = getattr(tile_class, 'TYPICAL_LENGTH', 2.0)
        tile_width = getattr(tile_class, 'TYPICAL_WIDTH', 2.0)
        
        # Create main tees (run parallel to length, spaced by tile width)
        main_tee_count = math.ceil(ceiling_width / tile_width) + 1
        
        for i in range(main_tee_count):
            # CREATE main tee element
            main_tee = Lumber2X4(length=ceiling_length, species="aluminum_main_tee")
            main_tee.name = f"{self.name}_main_tee_{i+1}"
            
            # MOVE: Position main tee at tile boundaries
            x_pos = 0.0
            y_pos = i * tile_width
            z_pos = 0.05  # Slightly above tiles to represent grid structure
            
            main_tee.move(dx=x_pos, dy=y_pos, dz=z_pos)
            elements.append(main_tee)
        
        # Create cross tees (run parallel to width, spaced by tile length)
        cross_tee_count = math.ceil(ceiling_length / tile_length) + 1
        
        for i in range(cross_tee_count):
            # CREATE cross tee element
            cross_tee = Lumber2X4(length=ceiling_width, species="aluminum_cross_tee")
            cross_tee.name = f"{self.name}_cross_tee_{i+1}"
            
            # ROTATE: Cross tees run perpendicular to main tees
            cross_tee.rotate_z(math.pi/2, [0, 0, 0])
            
            # MOVE: Position cross tee at tile boundaries
            x_pos = i * tile_length
            y_pos = 0.0
            z_pos = 0.05  # Same level as main tees
            
            cross_tee.move(dx=x_pos, dy=y_pos, dz=z_pos)
            elements.append(cross_tee)
        
        return elements


class SuspendedCeilingAssembly_2x2_Acoustic(BaseSuspendedCeilingAssembly):
    """2x2 acoustic tile suspended ceiling assembly"""
    FINISH_ELEMENT_CLASS = AcousticTile_2x2
    SUBSTRATE_ELEMENT_CLASS = Lumber2X4  # Grid framework
    MATERIAL_TYPE = "2x2 Acoustic Suspended Ceiling"


class SuspendedCeilingAssembly_2x4_Acoustic(BaseSuspendedCeilingAssembly):
    """2x4 acoustic tile suspended ceiling assembly"""
    FINISH_ELEMENT_CLASS = AcousticTile_2x4
    SUBSTRATE_ELEMENT_CLASS = Lumber2X4
    MATERIAL_TYPE = "2x4 Acoustic Suspended Ceiling"


class SuspendedCeilingAssembly_2x2_MineralFiber(BaseSuspendedCeilingAssembly):
    """2x2 mineral fiber tile suspended ceiling assembly"""
    FINISH_ELEMENT_CLASS = MineralFiberTile_2x2
    SUBSTRATE_ELEMENT_CLASS = Lumber2X4
    MATERIAL_TYPE = "2x2 Mineral Fiber Suspended Ceiling"


# PLASTER CEILING ASSEMBLIES
class BasePlasterCeilingAssembly(BaseCeilingAssembly):
    """Base class for plaster ceiling assemblies with lath substrate"""
    
    ASSEMBLY_TYPE = "plaster"
    
    def _create_finish_elements(self) -> List[Element]:
        """Create continuous plaster coating over the lath substrate"""
        elements = []
        ceiling_width = self.params['ceiling_width']
        ceiling_length = self.params['ceiling_length']
        
        # Create plaster coating element covering entire ceiling
        plaster_coating = self.FINISH_ELEMENT_CLASS(
            length=ceiling_length,
            width=ceiling_width
        )
        plaster_coating.name = f"{self.name}_plaster_coating"
        
        # Position plaster at Z=0 (finish surface level)
        plaster_coating.move(dx=0.0, dy=0.0, dz=0.0)
        elements.append(plaster_coating)
        
        return elements
    
    def _create_substrate_elements(self) -> List[Element]:
        """Create lath substrate for plaster application"""
        elements = []
        ceiling_width = self.params['ceiling_width']
        ceiling_length = self.params['ceiling_length']
        
        # Get lath dimensions from the lath class
        lath_class = self.SUBSTRATE_ELEMENT_CLASS
        lath_length = getattr(lath_class, 'TYPICAL_LENGTH', 4.0)
        lath_width = getattr(lath_class, 'TYPICAL_WIDTH', 0.125)  # Often very narrow strips
        
        # Calculate lath layout (typically run perpendicular to joists)
        joist_direction = self.params['joist_direction']
        
        if joist_direction == 'parallel_to_length':
            # Joists run along length, so lath runs along width
            lath_runs_along = 'width'
            lath_spacing = lath_width + (0.375 / 12.0)  # 3/8" spacing between lath strips
            num_lath_strips = math.ceil(ceiling_length / lath_spacing)
            
            for i in range(num_lath_strips):
                # For long spans, create multiple lath pieces end-to-end
                max_lath_length = getattr(self.SUBSTRATE_ELEMENT_CLASS, 'TYPICAL_LENGTH', 4.0)
                pieces_needed = math.ceil(ceiling_width / max_lath_length)
                
                for piece in range(pieces_needed):
                    # Calculate piece length (last piece may be shorter)
                    if piece == pieces_needed - 1:  # Last piece
                        piece_length = ceiling_width - (piece * max_lath_length)
                    else:
                        piece_length = max_lath_length
                    
                    # CREATE lath strip element
                    lath_strip = self.SUBSTRATE_ELEMENT_CLASS(length=piece_length)
                    lath_strip.name = f"{self.name}_lath_{i+1}_piece_{piece+1}"
                    
                    # ROTATE: Lath runs perpendicular to its natural orientation
                    lath_strip.rotate_z(math.pi/2, [0, 0, 0])
                    
                    # MOVE: Position lath strip
                    x_pos = i * lath_spacing
                    y_pos = piece * max_lath_length
                    z_pos = 0.03  # Lath slightly above finish surface
                    
                    lath_strip.move(dx=x_pos, dy=y_pos, dz=z_pos)
                    elements.append(lath_strip)
        else:
            # Joists run along width, so lath runs along length
            lath_spacing = lath_width + (0.375 / 12.0)
            num_lath_strips = math.ceil(ceiling_width / lath_spacing)
            
            for i in range(num_lath_strips):
                # For long spans, create multiple lath pieces end-to-end
                max_lath_length = getattr(self.SUBSTRATE_ELEMENT_CLASS, 'TYPICAL_LENGTH', 4.0)
                pieces_needed = math.ceil(ceiling_length / max_lath_length)
                
                for piece in range(pieces_needed):
                    # Calculate piece length (last piece may be shorter)
                    if piece == pieces_needed - 1:  # Last piece
                        piece_length = ceiling_length - (piece * max_lath_length)
                    else:
                        piece_length = max_lath_length
                    
                    # CREATE lath strip element
                    lath_strip = self.SUBSTRATE_ELEMENT_CLASS(length=piece_length)
                    lath_strip.name = f"{self.name}_lath_{i+1}_piece_{piece+1}"
                    
                    # MOVE: Position lath strip (no rotation needed)
                    x_pos = piece * max_lath_length
                    y_pos = i * lath_spacing
                    z_pos = 0.03
                    
                    lath_strip.move(dx=x_pos, dy=y_pos, dz=z_pos)
                    elements.append(lath_strip)
        
        return elements


class PlasterCeilingAssembly_WoodLath(BasePlasterCeilingAssembly):
    """Traditional wood lath and lime plaster ceiling assembly"""
    FINISH_ELEMENT_CLASS = LimePlaster_ThreeCoat
    SUBSTRATE_ELEMENT_CLASS = WoodLath_Standard
    MATERIAL_TYPE = "Wood Lath & Lime Plaster Ceiling"


class PlasterCeilingAssembly_MetalLath(BasePlasterCeilingAssembly):
    """Metal lath and gypsum plaster ceiling assembly"""
    FINISH_ELEMENT_CLASS = GypsumPlaster_ThreeCoat
    SUBSTRATE_ELEMENT_CLASS = MetalLath_Diamond_27ga
    MATERIAL_TYPE = "Metal Lath & Gypsum Plaster Ceiling"


# Example usage
if __name__ == "__main__":
    # Example 1: Drywall ceiling assembly
    drywall_ceiling = DrywallCeilingAssembly_0_5(
        ceiling_width=12.0,
        ceiling_length=16.0,
        joist_direction='parallel_to_length',
        finish_orientation='perpendicular_to_joists',
        stagger_joints=True
    )
    
    elements_drywall = drywall_ceiling.create_elements()
    print(f"Drywall ceiling assembly: Created {len(elements_drywall)} elements")
    
    # Example 2: Suspended ceiling assembly
    suspended_ceiling = SuspendedCeilingAssembly_2x2_Acoustic(
        ceiling_width=10.0,
        ceiling_length=12.0,
        stagger_joints=False  # Tiles align with grid
    )
    
    elements_suspended = suspended_ceiling.create_elements()
    print(f"Suspended ceiling assembly: Created {len(elements_suspended)} elements")
    
    # Count tiles vs grid elements
    tile_count = len([e for e in elements_suspended if 'tile' in e.name])
    grid_count = len([e for e in elements_suspended if 'tee' in e.name])
    print(f"  - {tile_count} ceiling tiles")
    print(f"  - {grid_count} grid elements")
    
    # Example 3: Plaster ceiling assembly
    plaster_ceiling = PlasterCeilingAssembly_WoodLath(
        ceiling_width=8.0,
        ceiling_length=10.0,
        joist_direction='parallel_to_width'
    )
    
    elements_plaster = plaster_ceiling.create_elements()
    print(f"Plaster ceiling assembly: Created {len(elements_plaster)} elements")
    # Count lath vs plaster elements
    lath_count = len([e for e in elements_plaster if 'lath' in e.name])
    plaster_count = len([e for e in elements_plaster if 'plaster' in e.name])
    print(f"  - {lath_count} lath elements")
    print(f"  - {plaster_count} plaster coating elements")
    
    # Visualize if plotting available
    try:
        from hierarchical.utils import plot_items
        print("Visualizing ceiling assemblies...")
        
        # Position assemblies side by side for comparison
        suspended_ceiling.move(dx=20.0)  # Move 20' to the right
        plaster_ceiling.move(dx=40.0)    # Move 40' to the right
        
        plot_items([drywall_ceiling, suspended_ceiling, plaster_ceiling], flatten_to_elements=True)
    except ImportError:
        print("Visualization not available")