"""
Floor assembly component classes following consistent assembly rules.
"""

from typing import Dict, List, Tuple
from hierarchical.catalog.base import ParametricComponent, Parameter
from hierarchical.catalog.elements.sub_flooring import (
    PlywoodSubflooring_0_5, PlywoodSubflooring_0_625, PlywoodSubflooring_0_75, 
    PlywoodSubflooring_1_0, PlywoodSubflooring_1_25,
    OSBSubflooring_0_5, OSBSubflooring_0_625, OSBSubflooring_0_75,
    OSBSubflooring_1_0, OSBSubflooring_1_25,
    AdvantechSubflooring
)
from hierarchical.catalog.elements.finish_flooring import (
    Hardwood_2_25, Hardwood_3_0, Hardwood_5_0,
    EngineeredHardwood_5_0, EngineeredHardwood_7_0,
    LVP_6x48, LVP_7x48, LVP_9x60,
    Tile_12x12, Tile_18x18, Tile_12x24, Tile_6x36,
    Carpet_12ft_Roll, Carpet_15ft_Roll,
    Laminate_5x47, Laminate_8x47
)
from hierarchical.items import Element
import math

class BaseFloorAssembly(ParametricComponent):
    """Base class for floor assemblies"""

    SUBFLOORING_CLASS = None  # To be set by subclasses
    FINISH_FLOORING_CLASS = None  # To be set by subclasses
    MATERIAL_TYPE = None  # To be set by subclasses
    SUBFLOORING_THICKNESS = None  # To be set by subclasses
    
    # Standard sheet dimensions following element creation rules
    SUBFLOORING_SHEET_LENGTH = 8.0  # feet (X-axis - longest dimension)
    SUBFLOORING_SHEET_WIDTH = 4.0   # feet (Y-axis - middle dimension)
    # thickness is Z-axis (shortest dimension)
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'floor_width': Parameter(
                name='floor_width',
                type=float,
                default=12.0,
                min_value=4.0,
                max_value=math.inf,
                unit='ft',
                description="Width of the floor to assemble"
            ),
            'floor_length': Parameter(
                name='floor_length',
                type=float,
                default=16.0,
                min_value=4.0,
                max_value=math.inf,
                unit='ft',
                description="Length of the floor to assemble"
            ),
            'subflooring_orientation': Parameter(
                name='subflooring_orientation',
                type=str,
                default='perpendicular_to_joists',
                description="Subflooring orientation: 'perpendicular_to_joists' or 'parallel_to_joists'"
            ),
            'finish_flooring_direction': Parameter(
                name='finish_flooring_direction',
                type=str,
                default='parallel_to_length',
                description="Finish flooring direction: 'parallel_to_length' or 'parallel_to_width'"
            ),
            'stagger_subflooring_joints': Parameter(
                name='stagger_subflooring_joints',
                type=bool,
                default=True,
                description="Stagger subflooring sheet joints for structural integrity"
            ),
            'stagger_finish_joints': Parameter(
                name='stagger_finish_joints',
                type=bool,
                default=True,
                description="Stagger finish flooring joints"
            ),
            'edge_gap': Parameter(
                name='edge_gap',
                type=float,
                default=0,
                min_value=0.0,
                max_value=0.25,
                unit='in',
                description="Gap between sheets for expansion"
            ),
            'deck_frame': Parameter(
                name='deck_frame',
                type=object,
                default=None,
                description="Optional deck frame component to align subflooring with joist locations"
            )
        }
    
    def _calculate_subflooring_layout(self) -> Tuple[int, int, List[Tuple[float, float]]]:
        """Calculate optimal subflooring layout to minimize waste"""
        floor_width = self.params['floor_width']
        floor_length = self.params['floor_length']
        orientation = self.params['subflooring_orientation']
        edge_gap_ft = self.params['edge_gap'] / 12.0  # Convert inches to feet
        
        if orientation == 'perpendicular_to_joists':
            # Sheets installed with 8' dimension across joists (typical)
            sheets_across = math.ceil(floor_length / self.SUBFLOORING_SHEET_LENGTH)
            sheets_up = math.ceil(floor_width / self.SUBFLOORING_SHEET_WIDTH)
            
            # Calculate actual sheet dimensions needed
            sheet_sizes = []
            for row in range(sheets_up):
                for col in range(sheets_across):
                    # Calculate sheet dimensions
                    if col == sheets_across - 1:  # Last column
                        sheet_length = floor_length - (col * self.SUBFLOORING_SHEET_LENGTH)
                        sheet_length = min(sheet_length, self.SUBFLOORING_SHEET_LENGTH)
                    else:
                        sheet_length = self.SUBFLOORING_SHEET_LENGTH
                    
                    if row == sheets_up - 1:  # Last row
                        sheet_width = floor_width - (row * self.SUBFLOORING_SHEET_WIDTH)
                        sheet_width = min(sheet_width, self.SUBFLOORING_SHEET_WIDTH)
                    else:
                        sheet_width = self.SUBFLOORING_SHEET_WIDTH
                    
                    sheet_sizes.append((sheet_length, sheet_width))
            
        else:  # parallel_to_joists
            # Sheets installed with 8' dimension parallel to joists
            sheets_across = math.ceil(floor_length / self.SUBFLOORING_SHEET_WIDTH)
            sheets_up = math.ceil(floor_width / self.SUBFLOORING_SHEET_LENGTH)
            
            # Calculate actual sheet dimensions needed
            sheet_sizes = []
            for row in range(sheets_up):
                for col in range(sheets_across):
                    # Calculate sheet dimensions
                    if col == sheets_across - 1:  # Last column
                        sheet_width = floor_length - (col * self.SUBFLOORING_SHEET_WIDTH)
                        sheet_width = min(sheet_width, self.SUBFLOORING_SHEET_WIDTH)
                    else:
                        sheet_width = self.SUBFLOORING_SHEET_WIDTH
                    
                    if row == sheets_up - 1:  # Last row
                        sheet_length = floor_width - (row * self.SUBFLOORING_SHEET_LENGTH)
                        sheet_length = min(sheet_length, self.SUBFLOORING_SHEET_LENGTH)
                    else:
                        sheet_length = self.SUBFLOORING_SHEET_LENGTH
                    
                    # Always return (length, width) - don't swap here
                    sheet_sizes.append((sheet_length, sheet_width))
        
        return sheets_across, sheets_up, sheet_sizes
    
    def _get_joist_locations_from_frame(self, deck_frame) -> List[float]:
        """Extract joist positions from a deck frame component"""
        joist_positions = []
        
        if deck_frame is None:
            return joist_positions
        
        try:
            # Get all elements from the deck frame
            elements = deck_frame.sub_items
            
            # Find joist elements (they contain "joist" in their name)
            for element in elements:
                if hasattr(element, 'name') and 'joist' in element.name.lower():
                    # Get the position of the joist (assuming perpendicular to length)
                    position = element.get_centroid().y  # Get Y position for joists running parallel to X
                    joist_positions.append(position)

            # Sort positions and remove duplicates
            joist_positions = sorted(list(set(joist_positions)))
            
        except Exception as e:
            print(f"Warning: Could not extract joist positions from frame: {e}")
            return []
        
        return joist_positions
    
    def _apply_joint_staggering(self, col: int, row: int, layer_type: str) -> Tuple[float, float]:
        """Apply joint staggering offset following building codes"""
        stagger_param = 'stagger_subflooring_joints' if layer_type == 'subflooring' else 'stagger_finish_joints'
        
        if not self.params[stagger_param]:
            return 0.0, 0.0
        
        # Stagger every other row by half sheet width
        stagger_offset = 0.0
        if row % 2 == 1:  # Odd rows get staggered
            if layer_type == 'subflooring':
                stagger_offset = self.SUBFLOORING_SHEET_WIDTH / 2.0
            else:
                # For finish flooring, use typical plank width for staggering
                stagger_offset = 2.0  # 2 feet stagger for finish flooring
        
        # Ensure stagger doesn't push elements beyond floor bounds
        max_x = self.params['floor_length'] - stagger_offset
        if stagger_offset > max_x:
            stagger_offset = 0.0
            
        return stagger_offset, 0.0
    
    def create_elements(self) -> List[Element]:
        """Create floor assembly elements following consistent assembly rules:
        Create element → rotate element → move element into place → repeat
        
        Creates both subflooring and finish flooring layers.
        """
        
        if self.SUBFLOORING_CLASS is None or self.FINISH_FLOORING_CLASS is None:
            raise ValueError(f"{self.__class__.__name__} must define SUBFLOORING_CLASS and FINISH_FLOORING_CLASS")
        
        elements = []
        deck_frame = self.params.get('deck_frame')
        
        # Create subflooring layer first (bottom layer at Z=0)
        subflooring_elements = self._create_subflooring_elements(deck_frame)
        elements.extend(subflooring_elements)
        
        # Create finish flooring layer (top layer at Z=subflooring_thickness)
        finish_flooring_elements = self._create_finish_flooring_elements()
        elements.extend(finish_flooring_elements)
        
        return elements
    
    def _create_subflooring_elements(self, deck_frame) -> List[Element]:
        """Create subflooring elements using sheet layout"""
        elements = []
        floor_width = self.params['floor_width']
        floor_length = self.params['floor_length']
        orientation = self.params['subflooring_orientation']
        edge_gap_ft = self.params['edge_gap'] / 12.0
        
        # Calculate optimal layout
        sheets_across, sheets_up, sheet_sizes = self._calculate_subflooring_layout()
        
        sheet_index = 0
        for row in range(sheets_up):
            for col in range(sheets_across):
                if sheet_index >= len(sheet_sizes):
                    break
                
                ## TODO handle sheet width and length for parallel_to_joists orientation
                ## Currently broken because it just changes the length and width rather than building the
                ## sheet with length as the longest dimension and then rotating the sheet and moving it
                sheet_length, sheet_width = sheet_sizes[sheet_index]
                
                # CREATE subflooring sheet element
                sheet = self.SUBFLOORING_CLASS(length=sheet_length, width=sheet_width)
                sheet.name = f"{self.name}_subflooring_{row+1}_{col+1}"
                
                # ROTATE: Orient sheet based on installation orientation
                if orientation == 'parallel_to_joists':
                    # Rotate 90° to run parallel to joists
                    sheet.rotate_z(math.pi/2, [0, 0, 0])
                
                # Apply joint staggering offset
                ## TODO improve staggering logic
                stagger_x, stagger_y = self._apply_joint_staggering(col, row, 'subflooring')
                stagger_x = stagger_y = 0  # Not using staggering right now


                # MOVE: Position sheet in final location
                # Following element creation rules: all elements in fully positive coordinate space
                if orientation == 'perpendicular_to_joists':
                    # Standard positioning - use standard sheet dimensions for grid spacing
                    x_pos = (col * self.SUBFLOORING_SHEET_LENGTH) + stagger_x
                    y_pos = row * self.SUBFLOORING_SHEET_WIDTH
                else:  # parallel_to_joists
                    # Rotated positioning - use rotated sheet dimensions for grid spacing
                    x_pos = (col * self.SUBFLOORING_SHEET_WIDTH) + stagger_x
                    y_pos = row * self.SUBFLOORING_SHEET_LENGTH
                
                # Ensure compliance with element creation rules: fully positive coordinate space
                x_pos = max(0.0, x_pos)
                y_pos = max(0.0, y_pos)
                z_pos = 0.0  # Subflooring starts at floor level
                
                # Ensure sheets stay within floor bounds
                if orientation == 'perpendicular_to_joists':
                    max_x = floor_length - sheet_length
                    max_y = floor_width - sheet_width
                else:  # parallel_to_joists - after rotation, dimensions are swapped
                    max_x = floor_length - sheet_width  # After rotation, width becomes X dimension
                    max_y = floor_width - sheet_length  # After rotation, length becomes Y dimension
                
                x_pos = min(x_pos, max(0.0, max_x))
                y_pos = min(y_pos, max(0.0, max_y))
                
                sheet.move(dx=x_pos, dy=y_pos, dz=z_pos)
                elements.append(sheet)
                
                sheet_index += 1
        
        return elements
    
    def _create_finish_flooring_elements(self) -> List[Element]:
        """Create finish flooring elements"""
        elements = []
        floor_width = self.params['floor_width']
        floor_length = self.params['floor_length']
        direction = self.params['finish_flooring_direction']
        edge_gap_ft = self.params['edge_gap'] / 12.0
        
        # Get typical dimensions from finish flooring class
        finish_class = self.FINISH_FLOORING_CLASS
        typical_length = getattr(finish_class, 'TYPICAL_LENGTH', 4.0)
        typical_width = getattr(finish_class, 'TYPICAL_WIDTH', 0.25)
        
        # Calculate number of pieces needed
        if direction == 'parallel_to_length':
            # Flooring runs parallel to floor length
            pieces_across = math.ceil(floor_width / typical_width)
            rows_needed = math.ceil(floor_length / typical_length)
        else:  # parallel_to_width
            # Flooring runs parallel to floor width
            pieces_across = math.ceil(floor_length / typical_width)
            rows_needed = math.ceil(floor_width / typical_length)
        
        piece_index = 0
        z_offset = self.SUBFLOORING_THICKNESS / 12.0  # Convert inches to feet
        
        for row in range(rows_needed):
            for col in range(pieces_across):
                # CREATE finish flooring piece
                piece = self.FINISH_FLOORING_CLASS()
                piece.name = f"{self.name}_finish_{row+1}_{col+1}"
                
                # ROTATE: Orient piece based on direction
                if direction == 'parallel_to_width':
                    # Rotate 90° to run parallel to width
                    piece.rotate_z(math.pi/2, [0, 0, 0])
                
                # Apply joint staggering offset

                ## TODO improve staggering logic
                stagger_x, stagger_y = self._apply_joint_staggering(col, row, 'finish')
                stagger_x = stagger_y = 0 ## not using it right now

                # MOVE: Position piece in final location
                if direction == 'parallel_to_length':
                    x_pos = row * typical_length + (row * edge_gap_ft)
                    y_pos = (col * typical_width) + (col * edge_gap_ft) + stagger_x
                else:  # parallel_to_width
                    x_pos = (col * typical_width) + (col * edge_gap_ft) + stagger_x
                    y_pos = row * typical_length + (row * edge_gap_ft)
                
                # Ensure compliance with element creation rules: fully positive coordinate space
                x_pos = max(0.0, x_pos)
                y_pos = max(0.0, y_pos)
                z_pos = z_offset  # Finish flooring sits on top of subflooring
                
                # Ensure pieces stay within floor bounds
                x_pos = min(x_pos, max(0.0, floor_length - typical_length))
                y_pos = min(y_pos, max(0.0, floor_width - typical_width))
                
                piece.move(dx=x_pos, dy=y_pos, dz=z_pos)
                elements.append(piece)
                
                piece_index += 1
        
        return elements


# SPECIFIC FLOOR ASSEMBLY CLASSES

class PlywoodHardwoodFloorAssembly_0_75(BaseFloorAssembly):
    """3/4" plywood subflooring with hardwood finish flooring"""
    SUBFLOORING_CLASS = PlywoodSubflooring_0_75
    FINISH_FLOORING_CLASS = Hardwood_2_25
    MATERIAL_TYPE = "Plywood + Hardwood"
    SUBFLOORING_THICKNESS = 0.75


class OSBHardwoodFloorAssembly_0_75(BaseFloorAssembly):
    """3/4" OSB subflooring with hardwood finish flooring"""
    SUBFLOORING_CLASS = OSBSubflooring_0_75
    FINISH_FLOORING_CLASS = Hardwood_2_25
    MATERIAL_TYPE = "OSB + Hardwood"
    SUBFLOORING_THICKNESS = 0.75


class AdvantechHardwoodFloorAssembly(BaseFloorAssembly):
    """Advantech subflooring with hardwood finish flooring"""
    SUBFLOORING_CLASS = AdvantechSubflooring
    FINISH_FLOORING_CLASS = Hardwood_3_0
    MATERIAL_TYPE = "Advantech + Hardwood"
    SUBFLOORING_THICKNESS = 0.75


class PlywoodLVPFloorAssembly_0_625(BaseFloorAssembly):
    """5/8" plywood subflooring with LVP finish flooring"""
    SUBFLOORING_CLASS = PlywoodSubflooring_0_625
    FINISH_FLOORING_CLASS = LVP_6x48
    MATERIAL_TYPE = "Plywood + LVP"
    SUBFLOORING_THICKNESS = 0.625


class OSBLVPFloorAssembly_0_625(BaseFloorAssembly):
    """5/8" OSB subflooring with LVP finish flooring"""
    SUBFLOORING_CLASS = OSBSubflooring_0_625
    FINISH_FLOORING_CLASS = LVP_7x48
    MATERIAL_TYPE = "OSB + LVP"
    SUBFLOORING_THICKNESS = 0.625


class PlywoodTileFloorAssembly_1_0(BaseFloorAssembly):
    """1" plywood subflooring with tile finish flooring"""
    SUBFLOORING_CLASS = PlywoodSubflooring_1_0
    FINISH_FLOORING_CLASS = Tile_12x12
    MATERIAL_TYPE = "Plywood + Tile"
    SUBFLOORING_THICKNESS = 1.0


class PlywoodEngineeredFloorAssembly_0_75(BaseFloorAssembly):
    """3/4" plywood subflooring with engineered hardwood finish flooring"""
    SUBFLOORING_CLASS = PlywoodSubflooring_0_75
    FINISH_FLOORING_CLASS = EngineeredHardwood_5_0
    MATERIAL_TYPE = "Plywood + Engineered Hardwood"
    SUBFLOORING_THICKNESS = 0.75


class OSBLaminateFloorAssembly_0_625(BaseFloorAssembly):
    """5/8" OSB subflooring with laminate finish flooring"""
    SUBFLOORING_CLASS = OSBSubflooring_0_625
    FINISH_FLOORING_CLASS = Laminate_5x47
    MATERIAL_TYPE = "OSB + Laminate"
    SUBFLOORING_THICKNESS = 0.625


class PlywoodCarpetFloorAssembly_0_5(BaseFloorAssembly):
    """1/2" plywood subflooring with carpet finish flooring"""
    SUBFLOORING_CLASS = PlywoodSubflooring_0_5
    FINISH_FLOORING_CLASS = Carpet_12ft_Roll
    MATERIAL_TYPE = "Plywood + Carpet"
    SUBFLOORING_THICKNESS = 0.5


# Example usage
if __name__ == "__main__":
    # Example 1: Standard hardwood floor assembly
    hardwood_floor = PlywoodHardwoodFloorAssembly_0_75(
        floor_width=12.0,
        floor_length=16.0,
        subflooring_orientation='perpendicular_to_joists',
        finish_flooring_direction='parallel_to_length',
        stagger_subflooring_joints=True,
        stagger_finish_joints=True,
        edge_gap=0.125
    )

    # Create a tile assembly
    tile_floor = PlywoodTileFloorAssembly_1_0(
        floor_width=10.0,
        floor_length=12.0,
        subflooring_orientation='perpendicular_to_joists',
        finish_flooring_direction='parallel_to_length',
        stagger_subflooring_joints=False,
        stagger_finish_joints=False,
        edge_gap=0.125
    )
    
    from hierarchical.utils import plot_items
    plot_items([tile_floor])
