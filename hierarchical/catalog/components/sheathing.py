"""
Wall sheathing component classes following consistent assembly rules.
"""
## STANDARDIZATION ##

# All items, components, and objects are built starting at 0,0,0 with the X axis being the longest dimension, 
# y being the second longest, and z being the 3rd (or up when up is important)
# This allows for easy alignment and positioning of items in a 3D space.
# This also standardizes how objects must be moved, rotated and scaled to work together.

from typing import Dict, List, Tuple
from hierarchical.catalog.base import ParametricComponent, Parameter
from hierarchical.catalog.elements.lumber import PlywoodSheet_0_25, PlywoodSheet_0_375, PlywoodSheet_0_5, PlywoodSheet_0_625, PlywoodSheet_0_75
from hierarchical.catalog.elements.lumber import OSBSheet_0_375, OSBSheet_0_5, OSBSheet_0_625, OSBSheet_0_75
from hierarchical.items import Element
import math

class BaseSheathing(ParametricComponent):
    """Base class for wall sheathing components"""
    
    SHEET_CLASS = None  # To be set by subclasses
    MATERIAL_TYPE = None  # To be set by subclasses
    THICKNESS = None  # To be set by subclasses
    
    # Standard sheet dimensions following element creation rules
    SHEET_LENGTH = 8.0  # feet (X-axis - longest dimension)
    SHEET_WIDTH = 4.0   # feet (Y-axis - middle dimension)
    # thickness is Z-axis (shortest dimension)
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'wall_height': Parameter(
                name='wall_height',
                type=float,
                default=8.0,
                min_value=7.0,
                max_value=12.0,
                unit='ft',
                description="Height of the wall to sheath"
            ),
            'wall_length': Parameter(
                name='wall_length',
                type=float,
                default=16.0,
                min_value=4.0,
                max_value=math.inf,
                unit='ft',
                description="Length of the wall to sheath"
            ),
            'installation_orientation': Parameter(
                name='installation_orientation',
                type=str,
                default='vertical',
                description="Sheet orientation: 'vertical' (8ft up) or 'horizontal' (8ft across)"
            ),
            'stagger_joints': Parameter(
                name='stagger_joints',
                type=bool,
                default=True,
                description="Stagger sheet joints for structural integrity"
            ),
            'edge_gap': Parameter(
                name='edge_gap',
                type=float,
                default=0.125,
                min_value=0.0,
                max_value=0.25,
                unit='in',
                description="Gap between sheets for expansion"
            ),
            'wall_frame': Parameter(
                name='wall_frame',
                type=object,
                default=None,
                description="Optional wall frame component to align sheathing with stud locations"
            )
        }
    
    def _calculate_sheet_layout(self) -> Tuple[int, int, List[Tuple[float, float]]]:
        """Calculate optimal sheet layout to minimize waste"""
        wall_height = self.params['wall_height']
        wall_length = self.params['wall_length']
        orientation = self.params['installation_orientation']
        edge_gap_ft = self.params['edge_gap'] / 12.0  # Convert inches to feet
        
        if orientation == 'vertical':
            # Sheets installed vertically (8' dimension goes up)
            sheets_across = math.ceil(wall_length / self.SHEET_WIDTH)
            sheets_up = math.ceil(wall_height / self.SHEET_LENGTH)
            
            # Calculate actual sheet dimensions needed
            sheet_sizes = []
            for col in range(sheets_across):
                for row in range(sheets_up):
                    # Calculate sheet dimensions
                    if col == sheets_across - 1:  # Last column
                        sheet_width = wall_length - (col * self.SHEET_WIDTH) - (col * edge_gap_ft)
                        sheet_width = min(sheet_width, self.SHEET_WIDTH)
                    else:
                        sheet_width = self.SHEET_WIDTH
                    
                    if row == sheets_up - 1:  # Top row
                        sheet_length = wall_height - (row * self.SHEET_LENGTH) - (row * edge_gap_ft)
                        sheet_length = min(sheet_length, self.SHEET_LENGTH)
                    else:
                        sheet_length = self.SHEET_LENGTH
                    
                    sheet_sizes.append((sheet_length, sheet_width))
            
        else:  # horizontal
            # Sheets installed horizontally (8' dimension goes across)
            sheets_across = math.ceil(wall_length / self.SHEET_LENGTH)
            sheets_up = math.ceil(wall_height / self.SHEET_WIDTH)
            
            # Calculate actual sheet dimensions needed
            sheet_sizes = []
            for col in range(sheets_across):
                for row in range(sheets_up):
                    # Calculate sheet dimensions
                    if col == sheets_across - 1:  # Last column
                        sheet_length = wall_length - (col * self.SHEET_LENGTH) - (col * edge_gap_ft)
                        sheet_length = min(sheet_length, self.SHEET_LENGTH)
                    else:
                        sheet_length = self.SHEET_LENGTH
                    
                    if row == sheets_up - 1:  # Top row
                        sheet_width = wall_height - (row * self.SHEET_WIDTH) - (row * edge_gap_ft)
                        sheet_width = min(sheet_width, self.SHEET_WIDTH)
                    else:
                        sheet_width = self.SHEET_WIDTH
                    
                    sheet_sizes.append((sheet_length, sheet_width))
        
        return sheets_across, sheets_up, sheet_sizes
    
    def _get_stud_locations_from_frame(self, wall_frame) -> List[float]:
        """Extract stud X-positions from a wall frame component"""
        stud_positions = []
        
        if wall_frame is None:
            return stud_positions
        
        try:
            # Get all elements from the wall frame
            elements = wall_frame.sub_items
            
            # Find stud elements (they contain "stud" in their name)
            for element in elements:
                if hasattr(element, 'name') and 'stud' in element.name.lower():
                    # Get the X position of the stud
                    # Assuming studs have been moved to their final positions (they should be already if following assembly rules)
                    position = element.get_centroid().x  # Get X position
                    stud_positions.append(position)

            # Sort positions and remove duplicates
            stud_positions = sorted(list(set(stud_positions)))
            
        except Exception as e:
            print(f"Warning: Could not extract stud positions from frame: {e}")
            return []
        
        return stud_positions

    def _create_elements_from_frame(self, frame_studs: List[Element]) -> List[Element]:
        """Create sheathing elements based on actual stud positions"""
        elements = []
        
        # Get stud data and validate orientation
        studs_data = [(stud.get_centroid().x, stud.ACTUAL_WIDTH) for stud in frame_studs]
        studs_data.sort()
        stud_positions = [pos for pos, _ in studs_data]
        orientation = self._validate_and_determine_orientation(stud_positions)
        
        edge_gap_ft = self.params['edge_gap'] / 12.0
        wall_height = self.params['wall_height']
        
        if orientation == 'vertical':
            current_x = 0.0
            sheet_num = 1
            
            while current_x < self.params['wall_length']:
                # Try standard sheet width
                natural_edge = current_x + self.SHEET_WIDTH
                sheet_width = self.SHEET_WIDTH
                
                # Check if natural edge lands on a stud (within tolerance)
                lands_on_stud = any(abs(natural_edge - pos) <= width/2 for pos, width in studs_data)
                
                if not lands_on_stud:
                    # Find closest stud before natural edge and trim to it
                    studs_before = [pos for pos, _ in studs_data if pos < natural_edge]
                    if not studs_before:
                        raise ValueError(f"No stud before natural edge at {natural_edge:.2f}'")
                    sheet_width = max(studs_before) - current_x
                
                # Don't exceed wall boundary
                sheet_width = min(sheet_width, self.params['wall_length'] - current_x)
                
                if sheet_width > 0.1:
                    # CREATE element
                    sheet = self.SHEET_CLASS(length=sheet_width)
                    sheet.name = f"{self.name}_sheet_{sheet_num}"
                    
                    # ROTATE element  
                    sheet.rotate_z(math.pi/2, [0, 0, 0])
                    
                    # MOVE element
                    sheet.move(dx=current_x, dz=0)
                    elements.append(sheet)
                    
                    current_x += sheet_width + edge_gap_ft
                    sheet_num += 1
                else:
                    break
        
        else:  # horizontal
            rows_needed = math.ceil(wall_height / self.SHEET_WIDTH)
            sheet_num = 1
            
            for row in range(rows_needed):
                row_height = min(self.SHEET_WIDTH, wall_height - (row * self.SHEET_WIDTH))
                current_x = 0.0
                
                while current_x < self.params['wall_length']:
                    # Try standard sheet length
                    natural_edge = current_x + self.SHEET_LENGTH
                    sheet_length = self.SHEET_LENGTH
                    
                    # Check if natural edge lands on a stud (within tolerance)
                    lands_on_stud = any(abs(natural_edge - pos) <= width/2 for pos, width in studs_data)
                    
                    if not lands_on_stud:
                        # Find closest stud before natural edge and trim to it
                        studs_before = [pos for pos, _ in studs_data if pos < natural_edge]
                        if not studs_before:
                            raise ValueError(f"No stud before natural edge at {natural_edge:.2f}'")
                        sheet_length = max(studs_before) - current_x
                    
                    # Don't exceed wall boundary
                    sheet_length = min(sheet_length, self.params['wall_length'] - current_x)
                    
                    if sheet_length > 0.1:
                        # CREATE element
                        sheet = self.SHEET_CLASS(length=sheet_length)
                        sheet.name = f"{self.name}_sheet_{sheet_num}"
                        
                        # ROTATE element (no rotation needed for horizontal)
                        # sheets are already in horizontal orientation by default
                        
                        # MOVE element
                        sheet.move(dx=current_x, dz=row * self.SHEET_WIDTH)
                        elements.append(sheet)
                        
                        current_x += sheet_length + edge_gap_ft
                        sheet_num += 1
                    else:
                        break
   

        
        return elements
    def _apply_joint_staggering(self, col: int, row: int) -> Tuple[float, float]:
        """Apply joint staggering offset following building codes
        Ensures all elements remain in fully positive coordinate space"""
        if not self.params['stagger_joints']:
            return 0.0, 0.0
        
        # Calculate stagger offset
        stagger_offset = 0.0
        if row % 2 == 1:  # Odd rows get staggered
            base_stagger = self.SHEET_WIDTH / 2.0 if self.params['installation_orientation'] == 'vertical' else self.SHEET_LENGTH / 2.0
            
            # Ensure stagger doesn't push sheets into negative space or beyond wall bounds
            current_x = col * (self.SHEET_WIDTH if self.params['installation_orientation'] == 'vertical' else self.SHEET_LENGTH)
            proposed_x = current_x + base_stagger
            
            # Check boundaries - must stay in positive space and within wall length
            if proposed_x >= 0 and proposed_x < self.params['wall_length']:
                stagger_offset = base_stagger
            else:
                stagger_offset = 0.0  # Don't stagger if it violates boundaries
        
        return stagger_offset, 0.0
        """Apply joint staggering offset following building codes"""
        if not self.params['stagger_joints']:
            return 0.0, 0.0
        
        # Stagger every other row by half sheet width
        stagger_offset = 0.0
        if row % 2 == 1:  # Odd rows get staggered
            stagger_offset = self.SHEET_WIDTH / 2.0 if self.params['installation_orientation'] == 'vertical' else self.SHEET_LENGTH / 2.0
        
        return stagger_offset, 0.0
    
    def create_elements(self) -> List[Element]:
        """Create sheathing elements following consistent assembly rules:
        Create element → rotate element → move element into place → repeat
        
        If wall_frame is provided, align sheathing with actual stud locations.
        Otherwise, use standard grid layout.
        """
        
        if self.SHEET_CLASS is None:
            raise ValueError(f"{self.__class__.__name__} must define SHEET_CLASS")
        
        elements = []
        wall_frame = self.params.get('wall_frame')
        
        # Check if we should use frame-based layout
        if wall_frame is not None:
            
            return self._create_elements_from_frame(wall_frame)
        
        # Fall back to standard grid layout
        return self._create_elements_standard_layout()

    def _create_elements_from_frame(self, frame: List[Element]) -> List[Element]:
        """Create sheathing elements based on actual stud positions"""
        elements = []

        # extract frame studs
        frame_studs = [s for s in frame.sub_items if "stud" in s.name]

        # Get stud data and validate orientation
        studs_data = [(stud.get_centroid().x, stud.ACTUAL_WIDTH) for stud in frame_studs]
        studs_data.sort()
        stud_positions = [pos for pos, _ in studs_data]
        orientation = self._validate_and_determine_orientation(stud_positions)
        
        edge_gap_ft = self.params['edge_gap'] / 12.0
        wall_height = self.params['wall_height']
        
        if orientation == 'vertical':
            
            rows_needed = math.ceil(wall_height / self.SHEET_LENGTH)
            sheet_num = 1
            for row in range(rows_needed):
                current_x = 0.0            
                while current_x < self.params['wall_length']:
                    # Try standard sheet width
                    natural_edge = current_x + self.SHEET_WIDTH
                    sheet_width = self.SHEET_WIDTH
                    sheet_length = self.SHEET_LENGTH
                    
                    # Check if natural edge lands on a stud (within tolerance)
                    lands_on_stud = any(abs(natural_edge - pos) <= width/2 for pos, width in studs_data)
                    
                    if not lands_on_stud:
                        # Find closest stud before natural edge and trim to it
                        studs_before = [pos for pos, _ in studs_data if pos < natural_edge]
                        if not studs_before:
                            raise ValueError(f"No stud before natural edge at {natural_edge:.2f}'")
                        sheet_width = max(studs_before) - current_x
                    
                    # Don't exceed wall boundary
                    sheet_width = min(sheet_width, self.params['wall_length'] - current_x)

                    # determine sheet length based on row * self.SHEET_LENGTH and wall height
                    if row == rows_needed - 1:  # Last row
                        sheet_length = wall_height - (row * self.SHEET_LENGTH)

                    if sheet_width > 0.1:
                        # CREATE element
                        sheet = self.SHEET_CLASS(width=sheet_width, length=sheet_length)
                        sheet.name = f"{self.name}_sheet_{sheet_num}"
                        
                        # ROTATE element  
                        sheet.rotate_z(math.pi/2, [0, 0, 0])
                        sheet.move(dx=sheet_width, dz=0)
                        sheet.rotate_x(math.pi/2, [0, 0, 0])  # Center on X-axis
                        # MOVE element
                        sheet.move(dx=current_x, dz=row * self.SHEET_LENGTH)
                        elements.append(sheet)
                        
                        current_x += sheet_width + edge_gap_ft
                        sheet_num += 1
                    else:
                        break
        
        else:  # horizontal
            rows_needed = math.ceil(wall_height / self.SHEET_WIDTH)
            sheet_num = 1
            
            for row in range(rows_needed):
                row_height = min(self.SHEET_WIDTH, wall_height - (row * self.SHEET_WIDTH))
                current_x = 0.0
                
                while current_x < self.params['wall_length']:
                    # Try standard sheet length
                    natural_edge = current_x + self.SHEET_LENGTH
                    sheet_length = self.SHEET_LENGTH
                    
                    # Check if natural edge lands on a stud (within tolerance)
                    lands_on_stud = any(abs(natural_edge - pos) <= width/2 for pos, width in studs_data)
                    
                    if not lands_on_stud:
                        # Find closest stud before natural edge and trim to it
                        studs_before = [pos for pos, _ in studs_data if pos < natural_edge]
                        if not studs_before:
                            raise ValueError(f"No stud before natural edge at {natural_edge:.2f}'")
                        sheet_length = max(studs_before) - current_x
                    
                    # Don't exceed wall boundary
                    sheet_length = min(sheet_length, self.params['wall_length'] - current_x)
                    
                    if sheet_length > 0.1:
                        # CREATE element
                        sheet = self.SHEET_CLASS(length=sheet_length)
                        sheet.name = f"{self.name}_sheet_{sheet_num}"
                        
                        # ROTATE element 
                        sheet.rotate_x(math.pi/2, [0, 0, 0])  # Center on X-axis
                        # sheets are already in horizontal orientation by default
                        
                        # MOVE element
                        sheet.move(dx=current_x, dz=row * self.SHEET_WIDTH)
                        elements.append(sheet)
                        
                        current_x += sheet_length + edge_gap_ft
                        sheet_num += 1
                    else:
                        break
        
        return elements
    
    def _create_elements_standard_layout(self) -> List[Element]:
        """Create sheathing elements using standard grid layout (original method)"""
        elements = []
        wall_height = self.params['wall_height']
        wall_length = self.params['wall_length']
        orientation = self.params['installation_orientation']
        edge_gap_ft = self.params['edge_gap'] / 12.0
        
        # Calculate optimal layout
        sheets_across, sheets_up, sheet_sizes = self._calculate_sheet_layout()
        
        sheet_index = 0
        for row in range(sheets_up):
            for col in range(sheets_across):
                if sheet_index >= len(sheet_sizes):
                    break
                
                sheet_length, sheet_width = sheet_sizes[sheet_index]
                
                # ELEMENT N: Create sheet
                # Sheet elements created following rules: X=longest, Y=middle, Z=shortest
                sheet = self.SHEET_CLASS(length=sheet_length)  # X-axis dimension
                sheet.name = f"{self.name}_sheet_{row+1}_{col+1}"
                
                # ROTATE: Orient sheet for installation
                if orientation == 'vertical':
                    # Rotate sheet to be vertical on wall (90° around Z-axis to orient properly)
                    sheet.rotate_z(math.pi/2, [0, 0, 0])
                else:  # horizontal
                    # For horizontal installation, may need different rotation
                    # Keep default orientation or rotate as needed
                    pass
                
                # Apply joint staggering offset
                stagger_x, stagger_y = self._apply_joint_staggering(col, row)
                
                # MOVE: Position sheet in final location
                # Following element creation rules: all elements in fully positive coordinate space
                if orientation == 'vertical':
                    x_pos = (col * self.SHEET_WIDTH) + (col * edge_gap_ft) + stagger_x
                    z_pos = row * self.SHEET_LENGTH + (row * edge_gap_ft)
                else:  # horizontal
                    x_pos = (col * self.SHEET_LENGTH) + (col * edge_gap_ft) + stagger_x
                    z_pos = row * self.SHEET_WIDTH + (row * edge_gap_ft)
                
                # Ensure compliance with element creation rules: fully positive coordinate space
                x_pos = max(0.0, x_pos)  # Never allow negative X
                z_pos = max(0.0, z_pos)  # Never allow negative Z
                
                # Ensure sheets stay within wall bounds
                max_x = wall_length - (sheet_width if orientation == 'vertical' else sheet_length)
                max_z = wall_height - (sheet_length if orientation == 'vertical' else sheet_width)
                
                x_pos = min(x_pos, max(0.0, max_x))  # Clamp to positive bounds
                z_pos = min(z_pos, max(0.0, max_z))  # Clamp to positive bounds
                
                sheet.move(dx=x_pos, dz=z_pos)
                elements.append(sheet)
                
                sheet_index += 1
        
        return elements
    
    def _validate_and_determine_orientation(self, stud_positions: List[float]) -> str:
        """Validate prescribed orientation and switch if necessary"""
        if len(stud_positions) < 2:
            return self.params['installation_orientation']
        
        prescribed = self.params['installation_orientation']
        
        # Calculate all bay widths
        bay_widths = []
        
        # First bay: wall start to first stud
        bay_widths.append(stud_positions[0])
        
        # Middle bays: between adjacent studs
        for i in range(len(stud_positions) - 1):
            bay_widths.append(stud_positions[i + 1] - stud_positions[i])
        
        # Last bay: last stud to wall end
        bay_widths.append(self.params['wall_length'] - stud_positions[-1])
        
        max_bay_width = max(bay_widths)
        
        # Check if prescribed orientation works
        prescribed_limit = self.SHEET_WIDTH if prescribed == 'vertical' else self.SHEET_LENGTH
        if max_bay_width <= prescribed_limit:
            return prescribed
        
        # Try alternative orientation
        alternative = 'horizontal' if prescribed == 'vertical' else 'vertical'
        alternative_limit = self.SHEET_WIDTH if alternative == 'vertical' else self.SHEET_LENGTH
        
        if max_bay_width <= alternative_limit:
            print(f"Warning: Switching from '{prescribed}' to '{alternative}' orientation. "
                    f"Max bay: {max_bay_width:.2f}' > {prescribed_limit}' limit.")
            return alternative
        
        # Neither works
        raise ValueError(f"Max bay width {max_bay_width:.2f}' exceeds both orientations (4' and 8')")


class PlywoodSheathing(BaseSheathing):
    """Base class for plywood sheathing"""
    MATERIAL_TYPE = "Plywood"


class PlywoodSheathing_0_25(PlywoodSheathing):
    """1/4" Plywood sheathing"""
    SHEET_CLASS = PlywoodSheet_0_25
    THICKNESS = 0.25


class PlywoodSheathing_0_375(PlywoodSheathing):
    """3/8" Plywood sheathing"""
    SHEET_CLASS = PlywoodSheet_0_375
    THICKNESS = 0.375


class PlywoodSheathing_0_5(PlywoodSheathing):
    """1/2" Plywood sheathing"""
    SHEET_CLASS = PlywoodSheet_0_5
    THICKNESS = 0.5


class PlywoodSheathing_0_625(PlywoodSheathing):
    """5/8" Plywood sheathing"""
    SHEET_CLASS = PlywoodSheet_0_625
    THICKNESS = 0.625


class PlywoodSheathing_0_75(PlywoodSheathing):
    """3/4" Plywood sheathing"""
    SHEET_CLASS = PlywoodSheet_0_75
    THICKNESS = 0.75


class OSBSheathing(BaseSheathing):
    """Base class for OSB sheathing"""
    MATERIAL_TYPE = "OSB"


class OSBSheathing_0_375(OSBSheathing):
    """3/8" OSB sheathing"""
    SHEET_CLASS = OSBSheet_0_375
    THICKNESS = 0.375


class OSBSheathing_0_5(OSBSheathing):
    """1/2" OSB sheathing"""
    SHEET_CLASS = OSBSheet_0_5
    THICKNESS = 0.5


class OSBSheathing_0_625(OSBSheathing):
    """5/8" OSB sheathing"""
    SHEET_CLASS = OSBSheet_0_625
    THICKNESS = 0.625


class OSBSheathing_0_75(OSBSheathing):
    """3/4" OSB sheathing"""
    SHEET_CLASS = OSBSheet_0_75
    THICKNESS = 0.75


# Example usage
if __name__ == "__main__":
    # Example 1: Standard grid layout
    sheathing_standard = PlywoodSheathing_0_5(
        wall_height=8.0,
        wall_length=16.0,
        installation_orientation='vertical',
        stagger_joints=True,
        edge_gap=0.125
    )
    
    elements_standard = sheathing_standard.create_elements()
    print(f"Standard layout: Created {len(elements_standard)} plywood sheets")
    
    # Example 2: Frame-aligned layout
    from hierarchical.catalog.components.wall_frames import WallFrame2X4, WallFrame2X6
    
    # Create a wall frame

    wall_frame = WallFrame2X6(
        height=10.0,
        length=16.0,
        stud_spacing=13.0,  # 16" on center
        species='SPF'
    )
    
    # Create sheathing aligned to the frame
    sheathing_aligned = PlywoodSheathing_0_5(
        wall_height=10.0,
        wall_length=16.0,
        installation_orientation='vertical',
        stagger_joints=True,
        edge_gap=0,
        wall_frame=wall_frame  # Pass the frame for alignment
    )
    
    elements_aligned = sheathing_aligned.create_elements()
    print(f"Frame-aligned layout: Created {len(elements_aligned)} plywood sheets")
    
    # Visualize if plotting available
    try:
        from hierarchical.utils import plot_items
        print("Visualizing frame-aligned sheathing...")
        plot_items([sheathing_aligned, wall_frame])
    except ImportError:
        print("Visualization not available")