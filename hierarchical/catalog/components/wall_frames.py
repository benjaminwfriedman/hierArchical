"""
Parametric wall frame components - Updated for consistent assembly rules.
All lumber elements follow standard creation rules: X=longest, Y=middle, Z=shortest dimension.
"""
## STANDARDIZATION ##

# All items, components, and objects are built starting at 0,0,0 with the X axis being the longest dimension, 
# y being the second longest, and z being the 3rd (or up when up is important)
# This allows for easy alignment and positioning of items in a 3D space.
# This also standardizes how objects must be moved, rotated and scaled to work together.

from typing import Dict, List
from hierarchical.catalog.base import ParametricComponent, Parameter
from hierarchical.catalog.elements.lumber import Lumber2X4, Lumber2X6
from hierarchical.items import Element
from hierarchical.utils import plot_items
import math

class WallFrame2X4(ParametricComponent):
    """Standard 2x4 wall frame assembly following consistent assembly rules"""
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'height': Parameter(
                name='height',
                type=float,
                default=8.0,
                min_value=7.0,
                max_value=12.0,
                unit='ft',
                description="Height of the wall frame"
            ),
            'length': Parameter(
                name='length',
                type=float,
                default=16.0,
                min_value=2.0,
                max_value=32.0,
                unit='ft',
                description="Length of the wall frame"
            ),
            'stud_spacing': Parameter(
                name='stud_spacing',
                type=float,
                default=16.0,
                min_value=12.0,
                max_value=24.0,
                unit='in',
                description="Stud spacing on center (12, 16, or 24 inches)"
            ),
            'species': Parameter(
                name='species',
                type=str,
                default='SPF',
                description="Wood species for all lumber"
            )
        }
    
    def create_elements(self) -> List[Element]:
        """Create all lumber elements following consistent assembly rules:
        Create element → rotate element → move element into place → repeat
        
        Lumber elements are created with X=length, Y=width, Z=thickness
        """
        elements = []
        
        height = self.params['height']
        length = self.params['length']
        stud_spacing_inches = self.params['stud_spacing']
        stud_spacing_feet = stud_spacing_inches / 12
        species = self.params['species']

        # ELEMENT 1: Create bottom plate
        # Lumber2X4 created with length along X-axis (wall length)
        bottom_plate = Lumber2X4(length=length, species=species)
        bottom_plate.name = f"{self.name}_bottom_plate"
        # Rotate bottom plate to lay flat (thickness along Z, length along X)
        # Default orientation is correct - no rotation needed
        # Move bottom plate into place (stays at Z=0)
        elements.append(bottom_plate)
        
        
        # Plate thickness (2x4 actual thickness is 1.5")
        plate_thickness = bottom_plate.ACTUAL_WIDTH  # feet

        
        # ELEMENT 2: Create first top plate
        top_plate_1 = Lumber2X4(length=length, species=species)
        top_plate_1.name = f"{self.name}_top_plate_1"
        # Rotate first top plate (default orientation is correct)
        # No rotation needed
        # Move first top plate into place
        top_plate_1.move(dz=height - plate_thickness)
        elements.append(top_plate_1)
        
        # ELEMENT 3: Create second top plate (double top plate)
        top_plate_2 = Lumber2X4(length=length, species=species)
        top_plate_2.name = f"{self.name}_top_plate_2"
        # Rotate second top plate (default orientation is correct)
        # No rotation needed
        # Move second top plate into place
        top_plate_2.move(dz=height)
        elements.append(top_plate_2)
        
        # Calculate stud positions and count
        num_studs = math.ceil(length / stud_spacing_feet) + 1
        stud_length = height - (2 * plate_thickness)  # Subtract bottom and top plate
        
        # ELEMENTS 4+: Create studs sequentially
        for i in range(num_studs):
            x_position = i * stud_spacing_feet
            if x_position < length:  # Don't exceed wall length
                # Create stud element (length = stud height)
                stud = Lumber2X4(length=stud_length, species=species)
                stud.name = f"{self.name}_stud_{i+1}"
                # Rotate stud to vertical position (90° around Y-axis)
                # This rotates the length from X-axis to Z-axis (vertical)
                stud.rotate_y(-math.pi/2, [0, 0, 0])
                # Move stud into place
                stud.move(dx=x_position + stud.ACTUAL_WIDTH, dz=plate_thickness)
                elements.append(stud)
            if x_position + stud.ACTUAL_WIDTH >= length:
                # move the stud to the end of the wall - stud.ACTUAL_WIDTH is the width of the stud
                stud = Lumber2X4(length=stud_length, species=species)
                stud.name = f"{self.name}_stud_{i+1}"
                # Rotate stud to vertical position (90° around Y-axis)
                # This rotates the length from X-axis to Z-axis (vertical)
                stud.rotate_y(-math.pi/2, [0, 0, 0])
                # Move stud into place
                stud.move(dx=self.params['length'], dz=plate_thickness)
                elements.append(stud)

        return elements


class WallFrame2X6(ParametricComponent):
    """Standard 2x6 wall frame assembly following consistent assembly rules"""
    MAX_PLATE_LENGTH = 16.0  # Maximum length for a single plate segment in feet

    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'height': Parameter(
                name='height',
                type=float,
                default=8.0,
                min_value=7.0,
                max_value=12.0,
                unit='ft',
                description="Height of the wall frame"
            ),
            'length': Parameter(
                name='length',
                type=float,
                default=16.0,
                min_value=2.0,
                max_value=math.inf,  # No upper limit for length
                unit='ft',
                description="Length of the wall frame"
            ),
            'stud_spacing': Parameter(
                name='stud_spacing',
                type=float,
                default=16.0,
                min_value=12.0,
                max_value=24.0,
                unit='in',
                description="Stud spacing on center (12, 16, or 24 inches)"
            ),
            'species': Parameter(
                name='species',
                type=str,
                default='SPF',
                description="Wood species for all lumber"
            )
        }
    
    def create_elements(self) -> List[Element]:
        """Create all lumber elements following consistent assembly rules:
        Create element → rotate element → move element into place → repeat
        
        Lumber elements are created with X=length, Y=width, Z=thickness
        """
        elements = []
        
        height = self.params['height']
        length = self.params['length']
        stud_spacing_inches = self.params['stud_spacing']
        stud_spacing_feet = stud_spacing_inches / 12
        species = self.params['species']

        # see how many plates we need
        num_plates = math.ceil(length / self.MAX_PLATE_LENGTH)

        for i in range(num_plates):
            plate_length = min(self.MAX_PLATE_LENGTH, length - i * self.MAX_PLATE_LENGTH)
            # ELEMENT 1: Create bottom plate
            bottom_plate = Lumber2X6(length=plate_length, species=species)
            bottom_plate.name = f"{self.name}_bottom_plate"
            # Rotate bottom plate (default orientation is correct - length along X)
            # No rotation needed
            # Move bottom plate into place (stays at Z=0)
            bottom_plate.move(dx=i * self.MAX_PLATE_LENGTH)
            elements.append(bottom_plate)
            
            # Plate thickness (2x6 actual thickness is 1.5")
            plate_thickness = bottom_plate.ACTUAL_WIDTH  # feet
            
            
            
            # ELEMENT 2: Create first top plate
            top_plate_1 = Lumber2X6(length=plate_length, species=species)
            top_plate_1.name = f"{self.name}_top_plate_1"
            # Rotate first top plate (default orientation is correct)
            # No rotation needed
            # Move first top plate into place
            top_plate_1.move(dz=height - plate_thickness, dx=i * self.MAX_PLATE_LENGTH)
            elements.append(top_plate_1)
            
            # ELEMENT 3: Create second top plate (double top plate)
            top_plate_2 = Lumber2X6(length=plate_length, species=species)
            top_plate_2.name = f"{self.name}_top_plate_2"
            # Rotate second top plate (default orientation is correct)
            # No rotation needed
            # Move second top plate into place
            top_plate_2.move(dz=height - 2 * plate_thickness, dx=i * self.MAX_PLATE_LENGTH)
            elements.append(top_plate_2)
        
        # Calculate stud positions and count
        num_studs = math.ceil(length / stud_spacing_feet) + 1
        stud_length = height - (3 * plate_thickness)  # Subtract bottom and top plate
        
        # ELEMENTS 4+: Create studs sequentially
        for i in range(num_studs):
            x_position = i * stud_spacing_feet
            if x_position < length:  # Don't exceed wall width
                # Create stud element (length = stud height)
                stud = Lumber2X6(length=stud_length, species=species)
                stud.name = f"{self.name}_stud_{i+1}"
                # Rotate stud to vertical position (90° around Y-axis)
                # This rotates the length from X-axis to Z-axis (vertical)
                stud.rotate_y(-math.pi/2, [0, 0, 0])
                # Move stud into place
                stud.move(dx=x_position + stud.ACTUAL_WIDTH, dz=plate_thickness)
                elements.append(stud)

            if x_position + stud.ACTUAL_WIDTH >= length:
                # move the stud to the end of the wall - stud.ACTUAL_WIDTH is the width of the stud
                stud = Lumber2X6(length=stud_length, species=species)
                stud.name = f"{self.name}_stud_{i+1}"
                # Rotate stud to vertical position (90° around Y-axis)
                # This rotates the length from X-axis to Z-axis (vertical)
                stud.rotate_y(-math.pi/2, [0, 0, 0])
                # Move stud into place
                stud.move(dx=self.params['length'], dz=plate_thickness)
                elements.append(stud)
        
        return elements
    



if __name__ == "__main__":
    frame = WallFrame2X6()
    from hierarchical.utils import plot_items
    plot_items([frame])