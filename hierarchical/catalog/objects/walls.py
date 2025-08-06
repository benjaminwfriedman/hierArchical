"""
Parametric wall objects - complete wall assemblies.
"""

## STANDARDIZATION ##

# All items, components, and objects are built starting at 0,0,0 with the X axis being the longest dimension, 
# y being the second longest, and z being the 3rd (or up when up is important)
# This allows for easy alignment and positioning of items in a 3D space.
# This also standardizes how objects must be moved, rotated and scaled to work together.

from typing import Dict, List
from hierarchical.catalog.base import ParametricObject, Parameter
from hierarchical.catalog.components.wall_frames import WallFrame2X4, WallFrame2X6
from hierarchical.items import Component, Wall
from hierarchical.catalog.elements.lumber import PlywoodSheet_0_25
from hierarchical.catalog.components.sheathing import PlywoodSheathing_0_5
from hierarchical.catalog.components.drywall_assemblies import BasicDryWallAssembly_0_25
import math

class ExteriorWall(ParametricObject, Wall):
    """Complete exterior wall assembly with frame, sheathing, and insulation"""
    
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
                description="Height of the wall"
            ),
            'length': Parameter(
                name='length',
                type=float,
                default=16.0,
                min_value=2.0,
                max_value=math.inf,
                unit='ft',
                description="Width of the wall"
            ),
            'wall_type': Parameter(
                name='wall_type',
                type=str,
                default='2x6_insulated',
                description="Wall construction type (2x4_basic, 2x6_insulated, etc.)"
            ),
            'r_value': Parameter(
                name='r_value',
                type=float,
                default=20.0,
                min_value=10.0,
                max_value=40.0,
                unit='hr-ft2-F/Btu',
                description="Target R-value for insulation"
            ),
            'stud_spacing': Parameter(
                name='stud_spacing',
                type=float,
                default=16.0,
                min_value=12.0,
                max_value=24.0,
                unit='in',
                description="Stud spacing on center"
            ),
            'species': Parameter(
                name='species',
                type=str,
                default='SPF',
                description="Wood species for framing"
            )
        }
    
    def create_components(self) -> List[Component]:
        """Create wall frame and sheathing components"""
        components = []
        
        height = self.params['height']
        length = self.params['length']
        wall_type = self.params['wall_type']
        stud_spacing = self.params['stud_spacing']
        species = self.params['species']
        
        # Create appropriate wall frame based on wall type
        if wall_type in ['2x4_basic', '2x4_insulated']:
            frame = WallFrame2X4(
                height=height,
                length=length,
                stud_spacing=stud_spacing,
                species=species
            )
        else:  # Default to 2x6
            frame = WallFrame2X6(
                height=height,
                length=length,
                stud_spacing=stud_spacing,
                species=species
            )
        
        frame.name = f"{self.name}_frame"
        components.append(frame)
        
        if wall_type in ['2x6_sheathed', '2x6_sheathed', '2x6_insulated', 
                         '2x4_sheathed', '2x4_insulated']:
            # Create sheathing component
            sheathing = PlywoodSheathing_0_5(
                wall_height=height,
                wall_length=length,
                wall_frame=frame  # Default rating
            )
            sheathing.name = f"{self.name}_sheathing"
            components.append(sheathing)
        

            # create internal sheathing 
            sheathing_2 = PlywoodSheathing_0_5(
                wall_height=height,
                wall_length=length,
                wall_frame=frame  # Default rating
            )
            sheathing_2.name = f"{self.name}_sheathing_2"

            sheathing_2.move(dy=frame.attributes.width)

            components.append(sheathing_2)

        
        return components
    

    def _create_insulation_component(self, height: float, width: float) -> Component:
        """Create a simplified insulation component"""
        # For now, create a placeholder component
        from ..elements.lumber import Lumber2X4
        
        insulation_elements = []
        
        # Single placeholder element representing insulation
        insulation_element = Lumber2X4(length=width, species="Fiberglass_Insulation")
        insulation_element.name = f"{self.name}_insulation"
        # Position inside wall cavity
        insulation_element.move(dy=0.25, dz=height/2)
        insulation_elements.append(insulation_element)
        
        return Component.from_elements(
            elements=tuple(insulation_elements),
            name=f"{self.name}_insulation_component",
            type="insulation"
        )


class InteriorWall(ParametricObject, Wall):
    """Complete interior wall assembly"""
    
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
                description="Height of the wall"
            ),
            'length': Parameter(
                name='length',
                type=float,
                default=16.0,
                min_value=2.0,
                max_value=math.inf,
                unit='ft',
                description="Width of the wall"
            ),
            'wall_type': Parameter(
                name='wall_type',
                type=str,
                default='2x4_standard',
                description="Wall construction type (2x4_standard, 2x6_acoustic, etc.)"
            ),
            'stud_spacing': Parameter(
                name='stud_spacing',
                type=float,
                default=16.0,
                min_value=12.0,
                max_value=24.0,
                unit='in',
                description="Stud spacing on center"
            ),
            'species': Parameter(
                name='species',
                type=str,
                default='SPF',
                description="Wood species for framing"
            )
        }
    
    def create_components(self) -> List[Component]:
        """Create interior wall frame"""
        components = []
        
        height = self.params['height']
        length = self.params['length']
        wall_type = self.params['wall_type']
        stud_spacing = self.params['stud_spacing']
        species = self.params['species']
        
        # Create wall frame (most interior walls are 2x4)
        if wall_type in ['2x6_acoustic', '2x6_standard']:
            frame = WallFrame2X6(
                height=height,
                length=length,
                stud_spacing=stud_spacing,
                species=species
            )
        else:  # Default to 2x4
            frame = WallFrame2X6(
                height=height,
                length=length,
                stud_spacing=stud_spacing,
                species=species
            )
        
        frame.name = f"{self.name}_frame"
        components.append(frame)

        # Create drywall component
        if wall_type in ['2x4_standard', '2x6_standard']:
            drywall = BasicDryWallAssembly_0_25(
                wall_height=height,
                wall_length=length,
                installation_orientation='vertical',
                wall_frame=frame
            )
            drywall.name = f"{self.name}_drywall"
            components.append(drywall)

            drywall_2 = BasicDryWallAssembly_0_25(
                wall_height=height,
                wall_length=length,
                installation_orientation='vertical',
                wall_frame=frame
            )
            drywall_2.name = f"{self.name}_drywall_2"
            drywall_2.move(dy=frame.attributes.width)
            components.append(drywall_2)

        return components
    

class ShearWall(ParametricObject, Wall):
    """Shear wall assembly with structural sheathing"""
    
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
                description="Height of the shear wall"
            ),
            'width': Parameter(
                name='width',
                type=float,
                default=8.0,
                min_value=4.0,
                max_value=16.0,
                unit='ft',
                description="Width of the shear wall"
            ),
            'shear_rating': Parameter(
                name='shear_rating',
                type=str,
                default='standard',
                description="Shear rating (standard, high, special)"
            ),
            'sheathing_type': Parameter(
                name='sheathing_type',
                type=str,
                default='plywood',
                description="Sheathing type (plywood, osb, steel)"
            ),
            'stud_spacing': Parameter(
                name='stud_spacing',
                type=float,
                default=16.0,
                min_value=12.0,
                max_value=16.0,  # Shear walls typically need closer spacing
                unit='in',
                description="Stud spacing on center"
            ),
            'species': Parameter(
                name='species',
                type=str,
                default='Douglas Fir',
                description="Wood species for framing"
            )
        }
    
    def create_components(self) -> List[Component]:
        """Create shear wall frame with structural sheathing"""
        components = []
        
        height = self.params['height']
        width = self.params['width']
        stud_spacing = self.params['stud_spacing']
        species = self.params['species']
        shear_rating = self.params['shear_rating']
        
        # Shear walls typically use 2x6 or larger for higher capacity
        frame = WallFrame2X6(
            height=height,
            width=width,
            stud_spacing=stud_spacing,
            species=species
        )
        frame.name = f"{self.name}_shear_frame"
        components.append(frame)
        
        # Create structural sheathing component
        structural_sheathing_1 = PlywoodSheathing_0_5(
            height=height,
            width=width,
            shear_rating=shear_rating
        )
        structural_sheathing_1.name = f"{self.name}_structural_sheathing_1"

        components.append(structural_sheathing_1)

        # structural_sheathing = self._create_structural_sheathing(height, width, shear_rating)
        if structural_sheathing:
            components.append(structural_sheathing)
        
        return components
    
    def _create_structural_sheathing(self, height: float, width: float, shear_rating: str) -> Component:
        """Create structural sheathing component"""
        from ..elements.lumber import Lumber2X4
        
        sheathing_elements = []
        
        # Placeholder structural sheathing
        sheathing = Lumber2X4(length=width, species=f"Structural_Sheathing_{shear_rating}")
        sheathing.name = f"{self.name}_structural_sheathing"
        sheathing.move(dy=0.6, dz=height/2)  # Exterior face
        sheathing_elements.append(sheathing)
        
        return Component.from_elements(
            elements=tuple(sheathing_elements),
            name=f"{self.name}_structural_sheathing",
            type="structural_sheathing"
        )
    
if __name__ == "__main__":
    # Example usage
    from hierarchical.utils import plot_items
    exterior_wall = ExteriorWall(height=10.0, width=20.0, wall_type='2x6_insulated', r_value=30.0, stud_spacing=16.0, species='Douglas Fir')
    components = exterior_wall.create_components()
    for comp in components:
        print(f"Created component: {comp.name} of type {comp.type}")

    interior_wall = InteriorWall(height=10.0, length=15.0, wall_type='2x4_standard', stud_spacing=16.0, species='SPF')
    interior_components = interior_wall.create_components()
    for comp in interior_components:
        print(f"Created component: {comp.name} of type {comp.type}")

    interior_wall.move(dy=5.0)  # Move the interior wall 5 feet in Y direction

    plot_items([exterior_wall, interior_wall], flatten_to_elements=True)


