"""
Parametric deck objects - complete deck assemblies with structure and optional floor/ceiling components.
"""
## TODO
# Implement WoodDeckFrame in deck_frames component
# Add ConcreteDeck with ConcreteSlab frame
# Add SteelFramedDeck with SteelDeckFrame


## STANDARDIZATION ##

# All items, components, and objects are built starting at 0,0,0 with the X axis being the longest dimension, 
# y being the second longest, and z being the 3rd (or up when up is important)
# This allows for easy alignment and positioning of items in a 3D space.
# This also standardizes how objects must be moved, rotated and scaled to work together.

# For decks: All components in positive Z space - ceiling at bottom, frame in middle, floor on top

from typing import Dict, List, Optional
from hierarchical.catalog.base import ParametricObject, Parameter
from hierarchical.catalog.components.floor_assemblies import (
    PlywoodHardwoodFloorAssembly_0_75, OSBHardwoodFloorAssembly_0_75, 
    PlywoodLVPFloorAssembly_0_625, PlywoodTileFloorAssembly_1_0,
    PlywoodCarpetFloorAssembly_0_5
)
from hierarchical.catalog.components.ceiling_assemblies import (
    DrywallCeilingAssembly_0_5, SuspendedCeilingAssembly_2x2_Acoustic,
    PlasterCeilingAssembly_WoodLath
)
from hierarchical.catalog.components.deck_frames import DeckFrame2X8, DeckFrame2X10, DeckFrame2X12
from hierarchical.items import Component, Deck
import math

class BaseDeck(ParametricObject, Deck):
    """Base class for all deck objects"""
    
    DECK_FRAME_CLASS = None  # To be set by subclasses
    DEFAULT_FLOOR_ASSEMBLY = None  # Default floor assembly type
    DEFAULT_CEILING_ASSEMBLY = None  # Default ceiling assembly type
    MATERIAL_TYPE = None  # To be set by subclasses
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'deck_width': Parameter(
                name='deck_width',
                type=float,
                default=12.0,
                min_value=4.0,
                max_value=math.inf,
                unit='ft',
                description="Width of the deck (Y-axis - second longest dimension)"
            ),
            'deck_length': Parameter(
                name='deck_length',
                type=float,
                default=16.0,
                min_value=4.0,
                max_value=math.inf,
                unit='ft',
                description="Length of the deck (X-axis - longest dimension)"
            ),
            'joist_spacing': Parameter(
                name='joist_spacing',
                type=float,
                default=16.0,
                min_value=12.0,
                max_value=24.0,
                unit='in',
                description="Joist spacing on center"
            ),
            'span_direction': Parameter(
                name='span_direction',
                type=str,
                default='parallel_to_length',
                description="Joist span direction: 'parallel_to_length' or 'parallel_to_width'"
            ),
            'wood_species': Parameter(
                name='wood_species',
                type=str,
                default='SPF',
                description="Wood species for framing"
            ),
            'include_floor_assembly': Parameter(
                name='include_floor_assembly',
                type=bool,
                default=True,
                description="Include floor assembly above structure"
            ),
            'include_ceiling_assembly': Parameter(
                name='include_ceiling_assembly',
                type=bool,
                default=True,
                description="Include ceiling assembly below structure"
            ),
            'floor_assembly_type': Parameter(
                name='floor_assembly_type',
                type=str,
                default='hardwood',
                description="Floor assembly type (hardwood, lvp, tile, carpet)"
            ),
            'ceiling_assembly_type': Parameter(
                name='ceiling_assembly_type',
                type=str,
                default='drywall',
                description="Ceiling assembly type (drywall, suspended, plaster)"
            ),
            'deck_frame': Parameter(
                name='deck_frame',
                type=object,
                default=None,
                description="Optional existing deck frame component to use"
            )
        }
    
    def create_components(self) -> List[Component]:
        """Create deck components: structure + optional floor/ceiling assemblies"""
        components = []
        ceiling_assembly = None
        floor_assembly = None

        deck_width = self.params['deck_width']
        deck_length = self.params['deck_length']
        joist_spacing = self.params['joist_spacing']
        span_direction = self.params['span_direction']
        wood_species = self.params['wood_species']
        
        # 1. CREATE STRUCTURAL DECK FRAME (always required)
        deck_frame = self._create_or_use_deck_frame()
        
        # deck comes in at z = 0
        
        # 2. CREATE FLOOR ASSEMBLY (optional - above structure)
        if self.params['include_floor_assembly']:
            floor_assembly = self._create_floor_assembly(deck_frame)
            # floor assembly comes in at z = 0

        # 3. CREATE CEILING ASSEMBLY (optional - below structure)
        if self.params['include_ceiling_assembly']:
            ceiling_assembly = self._create_ceiling_assembly(deck_frame)
            # ceiling assembly comes in at z = 0

        # based on the ceiling assembly we can position the deck frame and then the floor assembly based on the new deck position
        deck_frame.move(dz=ceiling_assembly.attributes.height if ceiling_assembly else 0)
        if floor_assembly:
            floor_assembly.move(dz=deck_frame.attributes.height if deck_frame else 0)

        components.append(deck_frame)
        if self.params['include_floor_assembly']:
            components.append(floor_assembly)
        if self.params['include_ceiling_assembly']:
            components.append(ceiling_assembly)

        return components
    
    def _create_or_use_deck_frame(self) -> Optional[Component]:
        """Create deck frame or use provided one"""
        deck_frame = self.params.get('deck_frame')
        
        if deck_frame is not None:
            # Use provided deck frame
            deck_frame.name = f"{self.name}_existing_frame"
            return deck_frame
        
        # Create new deck frame
        if self.DECK_FRAME_CLASS is None:
            raise ValueError(f"{self.__class__.__name__} must define DECK_FRAME_CLASS")
        
        frame = self.DECK_FRAME_CLASS(
            width=self.params['deck_width'],   # Frame width = deck width
            length=self.params['deck_length'], # Frame length = deck length
            joist_spacing=self.params['joist_spacing'],
            species=self.params['wood_species']
        )
        frame.name = f"{self.name}_frame"
                
        return frame
    
    def _create_floor_assembly(self, deck_frame: Optional[Component]) -> Optional[Component]:
        """Create floor assembly above the structure"""
        floor_type = self.params['floor_assembly_type']
        
        # Select floor assembly class based on type
        floor_assembly_classes = {
            'hardwood': PlywoodHardwoodFloorAssembly_0_75,
            'lvp': PlywoodLVPFloorAssembly_0_625, 
            'tile': PlywoodTileFloorAssembly_1_0,
            'carpet': PlywoodCarpetFloorAssembly_0_5
        }
        
        FloorAssemblyClass = floor_assembly_classes.get(floor_type)
        if FloorAssemblyClass is None:
            print(f"Warning: Unknown floor assembly type '{floor_type}', skipping floor assembly")
            return None
        
        # Determine subflooring orientation based on joist direction
        if self.params['span_direction'] == 'parallel_to_length':
            subflooring_orientation = 'perpendicular_to_joists'
        else:
            subflooring_orientation = 'parallel_to_joists'
        
        # Create floor assembly
        floor_assembly = FloorAssemblyClass(
            floor_width=self.params['deck_width'],
            floor_length=self.params['deck_length'],
            subflooring_orientation=subflooring_orientation,
            finish_flooring_direction='parallel_to_length',  # Default
            stagger_subflooring_joints=True,
            stagger_finish_joints=True,
            deck_frame=deck_frame  # Pass deck frame for alignment
        )
        floor_assembly.name = f"{self.name}_floor_assembly"
          
        return floor_assembly
    
    def _create_ceiling_assembly(self, deck_frame: Optional[Component]) -> Optional[Component]:
        """Create ceiling assembly below the structure"""
        ceiling_type = self.params['ceiling_assembly_type']
        
        # Select ceiling assembly class based on type  
        ceiling_assembly_classes = {
            'drywall': DrywallCeilingAssembly_0_5,
            'suspended': SuspendedCeilingAssembly_2x2_Acoustic,
            'plaster': PlasterCeilingAssembly_WoodLath
        }
        
        CeilingAssemblyClass = ceiling_assembly_classes.get(ceiling_type)
        if CeilingAssemblyClass is None:
            print(f"Warning: Unknown ceiling assembly type '{ceiling_type}', skipping ceiling assembly")
            return None
        
        # Create ceiling assembly
        ceiling_assembly = CeilingAssemblyClass(
            ceiling_width=self.params['deck_width'],
            ceiling_length=self.params['deck_length'],
            joist_direction=self.params['span_direction'],
            finish_orientation='perpendicular_to_joists',  # Default
            stagger_joints=True,
            ceiling_frame=deck_frame  # Pass deck frame for alignment
        )
        ceiling_assembly.name = f"{self.name}_ceiling_assembly"
        
        # Position ceiling assembly at the bottom (Z=0)
        # This provides the foundation layer for the deck assembly
        
        return ceiling_assembly


class WoodFramedDeck(BaseDeck):
    """Wood-framed deck with optional floor and ceiling assemblies"""
    
    DECK_FRAME_CLASS = DeckFrame2X10  # Default to 2X10 frame
    MATERIAL_TYPE = "Wood Framed Deck"
    


class WoodFramedDeck2X8(BaseDeck):
    """Wood-framed deck with 2X8 frame for smaller spans"""
    
    DECK_FRAME_CLASS = DeckFrame2X8
    MATERIAL_TYPE = "Wood Framed Deck (2X8)"


class WoodFramedDeck2X10(BaseDeck):
    """Wood-framed deck with 2X10 frame for medium spans"""
    
    DECK_FRAME_CLASS = DeckFrame2X10
    MATERIAL_TYPE = "Wood Framed Deck (2X10)"


class WoodFramedDeck2X12(BaseDeck):
    """Wood-framed deck with 2X12 frame for large spans"""
    
    DECK_FRAME_CLASS = DeckFrame2X12
    MATERIAL_TYPE = "Wood Framed Deck (2X12)"


# TODO: Add these deck types when corresponding frame classes are available
class ConcreteDeck(BaseDeck):
    """Concrete deck with optional floor and ceiling assemblies"""
    # TODO: DECK_FRAME_CLASS = ConcreteSlab  # When available
    MATERIAL_TYPE = "Concrete Deck"
    
    def create_components(self) -> List[Component]:
        """Override to show TODO message"""
        print("TODO: ConcreteDeck not yet implemented - requires ConcreteSlab frame class")
        return []


class SteelFramedDeck(BaseDeck):
    """Steel-framed deck with optional floor and ceiling assemblies"""
    # TODO: DECK_FRAME_CLASS = SteelDeckFrame  # When available  
    MATERIAL_TYPE = "Steel Framed Deck"
    
    def create_components(self) -> List[Component]:
        """Override to show TODO message"""
        print("TODO: SteelFramedDeck not yet implemented - requires SteelDeckFrame class")
        return []


# Example usage and demonstration
if __name__ == "__main__":
    print("Deck Objects - Complete Horizontal Assemblies")
    print("=" * 47)
    
    # Example 1: Second floor deck (structure + floor + ceiling)
    print("\nExample 1: Second Floor Deck (complete assembly)")
    second_floor = WoodFramedDeck(
        deck_width=12.0,
        deck_length=16.0,
        joist_spacing=16.0,
        span_direction='parallel_to_length',
        include_floor_assembly=True,
        include_ceiling_assembly=True,
        floor_assembly_type='hardwood',
        ceiling_assembly_type='drywall'
    )
    second_floor.name = "Second_Floor_Deck"
    
    components_2nd = second_floor.create_components()
    print(f"Created {len(components_2nd)} components:")
    for comp in components_2nd:
        print(f"  - {comp.name}: {comp.type}")
    
    # Example 2: Ground floor deck (structure + floor, no ceiling)
    print("\nExample 2: Ground Floor Deck (no ceiling below)")
    ground_floor = WoodFramedDeck(
        deck_width=14.0,
        deck_length=20.0,
        include_floor_assembly=True,
        include_ceiling_assembly=False,  # No ceiling below ground floor
        floor_assembly_type='tile',
        wood_species='Douglas Fir'
    )
    ground_floor.name = "Ground_Floor_Deck"
    
    components_ground = ground_floor.create_components()
    print(f"Created {len(components_ground)} components:")
    for comp in components_ground:
        print(f"  - {comp.name}: {comp.type}")
    
    # Example 3: Attic floor deck (minimal floor + ceiling below)
    print("\nExample 3: Attic Floor Deck (minimal floor + ceiling)")
    attic_floor = WoodFramedDeck(
        deck_width=10.0,
        deck_length=12.0,
        include_floor_assembly=True,
        include_ceiling_assembly=True,
        floor_assembly_type='hardwood',  # Minimal attic flooring
        ceiling_assembly_type='drywall'
    )
    attic_floor.name = "Attic_Floor_Deck"
    
    components_attic = attic_floor.create_components()
    print(f"Created {len(components_attic)} components:")
    for comp in components_attic:
        print(f"  - {comp.name}: {comp.type}")
    
    # Example 4: Using existing deck frame
    print("\nExample 4: Using Existing Deck Frame")
    # This would work when you have an actual deck frame component
    # existing_frame = WoodDeckFrame(...)
    # custom_deck = WoodFramedDeck(deck_frame=existing_frame, ...)
    
    print("TODO: Implement when deck frame components are available")
    
    # Visualize if plotting available
    try:
        from hierarchical.utils import plot_items
        print("Visualizing deck assemblies...")
        
        # Position decks side by side for comparison
        ground_floor.move(dx=25.0)   # Move 25' to the right
        attic_floor.move(dx=50.0)    # Move 50' to the right
        
        plot_items([second_floor, ground_floor, attic_floor], flatten_to_elements=True)
    except ImportError:
        print("Visualization not available")
    
