"""
Scalable parametric deck frame components using inheritance.
Preserves original business logic while eliminating code duplication.
"""

from typing import Dict, List, Type
from abc import ABC, abstractmethod
from hierarchical.catalog.base import ParametricComponent, Parameter
from hierarchical.catalog.elements.lumber import Lumber2X4, Lumber2X6, Lumber2X8, Lumber2X10, Lumber2X12, BaseLumber
from hierarchical.items import Element
import math

class BaseDeckFrame(ParametricComponent, ABC):
    """Base deck frame that contains the common assembly logic"""
    
    MAX_RIM_LENGTH = 16.0  # Maximum length for a single rim board segment in feet
    
    @property
    @abstractmethod
    def lumber_class(self) -> Type[BaseLumber]:
        """Return the lumber class to use for this frame type"""
        pass
    
    @property
    @abstractmethod 
    def default_width(self) -> float:
        """Return default width for this lumber size"""
        pass
    
    @property
    @abstractmethod
    def max_width(self) -> float:
        """Return maximum width for this lumber size"""
        pass
    
    @property
    @abstractmethod
    def default_double_rim(self) -> bool:
        """Return default double rim setting"""
        pass
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        # Create instance to access properties
        instance = cls.__new__(cls)
        return {
            'width': Parameter(
                name='width',
                type=float,
                default=instance.default_width,
                min_value=4.0,
                max_value=instance.max_width,
                unit='ft',
                description="Width of the deck frame (joist span direction)"
            ),
            'length': Parameter(
                name='length',
                type=float,
                default=16.0,
                min_value=4.0,
                max_value=math.inf,
                unit='ft',
                description="Length of the deck frame (rim board direction)"
            ),
            'joist_spacing': Parameter(
                name='joist_spacing',
                type=float,
                default=16.0,
                min_value=12.0,
                max_value=24.0,
                unit='in',
                description="Joist spacing on center (12, 16, 19.2, or 24 inches)"
            ),
            'species': Parameter(
                name='species',
                type=str,
                default='SPF',
                description="Wood species for all lumber"
            ),
            'double_rim': Parameter(
                name='double_rim',
                type=bool,
                default=instance.default_double_rim,
                description="Use double rim boards for added strength"
            )
        }
    
    def create_elements(self) -> List[Element]:
        """Create all lumber elements following consistent assembly rules:
        Create element → rotate element → move element into place → repeat
        
        Lumber elements are created with X=length, Y=width, Z=thickness
        Deck frame layout: rim boards run along length (X-axis), joists span width (Y-axis)
        """
        elements = []
        
        width = self.params['width']
        length = self.params['length']
        joist_spacing_inches = self.params['joist_spacing']
        joist_spacing_feet = joist_spacing_inches / 12
        species = self.params['species']
        double_rim = self.params['double_rim']

        # Calculate rim board segments for length
        num_rim_segments_l = math.ceil(length / self.MAX_RIM_LENGTH)

        # Create rim boards (single or double)
        rim_layers = 2 if double_rim else 1
        
        # Create front and back rim boards in segments
        for layer in range(rim_layers):
            y_offset = layer * self.lumber_class.ACTUAL_WIDTH if double_rim else 0

            for i in range(num_rim_segments_l):
                segment_length = min(self.MAX_RIM_LENGTH, length - i * self.MAX_RIM_LENGTH)
                
                # FRONT RIM SEGMENT
                front_rim = self.lumber_class(length=segment_length, species=species)
                rim_thickness = front_rim.ACTUAL_WIDTH
                front_rim.name = f"{self.name}_front_rim_L{layer+1}_seg_{i+1}"
                front_rim.rotate_x(math.pi/2, [0, 0, 0])
                front_rim.move(dy=front_rim.attributes.height)
                front_rim.move(dx=i * self.MAX_RIM_LENGTH, dy=y_offset)
                elements.append(front_rim)
                
                # BACK RIM SEGMENT
                back_rim = self.lumber_class(length=segment_length, species=species)
                back_rim.name = f"{self.name}_back_rim_L{layer+1}_seg_{i+1}"
                back_rim.rotate_x(math.pi/2, [0, 0, 0])
                back_rim.move(dx=i * self.MAX_RIM_LENGTH, dy=width - rim_thickness * (layer))
                elements.append(back_rim)
        
        # Create end rim boards
        for layer in range(rim_layers):
            x_offset = layer * self.lumber_class.ACTUAL_WIDTH if double_rim else 0

            left_rim = self.lumber_class(length=width - 2 * rim_layers * rim_thickness, species=species)
            left_rim.name = f"{self.name}_left_rim_L{layer+1}"
            left_rim.rotate_x(math.pi/2, [0, 0, 0])
            left_rim.rotate_z(math.pi/2, [0, 0, 0])
            left_rim.move(dx=x_offset, dy=rim_layers * rim_thickness)
            elements.append(left_rim)
            
            right_rim = self.lumber_class(length=width - 2 * rim_layers * rim_thickness, species=species)
            right_rim.name = f"{self.name}_right_rim_L{layer+1}"
            right_rim.rotate_x(math.pi/2, [0, 0, 0])
            right_rim.rotate_z(math.pi/2, [0, 0, 0])
            right_rim.move(dx=length - rim_thickness * (layer + 1), dy=rim_layers * rim_thickness)
            elements.append(right_rim)
        
        # Create joists
        num_joists = math.ceil(length / joist_spacing_feet)
        joist_length = width - 2 * rim_layers * rim_thickness
        
        for i in range(num_joists):
            x_position = (i + 1) * joist_spacing_feet
            if x_position <= length - rim_thickness:
                joist = self.lumber_class(length=joist_length, species=species)
                joist.name = f"{self.name}_joist_{i+1}"
                joist.rotate_x(math.pi/2, [0, 0, 0])
                joist.rotate_z(math.pi/2, [0, 0, 0])
                joist.move(dx=x_position, dy=rim_thickness * rim_layers)
                elements.append(joist)

        return elements


# Concrete implementations - just specify the lumber type and parameters
class DeckFrame2X8(BaseDeckFrame):
    """Standard 2x8 deck frame assembly following consistent assembly rules"""
    
    @property
    def lumber_class(self) -> Type[BaseLumber]:
        return Lumber2X8
    
    @property
    def default_width(self) -> float:
        return 12.0
    
    @property
    def max_width(self) -> float:
        return 20.0
    
    @property
    def default_double_rim(self) -> bool:
        return False


class DeckFrame2X10(BaseDeckFrame):
    """Standard 2x10 deck frame assembly following consistent assembly rules"""
    
    @property
    def lumber_class(self) -> Type[BaseLumber]:
        return Lumber2X10
    
    @property
    def default_width(self) -> float:
        return 14.0
    
    @property
    def max_width(self) -> float:
        return 24.0
    
    @property
    def default_double_rim(self) -> bool:
        return False


class DeckFrame2X12(BaseDeckFrame):
    """Standard 2x12 deck frame assembly for large spans following consistent assembly rules"""
    
    @property
    def lumber_class(self) -> Type[BaseLumber]:
        return Lumber2X12
    
    @property
    def default_width(self) -> float:
        return 16.0
    
    @property
    def max_width(self) -> float:
        return 28.0
    
    @property
    def default_double_rim(self) -> bool:
        return True


# Optional: Factory pattern for even cleaner creation
class DeckFrameFactory:
    """Factory for creating deck frames based on lumber size"""
    
    _FRAME_CLASSES = {
        '2x8': DeckFrame2X8,
        '2x10': DeckFrame2X10, 
        '2x12': DeckFrame2X12,
    }
    
    @classmethod
    def create_frame(cls, lumber_size: str, **params) -> BaseDeckFrame:
        """Create a deck frame of the specified lumber size"""
        if lumber_size not in cls._FRAME_CLASSES:
            raise ValueError(f"Unsupported lumber size: {lumber_size}. Available: {list(cls._FRAME_CLASSES.keys())}")
        
        frame_class = cls._FRAME_CLASSES[lumber_size]
        return frame_class(**params)
    
    @classmethod
    def get_available_sizes(cls) -> List[str]:
        """Get list of available lumber sizes"""
        return list(cls._FRAME_CLASSES.keys())


if __name__ == "__main__":
    # Example usage - all use the same business logic!
    
    # Direct instantiation
    deck_2x8 = DeckFrame2X8(width=12.0, length=16.0, joist_spacing=16.0)
    deck_2x10 = DeckFrame2X10(width=14.0, length=20.0, joist_spacing=16.0)
    deck_2x10.move(dy=3.0)  # Move second deck frame for visualization
    
    deck_2x12 = DeckFrame2X12(width=16.0, length=24.0, joist_spacing=16.0, double_rim=True)
    deck_2x12.move(dy=6.0)  # Move third deck frame for visualization
    # Factory pattern usage
    deck_factory = DeckFrameFactory.create_frame('2x12', width=20.0, length=32.0, double_rim=True)
    
    from hierarchical.utils import plot_items
    plot_items([deck_2x12, deck_2x8, deck_2x10])