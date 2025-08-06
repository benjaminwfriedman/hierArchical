"""
Base classes for parametric catalog items that inherit from hierarchical items.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from hierarchical.items import Element, Component, Object
from hierarchical.geometry import Geometry


@dataclass
class Parameter:
    """Definition of a parameter for a parametric item."""
    name: str
    type: type  # float, int, str, Enum
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: str = "m"
    description: str = ""


class ParametricElement(Element, ABC):
    """Base class for parametric Elements that inherit from items.Element"""
    
    def __init__(self, **params):
        # Validate and store parameters
        self.params = self._validate_parameters(params)
        
        # Create geometry and call parent Element constructor
        geometry = self.create_geometry()
        super().__init__(
            name=f"{self.__class__.__name__}_{hash(str(self.params))}",
            geometry=geometry,
            type=self.__class__.__name__.lower(),
            material=self.get_material_type()
        )
    
    @classmethod
    @abstractmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        """Return parameter definitions for this element type"""
        pass
    
    @classmethod
    @abstractmethod  
    def get_material_type(cls) -> str:
        """Return material type for this element"""
        pass
    
    @abstractmethod
    def create_geometry(self) -> Geometry:
        """Create geometry based on current parameters"""
        pass
    
    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set default parameters"""
        param_defs = self.get_parameters()
        validated = {}
        
        for param_name, param_def in param_defs.items():
            if param_name in params:
                value = params[param_name]
                # Add validation logic here
                if param_def.min_value is not None and isinstance(value, (int, float)):
                    if value < param_def.min_value:
                        raise ValueError(f"{param_name} must be >= {param_def.min_value}")
                if param_def.max_value is not None and isinstance(value, (int, float)):
                    if value > param_def.max_value:
                        raise ValueError(f"{param_name} must be <= {param_def.max_value}")
                validated[param_name] = value
            else:
                validated[param_name] = param_def.default
                
        return validated


class ParametricComponent(Component, ABC):
    """Base class for parametric Components that inherit from items.Component"""
    
    def __init__(self, **params):
        # Validate and store parameters
        self.params = self._validate_parameters(params)
        
        # Generate name for temporary use
        temp_name = f"{self.__class__.__name__}_{hash(str(self.params))}"
        
        # Create constituent elements first (using temp name)
        self.name = temp_name  # Temporarily set name for create_elements
        elements = self.create_elements()
        
        # Create component using from_elements to get proper geometry and materials
        component = Component.from_elements(
            elements=tuple(elements),
            name=temp_name,
            type=self.__class__.__name__.lower()
        )
        
        # Call parent Component constructor with proper geometry
        super().__init__(
            name=component.name,
            geometry=component.geometry,
            type=component.type,
            sub_items=component.sub_items,
            materials=component.materials,
            relationships=component.relationships,
            attributes=component.attributes,
            ontologies=component.ontologies,
            color=component.color,
            unit_system=component.unit_system,
            id=component.id,
            embeddable=component.embeddable
        )
    
    @classmethod
    @abstractmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        """Return parameter definitions for this component type"""
        pass
    
    @abstractmethod
    def create_elements(self) -> List[Element]:
        """Create the constituent elements for this component"""
        pass
    
    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set default parameters"""
        param_defs = self.get_parameters()
        validated = {}
        
        for param_name, param_def in param_defs.items():
            if param_name in params:
                value = params[param_name]
                # Add validation logic here
                if param_def.min_value is not None and isinstance(value, (int, float)):
                    if value < param_def.min_value:
                        raise ValueError(f"{param_name} must be >= {param_def.min_value}")
                if param_def.max_value is not None and isinstance(value, (int, float)):
                    if value > param_def.max_value:
                        raise ValueError(f"{param_name} must be <= {param_def.max_value}")
                validated[param_name] = value
            else:
                validated[param_name] = param_def.default
                
        return validated


class ParametricObject(Object, ABC):
    """Base class for parametric Objects that inherit from items.Object"""
    
    def __init__(self, **params):
        # Validate and store parameters
        self.params = self._validate_parameters(params)
        
        # Generate name for temporary use
        temp_name = f"{self.__class__.__name__}_{hash(str(self.params))}"
        
        # Create constituent components first (using temp name)
        self.name = temp_name  # Temporarily set name for create_components
        components = self.create_components()
        
        # Create object using from_components to get proper geometry and materials
        obj = Object.from_components(
            components=tuple(components),
            name=temp_name,
            type=self.__class__.__name__.lower()
        )
        
        # Call parent Object constructor with proper geometry
        super().__init__(
            name=obj.name,
            geometry=obj.geometry,
            type=obj.type,
            sub_items=obj.sub_items,
            materials=obj.materials,
            relationships=obj.relationships,
            attributes=obj.attributes,
            ontologies=obj.ontologies,
            color=obj.color,
            unit_system=obj.unit_system,
            id=obj.id,
            embeddable=obj.embeddable
        )
    
    @classmethod
    @abstractmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        """Return parameter definitions for this object type"""
        pass
    
    @abstractmethod
    def create_components(self) -> List[Component]:
        """Create the constituent components for this object"""
        pass
    
    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set default parameters"""
        param_defs = self.get_parameters()
        validated = {}
        
        for param_name, param_def in param_defs.items():
            if param_name in params:
                value = params[param_name]
                # Add validation logic here
                if param_def.min_value is not None and isinstance(value, (int, float)):
                    if value < param_def.min_value:
                        raise ValueError(f"{param_name} must be >= {param_def.min_value}")
                if param_def.max_value is not None and isinstance(value, (int, float)):
                    if value > param_def.max_value:
                        raise ValueError(f"{param_name} must be <= {param_def.max_value}")
                validated[param_name] = value
            else:
                validated[param_name] = param_def.default
                
        return validated