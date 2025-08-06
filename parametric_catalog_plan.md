# Parametric Building Materials Catalog Plan

## Objective
Create a robust, parametric catalog system that allows easy importing of common building materials, components, and objects with variable dimensions where appropriate (e.g., 2x4s parametric in length).

## Current State Analysis

### Existing Strengths
- **Geometry Creation**: `Geometry.from_prism()` and `from_primitive()` methods
- **Material System**: Automatic aggregation through Element → Component → Object hierarchy  
- **Unit Support**: Comprehensive metric/imperial conversion
- **Triple Representation**: Mesh, OpenCascade, TopologicPy for different use cases

### Current Limitations
- No catalog/library structure
- `from_primitive()` incomplete (cylinder, sphere)
- No parameter validation or constraints
- No standard lumber dimensions built-in

## Implementation Plan

### Phase 1: Core Catalog Infrastructure

#### 1.1 Create Base Parametric Classes that Inherit from Items Classes
```python
# catalog/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from hierarchical.items import Element, Component, Object
from hierarchical.geometry import Geometry

@dataclass
class Parameter:
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
                validated[param_name] = value
            else:
                validated[param_name] = param_def.default
                
        return validated

class ParametricComponent(Component, ABC):
    """Base class for parametric Components that inherit from items.Component"""
    
    def __init__(self, **params):
        # Validate and store parameters
        self.params = self._validate_parameters(params)
        
        # Create constituent elements
        elements = self.create_elements()
        
        # Call parent Component constructor
        super().__init__(
            name=f"{self.__class__.__name__}_{hash(str(self.params))}",
            type=self.__class__.__name__.lower()
        )
        
        # Set up the component using from_elements pattern
        component = Component.from_elements(
            elements=tuple(elements),
            name=self.name,
            type=self.type
        )
        
        # Copy properties from the created component
        self.geometry = component.geometry
        self.materials = component.materials
    
    @classmethod
    @abstractmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        """Return parameter definitions for this component type"""
        pass
    
    @abstractmethod
    def create_elements(self) -> List[Element]:
        """Create the constituent elements for this component"""
        pass

class ParametricObject(Object, ABC):
    """Base class for parametric Objects that inherit from items.Object"""
    
    def __init__(self, **params):
        # Validate and store parameters
        self.params = self._validate_parameters(params)
        
        # Create constituent components
        components = self.create_components()
        
        # Call parent Object constructor
        super().__init__(
            name=f"{self.__class__.__name__}_{hash(str(self.params))}",
            type=self.__class__.__name__.lower()
        )
        
        # Set up the object using from_components pattern
        obj = Object.from_components(
            components=tuple(components),
            name=self.name,
            type=self.type
        )
        
        # Copy properties from the created object
        self.geometry = obj.geometry
        self.materials = obj.materials
    
    @classmethod
    @abstractmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        """Return parameter definitions for this object type"""
        pass
    
    @abstractmethod
    def create_components(self) -> List[Component]:
        """Create the constituent components for this object"""
        pass
```

#### 1.2 Create Geometry Helper Functions
Create geometry creation helpers in catalog modules (not in core geometry):
```python
# hierarchical/catalog/geometry_helpers.py
def create_lumber_geometry(width: float, height: float, length: float) -> Geometry:
    """Create rectangular lumber cross-section geometry"""
    base_points = [(0, 0), (width, 0), (width, height), (0, height)]
    return Geometry.from_prism(base_points, length)

def create_i_beam_geometry(depth: float, flange_width: float, web_thickness: float, 
                          flange_thickness: float, length: float) -> Geometry:
    """Create I-beam profile geometry"""
    # Use existing from_prism with I-beam profile points
    profile_points = _calculate_i_beam_profile(depth, flange_width, web_thickness, flange_thickness)
    return Geometry.from_prism(profile_points, length)
```

### Phase 2: Individual Catalog Item Classes

#### 2.1 Lumber Classes
```python
# hierarchical/catalog/lumber.py
from hierarchical.catalog.base import ParametricItem, Parameter
from hierarchical.geometry import Geometry

class Lumber2X4(ParametricElement):
    """Standard 2x4 lumber (actual: 1.5" x 3.5")"""
    
    ACTUAL_WIDTH = 1.5 / 12   # feet
    ACTUAL_HEIGHT = 3.5 / 12  # feet
    
    @classmethod
    def get_parameters(cls):
        return {
            'length': Parameter('length', float, 8.0, min_value=0.5, max_value=20.0, unit='ft'),
            'species': Parameter('species', str, 'SPF', description="Wood species")
        }
    
    @classmethod
    def get_material_type(cls):
        return "wood"
    
    def create_geometry(self):
        base_points = [
            (0, 0), 
            (self.ACTUAL_WIDTH, 0), 
            (self.ACTUAL_WIDTH, self.ACTUAL_HEIGHT), 
            (0, self.ACTUAL_HEIGHT)
        ]
        return Geometry.from_prism(base_points, self.params['length'])

class Lumber2X6(ParametricElement):
    """Standard 2x6 lumber (actual: 1.5" x 5.5")"""
    
    ACTUAL_WIDTH = 1.5 / 12   # feet
    ACTUAL_HEIGHT = 5.5 / 12  # feet
    
    @classmethod
    def get_parameters(cls):
        return {
            'length': Parameter('length', float, 8.0, min_value=0.5, max_value=20.0, unit='ft'),
            'species': Parameter('species', str, 'SPF', description="Wood species")
        }
    
    @classmethod
    def get_material_type(cls):
        return "wood"
    
    def create_geometry(self):
        base_points = [
            (0, 0), 
            (self.ACTUAL_WIDTH, 0), 
            (self.ACTUAL_WIDTH, self.ACTUAL_HEIGHT), 
            (0, self.ACTUAL_HEIGHT)
        ]
        return Geometry.from_prism(base_points, self.params['length'])

class Lumber4X4(ParametricElement):
    """Standard 4x4 lumber (actual: 3.5" x 3.5")"""
    
    ACTUAL_WIDTH = 3.5 / 12   # feet
    ACTUAL_HEIGHT = 3.5 / 12  # feet
    
    @classmethod
    def get_parameters(cls):
        return {
            'length': Parameter('length', float, 8.0, min_value=0.5, max_value=20.0, unit='ft'),
            'species': Parameter('species', str, 'SPF', description="Wood species")
        }
    
    @classmethod
    def get_material_type(cls):
        return "wood"
    
    def create_geometry(self):
        base_points = [
            (0, 0), 
            (self.ACTUAL_WIDTH, 0), 
            (self.ACTUAL_WIDTH, self.ACTUAL_HEIGHT), 
            (0, self.ACTUAL_HEIGHT)
        ]
        return Geometry.from_prism(base_points, self.params['length'])
```

#### 2.2 Steel Catalog
```python
# hierarchical/catalog/steel.py
AISC_WIDE_FLANGE = {
    'W8x10': {'depth': 7.89, 'flange_width': 3.94, 'web_thickness': 0.17, 'flange_thickness': 0.21},
    'W8x13': {'depth': 7.99, 'flange_width': 4.00, 'web_thickness': 0.23, 'flange_thickness': 0.255},
    # ... more standard sizes
}

def create_steel_catalog():
    catalog = MaterialCatalog()
    
    for beam_name, props in AISC_WIDE_FLANGE.items():
        catalog.register_component(ParametricComponent(
            name=f"steel_{beam_name}",
            category="structural_steel",
            description=f"AISC {beam_name} Wide Flange Beam",
            parameters={
                'length': Parameter('length', float, 10.0, min_value=1.0, max_value=40.0, unit='ft'),
                'grade': Parameter('grade', str, 'A992', allowed_values=['A992', 'A572', 'A36'])
            },
            material_type="steel",
            create_function=lambda length, grade: _create_i_beam_geometry(
                depth=props['depth'] / 12,
                flange_width=props['flange_width'] / 12,
                web_thickness=props['web_thickness'] / 12,
                flange_thickness=props['flange_thickness'] / 12,
                length=length
            )
        ))
```

#### 2.3 Sheet Goods Catalog
```python
# hierarchical/catalog/sheet_goods.py
SHEET_STANDARDS = {
    'plywood_1/2': {'thickness': 0.5, 'standard_sizes': [(4, 8), (4, 9), (4, 10)]},
    'plywood_3/4': {'thickness': 0.75, 'standard_sizes': [(4, 8), (4, 9), (4, 10)]},
    'drywall_1/2': {'thickness': 0.5, 'standard_sizes': [(4, 8), (4, 9), (4, 10), (4, 12)]},
    'osb_7/16': {'thickness': 0.4375, 'standard_sizes': [(4, 8)]},
}

def create_sheet_goods_catalog():
    catalog = MaterialCatalog()
    
    for sheet_name, props in SHEET_STANDARDS.items():
        catalog.register_component(ParametricComponent(
            name=f"sheet_{sheet_name}",
            category="sheet_goods",
            description=f"Standard {sheet_name} sheet",
            parameters={
                'width': Parameter('width', float, 4.0, min_value=0.5, max_value=5.0, unit='ft'),
                'length': Parameter('length', float, 8.0, min_value=4.0, max_value=12.0, unit='ft'),
                'cut_to_size': Parameter('cut_to_size', bool, False)
            },
            material_type=sheet_name.split('_')[0],  # 'plywood', 'drywall', etc.
            create_function=lambda width, length, cut_to_size: _create_sheet_geometry(
                width=width,
                length=length, 
                thickness=props['thickness'] / 12
            )
        ))
```

#### 2.4 Hardware Catalog
```python
# hierarchical/catalog/hardware.py
BOLT_STANDARDS = {
    '1/4-20': {'diameter': 0.25, 'threads_per_inch': 20},
    '5/16-18': {'diameter': 0.3125, 'threads_per_inch': 18},
    '3/8-16': {'diameter': 0.375, 'threads_per_inch': 16},
    '1/2-13': {'diameter': 0.5, 'threads_per_inch': 13},
}

def create_hardware_catalog():
    catalog = MaterialCatalog()
    
    for bolt_name, props in BOLT_STANDARDS.items():
        catalog.register_component(ParametricComponent(
            name=f"bolt_{bolt_name}",
            category="hardware",
            description=f"Standard {bolt_name} hex bolt",
            parameters={
                'length': Parameter('length', float, 2.0, min_value=0.5, max_value=12.0, unit='in'),
                'material': Parameter('material', str, 'steel', allowed_values=['steel', 'stainless', 'galvanized'])
            },
            material_type="steel",
            create_function=lambda length, material: _create_bolt_geometry(
                diameter=props['diameter'] / 12,  # Convert to feet
                length=length / 12,
                threads_per_inch=props['threads_per_inch']
            )
        ))
```

### Phase 3: Usage Interface

#### 3.1 Hierarchical Module Structure
```python
# catalog/elements/lumber.py  
from ..base import ParametricElement, ParametricComponent, ParametricObject, Parameter
from hierarchical.geometry import Geometry

class Lumber2X4(ParametricElement):
    """Standard 2x4 lumber (actual: 1.5" x 3.5")"""
    # ... implementation

class Lumber2X6(ParametricElement):
    """Standard 2x6 lumber (actual: 1.5" x 5.5")"""
    # ... implementation

# catalog/elements/steel.py
from ..base import ParametricElement, ParametricComponent, ParametricObject, Parameter
from hierarchical.geometry import Geometry

class SteelW8X10(ParametricItem):
    """AISC W8x10 Wide Flange Beam"""
    # ... implementation

# catalog/elements/sheet_goods.py
from ..base import ParametricElement, ParametricComponent, ParametricObject, Parameter
from hierarchical.geometry import Geometry

class Plywood1_2(ParametricItem):
    """1/2 inch plywood sheet"""
    # ... implementation

# catalog/elements/hardware.py
from ..base import ParametricElement, ParametricComponent, ParametricObject, Parameter
from hierarchical.geometry import Geometry

class Bolt1_4_20(ParametricItem):
    """1/4-20 hex bolt"""
    # ... implementation

# catalog/components/wall_frames.py
from ..base import ParametricComponent, Parameter
from hierarchical.items import Component

class WallFrame2X4(ParametricComponent):
    """Standard 2x4 wall frame assembly"""
    # ... implementation

class WallFrame2X6(ParametricComponent):
    """Standard 2x6 wall frame assembly"""
    # ... implementation

# catalog/objects/walls.py
from ..base import ParametricObject, Parameter
from hierarchical.items import Wall

class ExteriorWall(ParametricObject):
    """Standard exterior wall assembly"""
    # ... implementation

class InteriorWall(ParametricObject):
    """Standard interior wall assembly"""  
    # ... implementation

# catalog/objects/doors.py
from ..base import ParametricObject, Parameter
from hierarchical.items import Door

class SwingDoor(ParametricObject):
    """Standard swing door"""
    # ... implementation

class SlidingDoor(ParametricObject):
    """Standard sliding door"""
    # ... implementation
```

#### 3.2 Direct Class Inheritance Usage
```python
# Import elements (directly inherit from Element)
from catalog.elements.lumber import Lumber2X4, Lumber2X6, Lumber2X8
from catalog.elements.steel import SteelW8X10, SteelW10X15
from catalog.elements.sheet_goods import Plywood1_2, Drywall1_2
from catalog.elements.hardware import Bolt1_4_20

# Import components (directly inherit from Component)
from catalog.components.wall_frames import WallFrame2X4, WallFrame2X6
from catalog.components.trusses import RoofTruss24OC, FloorTruss16OC
from catalog.components.panels import SIPPanel6_5, SIPPanel8_25

# Import objects (directly inherit from Object)
from catalog.objects.walls import ExteriorWall, InteriorWall, ShearWall
from catalog.objects.doors import SwingDoor, SlidingDoor, FrenchDoor
from catalog.objects.windows import CasementWindow, DoubleHungWindow

# Create elements - they ARE Elements (not .to_element() needed)
stud = Lumber2X4(length=8.0)  # This IS an Element
bolt = Bolt1_4_20(length=3.0)  # This IS an Element

# Create components - they ARE Components  
wall_frame = WallFrame2X4(
    height=8.0, 
    width=16.0, 
    stud_spacing=16.0
)  # This IS a Component

# Create objects - they ARE Objects
exterior_wall = ExteriorWall(
    height=8.0,
    width=16.0,
    wall_type="2x6_insulated",
    r_value=20
)  # This IS an Object

door = SwingDoor(
    width=3.0,
    height=6.67,
    swing_direction="inward_right",
    material="wood"
)  # This IS an Object

# They have all the Element/Component/Object methods directly
print(stud.get_height())  # Element method
print(wall_frame.materials)  # Component property
print(door.intersects_with(exterior_wall))  # Object method

# Get parameter info for any level
print(Lumber2X4.get_parameters())  # Element parameters
print(WallFrame2X4.get_parameters())  # Component parameters  
print(ExteriorWall.get_parameters())  # Object parameters
```

## Implementation Timeline

### Week 1: Core Infrastructure
- [ ] Create `ParametricComponent` and `MaterialCatalog` classes
- [ ] Add parameter validation system
- [ ] Complete geometry primitive methods

### Week 2: Lumber Catalog
- [ ] Implement lumber catalog with standard dimensions
- [ ] Add lumber geometry creation functions
- [ ] Create tests for lumber parametrics

### Week 3: Steel & Sheet Goods
- [ ] Implement steel catalog with AISC profiles
- [ ] Add sheet goods catalog
- [ ] Create I-beam and sheet geometry functions

### Week 4: Hardware & Integration  
- [ ] Implement hardware catalog
- [ ] Create unified `BuildingMaterialsCatalog` interface
- [ ] Add comprehensive examples and documentation

## Success Criteria

1. **Easy Item Creation**: `catalog.get_item('lumber', '2x4', length=10)` creates a 10-foot 2x4
2. **Parameter Validation**: Invalid parameters raise helpful error messages
3. **Standard Compliance**: All dimensions match industry standards (actual vs nominal)
4. **Extensible Design**: Easy to add new categories and items
5. **Unit Flexibility**: Works with both metric and imperial units
6. **Material Integration**: Items automatically get correct material properties

This plan builds on the existing robust geometry and item system while adding the catalog layer that makes common building materials easily accessible and parametric.