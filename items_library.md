# 🏗️ Hierarchical Items Library

## 📋 Overview

The Hierarchical Items Library provides a comprehensive framework for modeling architectural and construction elements in a **semantically-rich, relationship-aware hierarchy**. The library implements a three-tier system (Elements → Components → Objects) with specialized classes for common building elements and full geometric reasoning capabilities.

---

## 🏛️ Architecture

### Class Hierarchy

```
BaseItem (Abstract Base)
├── Element (Atomic building blocks)
├── Component (Built from Elements)  
└── Object (Built from Components)
    ├── Wall (Specialized Object)
    ├── Door (Specialized Object)
    ├── Window (Specialized Object)
    └── Deck (Specialized Object)
```

### 🧱 Core Components

#### 🔷 BaseItem (Foundation Class)
> **Location**: `hierarchical/items.py:45-350`

**Purpose**: Provides shared functionality for all items in the hierarchy

**Key Properties**:
- `name`, `type`, `geometry`, `id` (UUID)
- `unit_system`, `materials`, `attributes`
- `ontologies`, `relationships`, `color`

**Core Methods**:
- Geometric transformations: `move()`, `rotate_z()`, directional movement
- Spatial analysis: `intersects_with()`, `is_adjacent_to()`
- Material management and unit conversion
- Copy functionality with transformation support

#### 🔸 Element (Atomic Level)
> **Location**: `hierarchical/items.py:353-405`

**Purpose**: Smallest building blocks (studs, beams, panels)

**Features**:
- Single material assignment with automatic volume calculation
- Post-initialization material property setup
- Direct geometry-to-material mapping

#### 🔹 Component (Assembly Level)
> **Location**: `hierarchical/items.py:408-465`

**Purpose**: Assemblies of multiple Elements

**Creation**: `Component.from_elements(elements, name, type)`

**Features**:
- Automatic material aggregation from constituent elements
- Percentage-based material distribution
- Combined geometry from sub-elements

#### 🔶 Object (System Level)
> **Location**: `hierarchical/items.py:468-530`

**Purpose**: Complex assemblies built from Components

**Creation**: `Object.from_components(components, name, type)`

**Features**:
- Multi-level material aggregation
- IFC file import capability (`from_ifc()`)
- Hierarchical geometry composition

### 🏠 Specialized Objects

| Object | Location | Key Features |
|--------|----------|-------------|
| **🧱 Wall** | `hierarchical/items.py:533-690` | Center plane analysis using PCA, edge detection (top/bottom/left/right), boundary processing |
| **🚪 Door** | `hierarchical/items.py:693-750` | Swing direction support, embeddable properties, IFC import capability |
| **🪟 Window** | `hierarchical/items.py:753-785` | Basic window functionality, embeddable properties, lightweight implementation |
| **🏗️ Deck** | `hierarchical/items.py:788-820` | Horizontal surface specialization, center plane geometry analysis |

---

## ⚡ Key Features

### 📐 Geometric Operations
- **🔄 Transformations**: Move, rotate, directional movement (right/left/forward/back/up/down)
- **📊 Property Access**: Height, centroid, vertices, faces, bounding box
- **🔍 Spatial Analysis**: Intersection detection with overlap percentages

### 🔗 Spatial Relationships
The library supports comprehensive relationship types:

| Category | Relationships |
|----------|---------------|
| **🌐 Spatial** | AdjacentTo, Above, Below, InFrontOf, Behind, LeftOf, RightOf |
| **🏗️ Hierarchical** | Contains, IsPartOf, HasComponent, IsComponentOf |
| **⚙️ Functional** | EmbeddedIn, Embeds, PassesThrough, HasPassingThrough |
| **🔧 Structural** | Supports, SupportedBy, ConnectsTo |
| **🏢 System** | PartOfSystem, HasSystemComponent, FlowsTo, FlowsFrom |

### 🧪 Material Management
- ✅ Automatic material aggregation through hierarchy levels
- 📊 Volume-based calculations with percentage distribution
- 🔄 Full unit conversion support (metric/imperial)
- ✔️ Material conservation validation in tests

### 📏 Unit System Support
- **Supported Units**: `mm`, `cm`, `m`, `km`, `in`, `ft`, `yd`, `mi`
- **Methods**: `convert_units()`, `convert_to_metric()`, `convert_to_imperial()`
- **Queries**: `get_dimension_in_units()` for specific measurements

### 🎨 Geometry Integration
Integrates with **triple-representation geometry system**:

| Representation | Purpose | Features |
|----------------|---------|----------|
| **🔺 Mesh** | Performance | Lightweight representation for fast operations |
| **🔧 OpenCascade** | Precision | CAD-quality operations and precision |
| **🧠 TopologicPy** | Analysis | Topological analysis and reasoning |

**Additional Features**:
- 🚀 **Lazy Loading**: Representations generated on-demand with fallback chain
- 📁 **File Support**: OBJ, STL, primitive shapes, prisms, surfaces

---

## 💻 Usage Examples

### Creating Elements
```python
from hierarchical.items import Element
from hierarchical.geometry import Geometry

# Create geometry (box, cylinder, from file, etc.)
beam_geometry = Geometry.box(2.0, 0.2, 0.4)

# Create element
steel_beam = Element(
    name="steel_beam_001",
    geometry=beam_geometry,
    type="structural_beam",
    material="steel"
)
```

### Assembling Components
```python
from hierarchical.items import Component

# Create wall frame component from multiple elements
wall_frame = Component.from_elements(
    elements=(stud1, stud2, top_plate, bottom_plate),
    name="wall_frame_assembly",
    type="structural_frame",
    attributes={"spacing": 16, "height": 2.4}
)
```

### Building Objects
```python
from hierarchical.items import Wall

# Create wall from frame and sheathing components
exterior_wall = Wall.from_components(
    components=(wall_frame, exterior_sheathing, insulation),
    name="exterior_wall_north",
    type="exterior_wall",
    boundary_id="wall_n_001"
)
```

### Establishing Relationships
```python
# Embed door in wall
door.add_embedded_in_relationship(exterior_wall)

# Create adjacency between walls
wall_north.add_adjacent_to_relationship(wall_east)

# Check spatial relationships
if wall1.intersects_with(wall2):
    overlap_pct = wall1.intersects_with(wall2)
    print(f"Walls overlap by {overlap_pct:.1%}")
```

### Transformations and Positioning
```python
# Move and rotate items
wall.move(x=5.0, y=0, z=0)
wall.rotate_z(90)  # Rotate 90 degrees around Z-axis
wall.up(0.5)       # Move up by 0.5 units

# Directional movement
door.forward(0.1)  # Move door forward (out from wall)
window.right(2.0)  # Shift window position along wall
```

---

## 🚀 Implementation Plan

### 🎯 Phase 1: Core Enhancements (Immediate)

#### 📚 Documentation Completion
- [ ] Add comprehensive docstrings to all public methods
- [ ] Create usage examples for each specialized object type
- [ ] Document IFC integration patterns and limitations

#### ✅ Validation and Error Handling
- [ ] Implement geometry validation in constructors
- [ ] Add material assignment validation
- [ ] Create robust error messages for common failures

#### ⚡ Performance Optimization
- [ ] Profile material aggregation for large hierarchies
- [ ] Optimize intersection calculations using spatial indexing
- [ ] Implement geometry caching strategies

### 🔧 Phase 2: Feature Extensions (Short-term)

#### 🎨 Advanced Geometry Operations
- [ ] Complete OpenCascade conversion methods
- [ ] Add geometry repair and healing capabilities  
- [ ] Implement advanced intersection algorithms

#### 💾 Serialization and Persistence
- [ ] Add JSON/YAML export for item hierarchies
- [ ] Implement database storage integration
- [ ] Create checkpoint/restore functionality for large models

#### 🔗 Enhanced Relationships
- [ ] Add relationship queries and graph traversal
- [ ] Implement relationship validation rules
- [ ] Create relationship-based reporting tools

### 🎯 Phase 3: Advanced Features (Medium-term)

#### 🏗️ IFC Integration Enhancement
- [ ] Complete IFC import for all object types
- [ ] Add IFC export functionality
- [ ] Implement IFC property mapping and validation

#### 🤖 AI/ML Integration
- [ ] Add feature extraction methods for ML models
- [ ] Implement semantic similarity calculations
- [ ] Create embeddings for item classification

#### 📊 Visualization and Analysis
- [ ] Integrate with 3D visualization libraries
- [ ] Add interactive manipulation tools
- [ ] Create analysis and reporting dashboards

### 🌐 Phase 4: Ecosystem Integration (Long-term)

#### 🖥️ CAD Software Integration
- [ ] Add direct CAD software plugins (Rhino, Revit, etc.)
- [ ] Implement real-time synchronization
- [ ] Create collaborative editing capabilities

#### 🏢 Building Information Modeling (BIM)
- [ ] Full BIM workflow integration
- [ ] Construction sequencing support
- [ ] Cost estimation and material optimization

#### 🔄 Interoperability Standards
- [ ] STEP file format support
- [ ] Industry Foundation Classes (IFC) 4.3 compliance
- [ ] gbXML integration for energy analysis

---

## 🧪 Testing and Quality Assurance

The library includes comprehensive testing:

| Test Type | Location | Purpose |
|-----------|----------|---------|
| **🔬 Unit Tests** | `tests/unit/` | All classes and methods |
| **🔍 Property-Based Tests** | `tests/property/` | Material conservation, geometric properties |
| **🔄 Integration Tests** | `tests/integration/` | End-to-end workflows |
| **⚡ Performance Tests** | - | Large-scale model handling |
| **📊 Coverage** | - | >95% code coverage maintained |

---

## 🔧 Dependencies and Integration

### 🏗️ Core Dependencies
- **🎨 Geometry**: Custom triple-representation system
- **📏 Units**: Comprehensive unit conversion system  
- **🔗 Relationships**: Formal relationship type definitions
- **🧠 Abstractions**: Higher-level modeling concepts
- **🛠️ Helpers**: Utility functions and ID generation

### 🌐 External Integrations
- **🔧 OpenCascade**: CAD-quality geometric operations
- **🧠 TopologicPy**: Topological analysis and reasoning
- **📊 NumPy**: Numerical computations and transformations
- **🏢 IFC Libraries**: Building information exchange

---

## 🤝 Contributing Guidelines

1. **📝 Code Style**: Follow existing patterns and PEP 8
2. **🧪 Testing**: Add tests for all new functionality
3. **📚 Documentation**: Update docstrings and examples
4. **⚡ Performance**: Profile changes with large models
5. **🔄 Compatibility**: Maintain backward compatibility for public APIs

---

## 📄 License and Acknowledgments

This library builds upon established patterns in computational geometry, building information modeling, and semantic web technologies. See LICENSE file for usage terms and attribution requirements.