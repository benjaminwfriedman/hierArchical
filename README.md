# 🏗️ HierArchical

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**HierArchical** is a comprehensive Python library for **parametric building design and geometric analysis**. It provides a hierarchical modeling system where Elements aggregate into Components, which aggregate into Objects, enabling automatic material calculations, spatial analysis, and Building Information Modeling (BIM) workflows.

## 🎯 Key Features

### 🏗️ **Hierarchical Architecture**
- **Elements** → **Components** → **Objects** composition system  
- Automatic material aggregation and bill of materials generation
- Relationship-aware spatial modeling with graph database integration

### 📐 **Triple Geometry Representation**
- **Mesh**: Fast rendering and visualization using Trimesh
- **B-rep**: Precise CAD operations using OpenCascade  
- **Topology**: Advanced spatial analysis using TopologicPy
- Lazy loading with automatic fallback chains for performance

### 🧱 **Comprehensive Building Catalog**
- **150+ parametric building elements** (lumber, steel, flooring, etc.)
- **6 component assembly types** (walls, floors, ceilings)
- **Complete building objects** with automatic composition
- Industry-standard dimensions and material properties

### 🤖 **AI-Powered Analysis**
- Natural language queries of building models using OpenAI
- RAG (Retrieval-Augmented Generation) for building information
- Automated spatial relationship detection

### 📊 **Advanced Spatial Analysis**  
- Boundary healing and watertight model generation
- Space inference from wall boundaries
- Intersection detection with multiple algorithms
- Graph-based relationship modeling using KuzuDB

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hierArchical.git
cd hierArchical

# Create conda environment
conda env create -f environment.yaml
conda activate hierarchical

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from hierarchical.catalog.elements.lumber import Lumber2X4
from hierarchical.catalog.objects.walls import ExteriorWall
from hierarchical.catalog.objects.decks import WoodFramedDeck

# Create parametric building elements
stud = Lumber2X4(length=8.0, species='Douglas Fir')
print(f"Volume: {stud.geometry.compute_volume():.3f} cubic feet")

# Create complete building assemblies
exterior_wall = ExteriorWall(
    name="South Wall",
    length=20.0,    # 20-foot long wall
    height=8.0,     # 8-foot ceiling
    thickness=0.5   # 6-inch wall thickness
)

# Create floors with automatic framing
ground_floor = WoodFramedDeck(
    name="Ground Floor", 
    deck_width=20.0,
    deck_length=30.0,
    include_floor_assembly=True,
    floor_assembly_type='hardwood'
)

# Automatic bill of materials
print("\n=== BILL OF MATERIALS ===")
for material, quantity in exterior_wall.materials.items():
    print(f"{material}: {quantity:.2f}")
```

### Complete Building Example

```python
from hierarchical.abstractions import Model
from hierarchical.catalog.objects.walls import ExteriorWall, InteriorWall
from hierarchical.catalog.objects.decks import WoodFramedDeck
import math

# Building dimensions
building_width, building_depth = 20.0, 18.0
wall_height, wall_thickness = 8.0, 0.5

# Create floor
floor = WoodFramedDeck(
    name="Ground Floor",
    deck_width=building_depth,
    deck_length=building_width,
    include_floor_assembly=True,
    floor_assembly_type='tile'
)

# Create exterior walls
south_wall = ExteriorWall("South Wall", building_width, wall_height, wall_thickness)
west_wall = ExteriorWall("West Wall", building_depth, wall_height, wall_thickness)
north_wall = ExteriorWall("North Wall", building_width, wall_height, wall_thickness)  
east_wall = ExteriorWall("East Wall", building_depth, wall_height, wall_thickness)

# Position walls (following standardized coordinate system)
south_wall.move(dz=floor.attributes.height)
west_wall.rotate_z(math.pi / 2)
west_wall.move(dy=south_wall.attributes.width, dx=west_wall.attributes.width, dz=floor.attributes.height)
north_wall.rotate_z(math.pi)
north_wall.move(dy=building_depth, dx=building_width, dz=floor.attributes.height)
east_wall.rotate_z(math.pi / 2)  
east_wall.move(dx=building_width, dy=wall_thickness, dz=floor.attributes.height)

# Create interior wall
interior_wall = InteriorWall("Divider Wall", building_width-1, wall_height, wall_thickness)
interior_wall.move(dy=building_depth/2, dx=0.5, dz=floor.attributes.height)

# Create ceiling
ceiling = WoodFramedDeck(
    name="Ceiling",
    deck_width=building_depth, 
    deck_length=building_width,
    include_ceiling_assembly=True,
    ceiling_assembly_type='drywall'
)
ceiling.move(dz=wall_height + floor.attributes.height)

# Build complete model with AI analysis
objects = [floor, south_wall, west_wall, north_wall, east_wall, interior_wall, ceiling]
model = Model.from_objects("Two Room House", objects)

# AI-powered queries
print("Q: How many rooms are in this building?")
print(model.ask("How many rooms are in this building?"))

print("\nQ: What materials do I need to build this?")  
print(model.ask("What materials do I need to build this?"))

# Visualize the model
model.show()  # 3D visualization
model.show_spaces()  # Space analysis
```

## 📚 Available Building Components

### 🪵 **Elements** (150+ classes)
- **Lumber**: All standard sizes (2x4, 2x6, 2x8, 2x10, 2x12, 4x4, 6x6, etc.)
- **Steel**: AISC wide-flange beams (W8x10, W10x15, etc.) 
- **Flooring**: Hardwood, LVP, tile, carpet with thickness variants
- **Subflooring**: Plywood and OSB in standard thicknesses
- **Drywall**: All thicknesses with fire ratings
- **Ceiling Systems**: Drop ceiling tiles, acoustic panels
- **Plaster**: Traditional lime, gypsum, and clay systems

### 🔧 **Components** (Assembly types)
- **Wall Frames**: 2x4 and 2x6 framing with configurable stud spacing
- **Floor Assemblies**: Complete subfloor + finish combinations
- **Ceiling Assemblies**: Suspended, drywall, and plaster systems  
- **Deck Frames**: Engineered deck framing with proper joist sizing
- **Drywall Assemblies**: Complete wall finishing systems
- **Sheathing**: Structural sheathing with fastener patterns

### 🏠 **Objects** (Complete assemblies)  
- **Walls**: Exterior and interior walls with insulation
- **Decks**: Complete floor/ceiling assemblies with framing
- **Doors**: Swing, sliding, and pocket doors with frames

## 🎨 Design Philosophy

### **Standardized Coordinate System**
All components follow the **X=longest, Y=middle, Z=shortest** dimension convention:
- **X-axis**: Length (longest dimension)
- **Y-axis**: Width/Height (middle dimension) 
- **Z-axis**: Thickness/Up (shortest dimension)

This standardization enables consistent assembly, rotation, and positioning.

### **Create → Rotate → Move Pattern**
```python
# 1. Create element with standard orientation
wall_frame = WallFrame2X4(height=8.0, length=16.0)

# 2. Rotate if needed (around standard origin)  
wall_frame.rotate_z(math.pi / 2)  # Rotate 90 degrees

# 3. Move into final position
wall_frame.move(dx=10.0, dy=5.0, dz=1.0)
```

## 🧮 Advanced Features

### **Automatic Material Calculations**
```python
wall = ExteriorWall(length=16.0, height=8.0)
materials = wall.materials

# Automatically calculates:
# - Lumber quantities (studs, plates, headers)
# - Sheathing square footage  
# - Insulation volume
# - Drywall square footage
# - Fastener counts
```

### **Spatial Relationship Detection**
```python  
model = Model.from_objects("Building", [wall1, wall2, floor, ceiling])

# Automatically detects:
# - Adjacent elements (walls touching)
# - Embedded relationships (windows in walls)  
# - Intersecting components
# - Spatial boundaries and enclosed spaces
```

### **Graph Database Integration**
```python
# Query spatial relationships with Cypher-like syntax
model.building_graph.query("""
    MATCH (wall:Object)-[:ADJACENT_TO]->(other_wall:Object)
    WHERE wall.type = 'exterior_wall'  
    RETURN wall.name, other_wall.name
""")
```

### **AI-Powered Analysis**
```python
# Natural language building queries
model.ask("How much lumber do I need for framing?")
model.ask("What rooms have the most square footage?")
model.ask("Are there any structural issues with this design?")
```

## 📊 Performance Features

- **Lazy Loading**: Expensive operations only computed when needed
- **Multi-Library Integration**: Seamless fallbacks between geometry engines
- **Multiprocessing**: Parallel geometric analysis for large models  
- **Memory Optimization**: `@dataclass(slots=True)` for efficient storage
- **Caching**: Intelligent caching of geometric computations

## 🛠️ Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hierarchical

# Run performance benchmarks  
pytest --benchmark-only
```

### Code Style
```bash  
# Format code
black hierarchical/

# Check types
mypy hierarchical/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📖 Documentation

- **API Reference**: [docs/api/](docs/api/)
- **Examples**: [examples/](examples/)  
- **Tutorials**: [docs/tutorials/](docs/tutorials/)
- **Architecture Guide**: [docs/architecture.md](docs/architecture.md)

## 🎯 Use Cases

- **Architectural Design**: Rapid building design and iteration
- **Construction Planning**: Automated material takeoffs and cost estimation
- **BIM Workflows**: Programmatic building information modeling
- **Code Compliance**: Automated building code checking
- **Parametric Design**: Algorithm-driven architectural exploration  
- **Educational Tools**: Teaching computational design concepts

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCascade**: CAD kernel for precise geometric operations
- **TopologicPy**: Advanced topological analysis capabilities  
- **Trimesh**: Fast mesh processing and visualization
- **KuzuDB**: Graph database for spatial relationships
- **OpenAI**: AI-powered building analysis

---

**Built with ❤️ for the AEC industry and computational design community.**

*HierArchical transforms how we think about building design - from manual CAD drafting to intelligent, code-driven architecture.*