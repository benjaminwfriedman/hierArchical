# Triple Geometry Representation Implementation Plan

## Overview
This document outlines the implementation plan for adding three geometry representations to the hierarchical library:
1. **Mesh** (existing) - lightweight triangular mesh for basic operations
2. **OpenCascade** (new) - pythonocc-core shapes for precise CAD operations  
3. **TopologicPy** (enhanced) - topology objects for advanced geometric analysis

## Trade-offs Analysis

### Storing Raw Objects Directly

**Pros:**
- Direct access to full OpenCascade/TopologicPy functionality
- No conversion overhead during operations
- Preserves all precision and metadata
- Leverage existing OCC patterns in `abstractions.py`

**Cons:**
- **Memory usage**: ~3x storage (mesh + OCC + topologic objects)
- **Serialization complexity**: OCC/topologic objects don't pickle easily
- **Library coupling**: Tighter dependency on specific versions
- **Thread safety**: May need careful handling for concurrent access

**Decision**: Store raw objects with lazy serialization fallback

### Lazy vs Eager Generation

**Lazy (On-demand):**
**Pros:**
- Lower initial memory footprint
- Faster object creation (~2-5x faster)
- Only generate what's actually used

**Cons:**
- First access penalty (100-500ms per representation)
- Thread safety complexity
- Late failure detection
- Cache invalidation logic needed

**Eager (All at once):**
**Pros:**
- Fail-fast error detection
- Predictable performance
- Thread-safe by design
- Simpler logic

**Cons:**
- Higher memory usage (3x representation overhead)
- Slower object creation (~2-10x slower)
- Wasted computation for unused representations

**Decision**: Hybrid approach - eager for lightweight mesh, lazy for OCC/topologic

## Implementation Plan

### Phase 1: Core Infrastructure

#### 1. Update Dependencies
- Add `pythonocc-core=7.8.1` to requirements.txt
- Verify compatibility with existing topologicpy version

#### 2. New Geometry Class Structure
```python
@dataclass
class Geometry:
    # Legacy fields (maintain compatibility)
    sub_geometries: List['Geometry'] = field(default_factory=list)
    mesh_data: Dict[str, Any] = field(default_factory=dict)  # DEPRECATED - use .mesh property
    
    # New representation storage (private fields)
    _mesh_data: Optional[Dict[str, Any]] = None          # Eager - lightweight
    _opencascade_shape: Optional[TopoDS_Shape] = None    # Lazy - heavy
    _topologic_topology: Optional[topologicpy.Topology] = None  # Lazy - heavy
    
    # Generation flags for lazy loading
    _mesh_generated: bool = False
    _occ_generated: bool = False 
    _topologic_generated: bool = False
    
    # Metadata
    geometry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
```

#### 3. Property-Based Access Pattern
```python
@property 
def mesh(self) -> Dict[str, Any]:
    """Always available - fallback representation"""
    if not self._mesh_generated:
        self._generate_mesh()
    return self._mesh_data

@property
def opencascade(self) -> TopoDS_Shape:
    """Generated from topologic -> mesh if needed"""
    if not self._occ_generated:
        self._generate_opencascade()
    return self._opencascade_shape
    
@property  
def topologic(self) -> topologicpy.Topology:
    """Preferred representation when available"""
    if not self._topologic_generated:
        self._generate_topologic()
    return self._topologic_topology
```

### Phase 2: Conversion Logic

#### 4. Fallback Conversion Chain
**Primary Path**: TopologicPy → OpenCascade → Mesh
**Fallback Path**: Mesh → OpenCascade → TopologicPy (limited fidelity)

```python
def _generate_mesh(self):
    """Generate mesh from best available source"""
    if self._topologic_topology:
        self._mesh_data = self._topologic_to_mesh(self._topologic_topology)
    elif self._opencascade_shape:
        self._mesh_data = self._opencascade_to_mesh(self._opencascade_shape)
    elif self.mesh_data:  # Legacy compatibility
        self._mesh_data = self.mesh_data.copy()
    else:
        raise ValueError("No geometry data available")
    self._mesh_generated = True

def _generate_opencascade(self):
    """Generate OpenCascade shape from best available source"""
    if self._topologic_topology:
        self._opencascade_shape = self._topologic_to_opencascade(self._topologic_topology)
    elif self._mesh_data or self.mesh_data:
        mesh = self._mesh_data or self.mesh_data
        self._opencascade_shape = self._mesh_to_opencascade(mesh)
    else:
        raise ValueError("No geometry data available")
    self._occ_generated = True

def _generate_topologic(self):
    """Generate TopologicPy topology from best available source"""
    if self._opencascade_shape:
        self._topologic_topology = self._opencascade_to_topologic(self._opencascade_shape)
    elif self._mesh_data or self.mesh_data:
        mesh = self._mesh_data or self.mesh_data
        self._topologic_topology = self._mesh_to_topologic(mesh)
    else:
        raise ValueError("No geometry data available")
    self._topologic_generated = True
```

### Phase 3: Migration Strategy

#### 5. Replace brep_data Field
- **Current**: `brep_data: Dict[str, Any]` → **New**: `_opencascade_shape: TopoDS_Shape`
- Update all references in `abstractions.py` and other files
- Add migration helper for existing data

#### 6. Update Factory Methods
Update all geometry creation methods:
- `from_obj()` - Load and generate all representations
- `from_topology()` - Start with topologic, generate others
- `from_stl()` - Start with mesh, generate others  
- `from_primitive()` - Start with parametric → topologic → others
- `from_prism()` - Use existing topologic creation, enhance others
- `from_surface()` - Enhance mesh → topologic conversion

#### 7. Backward Compatibility Layer
```python
# Maintain existing mesh_data access with deprecation warning
@property
def mesh_data(self) -> Dict[str, Any]:
    warnings.warn("mesh_data is deprecated, use .mesh property", DeprecationWarning)
    return self.mesh

@mesh_data.setter  
def mesh_data(self, value: Dict[str, Any]):
    warnings.warn("mesh_data is deprecated, use .mesh property", DeprecationWarning)
    self._mesh_data = value
    self._mesh_generated = True
    # Invalidate other representations
    self._occ_generated = False
    self._topologic_generated = False
```

### Phase 4: Enhanced Operations

#### 8. Conversion Implementations
Leverage existing OCC expertise in `abstractions.py`:

```python
def _topologic_to_opencascade(self, topology: topologicpy.Topology) -> TopoDS_Shape:
    """Use existing topologic → OCC patterns from abstractions.py"""
    
def _opencascade_to_mesh(self, shape: TopoDS_Shape) -> Dict[str, Any]:
    """Use OCC meshing capabilities for high-quality triangulation"""
    
def _mesh_to_opencascade(self, mesh: Dict[str, Any]) -> TopoDS_Shape:
    """Build OCC shape from mesh (limited precision)"""
    
def _mesh_to_topologic(self, mesh: Dict[str, Any]) -> topologicpy.Topology:
    """Build topologic topology from mesh"""
```

#### 9. Enhanced Geometry Operations
Update existing methods to use best available representation:
- Volume calculation: OCC > topologic > mesh
- Intersection: topologic > OCC > mesh  
- Boolean operations: topologic > OCC > mesh
- Surface analysis: OCC > topologic > mesh

### Phase 5: Testing & Optimization

#### 10. Comprehensive Testing
- Unit tests for each representation type
- Conversion accuracy tests
- Performance benchmarks
- Memory usage analysis
- Backward compatibility tests

#### 11. Serialization Strategy
```python
def __getstate__(self):
    """Custom serialization - fallback to mesh for non-pickleable objects"""
    state = self.__dict__.copy()
    # Store OCC/topologic as mesh fallback if serialization fails
    if self._opencascade_shape and not _can_pickle(self._opencascade_shape):
        state['_opencascade_fallback_mesh'] = self._opencascade_to_mesh(self._opencascade_shape)
        state['_opencascade_shape'] = None
    return state
```

#### 12. Performance Optimizations
- **Lazy generation** for OCC/topologic (first access ~200-500ms)
- **Caching** with invalidation on geometry changes  
- **Memory pooling** for frequently created/destroyed objects
- **Weak references** for large object management

## Key Benefits

1. **Leverages existing OCC expertise** in `abstractions.py`
2. **Graceful degradation** - always have mesh fallback
3. **Memory efficient** - lazy load heavy representations
4. **Future-proof** - easy to add new representation types
5. **Backward compatible** - existing code continues working
6. **Performance optimized** - hybrid eager/lazy approach

## Implementation Timeline

- **Phase 1** (Core Infrastructure): 2-3 days
- **Phase 2** (Conversion Logic): 3-4 days  
- **Phase 3** (Migration): 2-3 days
- **Phase 4** (Enhanced Operations): 3-4 days
- **Phase 5** (Testing & Optimization): 2-3 days

**Total Estimated Time**: 12-17 days

## Migration Notes

- Existing `mesh_data` access will continue working with deprecation warnings
- New code should use `.mesh`, `.opencascade`, `.topologic` properties
- Gradual migration of `abstractions.py` OCC usage to new pattern
- Performance monitoring during rollout to identify bottlenecks

## Future Enhancements

- Additional representation types (CGAL, VTK, etc.)
- Streaming/chunked loading for large geometries
- Distributed geometry processing
- GPU-accelerated conversions