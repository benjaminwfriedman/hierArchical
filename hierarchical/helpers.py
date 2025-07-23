import random
import uuid 

def generate_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def random_color(seed=None):
    if seed is not None:
        random.seed(seed)
    r = random.randint(0, 250)
    g = random.randint(0, 250)
    b = random.randint(0, 250)
    return f"rgb({r},{g},{b})"

def normalize_ifc_enum(value: str) -> str:
    """
    Normalize IFC-style enum strings (e.g., '.SINGLE_SWING_LEFT.') to lowercase underscore format.
    """
    if not value:
        return ""
    return value.strip('.').lower()



### Functions for Geometry
def validate_healing(boundaries, healed_faces, tolerance=1e-6):
    """Validate that boundary healing creates enclosed shapes"""
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop_VolumeProperties
    
    print("=== BOUNDARY HEALING VALIDATION ===")
    
    # Create shape from healed faces
    healed_shape = create_shape_from_faces(healed_faces, tolerance)
    
    # 1. Basic shape validation
    print(f"\n1. BASIC VALIDATION:")
    print(f"   - Number of boundaries: {len(boundaries)}")
    print(f"   - Number of healed faces: {len(healed_faces)}")
    
    analyzer = BRepCheck_Analyzer(healed_shape)
    print(f"   - Shape is valid: {analyzer.IsValid()}")
    
    # 2. Face connectivity analysis
    print(f"\n2. FACE CONNECTIVITY:")
    face_count = count_topology_elements(healed_shape, TopAbs_FACE)
    edge_count = count_topology_elements(healed_shape, TopAbs_EDGE)
    vertex_count = count_topology_elements(healed_shape, TopAbs_VERTEX)
    
    print(f"   - Faces: {face_count}")
    print(f"   - Edges: {edge_count}")
    print(f"   - Vertices: {vertex_count}")
    
    # 3. Edge sharing analysis
    edge_sharing = analyze_edge_sharing(healed_shape)
    print(f"   - Shared edges: {edge_sharing['shared']}")
    print(f"   - Boundary edges: {edge_sharing['boundary']}")
    print(f"   - Is closed: {edge_sharing['boundary'] == 0}")
    
    # 4. Volume calculation
    print(f"\n3. VOLUME ANALYSIS:")
    try:
        props = GProp_GProps()
        brepgprop_VolumeProperties(healed_shape, props)
        volume = props.Mass()
        print(f"   - Volume: {volume:.6f}")
        print(f"   - Has volume: {volume > 1e-12}")
    except Exception as e:
        print(f"   - Volume calculation failed: {e}")
        volume = 0
    
    # 5. Gap analysis
    print(f"\n4. GAP ANALYSIS:")
    gaps = analyze_gaps(healed_faces, tolerance)
    print(f"   - Average gap: {gaps['average']:.6f}")
    print(f"   - Max gap: {gaps['max']:.6f}")
    print(f"   - Gaps within tolerance: {gaps['max'] <= tolerance}")
    
    # 6. Watertight test
    print(f"\n5. WATERTIGHT TEST:")
    is_watertight = is_watertight_shape(healed_shape)
    print(f"   - Is watertight: {is_watertight}")
    
    # 7. Visual validation
    print(f"\n6. VISUAL VALIDATION:")
    print("   - Plotting healed boundaries...")
    plot_healing_comparison(boundaries, healed_faces)
    
    return {
        'is_valid': analyzer.IsValid(),
        'is_closed': edge_sharing['boundary'] == 0,
        'has_volume': volume > 1e-12,
        'is_watertight': is_watertight,
        'within_tolerance': gaps['max'] <= tolerance,
        'face_count': face_count,
        'gap_analysis': gaps
    }

def create_shape_from_faces(healed_faces, tolerance):
    """Create a shape from the healed faces for validation"""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
    
    sewing = BRepBuilderAPI_Sewing(tolerance)
    
    for face in healed_faces:
        sewing.Add(face)
    
    sewing.Perform()
    return sewing.SewedShape()

def count_topology_elements(shape, topology_type):
    """Count topology elements of given type"""
    from OCC.Core.TopExp import TopExp_Explorer
    
    count = 0
    explorer = TopExp_Explorer(shape, topology_type)
    while explorer.More():
        count += 1
        explorer.Next()
    return count

def analyze_edge_sharing(shape):
    """Analyze how many edges are shared between faces"""
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCC.Core.TopExp import topexp_MapShapesAndAncestors
    
    # Map edges to faces
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp_MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
    
    shared_edges = 0
    boundary_edges = 0
    
    # Try different methods to get size
    try:
        extent = edge_face_map.Extent()
    except AttributeError:
        try:
            extent = edge_face_map.Size()
        except AttributeError:
            # Fallback - manually count edges
            extent = 0
            explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            while explorer.More():
                extent += 1
                explorer.Next()
    
    for i in range(1, extent + 1):
        try:
            face_list = edge_face_map.FindFromIndex(i)
            face_count = face_list.Size()
            
            if face_count == 2:
                shared_edges += 1
            elif face_count == 1:
                boundary_edges += 1
        except:
            # Skip problematic edges
            continue
    
    return {
        'shared': shared_edges,
        'boundary': boundary_edges,
        'total': extent
    }

def analyze_gaps(faces, tolerance):
    """Analyze gaps between faces"""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_VERTEX
    import numpy as np
    
    all_vertices = []
    
    for face in faces:
        vertex_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
        while vertex_explorer.More():
            vertex = vertex_explorer.Current()
            point = BRep_Tool.Pnt(vertex)
            all_vertices.append([point.X(), point.Y(), point.Z()])
            vertex_explorer.Next()
    
    if len(all_vertices) < 2:
        return {'average': 0, 'max': 0}
    
    vertices = np.array(all_vertices)
    
    # Calculate minimum distances between vertices
    gaps = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            dist = np.linalg.norm(vertices[i] - vertices[j])
            if dist > tolerance:  # Only consider actual gaps
                gaps.append(dist)
    
    if not gaps:
        return {'average': 0, 'max': 0}
    
    return {
        'average': np.mean(gaps),
        'max': np.max(gaps)
    }

def is_watertight_shape(shape):
    """Check if shape is watertight (no boundary edges)"""
    edge_analysis = analyze_edge_sharing(shape)
    return edge_analysis['boundary'] == 0

def plot_healing_comparison(original_faces, healed_faces):
    """Plot before and after healing comparison"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_VERTEX
    
    def extract_vertices_from_occ_face(face):
        """Extract vertices from OCC face"""
        vertices = []
        vertex_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
        while vertex_explorer.More():
            vertex = vertex_explorer.Current()
            point = BRep_Tool.Pnt(vertex)
            vertices.append((point.X(), point.Y(), point.Z()))
            vertex_explorer.Next()
        return vertices
    
    # Create side-by-side comparison
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Before Healing", "After Healing"),
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]]
    )
    
    # Plot original faces
    for i, face in enumerate(original_faces):
        vertices = extract_vertices_from_occ_face(face)
        if vertices:
            x, y, z = zip(*vertices)
            # Close the loop for plotting
            x = list(x) + [x[0]]
            y = list(y) + [y[0]]
            z = list(z) + [z[0]]
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                name=f'Original {i}',
                line=dict(width=3)
            ), row=1, col=1)
    
    # Plot healed faces
    for i, face in enumerate(healed_faces):
        vertices = extract_vertices_from_occ_face(face)
        if vertices:
            x, y, z = zip(*vertices)
            # Close the loop for plotting
            x = list(x) + [x[0]]
            y = list(y) + [y[0]]
            z = list(z) + [z[0]]
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                name=f'Healed {i}',
                line=dict(width=3)
            ), row=1, col=2)
    
    fig.update_layout(title="Boundary Healing Comparison")
    fig.show()
    
    return fig

def test_healing_validation(boundaries, healed_faces):
    """Test the healing validation"""
    print("Testing boundary healing validation...")

    # Convert boundaries to OCC faces
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCC.Core.gp import gp_Pnt

    occ_faces = []
    for boundary in boundaries.values() if hasattr(boundaries, 'values') else boundaries:
        vertices = boundary.geometry.get_vertices()
        if vertices and len(vertices) >= 3:
            # Create polygon from vertices
            polygon = BRepBuilderAPI_MakePolygon()
            
            for vertex in vertices:
                if hasattr(vertex, 'x'):
                    point = gp_Pnt(vertex.x, vertex.y, vertex.z)
                else:
                    point = gp_Pnt(float(vertex[0]), float(vertex[1]), float(vertex[2]))
                polygon.Add(point)
            
            polygon.Close()
            if polygon.IsDone():
                face = BRepBuilderAPI_MakeFace(polygon.Wire())
                if face.IsDone():
                    occ_faces.append(face.Face())

    # Run validation
    results = validate_healing(occ_faces, healed_faces, tolerance=1e-6)
    
    # Print summary
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Overall success: {all(results.values())}")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    return results