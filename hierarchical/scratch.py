def heal_boundaries(self, tolerance=15.0, version='occ'):
        """Heal boundaries with comprehensive shape fixing and gap filling"""
        if version == 'occ':
            import tempfile
            from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_Sewing, BRepBuilderAPI_MakePolygon, 
                                                BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeShell)
            from OCC.Core.gp import gp_Pnt, gp_Pln, gp_Dir, gp_Vec
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE
            from OCC.Core.ShapeFix import (ShapeFix_Shape, ShapeFix_Wireframe, 
                                        ShapeFix_Shell, ShapeFix_FixSmallFace)
            from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
            from OCC.Core.ShapeUpgrade import ShapeUpgrade_RemoveInternalWires
            from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
            from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
            from OCC.Core import TopoDS
            from OCC.Core.BRepOffset import BRepOffset_Analyse
            from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape
            from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume
            from OCC.Core.GeomAbs import GeomAbs_Intersection
            from OCC.Core.BRepOffset import BRepOffset_Skin
            from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffset
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            from OCC.Core.BRepTools import breptools_OuterWire
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_WIRE
            from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Splitter
            from OCC.Core.BRep import BRep_Builder
            from OCC.Core.TopoDS import TopoDS_Compound
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.TopTools import TopTools_ListOfShape
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.GeomLib import GeomLib_IsPlanarSurface
            from OCC.Core.gp import gp_Pln, gp_Vec, gp_Pnt
            from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            from OCC.Core.TopTools import TopTools_ListOfShape
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.BRepTools import breptools

            from topologicpy.Topology import Topology
            import math



            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing

            from hierarchical.utils import plot_opencascade_shapes


            def get_plane_normal_and_point(face):
                """Extract plane normal and point from a face"""
                try:
                    surface = BRep_Tool.Surface(face)
                    
                    # Check if it's planar
                    if GeomLib_IsPlanarSurface(surface):
                        # Get plane parameters
                        plane = surface.GetObject().Plane()
                        normal = plane.Axis().Direction()
                        point = plane.Location()
                        
                        return (normal.X(), normal.Y(), normal.Z()), (point.X(), point.Y(), point.Z())
                except:
                    pass
                return None, None

            def are_coplanar(face1, face2, tolerance=1e-6):
                """Check if two faces are coplanar within tolerance"""
                normal1, point1 = get_plane_normal_and_point(face1)
                normal2, point2 = get_plane_normal_and_point(face2)
                
                if normal1 is None or normal2 is None:
                    return False
                
                # Check if normals are parallel (or anti-parallel)
                dot_product = abs(normal1[0]*normal2[0] + normal1[1]*normal2[1] + normal1[2]*normal2[2])
                if abs(dot_product - 1.0) > tolerance:
                    return False
                
                # Check if points lie on the same plane
                # Vector from point1 to point2
                vec_12 = (point2[0]-point1[0], point2[1]-point1[1], point2[2]-point1[2])
                
                # Dot product with normal should be ~0 if points are coplanar
                distance = abs(vec_12[0]*normal1[0] + vec_12[1]*normal1[1] + vec_12[2]*normal1[2])
                
                return distance < tolerance

            def group_coplanar_faces(offset_faces, tolerance=1e-6):
                """Group faces that are coplanar"""
                valid_faces = [f for f in offset_faces if f is not None]
                groups = []
                used = set()
                
                for i, face1 in enumerate(valid_faces):
                    if i in used:
                        continue
                        
                    # Start new group with this face
                    group = [face1]
                    used.add(i)
                    
                    # Find all other faces coplanar with this one
                    for j, face2 in enumerate(valid_faces):
                        if j in used or j <= i:
                            continue
                            
                        if are_coplanar(face1, face2, tolerance):
                            group.append(face2)
                            used.add(j)
                    
                    groups.append(group)
                
                print(f"Grouped {len(valid_faces)} faces into {len(groups)} coplanar groups")
                return groups

            def merge_coplanar_group(face_group):
                """Merge a group of coplanar faces using boolean union"""
                if len(face_group) == 1:
                    return face_group[0]
                
                result = face_group[0]
                
                for face in face_group[1:]:
                    try:
                        # Use Fuse to union the faces
                        fuse_op = BRepAlgoAPI_Fuse(result, face)
                        fuse_op.Build()
                        
                        if fuse_op.IsDone():
                            result = fuse_op.Shape()
                        else:
                            print("Warning: Face merge failed, keeping separate")
                            # If merge fails, we'll just keep them separate
                    except Exception as e:
                        print(f"Warning: Exception during face merge: {e}")
                
                return result

            def preprocess_coplanar_faces(offset_faces, tolerance=0.1):
                """
                Merge coplanar faces before splitting to reduce artifacts
                """
                # Group coplanar faces
                coplanar_groups = group_coplanar_faces(offset_faces, tolerance)
                
                # Merge each group
                merged_faces = []
                for group in coplanar_groups:
                    merged_face = merge_coplanar_group(group)
                    
                    # Extract faces from the merged result (could be compound)
                    if merged_face:
                        explorer = TopExp_Explorer(merged_face, TopAbs_FACE)
                        while explorer.More():
                            merged_faces.append(explorer.Current())
                            explorer.Next()
                
                print(f"Merged down to {len(merged_faces)} faces")
                return merged_faces

            def offset_opencascade_face(face, offset):
                # Extract the outer wire of the face
                outer_wire = breptools_OuterWire(face)

                # Create 2D offset with sharp join type in constructor
                offset_maker = BRepOffsetAPI_MakeOffset(outer_wire, GeomAbs_Intersection)

                # No need for AddWire since we passed the wire in constructor
                offset_maker.Perform(offset)

                if offset_maker.IsDone():
                    offset_wire = offset_maker.Shape()
                    
                    # Convert wire back to face
                    face_maker = BRepBuilderAPI_MakeFace(offset_wire)
                    if face_maker.IsDone():
                        offset_face = face_maker.Face()
                        return offset_face
                    
            def split_faces_by_faces(offset_faces, tolerance=0.01):
                """
                Use BRepAlgoAPI_Splitter to split all offset faces by each other
                This creates a cell complex by finding all intersections
                """
                
                # Create a compound of all offset faces using BRep_Builder
                builder = BRep_Builder()
                compound = TopoDS_Compound()
                builder.MakeCompound(compound)
                
                # Add all offset faces to the compound
                for face in offset_faces:
                    if face is not None:  # Skip any None faces from failed offsets
                        builder.Add(compound, face)
                
                # Create the splitter
                splitter = BRepAlgoAPI_Splitter()
                
                # Set tolerance
                splitter.SetFuzzyValue(tolerance)

                # Create TopTools_ListOfShape for arguments and tools
                arguments_list = TopTools_ListOfShape()
                tools_list = TopTools_ListOfShape()
                
                # Add each face individually as both argument and tool
                for face in offset_faces:
                    if face is not None:
                        arguments_list.Append(face)
                        tools_list.Append(face)
            
                # Set arguments and tools
                splitter.SetArguments(arguments_list)
                splitter.SetTools(tools_list)
                
                # Perform the splitting operation
                splitter.Build()
                
                # Add the compound as both arguments and tools
                # Arguments = shapes to be split
                # Tools = shapes to split by
                
                
                if splitter.IsDone():
                    result_shape = splitter.Shape()
                    
                    # Extract all resulting faces
                    split_faces = []
                    explorer = TopExp_Explorer(result_shape, TopAbs_FACE)
                    
                    while explorer.More():
                        split_face = explorer.Current()
                        split_faces.append(split_face)
                        explorer.Next()
                    
                    print(f"Splitter created {len(split_faces)} faces from {len(offset_faces)} input faces")
                    return split_faces
                else:
                    print("Splitter operation failed")
                    return None
            
            all_boundaries = list(self.boundaries.values())
            
            # Step 1: Create initial faces with better polygon construction
            initial_faces = []
            for boundary in all_boundaries:
                face = self._create_robust_face(boundary, tolerance)
                if face:
                    initial_faces.append(face)

            print(f"Created {len(initial_faces)} initial faces")

            offset_opencascade_faces = [offset_opencascade_face(face, 0.2) for face in initial_faces]

            merged_opencascade_faces = preprocess_coplanar_faces(offset_opencascade_faces)

            # Create the volume maker
            volume_maker = BOPAlgo_MakerVolume()

            # Method 1: Add faces individually
            for face in merged_opencascade_faces:
                volume_maker.AddArgument(face)

            # Method 2: Or add them as a list (alternative approach)
            # face_list = TopTools_ListOfShape()
            # for face in merged_opencascade_faces:
            #     face_list.Append(face)
            # volume_maker.SetArguments(face_list)

            # Perform the volume creation
            volume_maker.Perform()

            # Check if operation was successful
            if volume_maker.HasErrors():
                print("Error creating volumes:")
                # You can inspect errors if needed
                print(volume_maker.GetReport())
            else:
                # Get the resulting shape(s)
                result = volume_maker.Shape()
                
                # The result might be a compound containing multiple solids
                # You can iterate through the result to get individual volumes
                from OCC.Core.TopExp import TopExp_Explorer
                from OCC.Core.TopAbs import TopAbs_SOLID
                
                explorer = TopExp_Explorer(result, TopAbs_SOLID)
                face_explorer = TopExp_Explorer(result, TopAbs_FACE)
                volumes = []
                faces = []
                while explorer.More():
                    solid = explorer.Current()
                    volumes.append(solid)
                    v_faces = []
                    while face_explorer.More():
                        face = face_explorer.Current()
                        v_faces.append(face)
                        face_explorer.Next()

                    ## TODO: Merge Coplaner Faces in the volume

                    ## TODO: Add merged faces to the faces list
                    
                    explorer.Next()
  
            # TODO Apply Healed Face Information back to original Boundaries