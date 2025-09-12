def find_shared_edges_simple(faces):
    edge_to_faces = defaultdict(list)

    # get all edges
    all_edges = set()
    for face in faces:
        edges = Topology.Edges(face)
        for edge in edges:
            all_edges.add(edge)
    
    
    for edge in all_edges:
        edge_to_faces[edge] = []
        # Iterate through each face and check edges
        for face_idx, face in enumerate(faces):
            # Check if edge is a super topology of each face
            super_topologies = Topology.SuperTopologies(edge, face, 'Face')
            if super_topologies:
                edge_to_faces[edge].extend(super_topologies)
    
    # Analyze the results
    naked_edges = []
    shared_edges = []
    overlapping_edges = []
    
    for edge_key, face_indices in edge_to_faces.items():
        if len(face_indices) == 1:
            naked_edges.append((edge_key, face_indices[0]))
        elif len(face_indices) == 2:
            shared_edges.append((edge_key, face_indices))
        else:
            overlapping_edges.append((edge_key, face_indices))
    
    print(f"Naked edges: {len(naked_edges)}")
    print(f"Shared edges: {len(shared_edges)}")
    print(f"Overlapping edges: {len(overlapping_edges)}")
    
    # Show some examples
    if naked_edges:
        print(f"Example naked edge on face {naked_edges[0][1]}")
    if shared_edges:
        print(f"Example shared edge between faces {shared_edges[0][1]}")
    if overlapping_edges:
        print(f"Example overlapping edge on faces {overlapping_edges[0][1]}")
    
    return edge_to_faces
    
