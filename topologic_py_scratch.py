from topologicpy.Topology import Topology
from topologicpy.Face import Face
from topologicpy.Vertex import Vertex

# create two faces that should intersect perpendicularly
face_1_vertices = [Vertex.ByCoordinates(0,0,0), Vertex.ByCoordinates(1,0,0), Vertex.ByCoordinates(1,1,0), Vertex.ByCoordinates(0,1,0)]
face_2_vertices = [Vertex.ByCoordinates(0.5,0.5,-1), Vertex.ByCoordinates(0.5,0.5,1), Vertex.ByCoordinates(1.5,0.5,1), Vertex.ByCoordinates(1.5,0.5,-1)]
face_1 = Face.ByVertices(face_1_vertices)
face_2 = Face.ByVertices(face_2_vertices)
faces = [face_1, face_2]



from hierarchical.utils import plot_topologic_objects

plot_topologic_objects(faces)

resp = Topology.Intersect(face_1, face_2)

plot_topologic_objects([resp, face_1, face_2])