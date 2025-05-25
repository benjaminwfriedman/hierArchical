from hierarchical.items import Element, Component, Wall, Deck, Window, Door, Object
from hierarchical.relationships import AdjacentTo
from collections import defaultdict
import networkx as nx
# Abstractions are things like spaces and zones. They are not elements but are 
# defined by elements. They are used to group elements together in ways that are meaningful 
# and speak to how people experience the building. For example, a room is a space that is defined by
# walls, floors, and ceilings. (Maybe boundary abstractions)

# Idea 1: We create boundaries out of our walls and floors -> We make spaces out of boundaries?

# Idea 2: We create spaces out of our walls and floors -> We make boundaries out of spaces?


class Model:
    """
    A class representing a building model, which can
    - contain objects and their components and elements.
    - contain spaces and zones.
    - contain relationships between all types of items
    """
    def __init__(self):
        self.elements = {}
        self.components = {}
        self.objects = {}
        self.spaces = {}
        self.zones = {}
        self.relationships = defaultdict(list)
        self.boundaries = {}
        self.building_graph = nx.DiGraph()

    @classmethod
    def from_objects(cls, objects):
        """
        Create a model from a list of objects.
        """
        model = cls()
        for obj in objects:
            if isinstance(obj, Element):
                model.elements[obj.id] = obj
            elif isinstance(obj, Component):
                model.components[obj.id] = obj
            elif isinstance(obj, Object):
                model.objects[obj.id] = obj

        """
        Add object ids as nodes in the graph and enrich it with features about the nodes like:
        - type
        - length
        - width
        - height
        - volume

        """

        for obj_id, obj in model.objects.items():
            # Attempt to extract geometric features if they exist
            features = {
                "type": type(obj).__name__,  # class name, e.g., Wall, Door, etc.
                "length": getattr(obj, "length", None),
                "width": getattr(obj, "width", None),
                "height": getattr(obj, "height", None),
                "volume": getattr(obj, "volume", None),
                "centroid_x": obj.get_centroid().x,
                "centroid_y": obj.get_centroid().y,
                "centroid_z": obj.get_centroid().z,
                "object": obj  # preserve full object for further use
            }

            model.building_graph.add_node(obj_id, **features)


        return model

    def create_adjacency_relationships(self, tolerance=0.01):
        """
        Add adjacency relationships between objects in the model to the building_graph.
        """
        for obj in self.objects.values():
            adjacent_items = obj.find_adjacent_items(self.objects.values(), tolerance=tolerance)
            for adjacent_item in adjacent_items:
                rel = AdjacentTo(obj.id, adjacent_item.id)
                self.relationships[obj.id].append(rel)

                # Add edge to graph
                self.building_graph.add_edge(obj.id, adjacent_item.id, relationship=rel.type)

    
    def show_building_graph(self, by: str = "type", view: str = "2d"):
        """
        Display a visualization of the building graph using matplotlib.
        
        Parameters:
        - by (str): The node attribute to color nodes by. Default is 'type'.
        - view (str): Either '2d' for 2D plot or '3d' for 3D plot. Default is '2d'.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        # Get centroid coordinates for positioning
        centroid_x = nx.get_node_attributes(self.building_graph, 'centroid_x')
        centroid_y = nx.get_node_attributes(self.building_graph, 'centroid_y')
        centroid_z = nx.get_node_attributes(self.building_graph, 'centroid_z')
        
        all_nodes = list(self.building_graph.nodes)
        
        # Check if centroid data is available
        has_centroids = all(node in centroid_x and node in centroid_y and node in centroid_z 
                        for node in all_nodes)
        
        if not has_centroids:
            print("Warning: Not all nodes have centroid coordinates. Using spring layout instead.")
            pos = nx.spring_layout(self.building_graph)
        else:
            # Create position dictionary using centroid coordinates
            if view == "2d":
                # Use x and y coordinates for 2D plot
                pos = {node: (centroid_x[node], centroid_y[node]) for node in all_nodes}
            else:
                # For 3D, we'll handle positioning differently
                pos = {node: (centroid_x[node], centroid_y[node], centroid_z[node]) 
                    for node in all_nodes}

        # Extract the node attribute values for coloring
        node_attrs = nx.get_node_attributes(self.building_graph, by)
        node_colors = []

        for node in all_nodes:
            value = node_attrs.get(node, "unknown")
            node_colors.append(hash(str(value)) % 256)  # crude hashing to int for color

        # Normalize to color range (matplotlib colormap)
        cmap = plt.cm.get_cmap("viridis")
        if node_colors:
            normed_colors = [c / max(node_colors) for c in node_colors]
        else:
            normed_colors = [0.5] * len(all_nodes)

        edge_labels = nx.get_edge_attributes(self.building_graph, 'relationship')

        if view == "3d" and has_centroids:
            # 3D visualization
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract coordinates for 3D plotting
            x_coords = [centroid_x[node] for node in all_nodes]
            y_coords = [centroid_y[node] for node in all_nodes]
            z_coords = [centroid_z[node] for node in all_nodes]
            
            # Plot nodes
            scatter = ax.scatter(x_coords, y_coords, z_coords, 
                            c=normed_colors, cmap=cmap, s=200, alpha=0.8)
            
            # Add node labels
            for i, node in enumerate(all_nodes):
                ax.text(x_coords[i], y_coords[i], z_coords[i], str(node), 
                    fontsize=8, ha='center', va='center')
            
            # Plot edges
            for edge in self.building_graph.edges():
                node1, node2 = edge
                if node1 in pos and node2 in pos:
                    x1, y1, z1 = pos[node1]
                    x2, y2, z2 = pos[node2]
                    ax.plot([x1, x2], [y1, y2], [z1, z2], 'k-', alpha=0.5, linewidth=1)
            
            ax.set_xlabel('Centroid X')
            ax.set_ylabel('Centroid Y')
            ax.set_zlabel('Centroid Z')
            ax.set_title(f"3D Building Graph Colored by '{by}'")
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
            
        else:
            # 2D visualization
            plt.figure(figsize=(12, 8))
            
            # Draw the graph
            nx.draw(
                self.building_graph, pos,
                with_labels=True,
                node_color=normed_colors,
                node_size=1500,
                cmap=cmap,
                font_size=10,
                font_weight='bold',
                edge_color='gray',
                alpha=0.8
            )
            
            # Add edge labels if they exist
            if edge_labels:
                nx.draw_networkx_edge_labels(self.building_graph, pos, 
                                        edge_labels=edge_labels, font_size=8)
            
            plt.xlabel('Centroid X' if has_centroids else 'X')
            plt.ylabel('Centroid Y' if has_centroids else 'Y')
            plt.title(f"Building Graph Colored by '{by}'" + 
                    (" (Positioned by Centroids)" if has_centroids else " (Spring Layout)"))
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            plt.colorbar(sm, label=f'Color by {by}')
        
        plt.tight_layout()
        plt.show()

    def show(self,
         show_coords=False,
         color_by_class=False,
         color_by_attribute=None,
         flatten_to_elements=False,
         item_types=None,
         show_building_graph=False,
         graph_color_by='type'):
        """
        Plot the B-rep surfaces of items in the model, grouped and colored by attribute or class.
        Optionally show the building graph alongside the model objects.

        Args:
            show_coords (bool): Show coordinate labels on vertices
            color_by_class (bool): Color items by their class name
            color_by_attribute (str): Color items by a specific attribute
            flatten_to_elements (bool): Flatten all items to elements before plotting
            item_types (list): Which item types to include ('objects', 'components', 'elements'). 
                            If None, includes all available types.
            show_building_graph (bool): Show the building graph alongside model objects
            graph_color_by (str): Node attribute to color graph nodes by. Default is 'type'.
        """
        from collections import defaultdict
        import plotly.graph_objects as go
        import networkx as nx
        from .helpers import random_color  # Assuming this exists in helpers

        fig = go.Figure()

        # Collect items from model based on item_types
        items = []
        if item_types is None:
            item_types = ['objects', 'components', 'elements']
        
        if 'objects' in item_types:
            items.extend(self.objects.values())
        if 'components' in item_types:
            items.extend(self.components.values())
        if 'elements' in item_types:
            items.extend(self.elements.values())

        # Plot model objects (B-rep surfaces)
        if items:
            # Flatten to elements if needed
            if flatten_to_elements:
                def flatten(item):
                    if isinstance(item, Element):
                        return [item]
                    elif hasattr(item, "sub_items"):
                        flattened = []
                        for sub in item.sub_items:
                            flattened.extend(flatten(sub))
                        return flattened
                    return []

                all_elements = []
                for top in items:
                    all_elements.extend(flatten(top))
                items = all_elements

            # Determine grouping key
            def get_group_key(item):
                if color_by_class:
                    return type(item).__name__
                elif color_by_attribute:
                    return getattr(item, color_by_attribute, "unknown")
                return "default"

            # Group items by key
            grouped_items = defaultdict(list)
            for item in items:
                grouped_items[get_group_key(item)].append(item)

            # Assign colors
            keys = sorted(grouped_items.keys())
            key_colors = {key: random_color(seed=idx + 10) for idx, key in enumerate(keys)}

            # Plot each group
            for key, group in grouped_items.items():
                vertices = []
                faces = []

                for item in group:
                    # Check if item has geometry and brep_data
                    if not hasattr(item, 'geometry') or not hasattr(item.geometry, 'brep_data'):
                        continue
                        
                    brep = item.geometry.brep_data
                    surfaces = brep.get("surfaces", [])
                    
                    for surf in surfaces:
                        vs = surf.get("vertices", [])
                        offset = len(vertices)
                        vertices.extend(vs)

                        if len(vs) >= 3:
                            for i in range(1, len(vs) - 1):
                                faces.append((offset, offset + i, offset + i + 1))

                if not vertices or not faces:
                    continue

                x, y, z = zip(*vertices)
                i, j, k = zip(*faces)

                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    opacity=0.5,
                    color=key_colors[key],
                    name=f"Objects: {str(key)}",  # legend entry
                    hoverinfo='skip',
                    showlegend=True
                ))

                if show_coords:
                    labels = [f"({round(xi, 2)}, {round(yi, 2)}, {round(zi, 2)})" 
                            for xi, yi, zi in vertices]
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode="text",
                        text=labels,
                        showlegend=False,
                        hoverinfo="none",
                        textfont=dict(size=9, color="black")
                    ))

        # Add building graph visualization if requested
        if show_building_graph and hasattr(self, 'building_graph'):
            # Get centroid coordinates for positioning
            centroid_x = nx.get_node_attributes(self.building_graph, 'centroid_x')
            centroid_y = nx.get_node_attributes(self.building_graph, 'centroid_y')
            centroid_z = nx.get_node_attributes(self.building_graph, 'centroid_z')
            
            all_nodes = list(self.building_graph.nodes)
            
            # Check if centroid data is available
            has_centroids = all(node in centroid_x and node in centroid_y and node in centroid_z 
                            for node in all_nodes)
            
            if has_centroids:
                # Extract the node attribute values for coloring
                node_attrs = nx.get_node_attributes(self.building_graph, graph_color_by)
                
                # Create color mapping for graph nodes
                unique_values = list(set(node_attrs.values()))
                graph_node_colors = {val: random_color(seed=hash(str(val)) % 100) 
                                for val in unique_values}
                
                # Group nodes by attribute value for separate traces
                node_groups = defaultdict(list)
                for node in all_nodes:
                    attr_value = node_attrs.get(node, "unknown")
                    node_groups[attr_value].append(node)
                
                # Plot nodes grouped by attribute
                for attr_value, nodes in node_groups.items():
                    if not nodes:
                        continue
                        
                    x_coords = [centroid_x[node] for node in nodes]
                    y_coords = [centroid_y[node] for node in nodes]
                    z_coords = [centroid_z[node] for node in nodes]
                    node_labels = [str(node) for node in nodes]
                    
                    fig.add_trace(go.Scatter3d(
                        x=x_coords,
                        y=y_coords,
                        z=z_coords,
                        mode='markers+text',
                        marker=dict(
                            size=12,
                            color=graph_node_colors.get(attr_value, 'gray'),
                            opacity=0.8
                        ),
                        text=node_labels,
                        textposition="middle center",
                        textfont=dict(size=10, color="white"),
                        name=f"Graph {graph_color_by}: {attr_value}",
                        hovertemplate=f"Node: %{{text}}<br>{graph_color_by}: {attr_value}<br>x: %{{x}}<br>y: %{{y}}<br>z: %{{z}}<extra></extra>",
                        showlegend=True
                    ))
                
                # Plot edges
                edge_x, edge_y, edge_z = [], [], []
                edge_labels = nx.get_edge_attributes(self.building_graph, 'relationship')
                
                for edge in self.building_graph.edges():
                    node1, node2 = edge
                    if node1 in centroid_x and node2 in centroid_x:
                        # Add edge line
                        edge_x.extend([centroid_x[node1], centroid_x[node2], None])
                        edge_y.extend([centroid_y[node1], centroid_y[node2], None])
                        edge_z.extend([centroid_z[node1], centroid_z[node2], None])
                
                # Add edges as a single trace
                if edge_x:
                    fig.add_trace(go.Scatter3d(
                        x=edge_x,
                        y=edge_y,
                        z=edge_z,
                        mode='lines',
                        line=dict(color='black', width=2),
                        name="Graph Edges",
                        hoverinfo='skip',
                        showlegend=True
                    ))
                
                # Add edge labels if they exist
                if edge_labels:
                    edge_label_x, edge_label_y, edge_label_z, edge_label_text = [], [], [], []
                    for edge, label in edge_labels.items():
                        node1, node2 = edge
                        if node1 in centroid_x and node2 in centroid_x:
                            # Position label at midpoint of edge
                            mid_x = (centroid_x[node1] + centroid_x[node2]) / 2
                            mid_y = (centroid_y[node1] + centroid_y[node2]) / 2
                            mid_z = (centroid_z[node1] + centroid_z[node2]) / 2
                            
                            edge_label_x.append(mid_x)
                            edge_label_y.append(mid_y)
                            edge_label_z.append(mid_z)
                            edge_label_text.append(str(label))
                    
                    if edge_label_text:
                        fig.add_trace(go.Scatter3d(
                            x=edge_label_x,
                            y=edge_label_y,
                            z=edge_label_z,
                            mode='text',
                            text=edge_label_text,
                            textfont=dict(size=8, color="white"),
                            name="Edge Labels",
                            hoverinfo='skip',
                            showlegend=False
                        ))
            else:
                print("Warning: Building graph nodes don't have centroid coordinates. Skipping graph visualization.")
        
        elif show_building_graph:
            print("Warning: No building graph found in model or building_graph parameter is False.")

        # Set title based on what's being shown
        title_parts = []
        if items:
            title_parts.append("B-rep Model Objects")
        if show_building_graph and hasattr(self, 'building_graph'):
            title_parts.append("Building Graph")
        
        title = " and ".join(title_parts) if title_parts else "Model Visualization"

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        fig.show()


    def show_objects(self, **kwargs):
        """Convenience method to show only objects from the model."""
        return self.show(item_types=['objects'], **kwargs)


    def show_components(self, **kwargs):
        """Convenience method to show only components from the model."""
        return self.show(item_types=['components'], **kwargs)


    def show_elements(self, **kwargs):
        """Convenience method to show only elements from the model."""
        return self.show(item_types=['elements'], **kwargs)


    def show_all_as_elements(self, **kwargs):
        """Convenience method to show all items flattened to elements."""
        return self.show(flatten_to_elements=True, **kwargs)