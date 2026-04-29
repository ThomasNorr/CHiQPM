"""
Graph caching for efficient tree re-rendering with different prediction sets.
"""

import networkx as nx
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt

from conformalPrediction.HierarchicalExplanation.graph_utils.helpers import (
    get_colors_per_feature, 
    draw_labels, 
    get_smaller_v_space_pos
)


class CachedGraphStructure:
    """
    Stores pre-computed graph structure for fast re-rendering with different prediction sets.
    """

    DPI = 150  # DPI for saved images
    
    def __init__(self, graph, pos, color_map, nodes_sizes, nodelabels, 
                 feature_mapper, full_new_mapper, root_idx, 
                 width_for_this_graph, alpha, global_plot):
        """
        Store all graph structure components that don't depend on prediction set.
        
        Args:
            graph: NetworkX DiGraph with base edges (no colored edges yet)
            pos: Node positions dictionary
            color_map: List of node colors
            nodes_sizes: List of node sizes
            nodelabels: Dict of node labels
            feature_mapper: Dict mapping (level, feat, prev) to node indices
            full_new_mapper: Dict mapping class indices to node indices
            root_idx: Index of root node
            width_for_this_graph: Base edge width
            alpha: Transparency value
            global_plot: Whether this is a global or local plot
        """
        self.graph = graph
        self.pos = pos
        self.color_map = color_map
        self.nodes_sizes = nodes_sizes
        self.nodelabels = nodelabels
        self.feature_mapper = feature_mapper
        self.full_new_mapper = full_new_mapper
        self.root_idx = root_idx
        self.width_for_this_graph = width_for_this_graph
        self.alpha = alpha
        self.global_plot = global_plot
        
        # Store base edges (before coloring)
        self.base_edges = list(graph.edges(data=True))
    
    def render_with_prediction_set(self, predicted_classes):
        """
        Fast render by only updating edge colors based on prediction set.
        
        Args:
            predicted_classes: Set of predicted class indices
            
        Returns:
            PIL.Image of the rendered tree
        """
        # Create a copy of the graph to avoid modifying the cached one
        graph_copy = self.graph.copy()
        
        # Reset all edges to base state
        graph_copy.clear_edges()
        for u, v, data in self.base_edges:
            graph_copy.add_edge(u, v, color="black", width=self.width_for_this_graph)
        
        # Color edges in prediction paths
        for sub_pred_idx in predicted_classes:
            if sub_pred_idx not in self.full_new_mapper:
                continue
                
            try:
                path = nx.shortest_path(graph_copy, source=self.root_idx, 
                                       target=self.full_new_mapper[sub_pred_idx])
                
                for i in range(1, len(path)):
                    u, v = path[i - 1], path[i]
                    graph_copy.remove_edge(u, v)
                    graph_copy.add_edge(u, v, color="lime", 
                                       width=self.width_for_this_graph + 0.5)
            except nx.NetworkXNoPath:
                # Path doesn't exist in this graph structure
                continue
        
        # Render to PIL Image
        edges = graph_copy.edges()
        colors = [graph_copy[u][v].get('color', "black") for u, v in edges]
        
        plt.figure(figsize=(10, 8))
        nx.draw(graph_copy, self.pos, node_color=self.color_map, 
                node_size=self.nodes_sizes, alpha=self.alpha, 
                edge_color=colors, width=self.width_for_this_graph)
        
        draw_labels(self.pos, self.nodelabels, 
                   font_size=8 - 2 * int(self.global_plot))
        
        # Save to BytesIO buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=self.DPI)
        buf.seek(0)
        tree_image = Image.open(buf)
        plt.close("all")
        
        return tree_image