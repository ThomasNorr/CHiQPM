import copy
from pathlib import Path
import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image

from conformalPrediction.HierarchicalExplanation.graph_utils.helpers import get_colors_per_feature, draw_labels, \
    get_smaller_v_space_pos, get_remapped_name


def visualize_explanation_tree(
        hierarchy_data: list[dict],
        label_idx: int,
        class_names: list[str],
        feature_activations: torch.Tensor,
        pred_idx: int,
        predicted_classes: set[int], # List of predicted class indices
        output_folder: Path,
        filename: str,
        feature_to_color_mapping ={},
        global_plot: bool = True,
        summarize_loc: bool = True
):
    """
    Builds and saves a hierarchical explanation graph for a model's prediction on a sample.

    This function creates a graph where nodes are either features and edges represent
    the hierarchical decision path.

    Args:
        hierarchy_data: A list of dictionaries representing the hierarchy levels.
        At every index in the list, the dictionary maps tuples of
        (feature_index, activation_value, previous_features, based_on_class) to a list of class indices.
        label_idx: The ground truth class index for the sample.
        class_names: A list mapping class indices to human-readable names.
        feature_activations: The feature activation tensor for this sample.
        pred_idx: The predicted class index for the sample.
        predicted_classes: A set of class indices that are considered as predictions for this sample.
        output_folder: The directory where the visualization files will be saved.
        filename: The base name for the saved visualization files (without extension).
        feature_to_color_mapping: A dictionary mapping feature indices to specific colors for visualization.

        global_plot: If True, generates a global plot considering all classes.
                     If False, generates a local plot focused on the predicted path.
        summarize_loc: If True, groups classes with identical explanation paths into a single node, summarizing their name.
    """


    alpha = 0.6

    # --- 1. Pre-process hierarchy data to find all unique features and classes ---
    unique_feats = set()
    unique_feats_independent = set()
    all_classes = set()
    n_depth = len(hierarchy_data)
    for i in range(n_depth):
        for (feat, act, prev, based_on_class), classes in hierarchy_data[i].items():
            if act > 0:
                unique_feats.add((i, feat, prev))
                unique_feats_independent.add(feat)
                all_classes.add(based_on_class)

    # --- 2. Build the graph structure (adjacency matrix) ---
    total_number_of_nodes = len(unique_feats) + len(all_classes)
    total_number_of_nodes += 1

    adjacency_matrix = np.zeros((total_number_of_nodes, total_number_of_nodes))

    # Mappers from graph elements to matrix indices
    feature_mapper = {(i, feat, prev): idx for idx, (i, feat, prev) in enumerate(unique_feats)}
    class_mapper = {cls: idx + len(feature_mapper) for idx, cls in enumerate(sorted(all_classes))}

    # Define the root node of the graph
    root_node_keys = [key for key in unique_feats if key[2] is None]
    root_idx = total_number_of_nodes - 1
    for node_key in root_node_keys:
        adjacency_matrix[root_idx, feature_mapper[node_key]] = 1


    # Populate adjacency matrix for connections
    connected_classes = set()
    for i in range(1, n_depth):
        for (feat, act, prev, based_on_class), classes in hierarchy_data[i].items():
            prev_prev = prev[:-1] if len(prev) > 1 else None
            key_for_prev = (i - 1, prev[-1], prev_prev)

            if key_for_prev in feature_mapper:
                prev_node_idx = feature_mapper[key_for_prev]
                if act > 0:
                    current_node_key = (i, feat, prev)
                    adjacency_matrix[prev_node_idx, feature_mapper[current_node_key]] = 1
                    if i == n_depth - 1:
                        for cls in classes:
                            adjacency_matrix[feature_mapper[current_node_key], class_mapper[cls]] = 1
                            connected_classes.add(cls)
                else:
                    adjacency_matrix[prev_node_idx, class_mapper[based_on_class]] = 1
                    connected_classes.add(based_on_class)

    assert len(connected_classes) == len(all_classes), "Mismatch between connected and discovered classes"

    # --- 3. Summarize class nodes if enabled ---
    class_mapper_new = class_mapper
    full_new_mapper = class_mapper
    remapped_labels, summarized_labels =  {}, {}
    if summarize_loc:
        summarized_indices, removed_classes = {},[]
        igno = set()
        all_class_keys = sorted(class_mapper.keys())
        indices_to_keep = np.arange(adjacency_matrix.shape[0])

        for i in range(len(all_class_keys)):
            if i in igno: continue
            parent_connections = adjacency_matrix[:, class_mapper[all_class_keys[i]]]
            is_identical = np.all(
                adjacency_matrix[:, [class_mapper[c] for c in all_class_keys]] == parent_connections[:, None], axis=0)
            summarize_indices = np.where(is_identical)[0]

            igno.update(summarize_indices)
            if len(summarize_indices) > 1:
                class_group = [all_class_keys[j] for j in summarize_indices]
                names = [class_names[c] for c in class_group]
                summarized_name = get_remapped_name(names)

                main_class = class_group[0]
                remapped_labels[main_class] = summarized_name
                summarized_labels[main_class] = names

                removed_classes.extend(class_group[1:])
                for entry in class_group[1:]:
                    summarized_indices[entry] = main_class

        if removed_classes:
            indices_to_remove = [class_mapper[c] for c in removed_classes]
            indices_to_keep = np.delete(indices_to_keep, indices_to_remove)
            adjacency_matrix = adjacency_matrix[indices_to_keep, :][:, indices_to_keep]

            root_idx -= len(removed_classes)

            remaining_classes = [c for c in sorted(class_mapper.keys()) if c not in removed_classes]
            class_mapper_new = {cls: idx + len(feature_mapper) for idx, cls in enumerate(remaining_classes)}
            full_new_mapper = copy.deepcopy(class_mapper_new)
            for old_cls, new_cls in summarized_indices.items():
                full_new_mapper[old_cls] = class_mapper_new[new_cls]
            assert set(full_new_mapper.keys()) == all_classes
    # --- 4. Create NetworkX graph and define node attributes ---
    graph = nx.DiGraph(adjacency_matrix)
    color_map, nodes_sizes, nodelabels = [], [], {}
    num_nodes = graph.number_of_nodes()
    color_map = [None] * num_nodes
    nodes_sizes = [None] * num_nodes
    feature_scaled_for_size = (feature_activations / feature_activations.max()) * 500
    if global_plot:
        feature_scaled_for_size = feature_scaled_for_size / 5
    colors_per_feature = get_colors_per_feature(feature_activations, list(unique_feats_independent), feature_to_color_mapping)

    for (i, feat, prev), idx in feature_mapper.items():
            nodelabels[idx] = ""
            graph.nodes[idx]["type"] = "feature"
            color_map[idx] = colors_per_feature[feat]
            nodes_sizes[idx] = feature_scaled_for_size[feat].item()

    for cls in sorted(class_mapper.keys()):
        if cls not in class_mapper_new: continue

        idx = class_mapper_new[cls]
        class_label = remapped_labels.get(cls, class_names[cls])
        graph.nodes[idx]["forSummary"] = summarized_labels.get(cls, [class_names[cls]])
        nodelabels[idx] = class_label
        if cls == pred_idx:
            nodelabels[idx] = f"Bold{class_label}"

        graph.nodes[idx]["type"] = "class"
        color_map[idx] = [0, 0, 0]# not shown anyway, since node size is 0 for classes.
        nodes_sizes[idx] = 0
    color_map[root_idx] = "black"
    nodes_sizes[root_idx] = 0
    nodelabels[root_idx] = ""
    width_for_this_graph = 2.5 - 1 * int(global_plot)
    # --- 5. Find and highlight the prediction path ---
    for sub_pred_idx in predicted_classes:
            path = nx.shortest_path(graph, source=root_idx, target=full_new_mapper[sub_pred_idx])


            subset_edges = []
            for i in range(1, len(path)):
                u, v = path[i - 1], path[i]
                graph.remove_edge(u, v)
                color =  "lime"
                graph.add_edge(u, v, color=color, width=width_for_this_graph + 0.5)
                subset_edges.append((u, v))

    # --- 6. Draw and Save the Main Plot ---
    edges = graph.edges()
    colors = [graph[u][v].get('color', "black") for u, v in edges]
    pos = get_smaller_v_space_pos(graph, root_idx)

    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, node_color=color_map, node_size=nodes_sizes, alpha=alpha, edge_color=colors,
            width=width_for_this_graph)


    draw_labels(pos, nodelabels, font_size=8 - 2 * int(global_plot))

    if global_plot:
        filename = "Global" + filename
    else:
        filename =  "Local"+ filename
    output_folder.mkdir(exist_ok=True)
    filename_base = output_folder / filename
    plt.savefig(f"{filename_base}.png", dpi=300)
    print(f"Saved main visualization to {filename_base}.png")
    plt.close("all")


def visualize_explanation_tree_to_pil(
        hierarchy_data: list[dict],
        label_idx: int,
        class_names: list[str],
        feature_activations: torch.Tensor,
        pred_idx: int,
        predicted_classes: set[int],
        feature_to_color_mapping={},
        global_plot: bool = True,
        summarize_loc: bool = True
) -> Image.Image:
    """
    Builds and returns a hierarchical explanation graph as PIL Image (in-memory, no file saving).
    
    This is a memory-efficient version of visualize_explanation_tree that returns the image
    directly instead of saving to disk. Use this for interactive applications like Gradio demos.

    Args:
        hierarchy_data: A list of dictionaries representing the hierarchy levels.
        label_idx: The ground truth class index for the sample.
        class_names: A list mapping class indices to human-readable names.
        feature_activations: The feature activation tensor for this sample.
        pred_idx: The predicted class index for the sample.
        predicted_classes: A set of class indices that are considered as predictions for this sample.
        feature_to_color_mapping: A dictionary mapping feature indices to specific colors for visualization.
        global_plot: If True, generates a global plot considering all classes.
        summarize_loc: If True, groups classes with identical explanation paths into a single node.
        
    Returns:
        PIL.Image: The rendered tree visualization
    """
    alpha = 0.6

    # --- 1. Pre-process hierarchy data to find all unique features and classes ---
    unique_feats = set()
    unique_feats_independent = set()
    all_classes = set()
    n_depth = len(hierarchy_data)
    for i in range(n_depth):
        for (feat, act, prev, based_on_class), classes in hierarchy_data[i].items():
            if act > 0:
                unique_feats.add((i, feat, prev))
                unique_feats_independent.add(feat)
                all_classes.add(based_on_class)

    # --- 2. Build the graph structure (adjacency matrix) ---
    total_number_of_nodes = len(unique_feats) + len(all_classes) + 1
    adjacency_matrix = np.zeros((total_number_of_nodes, total_number_of_nodes))

    # Mappers from graph elements to matrix indices
    feature_mapper = {(i, feat, prev): idx for idx, (i, feat, prev) in enumerate(unique_feats)}
    class_mapper = {cls: idx + len(feature_mapper) for idx, cls in enumerate(sorted(all_classes))}

    # Define the root node of the graph
    root_node_keys = [key for key in unique_feats if key[2] is None]
    root_idx = total_number_of_nodes - 1
    for node_key in root_node_keys:
        adjacency_matrix[root_idx, feature_mapper[node_key]] = 1

    # Populate adjacency matrix for connections
    connected_classes = set()
    for i in range(1, n_depth):
        for (feat, act, prev, based_on_class), classes in hierarchy_data[i].items():
            prev_prev = prev[:-1] if len(prev) > 1 else None
            key_for_prev = (i - 1, prev[-1], prev_prev)

            if key_for_prev in feature_mapper:
                prev_node_idx = feature_mapper[key_for_prev]
                if act > 0:
                    current_node_key = (i, feat, prev)
                    adjacency_matrix[prev_node_idx, feature_mapper[current_node_key]] = 1
                    if i == n_depth - 1:
                        for cls in classes:
                            adjacency_matrix[feature_mapper[current_node_key], class_mapper[cls]] = 1
                            connected_classes.add(cls)
                else:
                    adjacency_matrix[prev_node_idx, class_mapper[based_on_class]] = 1
                    connected_classes.add(based_on_class)

    assert len(connected_classes) == len(all_classes), "Mismatch between connected and discovered classes"

    # --- 3. Summarize class nodes if enabled ---
    class_mapper_new = class_mapper
    full_new_mapper = class_mapper
    remapped_labels, summarized_labels = {}, {}
    if summarize_loc:
        summarized_indices, removed_classes = {}, []
        igno = set()
        all_class_keys = sorted(class_mapper.keys())
        indices_to_keep = np.arange(adjacency_matrix.shape[0])

        for i in range(len(all_class_keys)):
            if i in igno: continue
            parent_connections = adjacency_matrix[:, class_mapper[all_class_keys[i]]]
            is_identical = np.all(
                adjacency_matrix[:, [class_mapper[c] for c in all_class_keys]] == parent_connections[:, None], axis=0)
            summarize_indices = np.where(is_identical)[0]

            igno.update(summarize_indices)
            if len(summarize_indices) > 1:
                class_group = [all_class_keys[j] for j in summarize_indices]
                names = [class_names[c] for c in class_group]
                summarized_name = get_remapped_name(names)

                main_class = class_group[0]
                remapped_labels[main_class] = summarized_name
                summarized_labels[main_class] = names

                removed_classes.extend(class_group[1:])
                for entry in class_group[1:]:
                    summarized_indices[entry] = main_class

        if removed_classes:
            indices_to_remove = [class_mapper[c] for c in removed_classes]
            indices_to_keep = np.delete(indices_to_keep, indices_to_remove)
            adjacency_matrix = adjacency_matrix[indices_to_keep, :][:, indices_to_keep]

            root_idx -= len(removed_classes)

            remaining_classes = [c for c in sorted(class_mapper.keys()) if c not in removed_classes]
            class_mapper_new = {cls: idx + len(feature_mapper) for idx, cls in enumerate(remaining_classes)}
            full_new_mapper = copy.deepcopy(class_mapper_new)
            for old_cls, new_cls in summarized_indices.items():
                full_new_mapper[old_cls] = class_mapper_new[new_cls]
            assert set(full_new_mapper.keys()) == all_classes

    # --- 4. Create NetworkX graph and define node attributes ---
    graph = nx.DiGraph(adjacency_matrix)
    color_map, nodes_sizes, nodelabels = [], [], {}
    num_nodes = graph.number_of_nodes()
    color_map = [None] * num_nodes
    nodes_sizes = [None] * num_nodes
    feature_scaled_for_size = (feature_activations / feature_activations.max()) * 500
    if global_plot:
        feature_scaled_for_size = feature_scaled_for_size / 5
    colors_per_feature = get_colors_per_feature(feature_activations, list(unique_feats_independent), feature_to_color_mapping)

    for (i, feat, prev), idx in feature_mapper.items():
        nodelabels[idx] = ""
        graph.nodes[idx]["type"] = "feature"
        color_map[idx] = colors_per_feature[feat]
        nodes_sizes[idx] = feature_scaled_for_size[feat].item()

    for cls in sorted(class_mapper.keys()):
        if cls not in class_mapper_new: continue

        idx = class_mapper_new[cls]
        class_label = remapped_labels.get(cls, class_names[cls])
        graph.nodes[idx]["forSummary"] = summarized_labels.get(cls, [class_names[cls]])
        nodelabels[idx] = class_label
        if cls == pred_idx:
            nodelabels[idx] = f"Bold{class_label}"

        graph.nodes[idx]["type"] = "class"
        color_map[idx] = [0, 0, 0]
        nodes_sizes[idx] = 0
        
    color_map[root_idx] = "black"
    nodes_sizes[root_idx] = 0
    nodelabels[root_idx] = ""
    width_for_this_graph = 2.5 - 1 * int(global_plot)
    
    # --- 5. Find and highlight the prediction path ---
    for sub_pred_idx in predicted_classes:
        path = nx.shortest_path(graph, source=root_idx, target=full_new_mapper[sub_pred_idx])

        subset_edges = []
        for i in range(1, len(path)):
            u, v = path[i - 1], path[i]
            graph.remove_edge(u, v)
            color = "lime"
            graph.add_edge(u, v, color=color, width=width_for_this_graph + 0.5)
            subset_edges.append((u, v))

    # --- 6. Draw and return as PIL Image (no file saving) ---
    edges = graph.edges()
    colors = [graph[u][v].get('color', "black") for u, v in edges]
    pos = get_smaller_v_space_pos(graph, root_idx)

    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, node_color=color_map, node_size=nodes_sizes, alpha=alpha, edge_color=colors,
            width=width_for_this_graph)

    draw_labels(pos, nodelabels, font_size=8 - 2 * int(global_plot))

    # Save to BytesIO buffer instead of file
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    tree_image = Image.open(buf)
    plt.close("all")
    
    return tree_image

