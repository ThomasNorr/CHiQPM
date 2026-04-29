from pathlib import Path

import torch

from conformalPrediction.HierarchicalExplanation.graphCode import visualize_explanation_tree, visualize_explanation_tree_to_pil
from conformalPrediction.cleanScoreFunction import HieraDiffNonConformityScore
from conformalPrediction.eval_cp import get_logits_and_labels
from conformalPrediction.utils import get_score, calibrate_predictor, get_predictions
from dataset_classes.cub200 import load_cub_class_mapping, CUB200Class
from evaluation.load_model import get_args_for_loading_model, load_model
from get_data import get_data
from visualization.compare_classes import viz_model
from visualization.localViz import generate_local_viz
from visualization.utils import get_feats_logits_labels, get_active_mean
import numpy as np
import networkx as nx
from conformalPrediction.HierarchicalExplanation.graph_utils.helpers import get_colors_per_feature, get_smaller_v_space_pos
from conformalPrediction.HierarchicalExplanation.graph_cache import CachedGraphStructure

class HierarchicalExplainer(HieraDiffNonConformityScore):
    def __init__(self, weight):
        super().__init__(weight)
        self.n_classes = weight.shape[0]
        self.class_class_similarity = (weight @ weight.T)

    def get_most_similar_classes(self, pred, max_classes = 5):
        class_sim = self.class_class_similarity[pred]
        sorted_similarities, sorted_indices = torch.sort(class_sim, descending=True)
        max_sim = sorted_similarities[1]
        answer = [pred, sorted_indices[1].item()]
        if max_sim == self.weights_per_class -1:
            equal_to_max = (sorted_similarities == max_sim).nonzero().flatten()
            for idx in equal_to_max[1:]:

                answer.append(sorted_indices[idx].item())
                if len(answer) >= max_classes:
                    break
        return tuple(answer)

    def generate_explanation(self, logits,features,prediction_set, gt_label = None, feature_to_color_mapping=None, folder = None, filename = None, global_plot = False):
        features = features.to(self.weight.device)
        sorted_values, sorted_indices = self.get_sorted_indices_and_values(features[None])
        pred = torch.argmax(logits)[None]
        delta_n = self.get_shared_with_preds_from_sorted(sorted_indices, pred)
        values_for_this_sample = sorted_values[0]
        indices_for_this_sample = sorted_indices[0]
        delta_n_for_this_sample = delta_n[0]

        hierarchy_data = [] # This is a list with one dict per level in the hierarchy. Level refers to numbers of features
        global_hierarchy_data = []

        # Local hierarchy data only includes classes that share the most important feature with the predicted class (delta_n[0] is True])
        for level in range(self.weights_per_class):
            level_dict = {}
            global_level_dict = {}
            for class_idx in range(self.n_classes):
                delta_n_for_this_class = self.get_shared_with_preds_from_sorted(sorted_indices, class_idx)
                predicted_classes_down_the_road = set(delta_n_for_this_class[0, level].nonzero().flatten().tolist())
                this_f_idx = indices_for_this_sample[level, class_idx].item()
                act = values_for_this_sample[level, class_idx].item()
                sharing_at_first_level = delta_n_for_this_sample[0, class_idx].item()
                prev = None
                if level > 0:
                    prev = tuple(indices_for_this_sample[:level, class_idx].tolist())

                tuple_key =(this_f_idx, act,prev ,class_idx)
                global_level_dict[tuple_key] = predicted_classes_down_the_road
                if sharing_at_first_level:
                    level_dict[tuple_key] = global_level_dict[tuple_key]
            hierarchy_data.append(level_dict)
            global_hierarchy_data.append(global_level_dict)
        
        # if class_names is None:
        #     class_names={}
            
        if global_plot:
            visualize_explanation_tree(global_hierarchy_data, gt_label, class_names, features, pred.item(),
                                       prediction_set, folder, filename, feature_to_color_mapping)
        else:
            visualize_explanation_tree(hierarchy_data, gt_label,class_names, features,pred.item(),prediction_set,  folder, filename, feature_to_color_mapping,global_plot = global_plot)

    def build_graph_structure(self, logits, features, global_plot=False, 
                             feature_to_color_mapping=None, class_names=None):
        """
        Build and cache the graph structure WITHOUT applying prediction set colors.
        This is the expensive operation that can be cached.
        
        Args:
            logits: Model output logits for the sample
            features: Feature activations for the sample
            global_plot: If True, build global tree; if False, build local tree
            feature_to_color_mapping: Dict mapping feature indices to colors
            class_names: Dict mapping class indices to names
            
        Returns:
            CachedGraphStructure object that can be quickly re-rendered
        """
        
        features = features.to(self.weight.device)
        sorted_values, sorted_indices = self.get_sorted_indices_and_values(features[None])
        pred = torch.argmax(logits)[None]
        delta_n = self.get_shared_with_preds_from_sorted(sorted_indices, pred)
        values_for_this_sample = sorted_values[0]
        indices_for_this_sample = sorted_indices[0]
        delta_n_for_this_sample = delta_n[0]
    
        hierarchy_data = []
        global_hierarchy_data = []
    
        # Build hierarchy data (same as before)
        for level in range(self.weights_per_class):
            level_dict = {}
            global_level_dict = {}
            for class_idx in range(self.n_classes):
                delta_n_for_this_class = self.get_shared_with_preds_from_sorted(sorted_indices, class_idx)
                predicted_classes_down_the_road = set(delta_n_for_this_class[0, level].nonzero().flatten().tolist())
                this_f_idx = indices_for_this_sample[level, class_idx].item()
                act = values_for_this_sample[level, class_idx].item()
                sharing_at_first_level = delta_n_for_this_sample[0, class_idx].item()
                prev = None
                if level > 0:
                    prev = tuple(indices_for_this_sample[:level, class_idx].tolist())
    
                tuple_key = (this_f_idx, act, prev, class_idx)
                global_level_dict[tuple_key] = predicted_classes_down_the_road
                if sharing_at_first_level:
                    level_dict[tuple_key] = global_level_dict[tuple_key]
            hierarchy_data.append(level_dict)
            global_hierarchy_data.append(global_level_dict)
    
        if class_names is None:
            class_names = {}
    
        # Choose which hierarchy to use
        chosen_hierarchy = global_hierarchy_data if global_plot else hierarchy_data
        
        # Build graph structure (extracted from visualize_explanation_tree_to_pil)
        cached_graph = self._build_graph_from_hierarchy(
            chosen_hierarchy, features, pred.item(), 
            feature_to_color_mapping, class_names, global_plot
        )
        
        return cached_graph

    def _build_graph_from_hierarchy(self, hierarchy_data, feature_activations, pred_idx,
                                    feature_to_color_mapping, class_names, global_plot):
        """
        Internal method to build graph structure from hierarchy data.
        This is the expensive part we want to cache.
        """
        from conformalPrediction.HierarchicalExplanation.graph_cache import CachedGraphStructure
        import copy
        from conformalPrediction.HierarchicalExplanation.graph_utils.helpers import get_remapped_name
        
        alpha = 0.6
        summarize_loc = True
        
        # --- 1. Pre-process hierarchy data ---
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

        # --- 2. Build adjacency matrix ---
        total_number_of_nodes = len(unique_feats) + len(all_classes) + 1
        adjacency_matrix = np.zeros((total_number_of_nodes, total_number_of_nodes))

        feature_mapper = {(i, feat, prev): idx for idx, (i, feat, prev) in enumerate(unique_feats)}
        class_mapper = {cls: idx + len(feature_mapper) for idx, cls in enumerate(sorted(all_classes))}

        root_node_keys = [key for key in unique_feats if key[2] is None]
        root_idx = total_number_of_nodes - 1
        for node_key in root_node_keys:
            adjacency_matrix[root_idx, feature_mapper[node_key]] = 1

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

        # --- 3. Summarize class nodes ---
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
                    names = [class_names.get(c, str(c)) for c in class_group]
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

        # --- 4. Create NetworkX graph ---
        graph = nx.DiGraph(adjacency_matrix)
        num_nodes = graph.number_of_nodes()
        color_map = [None] * num_nodes
        nodes_sizes = [None] * num_nodes
        nodelabels = {}
        
        feature_scaled_for_size = (feature_activations / feature_activations.max()) * 500
        if global_plot:
            feature_scaled_for_size = feature_scaled_for_size / 5
        colors_per_feature = get_colors_per_feature(feature_activations, 
                                                    list(unique_feats_independent), 
                                                    feature_to_color_mapping)

        for (i, feat, prev), idx in feature_mapper.items():
            nodelabels[idx] = ""
            graph.nodes[idx]["type"] = "feature"
            color_map[idx] = colors_per_feature[feat]
            nodes_sizes[idx] = feature_scaled_for_size[feat].item()

        for cls in sorted(class_mapper.keys()):
            if cls not in class_mapper_new: continue

            idx = class_mapper_new[cls]
            class_label = remapped_labels.get(cls, class_names.get(cls, str(cls)))
            graph.nodes[idx]["forSummary"] = summarized_labels.get(cls, [class_names.get(cls, str(cls))])
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

        # Compute positions (expensive!)
        pos = get_smaller_v_space_pos(graph, root_idx)

        # Return cached structure
        return CachedGraphStructure(
            graph=graph,
            pos=pos,
            color_map=color_map,
            nodes_sizes=nodes_sizes,
            nodelabels=nodelabels,
            feature_mapper=feature_mapper,
            full_new_mapper=full_new_mapper,
            root_idx=root_idx,
            width_for_this_graph=width_for_this_graph,
            alpha=alpha,
            global_plot=global_plot
        )


if __name__ == '__main__':
    args = get_args_for_loading_model()
    train_loader, test_loader = get_data(args.dataset, crop=args.cropGT, img_size=args.img_size)
    train_loader.dataset.transform = test_loader.dataset.transform
    model, folder = load_model(args.dataset, args.arch, args.seed, args.model_type, args.cropGT, args.n_features,
                               args.n_per_class, args.img_size, args.reduced_strides, args.folder)
    features_test, outputs_test, labels_test = get_feats_logits_labels(model, test_loader)
    cal_logits, cal_labels, cal_features, test_logits, test_labels, test_features, test_indices = get_logits_and_labels(
        features_test, outputs_test, labels_test, 10, )
    graph_folder = Path.home() / "tmp" / "CHiQPMExplanations"
    # Since the examples chosen in readme are for classes with indices 25,26, 176 and 181 we only show these
    test_labels_in_readme = [25, 26, 176, 181]
    kept_label = torch.zeros_like(test_labels)
    for lab in test_labels_in_readme:
        kept_label = kept_label + (test_labels == lab).long()
    test_logits = test_logits[kept_label.bool()]
    test_features = test_features[kept_label.bool()]
    test_labels = test_labels[kept_label.bool()]
    index_in_dataset = test_indices[kept_label.bool()]
    if args.dataset == "CUB2011":
        class_names = load_cub_class_mapping()
        class_names = {int(k): v for k, v in class_names.items()}
    # Reasonable cherrypicks for this model
    indices_of_test_samples = [ 23,43,3,59]

    explainer = HierarchicalExplainer(model.linear.weight)
    for acc in [ .9,.95, .925, .88, ]: #


        predictor, needs_feats = get_score("CHiQPM", model.linear.weight)

        calibrate_predictor(cal_logits, cal_labels, acc, cal_features, predictor, needs_feats)


        prediction_sets = get_predictions(test_logits, predictor, test_features, needs_feats)
        if indices_of_test_samples != []:
            for sample in indices_of_test_samples:


                prediction = torch.argmax(test_logits[sample]).item()
                class_pair_interesting = explainer.get_most_similar_classes(prediction,)
                index_in_dataset_of_sample = index_in_dataset[sample].item()
                filename_end = f"Sample_{index_in_dataset_of_sample}_label_{class_names[test_labels[sample].item()]}"
                image = test_loader.dataset.__getitem__(index_in_dataset_of_sample)[0]
                graph_folder_acc = graph_folder / f"SampleIndex_{str(index_in_dataset_of_sample)}" / f"acc_{acc}"
                graph_folder_acc.mkdir(parents=True, exist_ok=True)
                image_unnormalized = (image * test_loader.dataset.transform.transforms[-1].std[:, None, None] +
                                      test_loader.dataset.transform.transforms[-1].mean[:, None, None])
                active_mean = get_active_mean(model, train_loader, folder)
                class_indices = list(class_pair_interesting)
                colormapping = viz_model(model, train_loader, class_indices, "Train",
                                         active_mean, viz_folder=graph_folder_acc)
                generate_local_viz(image, image_unnormalized, model, colormapping,graph_folder_acc /
                                   f"Local_{filename_end}.png",active_mean=active_mean )

                explainer.generate_explanation(test_logits[sample], test_features[sample], prediction_sets[sample],feature_to_color_mapping=colormapping, gt_label = test_labels[sample], folder = graph_folder_acc, filename = f"Graph_{filename_end}")
        else:
            graph_folder_ac = graph_folder / f"acc_{acc}"
            graph_folder_ac.mkdir(parents=True, exist_ok=True)
            for sample in range(len(test_logits)):
                explainer.generate_explanation(test_logits[sample], test_features[sample], prediction_sets[sample], gt_label = test_labels[sample], folder = graph_folder_ac, filename = f"sample_{sample}_label_{test_labels[sample].item()}")

