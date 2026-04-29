"""
Tree generation module for CHiQPM demo.
Handles conformal prediction setup, calibration, and tree visualization.
"""

from conformalPrediction.visualize_tree import HierarchicalExplainer
from conformalPrediction.utils import get_score, get_predictions


class TreeGenerator:
    """
    Handles all tree generation logic with optimized caching.
    Caches graph structures and only recolors edges when accuracy changes.
    """
    
    def __init__(self, model, calibration_data, class_names):
        """
        Initialize tree generator with one-time calibration.
        
        Args:
            model: The trained CHiQPM model
            calibration_data: Dict with 'cal_logits', 'cal_labels', 'cal_features'
            class_names: Dict mapping class indices to names
        """
        self.model = model
        self.calibration_data = calibration_data
        self.class_names = class_names
        self.explainer = HierarchicalExplainer(model.linear.weight)
        
        # Create and calibrate predictor ONCE
        self.predictor, self.needs_feats = get_score("CHiQPM", self.model.linear.weight)
        self.predictor.calibrate_all_levels(
            self.calibration_data['cal_logits'],
            self.calibration_data['cal_labels'],
            self.calibration_data['cal_features']
        )
        
        # Cache for graph structures
        self.cached_graphs = {
            'local': None,
            'global': None
        }
        self.last_sample_hash = None  # To detect when sample changes
    
    def _get_sample_hash(self, output_logits, final_features):
        """
        Create a hash to detect if we're looking at a new sample.
        """
        logits_hash = output_logits.flatten()[:5].cpu().tolist()
        features_hash = final_features.flatten()[:5].cpu().tolist()
        return (tuple(logits_hash), tuple(features_hash))
    
    def generate_tree(self, output_logits, final_features, accuracy, colormapping):
        """
        Generate tree visualization as PIL Image with caching optimization.
        
        Args:
            output_logits: Model output logits [1, n_classes]
            final_features: Final layer features [1, n_features]
            accuracy: Target accuracy level (0.856-1.0, where 1.0 = all paths green)
            colormapping: Feature to color mapping dict
            
        Returns:
            PIL.Image of the tree visualization, or None if generation fails
        """
        # Special case: accuracy = 1.0 means show global tree with ALL paths in green
        if accuracy >= 0.9999:            
            # Check if sample changed
            current_hash = self._get_sample_hash(output_logits, final_features)
            if current_hash != self.last_sample_hash:
                self.cached_graphs['global'] = None
                self.last_sample_hash = current_hash
            
            # Prediction set = ALL classes makes all paths green
            all_classes = list(range(output_logits.shape[-1]))
            
            try:
                return self._generate_with_cache(
                    output_logits, final_features,
                    all_classes,  # All classes = all paths valid/green
                    colormapping,
                    global_plot=True
                )
            except Exception as e:
                print(f"Failed to generate global tree with all paths: {e}")
                return None
        
        # Normal case: accuracy-based prediction set
        self.predictor.update_predictor(1 - accuracy)
        prediction_set = get_predictions(
            output_logits,
            self.predictor,
            final_features,
            self.needs_feats
        )[0]

        max_level = getattr(self.predictor.score_function, "weights_per_class", None)
        selected_level = getattr(self.predictor.score_function, 'level', None)
        if selected_level is not None and max_level is not None and selected_level == max_level:
            top1_pred = int(output_logits.argmax(dim=-1).item())
            prediction_set = [top1_pred]
        
        # Check if sample changed
        current_hash = self._get_sample_hash(output_logits, final_features)
        if current_hash != self.last_sample_hash:
            self.cached_graphs['local'] = None
            self.cached_graphs['global'] = None
            self.last_sample_hash = current_hash

        sufficient_acc = self.predictor.score_function.total_accs >= accuracy
        generate_global_tree = not sufficient_acc.any()
        
        if generate_global_tree:
            try:
                return self._generate_with_cache(
                    output_logits, final_features, prediction_set,
                    colormapping, global_plot=True
                )
            except Exception as e:
                print(f"Global tree generation failed: {e}")
                return None
        else:
            # Try local tree first
            try:
                return self._generate_with_cache(
                    output_logits, final_features, prediction_set,
                    colormapping, global_plot=False
                )
            
            except Exception as e:
                print(f"Local tree failed: {e}, trying global tree instead")
                # Fallback to global tree
                try:
                    return self._generate_with_cache(
                        output_logits, final_features, prediction_set,
                        colormapping, global_plot=True
                    )
                except Exception as e2:
                    print(f"Global tree also failed: {e2}")
                    return None
    
    def _generate_with_cache(self, output_logits, final_features, 
                            prediction_set, colormapping, global_plot):
        """
        Generate tree using cache if available, otherwise build and cache.
        
        Args:
            output_logits: Model output logits [1, n_classes]
            final_features: Final layer features [1, n_features]
            prediction_set: Set of predicted class indices
            colormapping: Feature to color mapping dict
            global_plot: Whether to generate global or local tree
            
        Returns:
            PIL.Image of the tree
        """
        cache_key = 'global' if global_plot else 'local'
        
        # Check if we have cached graph structure
        if self.cached_graphs[cache_key] is None:
            # Build and cache the graph structure (expensive)
            print(f"Building {cache_key} graph structure (first time for this sample)")
            self.cached_graphs[cache_key] = self.explainer.build_graph_structure(
                output_logits.squeeze(0),
                final_features.squeeze(0),
                global_plot=global_plot,
                feature_to_color_mapping=colormapping,
                class_names=self.class_names
            )
        
        tree_image = self.cached_graphs[cache_key].render_with_prediction_set(
            set(prediction_set)
        )
        
        return tree_image