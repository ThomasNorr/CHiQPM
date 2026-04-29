import gradio as gr
import torch
import math
from typing import Tuple, Optional, List
from PIL import Image
from evaluation.load_model import get_args_for_loading_model, load_model
from dataset_classes.cub200 import load_cub_class_mapping
from get_data import get_data
from demonstrator.tree_generator import TreeGenerator
from demonstrator.feature_grid_generator import FeatureGridGenerator
from configs.demo_paths import CAL_DATA_ROOT


class BirdClassifierDemo:
    """Gradio demo for CHiQPM: Calibrated Hierarchical Interpretable Image Classification"""

    # CONSTANTS
    N_FEATURES_PER_CLASS = 5
    IMG_SIZE = 224
    ACCURACY_STEP = 0.001
    ACCURACY_MAX = 1.0
    DEFAULT_ACCURACY_INDEX = 2
    DEFAULT_K_PREDICTIONS = 3
    MIN_REQUIRED_ACCURACY_LEVELS = 5

    def __init__(self):
        """Initialize and load all required resources"""
        try:
            print("Initializing demo...")
            
            # Load configuration
            args = get_args_for_loading_model()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model
            self.model, self.folder = load_model(
                args.dataset, args.arch, args.seed, args.model_type,
                args.cropGT, args.n_features, args.n_per_class,
                args.img_size, args.reduced_strides, args.folder
            )
            self.model = self.model.to(self.device)
            
            # Load calibration data
            cal_path = CAL_DATA_ROOT / f"seed_{args.seed}" / "calibration_data.pt"
            if not cal_path.exists():
                raise FileNotFoundError(f"Calibration data not found at: {cal_path}")
            
            self.calibration_data = torch.load(cal_path)
            
            # Load training data
            self.train_loader, _ = get_data( args.dataset, crop=args.cropGT, img_size=args.img_size)
            
            # Load class names
            class_mapping = load_cub_class_mapping()
            self.class_names = {int(k): v for k, v in class_mapping.items()}
            print(f"Loaded {len(self.class_names)} classes")
            
            # Initialize generators
            self.tree_generator = TreeGenerator(self.model, self.calibration_data, self.class_names)

            raw_accs = self.tree_generator.predictor.score_function.total_accs.cpu().tolist()
            self.total_accs = [math.floor(acc * 1000) / 1000 for acc in raw_accs]
            self.accuracy_min = self.total_accs[-1]
            print(f"Total accuracies (truncated): {self.total_accs}")
            
            self.feature_grid_generator = FeatureGridGenerator(
                self.model, 
                self.train_loader, 
                self.folder, 
                self.class_names, 
                self.device,
                img_size=self.IMG_SIZE, 
                n_features_per_class=self.N_FEATURES_PER_CLASS
            )
            
            # Validate total_accs length
            if len(self.total_accs) < self.MIN_REQUIRED_ACCURACY_LEVELS:
                raise ValueError(
                    f"Expected at least {self.MIN_REQUIRED_ACCURACY_LEVELS} accuracy levels, got {len(self.total_accs)}"
                )
            
            print("Demo ready!")
            
        except FileNotFoundError as e:
            print(f"Error: Required file not found - {e}")
            raise
        except Exception as e:
            print(f"Error initializing demo: {type(e).__name__}: {e}")
            raise
   
    def _create_empty_result(self) -> Tuple:
        """
        Create empty result tuple for when no image is uploaded.
        
        Returns:
            Tuple matching the output format of process_input_image
        """
        n_buttons = len(self.total_accs) + 1  # +1 for "no limit" button
        empty_button_updates = [gr.update(variant="secondary")] * n_buttons
        
        return (
            None,  # input_unnormalized
            None,  # input_normalized
            None,  # output_logits
            None,  # final_features
            None,  # prediction
            None,  # predicted_class_name
            None,  # colormapping
            gr.update(visible=False, minimum=self.accuracy_min, maximum=self.ACCURACY_MAX),  # accuracy_slider
            None,  # comparison_grid (viz_grid)
            None,  # tree
            gr.update(visible=False),  # grid_section
            gr.update(visible=False),  # tree_controls
            gr.update(visible=False),  # tree_visualization
            None,  # k_predictions state
            gr.update(value=self.DEFAULT_K_PREDICTIONS),  # k_predictions_input
            *empty_button_updates  # button_components
        )

    def forward_pass(self, input_image_pil: Optional[Image.Image]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[int]]:
        """Run forward pass once and return all needed outputs"""
        if input_image_pil is None:
            return None, None, None, None, None
        
        input_image, input_image_normalized = self.feature_grid_generator.preprocess_image(input_image_pil)
        
        with torch.no_grad():
            output_logits, feature_maps, final_features = self.model(
                input_image_normalized,
                with_feature_maps=True,
                with_final_features=True
            )
        
        prediction = torch.argmax(output_logits).item()

        print(f"prediction = {prediction}")
        
        return input_image, input_image_normalized, output_logits, final_features, prediction

    def generate_tree_from_outputs(self, output_logits: torch.Tensor, final_features: torch.Tensor, accuracy: float, colormapping: dict) -> Optional[Image.Image]:
        """Generate tree from already computed forward pass outputs"""
        return self.tree_generator.generate_tree(
            output_logits, 
            final_features, 
            accuracy, 
            colormapping
        )

    def process_input_image(self, input_image_pil: Optional[Image.Image], k_predictions: int) -> Tuple:
        """Initial processing when input image is uploaded"""        
        if input_image_pil is None:
            return self._create_empty_result()
        
        # Run forward pass
        input_image, input_image_normalized, output_logits, final_features, prediction = \
            self.forward_pass(input_image_pil)
        
        predicted_class_name = self.class_names.get(prediction, f"Class {prediction}")
        
        # Generate feature grid
        viz_grid, colormapping = self.feature_grid_generator.generate_feature_grid(
            input_image,
            input_image_normalized,
            output_logits,
            final_features,
            k_predictions
        )
        
        # Generate tree with default accuracy
        accuracy_default = self.total_accs[self.DEFAULT_ACCURACY_INDEX]
        tree_image = self.generate_tree_from_outputs(
            output_logits, 
            final_features, 
            accuracy=accuracy_default, 
            colormapping=colormapping
        )
        
        button_updates = self.update_button_highlights(accuracy_default)
        
        return (
            input_image,
            input_image_normalized,
            output_logits,
            final_features,
            prediction,
            predicted_class_name,
            colormapping,
            gr.update(
                value=accuracy_default, 
                minimum=self.accuracy_min,
                maximum=self.ACCURACY_MAX
            ),
            viz_grid,
            tree_image,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            k_predictions,
            gr.update(value=self.DEFAULT_K_PREDICTIONS),
            *button_updates,
        )

    def update_button_highlights(self, current_accuracy: float) -> List[gr.update]:
        """Return button variant updates based on current accuracy"""
        updates = []
        
        # Iterate through accuracy levels in reverse
        for i in range(len(self.total_accs) - 1, -1, -1):
            acc = self.total_accs[i]
            acc_prev = 0 if i == len(self.total_accs) - 1 else self.total_accs[i + 1]
            
            if acc_prev < current_accuracy <= acc:
                updates.append(gr.update(variant="primary"))
            else:
                updates.append(gr.update(variant="secondary"))
        
        # Handle "no limit" button
        n1_limit = self.total_accs[0]
        if n1_limit < current_accuracy:
            updates.append(gr.update(variant="primary"))
        else:
            updates.append(gr.update(variant="secondary"))
        
        return updates

    def regenerate_feature_grid_only(self, input_image_pil: Optional[Image.Image], k_predictions: int, output_logits: torch.Tensor, final_features: torch.Tensor, input_unnorm: torch.Tensor, input_norm: torch.Tensor) -> Tuple[Optional[Image.Image], Optional[dict]]:
        """
        Regenerate only the feature grid without running forward pass.
        Used when user changes k_predictions but image stays the same.
        """
        if input_image_pil is None or output_logits is None:
            return None, None
        
        viz_grid, colormapping = self.feature_grid_generator.generate_feature_grid(
            input_unnorm,
            input_norm,
            output_logits,
            final_features,
            k_predictions
        )
        
        return viz_grid, colormapping
    
    def _create_button_click_handler(self, acc_value: float):
        """Create a button click handler for a specific accuracy value"""
        def handler(logits: torch.Tensor, feats: torch.Tensor, cmap: dict):
            return (
                acc_value,
                self.generate_tree_from_outputs(logits, feats, acc_value, cmap),
                *self.update_button_highlights(acc_value)
            )
        return handler
    
    def _create_slider_handler(self):
        """Create slider release handler"""
        def handler(acc: float, logits: torch.Tensor, feats: torch.Tensor, cmap: dict):
            return (
                self.generate_tree_from_outputs(logits, feats, acc, cmap),
                *self.update_button_highlights(acc)
            )
        return handler
    
    def create_ui(self):
        """Build and return the Gradio interface"""
        with gr.Blocks(css=".reset-button { display: none !important; }") as demo:
            gr.Markdown("### CHiQPM - Interpretable Bird Classifier")
            
            # Always visible - input section
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        type="pil",
                        label="Upload Bird Image",
                        height=200,
                        show_download_button=False
                    )
                with gr.Column(scale=3):
                    with gr.Row():
                        k_predictions_input = gr.Number(
                            value=self.DEFAULT_K_PREDICTIONS,
                            label="Top k Predicted Classes",
                            minimum=1,
                            maximum=5,
                            step=1,
                            precision=0
                        )
                        classify_btn = gr.Button("Classify", variant="secondary")
                    predicted_class_name = gr.Textbox(
                        label="Predicted Species", 
                        interactive=False
                    )
            
            # Hidden until image uploaded - feature visualization
            with gr.Column(visible=False) as grid_section:
                gr.Markdown("#### Activated Features")
                comparison_grid = gr.Image(
                    type="pil",
                    show_download_button=False,
                    show_share_button=False,
                    show_label=False,
                    interactive=False,
                    show_fullscreen_button=True
                )
            
            # Hidden until image uploaded - tree controls
            with gr.Column(visible=False) as tree_controls:
                gr.Markdown("#### Decision Tree")
                
                accuracy_slider = gr.Slider(
                    minimum=self.accuracy_min,
                    maximum=self.ACCURACY_MAX,
                    value=self.total_accs[self.DEFAULT_ACCURACY_INDEX],
                    step=self.ACCURACY_STEP,
                    label="Accuracy",
                    show_label=True,
                    interactive=True
                )
                
                gr.Markdown("**Quick select hierarchy levels:**")
                with gr.Row():
                    level_buttons = []
                    button_components = []
                    
                    # Create buttons for each accuracy level
                    for i in range(len(self.total_accs) - 1, -1, -1):
                        acc = self.total_accs[i]
                        acc_percentage = acc * 100
                        
                        if i == len(self.total_accs) - 1:
                            acc_prev = 0
                            acc_range = f"({acc_prev} - {acc_percentage:.1f}) %"
                        else:
                            acc_prev = self.total_accs[i + 1] * 100
                            acc_range = f"({acc_prev:.1f} - {acc_percentage:.1f}) %"
                        
                        btn = gr.Button(
                            f"nˡⁱᵐⁱᵗ = {i + 1}\u2003{acc_range}",
                            size="sm",
                            variant="secondary"
                        )
                        level_buttons.append((btn, acc))
                        button_components.append(btn)
                    
                    # "No limit" button
                    acc = self.ACCURACY_MAX
                    n1_limit = self.total_accs[0] * 100
                    acc_range = f"({n1_limit:.1f} - 100) %"
                    btn = gr.Button(
                        f"no nˡⁱᵐⁱᵗ \u2003{acc_range}",
                        size="sm",
                        variant="secondary"
                    )
                    level_buttons.append((btn, acc))
                    button_components.append(btn)
            
            # Hidden until image uploaded - tree display
            with gr.Column(visible=False) as tree_visualization:
                tree = gr.Image(
                    type="pil",
                    show_download_button=False,
                    show_share_button=False,
                    show_label=False,
                    height=600,
                    interactive=False,
                    show_fullscreen_button=True
                )
            
            # Hidden state variables to store forward pass outputs
            input_unnormalized = gr.State(None)
            input_normalized = gr.State(None)
            output_logits = gr.State(None)
            final_features = gr.State(None)
            prediction = gr.State(None)
            colormapping = gr.State(None)
            k_predictions = gr.State(self.DEFAULT_K_PREDICTIONS)
            
            # Connect button click handlers
            for btn, acc_value in level_buttons:
                handler = self._create_button_click_handler(acc_value)
                btn.click(
                    fn=handler,
                    inputs=[output_logits, final_features, colormapping],
                    outputs=[accuracy_slider, tree] + button_components
                )
            
            # Classify button handler
            classify_btn.click(
                fn=self.process_input_image,
                inputs=[input_image, k_predictions_input],
                outputs=[
                    input_unnormalized,
                    input_normalized,
                    output_logits,
                    final_features,
                    prediction,
                    predicted_class_name,
                    colormapping,
                    accuracy_slider,
                    comparison_grid,
                    tree,
                    grid_section,
                    tree_controls,
                    tree_visualization,
                    k_predictions,
                    k_predictions_input,
                    *button_components,
                ]
            )
            
            # K predictions change handler
            k_predictions_input.change(
                fn=self.regenerate_feature_grid_only,
                inputs=[
                    input_image, 
                    k_predictions_input, 
                    output_logits, 
                    final_features,
                    input_unnormalized, 
                    input_normalized
                ],
                outputs=[comparison_grid, colormapping]
            )
            
            # Accuracy slider handler
            slider_handler = self._create_slider_handler()
            accuracy_slider.release(
                fn=slider_handler,
                inputs=[accuracy_slider, output_logits, final_features, colormapping],
                outputs=[tree] + button_components
            )
        
        return demo


if __name__ == '__main__':
    try:
        demo_app = BirdClassifierDemo()
        ui = demo_app.create_ui()
        ui.launch()
        # ui.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        print(f"Failed to start demo: {e}")
        raise