"""
Feature visualization grid generation module for CHiQPM demo.
Handles feature selection, representative image loading, and grid creation.
"""

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
from get_data import DeepLabBirdCrop
from visualization.get_heatmaps import get_visualizations
from visualization.utils import get_active_mean
from configs.dataset_params import normalize_params
from configs.demo_paths import DATASET_ROOT, REPRESENTATIVES_ROOT

class FeatureGridGenerator:
    """
    Handles all feature visualization grid logic including:
    - Loading representative class images
    - Feature selection across multiple predictions
    - Grid visualization creation
    """

    DEFAULT_PADDING = 20
    DEFAULT_GAMMA = 3.0  # Gamma for visualization intensity
    DPI = 150 # DPI for saved images
    
    def __init__(self, model, train_loader, folder, class_names, device, img_size=224, n_features_per_class=5, padding=None, gamma=None):
        """
        Initialize feature grid generator.
        
        Args:
            model: The trained CHiQPM model
            train_loader: DataLoader for training data (used for active mean calculation)
            folder: Model folder path
            class_names: Dict mapping class indices to names
            device: torch.device to use
            img_size: Image size for resizing (default: 224)
            n_features_per_class: Number of features to show per class (default: 5)
            padding: Padding for image cropping (default: DEFAULT_PADDING)
            gamma: Gamma for visualization intensity (default: DEFAULT_GAMMA)
        """
        self.padding = padding if padding is not None else self.DEFAULT_PADDING
        self.gamma = gamma if gamma is not None else self.DEFAULT_GAMMA
        self.model = model
        self.train_loader = train_loader
        self.folder = folder
        self.class_names = class_names
        self.device = device
        self.img_size = img_size
        self.n_features_per_class = n_features_per_class
        
        self.active_mean = get_active_mean(model, train_loader, folder)
        
        self.dataset_path = DATASET_ROOT
        self.representatives_path = REPRESENTATIVES_ROOT

        self.representative_cache = {}
    
    def preprocess_image(self, image_pil):
        """
        Crops, resizes, normalizes the image and converts to tensor.
        
        Args:
            image_pil: PIL.Image - input image
            
        Returns:
            Tuple of:
                - unnormalized image tensor [1,3,H,W]
                - normalized image tensor [1,3,H,W]
        """
        cropper = DeepLabBirdCrop(padding=self.padding)
        cropped_image = cropper(image_pil)

        preprocess = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        
        image_tensor = preprocess(cropped_image).unsqueeze(0).to(self.device)

        normalize = transforms.Normalize(**normalize_params["CUB2011"])
        normalized_image_tensor = normalize(image_tensor)

        return image_tensor, normalized_image_tensor

    def get_representative_class_image(self, predicted_class):
        """
        Load and return the preprocessed representative image for a predicted class.
        Uses caching to avoid repeated disk I/O and preprocessing.
        
        Args:
            predicted_class: Class index
            
        Returns:
            Tuple of (unnormalized_tensor, normalized_tensor) or (None, None) if not found
        """
        # Check cache first
        if predicted_class in self.representative_cache:
            return self.representative_cache[predicted_class]
        
        # Not in cache - load from disk
        class_folders = [f for f in os.listdir(self.dataset_path) if f.startswith(f"{predicted_class+1:03d}.")]
        
        if not class_folders:
            # Cache None to avoid repeated lookups
            self.representative_cache[predicted_class] = (None, None)
            return None, None

        class_folder = class_folders[0]
        representatives_path = self.representatives_path / class_folder
        
        if representatives_path.exists():
            cropped_images = [f for f in os.listdir(representatives_path) if f.endswith('_cropped.jpg')]
            
            if cropped_images:
                representative_image_path = representatives_path / cropped_images[0]
                pil_image = Image.open(representative_image_path)
                
                # Preprocess immediately and cache tensors
                unnorm_tensor, norm_tensor = self.preprocess_image(pil_image)
                self.representative_cache[predicted_class] = (unnorm_tensor, norm_tensor)
                
                return unnorm_tensor, norm_tensor
        
        # Not found - cache None
        print(f"Warning: No representative found for class {class_folder}")
        self.representative_cache[predicted_class] = (None, None)
        return None, None
    
    def _create_visualization_grid(self, viz_all, all_features, input_unnormalized, 
                                   class_images_unnormalized_list, prediction_list):
        """
        Create a grid visualization of feature heatmaps.
        
        Args:
            viz_all: List of visualization tensors for each feature
            all_features: List of feature indices
            input_unnormalized: Unnormalized input image tensor
            class_images_unnormalized_list: List of unnormalized representative images
            prediction_list: List of predicted class indices
            
        Returns:
            PIL.Image of the visualization grid
        """
        n_rows = 1 + len(prediction_list)   # first row = input
        n_cols = 1 + len(all_features)      # first col = original images
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))

        # First column: original images (input + reps)
        axes[0, 0].imshow(input_unnormalized[0].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[0, 0].axis('off')

        for row_idx, (pred_class, class_img_unnorm) in enumerate(zip(prediction_list, class_images_unnormalized_list), 1):
            ax = axes[row_idx, 0]
            if class_img_unnorm is not None:
                ax.imshow(class_img_unnorm[0].cpu().permute(1, 2, 0).clamp(0, 1))
            else:
                # fallback to grayscale input if no representative available
                gray_img = input_unnormalized[0].cpu().mean(dim=0, keepdim=True).repeat(3, 1, 1)
                ax.imshow(gray_img.permute(1, 2, 0).clamp(0, 1), cmap='gray')
            ax.axis('off')

        # Feature columns: directly render overlays from viz_all
        for col_idx, feat_samples in enumerate(viz_all, start=1):
            # If we had fewer class images because some reps were None, feat_samples may have fewer rows;
            # align by min length so the grid still renders.
            max_rows_available = min(n_rows, feat_samples.shape[0])
            for row_idx in range(max_rows_available):
                ax = axes[row_idx, col_idx]
                ax.imshow(feat_samples[row_idx].cpu().permute(1, 2, 0))
                ax.axis('off')
            # If any rows had no visualization (missing reps), hide their axes in this column
            for row_idx in range(max_rows_available, n_rows):
                axes[row_idx, col_idx].axis('off')

        # Row labels - place bird names to the left of each row using text annotations
        for row_idx in range(n_rows):
            if row_idx == 0:
                label_text = "Input"
            else:
                pred_class = prediction_list[row_idx - 1]
                label_text = self.class_names.get(pred_class, f"Class {pred_class}").replace("_", " ")
            
            # Place text to the left of the first column
            axes[row_idx, 0].text(
                -0.075, 0.5, label_text,
                transform=axes[row_idx, 0].transAxes,
                fontsize=12,
                rotation=90,
                ha='center',
                va='center'
            )

        # Column titles
        for col_idx, feature in enumerate(all_features, start=1):
            axes[0, col_idx].set_title(f"Feature {feature}", color='black', fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=self.DPI)
        buf.seek(0)
        comparison_image = Image.open(buf)
        plt.close("all")
        return comparison_image
    
    def generate_feature_grid(self, input_image, input_image_normalized, output_logits, 
                         final_features, k_predictions):
        """
        Generate the complete feature visualization grid.
        
        Args:
            input_image: Unnormalized input image tensor [1,3,H,W]
            input_image_normalized: Normalized input image tensor [1,3,H,W]
            output_logits: Model output logits [1, n_classes]
            final_features: Final layer features [1, n_features]
            k_predictions: Number of top predictions to visualize
            
        Returns:
            Tuple of:
                - PIL.Image: The visualization grid
                - dict: Feature to color mapping for tree visualization
        """
        features_for_prediction = torch.topk(
            final_features.flatten(), 
            k=min(self.n_features_per_class, final_features.numel())
        ).indices.tolist()
        
        predictions = torch.topk(
            output_logits, 
            k=min(k_predictions, output_logits.numel())
        ).indices.flatten().tolist()

        predicted_classes_images = []
        predicted_classes_images_normalized = []

        for p in predictions:
            rep_unnorm, rep_norm = self.get_representative_class_image(p)
            predicted_classes_images.append(rep_unnorm)
            predicted_classes_images_normalized.append(rep_norm)

        all_features = features_for_prediction.copy()
        for p in predictions:
            features = self.model.linear.weight[p].nonzero().flatten()[:self.n_features_per_class].tolist()
            for f in features:
                if f not in all_features:
                    all_features.append(f)

        # Build a batch [input, pred#1, pred#2, ...]
        all_images = [input_image] + [img for img in predicted_classes_images if img is not None]
        all_images_normalized = [input_image_normalized] + [img for img in predicted_classes_images_normalized if img is not None]

        all_images = torch.cat(all_images, dim=0)
        all_images_normalized = torch.cat(all_images_normalized, dim=0)

        viz_all, colormapping = get_visualizations(
            all_features,
            all_images_normalized,
            all_images,
            self.model,
            gamma=self.gamma,
            with_color=True,
            active_means=self.active_mean
        )

        comparison_visualization = self._create_visualization_grid(
            viz_all,
            all_features,
            input_image,
            predicted_classes_images,
            predictions
        )

        return comparison_visualization, colormapping
