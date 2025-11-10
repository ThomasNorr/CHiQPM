"""
Evaluation Metrics Module

This module provides comprehensive evaluation metrics for interpretable models.
It computes metrics related to:
- Spatial diversity of features (SID@k)
- Feature contrastiveness (GMM overlap)
- Class independence
- Structural grounding (alignment with ground truth class similarities)
- Feature correlation

These metrics assess the quality and interpretability of learned features.
"""
import numpy as np
import sklearn
import torch
from tqdm import trange

from evaluation.Metrics.Contrastiveness import gmm_overlap_per_feature
from evaluation.Metrics.Correlation import get_correlation
from evaluation.Metrics.Dependence import compute_contribution_top_feature, compute_class_independence
from evaluation.Metrics.StructuralGrounding import get_structural_grounding_for_weight_matrix
from evaluation.diversity import MultiKCrossChannelMaxPooledSum
from evaluation.utils import get_metrics_for_model


def evaluateALLMetricsForComps(features_train,  outputs_train,  feature_maps_test,
                               outputs_test, linear_matrix,  labels_train, labels_test, features_test):
    """
    Evaluate model on all interpretability metrics.
    
    Computes a comprehensive set of metrics that measure the quality and
    interpretability of learned features including diversity, dependency,
    contrastiveness, and alignment with ground truth.
    
    Args:
        features_train (Tensor): Training set features after pooling and normalization, shape (n_train, n_features)
        outputs_train (Tensor): Training set model outputs (logits), shape (n_train, n_classes)
        feature_maps_test (Tensor): Test set feature maps before pooling, 
                                     shape (n_test, n_features, height, width)
        outputs_test (Tensor): Test set model outputs (logits), shape (n_test, n_classes)
        linear_matrix (Tensor): Weight matrix of final layer, shape (n_classes, n_features)
        labels_train (Tensor): Training set ground truth labels, shape (n_train,)
        labels_test (Tensor): Test set ground truth labels, shape (n_test,)
        features_test (Tensor): Test set features after pooling and normalization, shape (n_test, n_features)
    
    Returns:
        dict: Dictionary containing computed metrics:
            - "SID@5": Spatial Independence Diversity at k=5
            - "Class-Independence": Feature independence from class predictions
            - "Contrastiveness": GMM-based feature contrastiveness (1 - overlap)
            - "Structural Grounding": Alignment with ground truth class similarities (CUB only)
            - "Correlation": Mean absolute correlation between features
    
    Note:
        - Structural grounding is only computed for CUB dataset (train size < 7000)
        - Contrastiveness computation is skipped for dense models (>1000 features)
          as it's computationally expensive
    """
    # Calculate Diversity, Dependency, GMM Overlap and similarity with CUB GT for given features

    with torch.no_grad():
        # Compute structural grounding only for CUB (identified by training set size)
        if len(features_train) < 7000:
            cub_overlap = get_structural_grounding_for_weight_matrix(linear_matrix)
        else:
            cub_overlap = 0
        print("cub_overlap: ", cub_overlap)
        
        # Compute spatial diversity using SumNMax localization at multiple k values
        soft_max_scaled_localizer = MultiKCrossChannelMaxPooledSum(range(1, 6), linear_matrix, None,
                                                                   func="SumNMax")
        batch_size = 300
        for i in range(np.floor(len(features_train) / batch_size).astype(int)):
            soft_max_scaled_localizer(outputs_test[i * batch_size:(i + 1) * batch_size].to("cuda"),
                                      feature_maps_test[i * batch_size:(i + 1) * batch_size].to("cuda"))
        diversity_sm_scaled = soft_max_scaled_localizer.get_result()[0][4].item()
        print("SID@5: ", diversity_sm_scaled)
        
        # Compute contrastiveness (skip for dense models as it's expensive)
        if features_train.shape[1] > 1000:
            print("Skipping Contrastiveness for dense model as it takes a while")
            overlap_mean = -1
        else:
            overlap_mean = 1 - gmm_overlap_per_feature(features_train).mean()
        
        # Compute class independence and feature correlation
        class_independence = compute_class_independence(features_train,  linear_matrix,
                                                                     labels_train)
        correlation_features = get_correlation(features_train).item()
        answer_dict = {"SID@5": diversity_sm_scaled,"Class-Independence": class_independence, "Contrastiveness": overlap_mean,"Structural Grounding": cub_overlap, "Correlation":correlation_features}
    return answer_dict

def eval_model_on_all_qpm_metrics(model, test_loader, train_loader):
    """
    Evaluate a model on all interpretability metrics.
    
    This is the main entry point for evaluating a trained model.
    It extracts features and predictions from the model, then computes all metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        train_loader: DataLoader for training data
    
    Returns:
        dict: Dictionary containing all computed metrics
    """
    return get_metrics_for_model(train_loader, test_loader, model, evaluateALLMetricsForComps)
