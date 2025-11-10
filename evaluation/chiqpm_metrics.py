"""
CHiQPM Evaluation Metrics Module

This module provides evaluation functions specific to CHiQPM,
including conformal prediction set metrics alongside standard metrics.

The evaluation includes:
- Standard metrics (diversity, contrastiveness, correlation, etc.)
- Conformal prediction set metrics (coverage, set size, coherence)
- Support for multiple conformal prediction methods (CHiQPM, APS, THR)
"""
import numpy as np
import sklearn
import torch
from torchcp.classification import Metrics
from tqdm import trange

from conformalPrediction.utils import get_predictions, get_score, calibrate_predictor
from conformalPrediction.eval_cp import get_logits_and_labels
from evaluation.Metrics.Contrastiveness import gmm_overlap_per_feature
from evaluation.Metrics.Correlation import get_correlation
from evaluation.Metrics.Dependence import compute_contribution_top_feature, compute_class_independence
from evaluation.Metrics.SetCoherence import get_set_coherence
from evaluation.Metrics.StructuralGrounding import get_structural_grounding_for_weight_matrix
from evaluation.diversity import MultiKCrossChannelMaxPooledSum
from evaluation.qpm_metrics import evaluateALLMetricsForComps
from evaluation.utils import get_metrics_for_model

def evaluate_ChiQPMMetrics(features_train,  outputs_train,  feature_maps_test,
                               outputs_test, linear_matrix,  labels_train, labels_test, features_test):
    """
    Evaluate CHiQPM model on all metrics including conformal prediction.
    
    Computes both standard QPM metrics and conformal prediction set metrics
    for the CHiQPM model.
    
    Args:
        features_train (Tensor): Training set features after pooling and normalization
        outputs_train (Tensor): Training set model outputs (logits)
        feature_maps_test (Tensor): Test set feature maps before pooling
        outputs_test (Tensor): Test set model outputs (logits)
        linear_matrix (Tensor): Weight matrix of the final linear layer
        labels_train (Tensor): Training set ground truth labels
        labels_test (Tensor): Test set ground truth labels
        features_test (Tensor): Test set features after pooling and normalization
    
    Returns:
        dict: Dictionary containing all computed metrics including:
            - Standard metrics (diversity, contrastiveness, correlation, etc.)
            - CP metrics for different methods and accuracy levels
    """
    cp_set_metrics = get_set_metrics(features_test, outputs_test,labels_test, linear_matrix)
    print("CP Set Metrics: ", cp_set_metrics)
    qpm_answer_dict = evaluateALLMetricsForComps(features_train,  outputs_train,  feature_maps_test,
                               outputs_test, linear_matrix,  labels_train, labels_test, features_test)
    qpm_answer_dict.update(cp_set_metrics)
    return qpm_answer_dict


def get_set_metrics(features_test, outputs_test,labels_test,  weight):
    """
    Compute conformal prediction set metrics for different methods and accuracy levels.
    
    Evaluates conformal prediction performance using multiple methods (CHiQPM, APS, THR)
    at various target accuracy levels (88%, 90%, 92.5%, 95%).
    
    Args:
        features_test (Tensor): Test set features after pooling and normalization
        outputs_test (Tensor): Test set model outputs (logits)
        labels_test (Tensor): Test set ground truth labels
        weight (Tensor): Weight matrix of the final linear layer
    
    Returns:
        dict: Nested dictionary with structure:
            {method: {accuracy_level: {metric_name: value}}}
            where metrics include coverage_rate, average_size, and optionally set_coherence
    
    Note:
        The data is split using 10 samples per class for calibration.
        Set coherence is only computed for CUB dataset (identified by test set size of 5794).
    """
    answer = {}
    # Split using 10 samples per class for calibration
    cal_logits, cal_labels, cal_features, test_logits, test_labels, test_features, test_indices = get_logits_and_labels(
        features_test, outputs_test, labels_test, 10,)
    for method in [ "CHiQPM", "APS", "THR",]:
        answer[method] = {}
        for acc in [.88, .9, .925, .95]:
            answer[method][acc] = {}

            predictor, needs_feats = get_score(method, weight)

            # Calibrate the predictor on the calibration set
            calibrate_predictor(cal_logits, cal_labels, acc, cal_features, predictor, needs_feats)

            # Get predictions on the test set
            prediction_sets = get_predictions(test_logits, predictor, test_features, needs_feats)
            metrics = Metrics()
            Coverage_rate = metrics("coverage_rate")(prediction_sets, test_labels)
            Average_size = metrics("average_size")(prediction_sets, test_labels)
            answer[method][acc]["Coverage_rate"] = Coverage_rate
            answer[method][acc]["Average_size"] = Average_size
            
            # Compute set coherence only for CUB dataset
            if len(features_test) == 5794:
                answer[method][acc]["SetCoherence"] = get_set_coherence(prediction_sets)
    return answer

def eval_model_on_all_chiqpm_metrics(model, test_loader, train_loader):
    """
    Evaluate a CHiQPM model on all metrics.
    
    This is the main entry point for evaluating a trained CHiQPM model.
    It extracts features and predictions from the model, then computes all metrics.
    
    Args:
        model: Trained CHiQPM PyTorch model
        test_loader: DataLoader for test data
        train_loader: DataLoader for training data
    
    Returns:
        dict: Dictionary containing all computed metrics
    """
    return get_metrics_for_model(train_loader, test_loader, model, evaluate_ChiQPMMetrics)
