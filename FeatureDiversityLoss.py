"""
Feature Diversity Loss Module

This module implements the Feature Diversity Loss (FDL), which encourages
different features to activate in different spatial regions of feature maps.
This promotes diversity and reduces redundancy in the learned features.

The loss works by:
1. Identifying relevant features for each predicted class
2. Computing softmax-normalized feature maps
3. Maximizing the diversity of feature activations across spatial locations

Usage:
    fdl = FeatureDiversityLoss(0.196, model.linear)
    # During training:
    loss = fdl(feature_maps, outputs)

Reference:
    This loss is described in the CHiQPM/QPM papers for promoting interpretable
    feature diversity in deep neural networks.
"""
import torch
from torch import nn


class FeatureDiversityLoss(nn.Module):
    """
    Feature Diversity Loss for promoting spatially diverse feature activations.
    
    Attributes:
        scaling_factor (float): Multiplier for the loss value
        linearLayer (nn.Linear): The final linear classification layer
    """
    def __init__(self, scaling_factor, linear):
        """
        Initialize the Feature Diversity Loss.
        
        Args:
            scaling_factor (float): Multiplier for the loss (typically 0.196 as in paper)
            linear (nn.Linear): The final linear classification layer of the model
        """
        super().__init__()
        self.scaling_factor = scaling_factor #* 0
        print("Scaling Factor: ", self.scaling_factor)
        self.linearLayer = linear

    def initialize(self, linearLayer):
        """
        Update the linear layer reference (used when model structure changes).
        
        Args:
            linearLayer (nn.Linear): New linear layer to use
        """
        self.linearLayer = linearLayer

    def get_weights(self, outputs):
        """
        Extract relevant feature weights for the predicted classes.
        
        Args:
            outputs (Tensor): Model outputs (logits) of shape (batch_size, n_classes)
        
        Returns:
            Tensor: Absolute weights for top predicted classes, shape (batch_size, n_features)
        """
        weight_matrix = self.linearLayer.weight
        weight_matrix = torch.abs(weight_matrix)
        top_classes = torch.argmax(outputs, dim=1)
        relevant_weights = weight_matrix[top_classes]
        return relevant_weights

    def forward(self, feature_maps, outputs):
        """
        Compute the Feature Diversity Loss.
        
        The loss encourages features to activate in diverse spatial locations by
        maximizing the sum of peak activations across relevant features.
        
        Args:
            feature_maps (Tensor): Feature maps from the last conv layer,
                                   shape (batch_size, n_features, height, width)
            outputs (Tensor): Model outputs (logits), shape (batch_size, n_classes)
        
        Returns:
            Tensor: Scalar diversity loss value (negative for maximization)
        
        Note:
            The loss is negated because we want to maximize diversity, but PyTorch
            minimizes loss values.
        """
        relevant_weights = self.get_weights(outputs)
        relevant_weights = norm_vector(relevant_weights)
        feature_maps = preserve_avg_func(feature_maps)
        flattened_feature_maps = feature_maps.flatten(2)
        batch, features, map_size = flattened_feature_maps.size()
        relevant_feature_maps = flattened_feature_maps * relevant_weights[..., None]
        diversity_loss = torch.sum(
            torch.amax(relevant_feature_maps, dim=1))
        return -diversity_loss / batch * self.scaling_factor


def norm_vector(x):
    """
    L2-normalize vectors along the feature dimension.
    
    Args:
        x (Tensor): Input tensor of shape (batch_size, n_features)
    
    Returns:
        Tensor: L2-normalized tensor, same shape as input
    """
    return x / (torch.norm(x, dim=1) + 1e-5)[:, None]


def preserve_avg_func(x):
    """
    Apply softmax normalization while preserving relative average activations.
    
    This function normalizes feature maps using softmax across spatial locations
    while scaling to preserve the relative importance of different features based
    on their average activations.
    
    Args:
        x (Tensor): Feature maps of shape (batch_size, n_features, height, width)
    
    Returns:
        Tensor: Normalized and scaled feature maps, same shape as input
    """
    avgs = torch.mean(x, dim=[2, 3])
    max_avgs = torch.max(avgs, dim=1)[0]
    scaling_factor = avgs / torch.clamp(max_avgs[..., None], min=1e-6)
    softmaxed_maps = softmax_feature_maps(x)
    scaled_maps = softmaxed_maps * scaling_factor[..., None, None]
    return scaled_maps


def softmax_feature_maps(x):
    """
    Apply softmax normalization across spatial locations of feature maps.
    
    Args:
        x (Tensor): Feature maps of shape (batch_size, n_features, height, width)
    
    Returns:
        Tensor: Softmax-normalized feature maps, same shape as input
    """
    return torch.softmax(x.reshape(x.size(0), x.size(1), -1), 2).view_as(x)

