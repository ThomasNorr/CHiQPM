"""
Final Layer Module

This module implements the final classification layer used in CHiQPM and related models.
It handles feature pooling, dropout, and linear classification, with support for
feature selection and SLDD-specific transformations.
"""
import torch
from torch import nn

from architectures.SLDDLevel import SLDDLevel


class FinalLayer():
    """
    Final layer mixin class for classification models.
    
    This class provides the final processing stage of the model, including:
    - Adaptive average pooling to convert feature maps to vectors
    - Optional feature selection
    - Dropout for regularization
    - Linear classification layer
    - Support for SLDD-specific feature normalization
    
    Attributes:
        avgpool (nn.AdaptiveAvgPool2d): Pooling layer to reduce feature maps to vectors
        linear (nn.Linear): Classification layer
        featureDropout (nn.Dropout): Dropout layer for regularization
        selection (Tensor or None): Indices of selected features (None means use all)
    """
    def __init__(self, num_classes,  n_features):
        """
        Initialize the final layer.
        
        Args:
            num_classes (int): Number of output classes
            n_features (int): Number of input features
        """
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(n_features, num_classes)
        self.featureDropout = torch.nn.Dropout(0.2)
        self.selection = None

    def transform_output(self,  feature_maps, with_feature_maps,
                         with_final_features):
        """
        Transform feature maps to classification outputs with optional intermediate values.
        
        This method processes feature maps through pooling, dropout, and classification,
        with flexible output based on what intermediate values are requested.
        
        Args:
            feature_maps (Tensor): Feature maps from the last convolutional layer,
                                   shape (batch_size, n_features, height, width)
            with_feature_maps (bool): If True, include feature maps in the output
            with_final_features (bool): If True, include final features (after pooling) in output
        
        Returns:
            Tensor or tuple: Depending on flags, returns one of:
                - outputs only (single Tensor)
                - (outputs, feature_maps) if with_feature_maps=True
                - (outputs, feature_maps, final_features) if both flags=True
        
        Note:
            If feature selection is active (self.selection is not None), only selected
            features are used. For SLDD models, the linear layer returns both outputs
            and normalized features as a tuple.
        """
        if self.selection is not None:
            feature_maps = feature_maps[:, self.selection]
        x = self.avgpool(feature_maps)
        pre_out = torch.flatten(x, 1)
        final_features = self.featureDropout(pre_out)
        final = self.linear(final_features)
        # In case of SLDD, features are also returned, as they are scaled for normalization
        if isinstance(final, tuple):
            final, final_features = final
        final = [final]
        if with_feature_maps:
            final.append(feature_maps)
        if with_final_features:
            final.append(final_features)
        if len(final) == 1:
            final = final[0]
        return final


    def set_model_sldd(self, selection, weight, mean, std, bias = None, retrain_normalisation = True, relu=False):
        """
        Convert the model to SLDD (Sparse Linear Decomposition) configuration.
        
        This method replaces the standard linear layer with an SLDD-specific layer
        that includes feature normalization and optional ReLU activation.
        
        Args:
            selection (array-like): Indices of selected features
            weight (Tensor): Weight matrix for the SLDD layer
            mean (Tensor): Mean values for feature normalization
            std (Tensor): Standard deviation values for feature normalization
            bias (Tensor, optional): Bias values for the linear layer. If None, no bias is used.
            retrain_normalisation (bool): Whether to allow retraining of normalization parameters
            relu (bool): Whether to apply ReLU activation to normalized features
        
        Note:
            SLDD uses lower dropout (0.1) compared to the default (0.2).
        """
        self.selection = selection
        self.linear = SLDDLevel(selection, weight, mean, std, bias, retrain_normalisation, relu)
        self.featureDropout = torch.nn.Dropout(0.1)