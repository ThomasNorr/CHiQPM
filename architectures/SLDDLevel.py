"""
SLDD Level Module

This module implements the SLDD (Sparse Linear Decomposition with Diversity) layer,
which performs feature normalization followed by linear classification. The layer
maintains learnable normalization parameters and supports optional ReLU activation.
"""
import torch.nn


class SLDDLevel(torch.nn.Module):
    """
    SLDD classification layer with feature normalization.
    
    This layer normalizes input features using learnable mean and standard deviation,
    optionally applies ReLU, then performs linear classification. The normalization
    helps stabilize the feature values and improve interpretability.
    
    Attributes:
        selection (Tensor): Buffer storing indices of selected features (not trainable)
        mean (Parameter): Learnable mean for normalization
        std (Parameter): Learnable standard deviation for normalization
        layer (nn.Linear): Linear classification layer
        relu (bool): Whether to apply ReLU to normalized features
    """
    def __init__(self, selection, weight_at_selection,mean, std, bias=None, retrain_normalisation=True, relu=False):
        """
        Initialize the SLDD layer.
        
        Args:
            selection (array-like): Indices of selected features
            weight_at_selection (Tensor): Weight matrix of shape (n_classes, n_features)
            mean (Tensor): Initial mean values for normalization
            std (Tensor): Initial standard deviation values for normalization
            bias (Tensor, optional): Bias values for the linear layer. If None, no bias is used.
            retrain_normalisation (bool): Whether normalization parameters should be trainable
            relu (bool): Whether to apply ReLU activation to normalized features
        
        Note:
            The mean and std can be either full-size or already selected. If full-size,
            they will be automatically indexed by the selection.
        """
        super().__init__()
        self.register_buffer('selection', torch.tensor(selection, dtype=torch.long))
        num_classes,        n_features = weight_at_selection.shape
        selected_mean = mean
        selected_std = std
        # Handle both pre-selected and full-size mean/std
        if len(selected_mean) != len(selection):
            selected_mean = selected_mean[selection]
            selected_std = selected_std[selection]
        self.mean = torch.nn.Parameter(selected_mean, requires_grad=retrain_normalisation)
        self.std = torch.nn.Parameter(selected_std, requires_grad=retrain_normalisation)
        if bias is not None:
            self.layer = torch.nn.Linear(n_features, num_classes)
            self.layer.bias = torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.layer = torch.nn.Linear(n_features, num_classes, bias=False)
        self.layer.weight = torch.nn.Parameter(weight_at_selection, requires_grad=False)
        self.relu = relu

    @property
    def weight(self):
        """Get the weight matrix of the linear layer."""
        return self.layer.weight

    @property
    def bias(self):
        """
        Get the bias of the linear layer.
        
        Returns:
            Tensor: Bias values, or zeros if no bias exists
        """
        if self.layer.bias is None:
            return torch.zeros(self.layer.out_features)
        else:
            return self.layer.bias


    def forward(self, input):
        """
        Forward pass through the SLDD layer.
        
        Args:
            input (Tensor): Input features of shape (batch_size, n_features)
        
        Returns:
            tuple: (outputs, normalized_features)
                - outputs: Classification logits of shape (batch_size, n_classes)
                - normalized_features: Normalized (and optionally ReLU'd) features
        
        Note:
            The standard deviation is clamped to a minimum of 1e-6 to avoid division by zero.
        """
        input = (input - self.mean) / torch.clamp(self.std, min=1e-6)
        if self.relu:
            input = torch.nn.functional.relu(input)
        return self.layer(input), input
