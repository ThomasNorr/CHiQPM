"""
ImageNet-specific Training Configuration

This module is intended to provide ImageNet-specific optimizer and learning rate
scheduler configurations. Currently, these functions are not implemented as the
default configuration is handled in training/optim.py.

Note:
    When training on ImageNet with do_dense=False, the system uses pretrained
    models and skips these functions. For ImageNet training from scratch, these
    functions would need to be implemented following standard ImageNet training
    recipes (e.g., SGD with momentum, cosine annealing, etc.).
"""


def get_default_img_optimizer(model):
    """
    Get default optimizer for ImageNet training.
    
    Args:
        model: PyTorch model to optimize
    
    Returns:
        NotImplemented: This function is not yet implemented
    
    Raises:
        NotImplementedError: Always raised as this is a placeholder
    
    Note:
        For ImageNet training, the system currently relies on pretrained models
        or uses the configuration from training/optim.py when lr is None.
    """
    raise NotImplementedError("TODO: Implement get_default_img_optimizer")


def get_default_img_schedule(default_img_optimizer):
    """
    Get default learning rate scheduler for ImageNet training.
    
    Args:
        default_img_optimizer: Optimizer instance to schedule
    
    Returns:
        NotImplemented: This function is not yet implemented
    
    Raises:
        NotImplementedError: Always raised as this is a placeholder
    
    Note:
        For ImageNet training, the system currently relies on pretrained models
        or uses the configuration from training/optim.py when lr is None.
    """
    raise NotImplementedError("TODO: Implement get_default_img_schedule")