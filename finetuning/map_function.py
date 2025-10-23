"""
Model Finetuning Dispatcher Module

This module provides a unified interface for finetuning different interpretable
model variants (CHiQPM, QPM, Q-SENN, SLDD) after the initial dense model training.

Each model type has its own specific finetuning procedure:
- CHiQPM: Hierarchical feature selection with conformal prediction
- QPM: Quadratic programming-based feature selection
- Q-SENN: Quantized self-explaining neural network
- SLDD: Sparse linear decomposition
"""
from finetuning.chiqpm import finetune_chiqpm
from finetuning.qpm import finetune_qpm
from finetuning.qsenn import finetune_qsenn
from finetuning.sldd import finetune_sldd


def finetune(key, model, train_loader, test_loader, log_dir, n_classes, seed, beta, optimization_schedule, per_class, n_features):
    """
    Dispatch to the appropriate finetuning function based on model type.
    
    This function acts as a router that calls the appropriate finetuning procedure
    based on the model type specified. Each model variant has different feature
    selection and assignment strategies.
    
    Args:
        key (str): Model type identifier, one of ["chiqpm", "qpm", "qsenn", "sldd"]
        model: Trained dense PyTorch model to finetune
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        log_dir (Path): Directory for saving intermediate results and logs
        n_classes (int): Number of classes in the dataset
        seed (int): Random seed for reproducibility
        beta (float): Scaling factor for feature diversity loss
        optimization_schedule: Learning rate schedule configuration
        per_class (int): Number of features to assign per class
        n_features (int): Total number of features to select
    
    Returns:
        PyTorch model: Finetuned sparse model with selected features
    
    Raises:
        ValueError: If an unknown model type key is provided
    
    Note:
        The model is set to eval mode before finetuning begins.
        CHiQPM uses additional hierarchical parameters (3, 30) for tree construction.
    """
    model.eval()
    if key == 'sldd':
        return finetune_sldd(model, train_loader, test_loader, log_dir, n_classes, seed, beta, optimization_schedule,per_class, n_features)
    elif key == 'qsenn':
        return finetune_qsenn(model, train_loader, test_loader, log_dir, n_classes, seed, beta, optimization_schedule,n_features,per_class, )
    elif key == "qpm":
        return finetune_qpm(model, train_loader, test_loader, log_dir, n_classes, seed, beta, optimization_schedule,
                            n_features, per_class, )
    elif key == "chiqpm":
        return finetune_chiqpm(model, train_loader, test_loader, log_dir, n_classes, seed, beta, optimization_schedule,
                            n_features, per_class,(3, 30) )
    else:
        raise ValueError(f"Unknown Model key {key}")