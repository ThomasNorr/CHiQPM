"""
QPM Feature Selection and Assignment Module

This module implements the core Quadratic Programming-based feature selection
and assignment algorithm for QPM (Quantized Prototype Model). It computes and
solves a constrained optimization problem to select the most discriminative and
diverse features.

The QPM problem optimizes:
- Feature-class correlation (A matrix)
- Feature diversity via cosine similarity (R matrix)
- Spatial locality bias (B vector)

Reference:
    Norrenbrock et al., "QPM: Discrete Optimization for Globally Interpretable
    Image Classification", ICLR 2025
"""
import os
import sys

import numpy as np
import torch.utils.data

from sparsification.qpm.qpm_solving import solve_qp
from sparsification.qpm_constants.compute_A import compute_feat_class_corr_matrix
from sparsification.qpm_constants.compute_B import compute_locality_bias
from sparsification.qpm_constants.compute_R import compute_cos_sim_matrix
from sparsification.utils import get_feature_loaders

def compute_qpm_feature_selection_and_assignment(model, train_loader, test_loader, log_dir, n_classes, seed, n_features, per_class, rho=0):
    """
    Compute QPM feature selection and class assignment using quadratic programming.
    
    This function performs the complete QPM feature selection pipeline:
    1. Extract features from the dense model
    2. Compute correlation matrix A (feature-class relationships)
    3. Compute diversity matrix R (feature-feature similarity)
    4. Compute locality bias B (spatial grounding)
    5. Solve QP problem to select features and assign to classes
    
    Results are cached to disk to avoid recomputation.
    
    Args:
        model: Trained dense PyTorch model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        log_dir (Path): Directory for saving computed matrices and results
        n_classes (int): Number of classes in the dataset
        seed (int): Random seed for feature loader
        n_features (int): Total number of features to select
        per_class (int): Number of features to assign to each class
        rho (float): Regularization parameter for hierarchical constraints (default: 0)
    
    Returns:
        tuple: (feature_sel, weight, mean, std)
            - feature_sel: Indices of selected features
            - weight: Binary weight matrix (n_classes, n_features) indicating feature-class assignments
            - mean: Mean values for feature normalization
            - std: Std values for feature normalization
    
    Note:
        The function caches intermediate results (A, R, B matrices) and final results
        (feature_sel, weight) to disk. On subsequent runs, cached values are loaded
        instead of recomputing.
        
        If no GPU is available after solving QP, the function exits as further
        processing requires GPU acceleration.
    """
    feature_loaders, metadata, device,args =  get_feature_loaders(seed, log_dir,train_loader, test_loader, model, n_classes, )
    full_train_dataset = torch.utils.data.ConcatDataset([feature_loaders['train'].dataset, feature_loaders['val'].dataset])
    full_train_dataset_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=feature_loaders['train'].batch_size, shuffle=False, # Shuffling does not matter here
                                                            num_workers=feature_loaders['train'].num_workers)
    save_folder = log_dir / "qpm_constants_saved"
    save_folder.mkdir(parents=True, exist_ok=True)
    
    # Load or compute A, R, B matrices
    if  os.path.exists(save_folder / "A.pt") and os.path.exists(save_folder / "R.pt") and os.path.exists(save_folder / "B.pt"):
        a_matrix = torch.load(save_folder / "A.pt",map_location=torch.device('cpu') )
        r_matrix = torch.load(save_folder / "R.pt",map_location=torch.device('cpu') )
        b = torch.load(save_folder / "B.pt",map_location=torch.device('cpu') )
    else:
        # A: Feature-class correlation matrix
        a_matrix = compute_feat_class_corr_matrix(full_train_dataset_loader)
        a_matrix =  a_matrix / np.abs(a_matrix).max()
        # R: Feature similarity (cosine) matrix  
        r_matrix = compute_cos_sim_matrix(a_matrix)
        r_matrix = r_matrix / r_matrix.abs().max()
        # B: Locality bias vector
        b = compute_locality_bias(train_loader, model)

        torch.save(a_matrix, save_folder / "A.pt")
        torch.save(r_matrix, save_folder / "R.pt")
        torch.save(b, save_folder / "B.pt")
    
    # Only keep upper triangle and non-negative values for R
    r_matrix = torch.triu(torch.tensor(r_matrix))
    r_matrix[r_matrix < 0] = 0
    
    qpm_key  = f"{n_features}_{per_class}"
    res_folder = save_folder if rho == 0 else save_folder / f"rho_{rho}"
    res_folder.mkdir(parents=True, exist_ok=True)
    
    # Load or solve QP problem
    if  os.path.exists(res_folder / f"{qpm_key}_sel.pt") and os.path.exists(res_folder / f"{qpm_key}_weight.pt"):
        feature_sel = torch.load(res_folder / f"{qpm_key}_sel.pt",map_location=torch.device('cpu') )
        weight = torch.load(res_folder / f"{qpm_key}_weight.pt",map_location=torch.device('cpu') )
    else:
        feature_sel, weight = solve_qp(np.array(a_matrix),np.array(r_matrix), np.array(b), n_features, per_class, save_folder=save_folder, rho = rho)
        torch.save(feature_sel, res_folder / f"{qpm_key}_sel.pt")
        torch.save(weight, res_folder / f"{qpm_key}_weight.pt")
        if not torch.cuda.is_available():
            print("No GPU available, returning")
            sys.exit(0)
    mean, std = metadata["X"]['mean'], metadata["X"]['std']
    return feature_sel, weight.float(),  mean, std


