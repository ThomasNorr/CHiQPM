"""
Training and Testing Module for CHiQPM

This module provides the core training and testing functions used during
the dense model training phase. It handles:
- Training loop with multiple loss components
- Validation/testing without gradient updates
- Accuracy computation
- Progress logging with tqdm
"""
import torch
from tqdm import tqdm

from FeatureGroundingLoss import get_FeatureGroundingLoss
from training.utils import VariableLossLogPrinter


def get_acc(outputs, targets):
    """
    Calculate classification accuracy.
    
    Args:
        outputs: Tensor of shape (batch_size, n_classes) containing logits
        targets: Tensor of shape (batch_size,) containing target class indices
    
    Returns:
        float: Accuracy as a percentage (0-100)
    """
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return correct / total * 100



def train(model, train_loader, optimizer, fdl, lambda_feat, epoch):
    """
    Train the model for one epoch.
    
    Performs a single training epoch with combined loss functions including
    cross-entropy, feature diversity loss, and optionally feature grounding loss.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        optimizer: PyTorch optimizer
        fdl: FeatureDiversityLoss instance for promoting feature diversity
        lambda_feat: Scaling factor for Feature Grounding Loss (0 disables it)
        epoch: Current epoch number for logging
    
    Returns:
        model: Trained model (in training mode)
    
    Note:
        The function automatically detects CUDA availability and moves data accordingly.
    """
    model.train()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    VariableLossPrinter = VariableLossLogPrinter()
    model = model.to(device)
    iterator = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data, target) in iterator:
        on_device = data.to(device)
        target_on_device = target.to(device)

        output, feature_maps, final_features = model(on_device, with_feature_maps=True,with_final_features=True)
        loss = torch.nn.functional.cross_entropy(output, target_on_device)

        fdl_loss = fdl(feature_maps, output)
        total_loss = loss + fdl_loss

        if lambda_feat > 0:
            fgloss = get_FeatureGroundingLoss(final_features, target, model.linear.weight,) * lambda_feat
            total_loss = total_loss + fgloss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        acc = get_acc(output, target_on_device)
        VariableLossPrinter.log_loss("Train Acc", acc, on_device.size(0))
        VariableLossPrinter.log_loss("CE-Loss", loss.item(), on_device.size(0))
        VariableLossPrinter.log_loss("FDL", fdl_loss.item(), on_device.size(0))
        if lambda_feat > 0:
            VariableLossPrinter.log_loss("FGL", fgloss.item(), on_device.size(0))
        VariableLossPrinter.log_loss("Total-Loss", total_loss.item(), on_device.size(0))
        iterator.set_description(f"Train Epoch:{epoch} Metrics: {VariableLossPrinter.get_loss_string()}")
    print("Trained model for one epoch ",  epoch," with lr group 0: ", optimizer.param_groups[0]["lr"])
    return model


def test(model, test_loader, epoch):
    """
    Evaluate the model on test data.
    
    Runs the model in evaluation mode (no gradient computation) and computes
    accuracy and cross-entropy loss on the test set.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        epoch: Current epoch number for logging
    
    Returns:
        None. Prints test metrics to console.
    
    Note:
        This function does not return metrics but prints them via tqdm progress bar.
    """
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    VariableLossPrinter = VariableLossLogPrinter()
    iterator = tqdm(enumerate(test_loader), total=len(test_loader))
    with torch.no_grad():
        for batch_idx, (data, target) in iterator:
            on_device = data.to(device)
            target_on_device = target.to(device)
            output, feature_maps = model(on_device, with_feature_maps=True)
            loss = torch.nn.functional.cross_entropy(output, target_on_device)
            acc = get_acc(output, target_on_device)
            VariableLossPrinter.log_loss("Test Acc", acc, on_device.size(0))
            VariableLossPrinter.log_loss("CE-Loss", loss.item(), on_device.size(0))
            iterator.set_description(f"Test Epoch:{epoch} Metrics: {VariableLossPrinter.get_loss_string()}")
