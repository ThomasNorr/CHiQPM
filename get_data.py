"""
Data Loading and Augmentation Module

This module handles dataset loading and preprocessing for various fine-grained
classification datasets including CUB-200-2011, Stanford Cars, Traveling Birds,
and ImageNet.

Supports:
- Multiple dataset types with custom augmentation strategies
- Optional cropping based on ground truth bounding boxes
- Configurable image sizes and augmentation parameters
- Lighting transformations for ImageNet (PCA-based color augmentation)

Example usage:
    train_loader, test_loader = get_data("CUB2011", crop=True, img_size=224)
"""
from pathlib import Path

import torch
import torchvision
from torchvision.transforms import transforms, TrivialAugmentWide

from configs.dataset_params import normalize_params
from dataset_classes.cub200 import CUB200Class
from dataset_classes.stanfordcars import StanfordCarsClass
from dataset_classes.travelingbirds import TravelingBirds


def get_data(dataset, crop = True, img_size=448):
    """
    Create data loaders for training and testing.
    
    Loads the specified dataset and creates PyTorch DataLoaders with appropriate
    augmentation and preprocessing. Automatically handles dataset-specific settings
    like normalization parameters and batch sizes.
    
    Args:
        dataset (str): Dataset name, one of ["CUB2011", "TravelingBirds", "StanfordCars", "ImageNet"]
        crop (bool): Whether to use cropped images based on ground truth bounding boxes.
                     Only applies to CUB2011 and TravelingBirds. Default: True
        img_size (int): Target image size for resizing/cropping. Default: 448
                        Note: ImageNet only supports 224
    
    Returns:
        tuple: (train_loader, test_loader) - PyTorch DataLoaders for training and testing
    
    Raises:
        NotImplementedError: If FGVCAircraft dataset is requested (not yet implemented)
        NotImplementedError: If ImageNet is used with img_size != 224
        ValueError: If invalid dataset name is provided
    
    Note:
        - Batch size is 16 for fine-grained datasets and 64 for ImageNet
        - ImageNet uses a specific augmentation strategy from the robustness package
        - Cropped images should be pre-computed following ProtoTree's preprocessing instructions
    """
    # Validate dataset name
    valid_datasets = ["CUB2011", "TravelingBirds", "StanfordCars", "ImageNet", "FGVCAircraft"]
    if dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset '{dataset}'. Must be one of {valid_datasets[:-1]}")
    
    batchsize = 16
    if dataset == "CUB2011":
        train_transform = get_augmentation(0.1, img_size, True,not crop, True, True, normalize_params["CUB2011"])
        test_transform = get_augmentation(0.1, img_size, False, not crop, True, True, normalize_params["CUB2011"])
        train_dataset = CUB200Class(True, train_transform, crop)
        test_dataset = CUB200Class(False, test_transform, crop)
    elif dataset == "TravelingBirds":
        train_transform = get_augmentation(0.1, img_size, True, not crop, True, True, normalize_params["TravelingBirds"])
        test_transform = get_augmentation(0.1, img_size, False, not crop, True, True, normalize_params["TravelingBirds"])
        train_dataset = TravelingBirds(True, train_transform, crop)
        test_dataset = TravelingBirds(False, test_transform, crop)

    elif dataset == "StanfordCars":
        train_transform = get_augmentation(0.1, img_size, True, True, True, True, normalize_params["StanfordCars"])
        test_transform = get_augmentation(0.1, img_size, False, True, True, True, normalize_params["StanfordCars"])
        train_dataset = StanfordCarsClass(True, train_transform)
        test_dataset = StanfordCarsClass(False, test_transform)
    elif dataset == "FGVCAircraft":
        raise NotImplementedError("FGVCAircraft dataset is not yet implemented")

    elif dataset == "ImageNet":
        # Defaults from the robustness package
        if img_size != 224:
            raise NotImplementedError("ImageNet is setup to only work with 224x224 images")
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            transforms.ToTensor(),
            Lighting(0.05, IMAGENET_PCA['eigval'],
                     IMAGENET_PCA['eigvec'])
        ])
        """
        Standard training data augmentation for ImageNet-scale datasets: Random crop,
        Random flip, Color Jitter, and Lighting Transform (see https://git.io/fhBOc)
        """
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        imgnet_root = Path.home()/ "tmp" /"Datasets"/ "imagenet"
        train_dataset = torchvision.datasets.ImageNet(root=imgnet_root, split='train',  transform=train_transform)
        test_dataset = torchvision.datasets.ImageNet(root=imgnet_root, split='val',  transform=test_transform)
        batchsize = 64

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=8)
    return train_loader, test_loader

def get_augmentation(jitter,  size,  training,  random_center_crop, trivialAug, hflip, normalize):
    """
    Create a composition of image augmentation transforms.
    
    Builds a torchvision transform pipeline with various augmentation options
    that can be enabled/disabled based on the arguments.
    
    Args:
        jitter (float): Amount of color jitter (brightness, contrast, saturation).
                        Set to 0 to disable.
        size (int): Target image size
        training (bool): If True, apply training augmentations (crop, flip, color jitter).
                        If False, only resize and center crop.
        random_center_crop (bool): If True, use random/center crop. If False, resize to exact size.
        trivialAug (bool): Whether to apply TrivialAugmentWide (AutoAugment variant)
        hflip (bool): Whether to apply random horizontal flip during training
        normalize (dict): Normalization parameters with 'mean' and 'std' tensors
    
    Returns:
        transforms.Compose: Composed transformation pipeline
    
    Note:
        The transform always ends with ToTensor() and Normalize() operations.
    """
    augmentation = []
    if random_center_crop:
        augmentation.append(transforms.Resize(size))
    else:
        augmentation.append(transforms.Resize((size, size)))
    if training:
        if random_center_crop:
                augmentation.append(transforms.RandomCrop(size, padding=4))
    else:
        if random_center_crop:
            augmentation.append(transforms.CenterCrop(size))
    if training:
        if hflip:
            augmentation.append(transforms.RandomHorizontalFlip())
        if jitter:
            augmentation.append(transforms.ColorJitter(jitter, jitter, jitter))
        if trivialAug:
            augmentation.append(TrivialAugmentWide())
    augmentation.append(transforms.ToTensor())
    augmentation.append(transforms.Normalize(**normalize))
    return transforms.Compose(augmentation)

class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        """
        Apply lighting augmentation to an image tensor.
        
        Args:
            img (Tensor): Input image tensor
            
        Returns:
            Tensor: Augmented image tensor
        """
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

# PCA statistics computed from ImageNet training set
IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}
