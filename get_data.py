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

class DeepLabBirdCrop(object):
    """
    Crop bird using DeepLabV3 semantic segmentation with padding allowance
    """
    def __init__(self, padding=20, device=None):
        from torchvision.models import segmentation
        
        self.model = segmentation.deeplabv3_resnet50(pretrained=True)
        # self.model = self.model.to(self.device)
        self.model.eval()
        self.padding = padding
        
        self.deeplab_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, img):
        if hasattr(img, 'mode'):  # PIL Image
            input_tensor = self.deeplab_preprocess(img).unsqueeze(0)
            # input_tensor = input_tensor.to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
            
            # Bird class in Pascal VOC is class 3
            bird_mask = output.argmax(0) == 3
            
            if bird_mask.any():
                # Find bounding box of bird
                bird_pixels = torch.where(bird_mask)
                y_min, y_max = bird_pixels[0].min(), bird_pixels[0].max()
                x_min, x_max = bird_pixels[1].min(), bird_pixels[1].max()
                
                # Add padding allowance around the bird
                img_width, img_height = img.size
                x_min = max(0, int(x_min) - self.padding)
                y_min = max(0, int(y_min) - self.padding)
                x_max = min(img_width, int(x_max) + self.padding)
                y_max = min(img_height, int(y_max) + self.padding)
                
                return img.crop((x_min, y_min, x_max, y_max))
        
        return img
    
class CUB200DeepLabPreprocessed(CUB200Class):
    """CUB200 dataset using pre-processed DeepLab crops (fast loading)"""
    
    def __init__(self, train, transform, crop=False):
        # Override the root to use preprocessed DeepLab crops
        self.original_root = self.root
        self.root = Path.home() / "tmp/Datasets/CUB200_DeepLab"
        
        # Initialize with the new root (never use PPCUB200 crops for DeepLab)
        super().__init__(train, transform, crop=False)
    
    def __getitem__(self, idx):
        # Regular loading - images are already DeepLab cropped
        sample = self.data.iloc[idx]
        path = self.root / self.base_folder / sample.filepath
        target = sample.target - 1
        
        img = self.loader(path)
        img = self.transform(img)
        return img, target

def get_deeplab_preprocessed_data(img_size=224):
    """Get DataLoaders using pre-processed DeepLab cropped images"""
    
    # No DeepLab cropping in transforms since images are already cropped
    train_transform = get_augmentation(
        jitter=0.1, size=img_size, training=True, 
        random_center_crop=False, trivialAug=True, hflip=True, 
        normalize=normalize_params["CUB2011"]
    )
    test_transform = get_augmentation(
        jitter=0.1, size=img_size, training=False, 
        random_center_crop=False, trivialAug=True, hflip=True, 
        normalize=normalize_params["CUB2011"]
    )
    
    train_dataset = CUB200DeepLabPreprocessed(True, train_transform)
    test_dataset = CUB200DeepLabPreprocessed(False, test_transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=8
    )
    
    return train_loader, test_loader
    

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    import os

    # Get all images from the Summerr Tanager directory
    img_dir = Path.home() / "tmp" / "Datasets" / "CUB200" / "CUB_200_2011" / "images" / "140.Summer_Tanager"
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    image_files.sort()  # Sort for consistent ordering
    
    n_images = len(image_files)
    cols = 3  # 3 columns: Original, DeepLab, Center Crop
    rows = n_images
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    # Handle case where there's only one image
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    cropper = DeepLabBirdCrop(padding=20)
    center_crop_transform = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448)
    ])
    
    for i, img_file in enumerate(image_files):
        img_path = img_dir / img_file
        img = Image.open(img_path).convert("RGB")
        
        # DeepLab cropping
        deeplab_cropped = cropper(img)
        
        # Center cropping
        center_cropped = center_crop_transform(img)
        
        # Plot original
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original: {img_file}")
        axes[i, 0].axis('off')
        
        # Plot DeepLab cropped
        axes[i, 1].imshow(deeplab_cropped)
        axes[i, 1].set_title("DeepLab Cropped")
        axes[i, 1].axis('off')
        
        # Plot center cropped
        axes[i, 2].imshow(center_cropped)
        axes[i, 2].set_title("Center Cropped")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("all_summer_tanager_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Processed {n_images} images from 140.Summer_Tanager directory")