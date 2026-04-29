import torch
import shutil
from evaluation.load_model import get_args_for_loading_model, load_model
from get_data import get_data
from evaluation.diversity import MultiKCrossChannelMaxPooledSum
from configs.demo_paths import DATASET_ROOT, REPRESENTATIVES_ROOT, CROPPED_TRAINED_DATA


def find_representative_images(model, train_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    
    linear_matrix = model.linear.weight
    
    # Dictionary to store SID@5 values per class
    # Key: class_name (e.g., "001.Black_footed_Albatross")
    # Value: list of tuples (sid5_score, image_path, is_correct)
    class_sid_dict = {}
    
    global_idx = 0  # Track actual position in dataset
    
    with torch.no_grad():
        for _, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            # Get predictions and features
            output, feature_maps, final_features = model(
                data, 
                with_feature_maps=True, 
                with_final_features=True
            )
            
            _, predicted = output.max(1)
            
            for i in range(data.size(0)):
                # Get image path from dataset
                image_path = train_loader.dataset.data.iloc[global_idx].filepath
                class_name = image_path.split('/')[0]  # e.g., "001.Black_footed_Albatross"
                
                # Check if correctly classified
                is_correct = (predicted[i] == target[i]).item()
                
                # Calculate SID@5 for this single image
                localizer = MultiKCrossChannelMaxPooledSum(
                    range(1, 6), 
                    linear_matrix, 
                    None, 
                    func="SumNMax"
                )
                
                # Add batch dimension for single image
                localizer(
                    output[i:i+1],  # Shape: [1, num_classes]
                    feature_maps[i:i+1]  # Shape: [1, features, H, W]
                )
                
                # Get SID@5 score (index [0][4] for k=5)
                sid5_score = localizer.get_result()[0][4].item()
                
                # Store in dictionary
                if class_name not in class_sid_dict:
                    class_sid_dict[class_name] = []
                
                class_sid_dict[class_name].append({
                    'sid5_score': sid5_score,
                    'image_path': image_path,
                    'is_correct': is_correct
                })
                
                global_idx += 1
    
    # Select best representative for each class
    representatives = {}
    
    for class_name, candidates in class_sid_dict.items():
        correct_candidates = [c for c in candidates if c['is_correct']]
        
        if not correct_candidates:
            print(f"Warning: No correctly classified images for class {class_name}")
            continue
        
        best = max(correct_candidates, key=lambda x: x['sid5_score'])
        
        representatives[class_name] = {
            'image_path': best['image_path'],
            'sid5_score': best['sid5_score']
        }
    
    return representatives

def save_representatives(representatives, original_root, output_root, cropped_root):
    """Save both original and cropped representative images"""
    output_root.mkdir(exist_ok=True, parents=True)
    
    for class_name, rep_info in representatives.items():
        image_path = rep_info['image_path']
        filename = image_path.split('/')[-1]
        filename_base = filename.rsplit('.', 1)[0]  # Remove extension
        filename_ext = filename.rsplit('.', 1)[1]   # Get extension
        
        # src_original = original_root / "CUB_200_2011/images" / image_path
        # src_cropped = cropped_root / "CUB_200_2011/train_cropped" / image_path
        src_original = original_root / image_path
        src_cropped = cropped_root / image_path


        output_class_path = output_root / class_name
        output_class_path.mkdir(exist_ok=True, parents=True)

        dst_original = output_class_path / filename
        dst_cropped = output_class_path / f"{filename_base}_cropped.{filename_ext}"
        
        # Copy original image
        if src_original.exists():
            shutil.copy2(src_original, dst_original)
            print(f"Saved {class_name}: {filename} (original)")
        else:
            print(f"Error: Original not found: {src_original}")
        
        # Copy cropped image
        if src_cropped.exists():
            shutil.copy2(src_cropped, dst_cropped)
            print(f"Saved {class_name}: {filename_base}_cropped.{filename_ext} (cropped, SID@5={rep_info['sid5_score']:.4f})")
        else:
            print(f"Error: Cropped not found: {src_cropped}")


if __name__ == "__main__":
    args = get_args_for_loading_model()
    
    # Load model using args
    model, folder = load_model(
        args.dataset, 
        args.arch, 
        args.seed, 
        args.model_type,
        args.cropGT, 
        args.n_features, 
        args.n_per_class, 
        args.img_size, 
        args.reduced_strides, 
        args.folder
    )
    
    train_loader, _ = get_data(args.dataset, crop=args.cropGT, img_size=args.img_size)
    
    representatives = find_representative_images(model, train_loader)

    # original_root = DATASET_ROOT.parent.parent  # Go up to CUB200 level
    save_representatives(representatives, DATASET_ROOT, REPRESENTATIVES_ROOT, CROPPED_TRAINED_DATA)
