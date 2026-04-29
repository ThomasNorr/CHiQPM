"""
Run this once to generate calibration data for the demo.
Usage: python prepare_calibration_data.py
"""
import torch
from pathlib import Path
from evaluation.load_model import get_args_for_loading_model, load_model
from get_data import get_data
from visualization.utils import get_feats_logits_labels
from conformalPrediction.eval_cp import get_logits_and_labels

from configs.demo_paths import CAL_DATA_ROOT

def prepare_calibration_data():
    args = get_args_for_loading_model()
    train_loader, test_loader = get_data(args.dataset, crop=args.cropGT, img_size=args.img_size)
    train_loader.dataset.transform = test_loader.dataset.transform
    
    model, folder = load_model(args.dataset, args.arch, args.seed, args.model_type, 
                               args.cropGT, args.n_features, args.n_per_class, 
                               args.img_size, args.reduced_strides, args.folder)
    
    # Get features and logits from test set
    features_test, outputs_test, labels_test = get_feats_logits_labels(model, test_loader)
    
    # Split into calibration and test sets (10% calibration)
    cal_logits, cal_labels, cal_features, _, _, _, _ = get_logits_and_labels(
        features_test, outputs_test, labels_test, 10
    )
    
    # Save calibration data
    save_path = CAL_DATA_ROOT / f"seed_{args.seed}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'cal_logits': cal_logits,
        'cal_labels': cal_labels,
        'cal_features': cal_features,
        'args': vars(args)  # Save args for verification
    }, save_path / "calibration_data.pt")
    
    print(f"Calibration data saved to {save_path}")
    print(f"Calibration set size: {len(cal_logits)}")

if __name__ == '__main__':
    prepare_calibration_data()