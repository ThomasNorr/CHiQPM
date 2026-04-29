# Create configs/paths.py
from pathlib import Path

DATASET_ROOT = Path.home() / "tmp/Datasets/CUB200/CUB_200_2011/images"
CROPPED_DATASET_ROOT = Path.home() / "tmp/Datasets/PPCUB200"
CROPPED_TRAINED_DATA = CROPPED_DATASET_ROOT / "CUB_200_2011/train_cropped"
REPRESENTATIVES_ROOT = Path.home() / "tmp/Datasets/CUB200_representatives"
CAL_DATA_ROOT = Path.home() / "tmp/CHiQPM_calibration"