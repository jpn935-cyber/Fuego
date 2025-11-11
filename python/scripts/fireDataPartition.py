import os
import shutil
import random
from pathlib import Path

# ====== CONFIGURATION ======
BASE_DIR = Path(r"D:\Fuego\dataset")
OUTPUT_DIR = Path(r"D:\Fuego\split_dataset")

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

random.seed(42)  # for reproducibility

# ====== FUNCTION TO SPLIT AND COPY ======
def split_and_copy(class_dirs, output_root):
    for class_name, img_dir in class_dirs.items():
        img_paths = list(Path(img_dir).glob("*.png")) + list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.jpeg"))
        random.shuffle(img_paths)

        n_total = len(img_paths)
        n_train = int(SPLIT_RATIOS["train"] * n_total)
        n_val = int(SPLIT_RATIOS["val"] * n_total)

        splits = {
            "train": img_paths[:n_train],
            "val": img_paths[n_train:n_train + n_val],
            "test": img_paths[n_train + n_val:]
        }

        for split_name, files in splits.items():
            split_dir = output_root / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, split_dir / f.name)

        print(f"✅ {class_name}: {n_total} images split into train/val/test.")

# ====== FIRE AND SMOKE DATASET ======
fire_and_smoke_img_dir = BASE_DIR / "fireAndSmoke" / "DataCluster Fire And Smoke Sample" / "DataCluster Fire And Smoke Sample"
# Assuming XML labels are not used right now — if you want to use them later, we can handle them separately.

fire_and_smoke_classes = {
    "fire_smoke": fire_and_smoke_img_dir
}

split_and_copy(fire_and_smoke_classes, OUTPUT_DIR / "fireAndSmoke_split")

# # ====== FIRE DATASET (fire vs non-fire) ======
# fire_dataset_base = BASE_DIR / "fireDataset" / "fire_dataset"
# fire_dataset_classes = {
#     "fire": fire_dataset_base / "fire_images",
#     "normal": fire_dataset_base / "non_fire_images"
# }

# split_and_copy(fire_dataset_classes, OUTPUT_DIR / "fireDataset_split")

# print("\n🎉 Done! Your split datasets are in:", OUTPUT_DIR)