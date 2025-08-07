# import os
# import shutil
# import random
# import argparse
# from pathlib import Path

# CLASSES = ["eczema", "ringworm", "psoriasis", "normal", "others"]

# def split_dataset(raw_dir, out_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, force=False):
#     random.seed(seed)
#     raw_dir = Path(raw_dir)
#     out_dir = Path(out_dir)

#     if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
#         raise ValueError("Train/Val/Test ratios must sum to 1.")
#     if force and out_dir.exists():
#         shutil.rmtree(out_dir)
#     for split in ["train", "val", "test"]:
#         for cls in CLASSES:
#             (out_dir / split / cls).mkdir(parents=True, exist_ok=True)
#     for cls in CLASSES:
#         src_dir = raw_dir / cls
#         if not src_dir.exists():
#             print(f"Warning: Class folder {src_dir} does not exist!")
#             continue

#         images = [p for p in src_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
#         random.shuffle(images)

#         n_total = len(images)
#         n_train = int(train_ratio * n_total)
#         n_val = int(val_ratio * n_total)
#         n_test = n_total - n_train - n_val

#         train_imgs = images[:n_train]
#         val_imgs = images[n_train:n_train + n_val]
#         test_imgs = images[n_train + n_val:]

#         for p in train_imgs:
#             shutil.copy(p, out_dir / "train" / cls / p.name)
#         for p in val_imgs:
#             shutil.copy(p, out_dir / "val" / cls / p.name)
#         for p in test_imgs:
#             shutil.copy(p, out_dir / "test" / cls / p.name)

#         print(f"[{cls}] Total: {n_total}  Train: {len(train_imgs)}  Val: {len(val_imgs)}  Test: {len(test_imgs)}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--raw_dir", type=str, default="all_data", help="Path to the raw dataset")
#     parser.add_argument("--out_dir", type=str, default="dataset", help="Path to the output dataset")
#     parser.add_argument("--train", type=float, default=0.7, help="Training split ratio")
#     parser.add_argument("--val", type=float, default=0.15, help="Validation split ratio")
#     parser.add_argument("--test", type=float, default=0.15, help="Test split ratio")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed")
#     parser.add_argument("--force", action="store_true", help="Delete existing dataset folder before split")

#     args = parser.parse_args()
#     split_dataset(args.raw_dir, args.out_dir, args.train, args.val, args.test, args.seed, args.force)

# split segmented data

import os
import shutil
import random

data_dir = '/Users/desilvaanuradha/Documents/FYP/Ezcema-Detection/seg_dataset'

# Create output folders
train_img_dir = os.path.join(data_dir, 'train', 'images')
train_mask_dir = os.path.join(data_dir, 'train', 'masks')
val_img_dir = os.path.join(data_dir, 'val', 'images')
val_mask_dir = os.path.join(data_dir, 'val', 'masks')

for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
    os.makedirs(d, exist_ok=True)

# Get all files (exclude the train/val folders just in case)
all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

# Separate images and masks
masks = [f for f in all_files if f.endswith('_Segmentation.png')]
images = [f for f in all_files if (f.endswith('.jpg') or f.endswith('.png')) and not f.endswith('_Segmentation.png')]

# Extract basenames for matching
mask_basenames = set([f.replace('_Segmentation.png', '') for f in masks])
image_basenames = set([os.path.splitext(f)[0] for f in images])

# Keep only matching pairs
common_basenames = list(mask_basenames.intersection(image_basenames))

if not common_basenames:
    print("⚠️ No matching image-mask pairs found!")

# Create pairs
pairs = []
for base in common_basenames:
    img_file = base + '.jpg' if base + '.jpg' in images else base + '.png'
    mask_file = base + '_Segmentation.png'
    pairs.append((img_file, mask_file))

# Shuffle and split pairs into train/val
random.seed(42)
random.shuffle(pairs)

split_ratio = 0.8
split_idx = int(len(pairs) * split_ratio)
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]

def move_pairs(pairs, img_dir, mask_dir):
    for img, mask in pairs:
        shutil.move(os.path.join(data_dir, img), os.path.join(img_dir, img))
        shutil.move(os.path.join(data_dir, mask), os.path.join(mask_dir, mask))

move_pairs(train_pairs, train_img_dir, train_mask_dir)
move_pairs(val_pairs, val_img_dir, val_mask_dir)

print(f"✅ Moved {len(train_pairs)} train pairs and {len(val_pairs)} val pairs successfully.")
