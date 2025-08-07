# import os
# from PIL import Image
# from torch.utils.data import Dataset

# class ClassificationDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform

#         # Filter only valid class directories
#         self.classes = sorted([
#             d for d in os.listdir(root_dir)
#             if os.path.isdir(os.path.join(root_dir, d))
#         ])
#         self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

#         self.samples = []
#         for cls in self.classes:
#             cls_folder = os.path.join(root_dir, cls)
#             for fname in os.listdir(cls_folder):
#                 if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     rel_path = os.path.join(cls, fname)
#                     full_path = os.path.join(root_dir, rel_path)
#                     if os.path.exists(full_path):
#                         self.samples.append((rel_path, self.class_to_idx[cls]))
#                     else:
#                         print(f"Skipped missing file: {full_path}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         rel_path, label = self.samples[idx]
#         img_path = os.path.join(self.root_dir, rel_path)
#         try:
#             img = Image.open(img_path).convert("RGB")
#         except Exception as e:
#             print(f"Error loading image {img_path}: {e}")
#             raise e
#         if self.transform:
#             img = self.transform(img)
#         return img, label

# scripts/dataset_loader.py
import os
from torch.utils.data import DataLoader
from torchvision import datasets

def get_datasets(data_dir, transform_train, transform_eval):
    """
    Return ImageFolder datasets for train, val, and test using provided transforms.
    """
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=transform_eval)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=transform_eval)
    return train_ds, val_ds, test_ds

def get_loaders(train_ds, val_ds, test_ds, batch_size=32, num_workers=2):
    """
    Wrap datasets in DataLoaders.
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

# import torch
# from torchvision import transforms, datasets
# from torch.utils.data import DataLoader

# def get_dataloaders(data_dir="dataset", batch_size=32, img_size=224):
#     transform = transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
#     val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
#     test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, val_loader, test_loader, train_dataset.classes
