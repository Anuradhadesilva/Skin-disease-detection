# scripts/train_unet.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from unet_model import UNet

# Device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform_img=None, transform_mask=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask

def train_unet(data_dir, epochs=20, batch_size=4, lr=1e-4, checkpoint="models/unet.pth"):
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    transform_img = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    transform_mask = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = SegmentationDataset(img_dir, mask_dir, transform_img, transform_mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet().to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss / len(loader):.4f}")
        torch.save(model.state_dict(), checkpoint)

    print(f"U-Net model saved to {checkpoint}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="lesion_dataset")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--checkpoint", type=str, default="models/unet.pth")
    args = ap.parse_args()

    train_unet(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint=args.checkpoint
    )
