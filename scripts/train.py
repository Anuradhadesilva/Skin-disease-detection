# scripts/train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from cbam import CBAM
from torchvision import models, transforms
from dataset_loader import get_datasets, get_loaders
from utils import plot_curves, save_checkpoint


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: Apple MPS GPU")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

def build_transforms(img_size=224, augment=True):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        transform_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    transform_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # ss
    model.features[4] = nn.Sequential(
    model.features[4],
    CBAM(channel=model.features[4][0].out_channels)
)
    return transform_train, transform_eval

def train(
    data_dir="dataset",
    epochs=20,
    batch_size=8,
    lr=1e-4,
    img_size=224,
    checkpoint_path="models/efficientnet_best.pth",
    freeze_backbone=False,
):
    # transform
    t_train, t_eval = build_transforms(img_size=img_size, augment=True)

    # Datasets & loaders
    train_ds, val_ds, test_ds = get_datasets(data_dir, t_train, t_eval)
    train_loader, val_loader, _ = get_loaders(train_ds, val_ds, test_ds, batch_size=batch_size)
    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Detected classes: {class_names}")


    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    model.to(device)

    # Loss / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_acc = 0.0
    train_losses, val_losses, val_accuracies = [], [], []
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        correct, total = 0, 0
        running_val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_loss = running_val_loss / len(val_loader)
        val_acc = correct / total if total else 0.0
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch}/{epochs} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            save_checkpoint(model, checkpoint_path)
            best_val_acc = val_acc
            print(f"  ✔ Saved best model (val_acc={val_acc:.4f})")

    plot_curves(train_losses, val_losses, val_accuracies, out_path="training_curves.png")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {checkpoint_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--checkpoint", type=str, default="models/efficientnet_best.pth")
    ap.add_argument("--freeze_backbone", action="store_true")
    args = ap.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        checkpoint_path=args.checkpoint,
        freeze_backbone=args.freeze_backbone,
    )

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models
# from dataset_loader import get_dataloaders
# from utils import plot_curves
# import os

# def train_model():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     train_loader, val_loader, _, class_names = get_dataloaders()
#     num_classes = len(class_names)

#     model = models.resnet18(pretrained=True)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     model = model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.0001)
#     epochs = 10

#     best_acc = 0.0
#     train_losses, val_losses, val_accuracies = [], [], []

#     os.makedirs("models", exist_ok=True)

#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         train_losses.append(running_loss / len(train_loader))

#         # Validation
#         model.eval()
#         val_loss, correct = 0.0, 0
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#                 preds = torch.argmax(outputs, dim=1)
#                 correct += (preds == labels).sum().item()

#         val_accuracy = correct / len(val_loader.dataset)
#         val_losses.append(val_loss / len(val_loader))
#         val_accuracies.append(val_accuracy)

#         print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, "
#               f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracy:.4f}")

#         if val_accuracy > best_acc:
#             torch.save(model.state_dict(), "models/best_model.pth")
#             best_acc = val_accuracy

#     plot_curves(train_losses, val_losses, val_accuracies)

# if __name__ == "__main__":
#     train_model()
