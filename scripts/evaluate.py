import argparse
import torch
from torchvision import models, transforms
from dataset_loader import get_datasets, get_loaders
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: Apple MPS GPU")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

def build_eval_transform(img_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def evaluate(data_dir="dataset", checkpoint_path="models/best_model.pth", batch_size=32, img_size=224):
    # Prepare dataset and loader
    t_eval = build_eval_transform(img_size=img_size)
    train_ds, _, test_ds = get_datasets(data_dir, t_eval, t_eval)
    _, _, test_loader = get_loaders(train_ds, test_ds, test_ds, batch_size=batch_size)
    class_names = test_ds.classes
    num_classes = len(class_names)

    # Load model and weights
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Evaluate
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Results
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    print(f"\nTest Accuracy     : {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall   : {recall:.4f}")
    print(f"Weighted F1-score : {f1:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--checkpoint", type=str, default="models/efficientnet_best.pth")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    evaluate(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )


# import torch
# from torchvision import models
# from dataset_loader import get_dataloaders
# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt

# def evaluate_model():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     _, _, test_loader, class_names = get_dataloaders()

#     model = models.resnet18()
#     model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
#     model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
#     model.to(device)
#     model.eval()

#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     cm = confusion_matrix(all_labels, all_preds)
#     print("Confusion Matrix:\n", cm)
#     print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))

# if __name__ == "__main__":
#     evaluate_model()
