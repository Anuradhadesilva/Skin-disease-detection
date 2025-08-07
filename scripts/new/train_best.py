import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset_loader import get_dataloaders
from utils import plot_curves
import os

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, class_names = get_dataloaders()
    num_classes = len(class_names)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 10

    best_acc = 0.0
    train_losses, val_losses, val_accuracies = [], [], []

    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()

        val_accuracy = correct / len(val_loader.dataset)
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracy:.4f}")

        if val_accuracy > best_acc:
            torch.save(model.state_dict(), "models/best_model.pth")
            best_acc = val_accuracy

    plot_curves(train_losses, val_losses, val_accuracies)

if __name__ == "__main__":
    train_model()
