import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset_loader import ClassificationDataset
from combine_model import UNet, EfficientNetClassifier, CombinedModel

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: Apple MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    train_dataset = ClassificationDataset(root_dir="dataset/train", transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    unet_model = UNet()
    unet_model.load_state_dict(torch.load("models/unet_best.pth", map_location=device))
    unet_model.to(device)

    classifier_model = EfficientNetClassifier(num_classes=5).to(device)

    combined_model = CombinedModel(unet_model, classifier_model, freeze_unet=True).to(device)

    criterion_cls = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, combined_model.parameters()), lr=1e-4)

    for epoch in range(1, 11):
        combined_model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = combined_model(images)
            loss = criterion_cls(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} Loss: {total_loss / len(train_loader):.4f}")

    torch.save(combined_model.state_dict(), "models/combined_model_best.pth")

if __name__ == "__main__":
    main()
