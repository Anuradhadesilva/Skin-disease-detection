# scripts/predict_lesion.py
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from combine_model import UNet, EfficientNetClassifier, CombinedModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models
unet_model = UNet()
unet_model.load_state_dict(torch.load("models/unet_best.pth", map_location=device))
unet_model.to(device)

classifier_model = EfficientNetClassifier(num_classes=5).to(device)

combined_model = CombinedModel(unet_model, classifier_model, freeze_unet=False).to(device)
combined_model.load_state_dict(torch.load("models/combined_model_best.pth", map_location=device))
combined_model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class_names = ['eczema', 'psoriasis', 'ringworm', 'normal', 'others']

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, mask_logits = combined_model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        pred_mask = torch.sigmoid(mask_logits).squeeze().cpu().numpy()

    pred_mask_bin = (pred_mask > 0.5).astype("uint8")


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[pred_class]} ({confidence:.2f})")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask_bin, cmap='gray')
    plt.title("Predicted Lesion Mask")
    plt.tight_layout()
    plt.show()


predict("/Users/desilvaanuradha/Documents/FYP/Skin-diseases-detection/test_images/249.jpg")
