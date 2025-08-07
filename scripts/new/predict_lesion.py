import torch
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import os
from unet_model import UNet

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: Apple MPS GPU")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

def predict_image(image_path, model, save_path=None):
    model.eval()

    # Load image and preprocess
    img = Image.open(image_path).convert("RGB")
    original_size = img.size

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Predict mask
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        output = output.squeeze().cpu().numpy()

    # Threshold
    mask = (output > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).resize(original_size)

    # Create red overlay
    red_overlay = ImageOps.colorize(mask_img.convert("L"), black="black", white="red")
    red_overlay.putalpha(100)  # semi-transparent

    # Combine original with overlay
    img = img.convert("RGBA")
    highlighted = Image.alpha_composite(img, red_overlay)

    # Show result
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img.convert("RGB"))
    plt.title("Original Image")
    plt.subplot(1,2,2)
    plt.imshow(highlighted.convert("RGB"))
    plt.title("Highlighted Disease Area")
    plt.tight_layout()
    plt.show()

    # Save if path given
    if save_path:
        highlighted.convert("RGB").save(save_path)
        print(f"Saved highlighted image to {save_path}")

# === Example usage ===
model = UNet().to(device)
model.load_state_dict(torch.load('/Users/desilvaanuradha/Documents/FYP/Ezcema-Detection/models/unet_best.pth', map_location=device))
model.to(device)

predict_image(
    image_path='/Users/desilvaanuradha/Documents/FYP/Ezcema-Detection/test_images/eczema21.jpg',
    model=model,
    save_path='/Users/desilvaanuradha/Documents/FYP/Ezcema-Detection/results/result1.jpg'
)