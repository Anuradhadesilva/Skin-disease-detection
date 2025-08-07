# scripts/predict.py
import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from dataset_loader import get_datasets

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: Apple MPS GPU")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

def build_eval_transform(img_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def load_model(num_classes, checkpoint_path, device):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(path, model, transform, class_names, device):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
        idx = probs.argmax()
    return class_names[idx], float(probs[idx])

def collect_image_paths(image_path=None, image_dir=None):
    paths = []
    if image_path:
        paths.append(image_path)
    if image_dir:
        for f in os.listdir(image_dir):
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
                paths.append(os.path.join(image_dir, f))
    return paths

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", type=str, help="Path to one image.")
    ap.add_argument("--image_dir",  type=str, help="Folder of images to predict.")
    ap.add_argument("--data_dir",   type=str, default="dataset")
    ap.add_argument("--checkpoint", type=str, default="models/efficientnet_best.pth")
    ap.add_argument("--img_size",   type=int, default=224)
    args = ap.parse_args()

    if not args.image_path and not args.image_dir:
        raise SystemExit("Please provide --image_path or --image_dir.")

    t_eval = build_eval_transform(img_size=args.img_size)
    train_ds, _, _ = get_datasets(args.data_dir, t_eval, t_eval)
    class_names = train_ds.classes

    model = load_model(len(class_names), args.checkpoint, device)
    transform = build_eval_transform(img_size=args.img_size)

    paths = collect_image_paths(args.image_path, args.image_dir)
    for p in paths:
        try:
            label, conf = predict_image(p, model, transform, class_names, device)
            print(f"{p} -> {label} ({conf:.4f})")
        except Exception as e:
            print(f"{p} -> ERROR: {e}")

def predict_from_uploaded_image(uploaded_file, checkpoint_path="models/efficientnet_best.pth", data_dir="dataset", img_size=224):
    transform = build_eval_transform(img_size=img_size)
    train_ds, _, _ = get_datasets(data_dir, transform, transform)
    class_names = train_ds.classes

    model = load_model(len(class_names), checkpoint_path, device)

    img = Image.open(uploaded_file).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
        idx = probs.argmax()

    return class_names[idx], float(probs[idx])


            

# import torch
# from torchvision import models, transforms
# from PIL import Image
# import argparse
# from dataset_loader import get_dataloaders

# def predict(image_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     _, _, _, class_names = get_dataloaders()

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     img = Image.open(image_path).convert('RGB')
#     img = transform(img).unsqueeze(0).to(device)

#     model = models.resnet18()
#     model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
#     model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
#     model.to(device)
#     model.eval()

#     with torch.no_grad():
#         outputs = model(img)
#         _, predicted = torch.max(outputs, 1)

#     print(f"Predicted class: {class_names[predicted.item()]}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image_path", type=str, required=True, help="Path to the image")
#     args = parser.parse_args()
#     predict(args.image_path)

# import sys
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model

# def preprocess_image(image_path, target_size=(256, 256)):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, target_size)
#     return img / 255.0

# unet_model = load_model("models/eczema_unet.h5")
# clf_model = load_model("models/eczema_classifier.h5")

# def predict(image_path):
#     img = preprocess_image(image_path)
#     mask = unet_model.predict(np.expand_dims(img, 0))[0]
#     masked_img = img * mask
#     prediction = clf_model.predict(np.expand_dims(masked_img, 0))[0][0]
#     return "Eczema" if prediction > 0.5 else "No Eczema"

# if __name__ == "__main__":
#     img_path = sys.argv[1]
#     result = predict(img_path)
#     print(f"Prediction for '{img_path}': {result}")
