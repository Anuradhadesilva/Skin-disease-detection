# scripts/combine_predict.py
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import os
from unet_model import UNet
from dataset_loader import get_datasets


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


def load_classifier(num_classes, checkpoint_path, device):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# seg
def load_segmentation_model(ckpt_path):
    model = UNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def predict_image(image_path, seg_model, clf_model, transform, class_names):
#   load images
    img = Image.open(image_path).convert("RGB")
    original_size = img.size


    seg_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = seg_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = seg_model(input_tensor)
        output = torch.sigmoid(output)
        output = output.squeeze().cpu().numpy()
    mask = (output > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).resize(original_size)


    red_overlay = ImageOps.colorize(mask_img.convert("L"), black="black", white="red")
    red_overlay.putalpha(100)
    highlighted = Image.alpha_composite(img.convert("RGBA"), red_overlay)


    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = clf_model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
        idx = probs.argmax()
    label, conf = class_names[idx], float(probs[idx])

    return highlighted.convert("RGB"), label, conf


def predict_uploaded_image(uploaded_file, unet_ckpt, clf_ckpt, data_dir, img_size=224):
    img = Image.open(uploaded_file).convert("RGB")

    # Transforms and classes
    transform = build_eval_transform(img_size=img_size)
    train_ds, _, _ = get_datasets(data_dir, transform, transform)
    class_names = train_ds.classes


    seg_model = load_segmentation_model(unet_ckpt)
    clf_model = load_classifier(len(class_names), clf_ckpt, device)


    temp_path = "temp.jpg"
    img.save(temp_path)
    highlighted_img, label, conf = predict_image(temp_path, seg_model, clf_model, transform, class_names)
    os.remove(temp_path)
    return highlighted_img, label, conf

if __name__ == "__main__":
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt

    ap = ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to image")
    ap.add_argument("--unet_ckpt", default="models/unet_best.pth")
    ap.add_argument("--clf_ckpt", default="models/efficientnet_best.pth")
    ap.add_argument("--data_dir", default="dataset")
    args = ap.parse_args()

    transform = build_eval_transform()
    train_ds, _, _ = get_datasets(args.data_dir, transform, transform)
    class_names = train_ds.classes

    seg_model = load_segmentation_model(args.unet_ckpt)
    clf_model = load_classifier(len(class_names), args.clf_ckpt, device)

    result_img, pred_label, conf = predict_image(args.image, seg_model, clf_model, transform, class_names)

    print(f"Prediction: {pred_label} ({conf:.2f})")
    plt.imshow(result_img)
    plt.title(f"{pred_label} ({conf:.2f})")
    plt.axis("off")
    plt.show()

# import os
# import torch
# import torch.nn.functional as F
# import numpy as np
# from PIL import Image, ImageOps
# import matplotlib.pyplot as plt
# from torchvision import transforms, models

# from unet_model import UNet
# from dataset_loader import get_datasets

# # ==== Setup Device ====
# device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # ==== Image Preprocessing ====
# def build_eval_transform(img_size=224):
#     return transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

# def load_classifier_model(num_classes, checkpoint_path):
#     model = models.efficientnet_b0(weights=None)
#     in_features = model.classifier[1].in_features
#     model.classifier[1] = torch.nn.Linear(in_features, num_classes)
#     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#     model.to(device)
#     model.eval()
#     return model

# def load_segmentation_model(checkpoint_path):
#     model = UNet().to(device)
#     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#     model.eval()
#     return model

# # ==== Prediction Function ====
# def predict_combined(image_path, unet_model, classifier_model, class_names, img_size=224, save_path=None):
#     # Load and preprocess
#     img = Image.open(image_path).convert("RGB")
#     original_size = img.size

#     transform_unet = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor()
#     ])
#     transform_classifier = build_eval_transform(img_size)

#     input_unet = transform_unet(img).unsqueeze(0).to(device)
#     input_cls = transform_classifier(img).unsqueeze(0).to(device)

#     # === Predict mask ===
#     with torch.no_grad():
#         mask_logits = unet_model(input_unet)
#         mask_prob = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
#         mask_bin = (mask_prob > 0.5).astype(np.uint8) * 255

#     mask_img = Image.fromarray(mask_bin).resize(original_size)
#     red_overlay = ImageOps.colorize(mask_img.convert("L"), black="black", white="red")
#     red_overlay.putalpha(100)

#     img_rgba = img.convert("RGBA")
#     highlighted = Image.alpha_composite(img_rgba, red_overlay)

#     # === Predict class ===
#     with torch.no_grad():
#         logits = classifier_model(input_cls)
#         probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
#         idx = probs.argmax()
#         label = class_names[idx]
#         confidence = float(probs[idx])

#     # === Display ===
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(img)
#     plt.title("Original Image")

#     plt.subplot(1, 2, 2)
#     plt.imshow(highlighted.convert("RGB"))
#     plt.title(f"Predicted: {label} ({confidence:.2f})")
#     plt.tight_layout()
#     plt.show()

#     # Save if needed
#     if save_path:
#         highlighted.convert("RGB").save(save_path)
#         print(f"Saved to: {save_path}")

#     return label, confidence

# # ==== Example Usage ====
# if __name__ == "__main__":
#     image_path = "/Users/desilvaanuradha/Documents/FYP/Ezcema-Detection/test_images/img_27.jpg"
#     save_path  = "/Users/desilvaanuradha/Documents/FYP/Ezcema-Detection/results/result_combined1.jpg"

#     # Load models
#     unet_model = load_segmentation_model("models/unet_best.pth")
#     transform_eval = build_eval_transform()
#     train_ds, _, _ = get_datasets("dataset", transform_eval, transform_eval)
#     class_names = train_ds.classes
#     classifier_model = load_classifier_model(len(class_names), "models/efficientnet_best.pth")

#     # Predict
#     label, confidence = predict_combined(image_path, unet_model, classifier_model, class_names, save_path=save_path)

